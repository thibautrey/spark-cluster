import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .capacity import calculate_capacity
from .cluster_control import (
    InvalidParallelism,
    ModelNotManaged,
    RuntimeDeploymentService,
    RuntimeLogService,
)
from .cluster_setup import ClusterSetupService, SetupFailure
from .config import Settings
from .model_manager import (
    ModelAlreadyInstalled,
    ModelDoesNotFit,
    ModelManager,
)
from .models import (
    CapacityPlan,
    CapacityRequest,
    ClusterSnapshot,
    DownloadJob,
    LocalModel,
    ModelDownloadRequest,
    ModelFitPlan,
    ModelFitRequest,
    ModelLaunchJob,
    ModelLaunchRequest,
    ModelRuntime,
    NodeSnapshot,
    RuntimeEndpoint,
    RuntimeLog,
)
from .prometheus import render_metrics
from .setup_models import DiscoveryRequest, SetupRequest
from .simulator import ClusterSimulator
from .store import SnapshotStore


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = SnapshotStore(settings.database_path)
        self.simulator = ClusterSimulator(settings.cluster_name)
        self.latest = self.store.latest()
        self.subscribers: set[asyncio.Queue[ClusterSnapshot]] = set()
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.settings.mode == "demo":
            self.task = asyncio.create_task(self._simulation_loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        self.store.close()

    async def publish(self, snapshot: ClusterSnapshot) -> None:
        self.latest = snapshot
        await asyncio.to_thread(self.store.append, snapshot)
        for queue in tuple(self.subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(snapshot)

    async def _simulation_loop(self) -> None:
        while True:
            await self.publish(self.simulator.snapshot())
            if self.simulator.tick % 120 == 0:
                await asyncio.to_thread(self.store.prune, self.settings.history_retention_hours)
            await asyncio.sleep(self.settings.sample_interval_seconds)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    runtime = Runtime(settings)
    model_manager = ModelManager(settings, runtime.store, runtime)
    log_service = RuntimeLogService(settings, runtime.store)
    deployment_service = RuntimeDeploymentService(
        settings, runtime.store, model_manager, log_service
    )
    setup_service = ClusterSetupService(settings, runtime.store)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        await model_manager.start()
        await log_service.start()
        await deployment_service.start()
        await setup_service.start()
        yield
        await setup_service.stop()
        await deployment_service.stop()
        await log_service.stop()
        await model_manager.stop()
        await runtime.stop()

    app = FastAPI(title=settings.title, version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.model_manager = model_manager
    app.state.log_service = log_service
    app.state.deployment_service = deployment_service
    app.state.setup_service = setup_service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    static_dir = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        expected = settings.ingest_token
        if expected and authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid ingest token")

    def require_model_token(authorization: str | None = Header(default=None)) -> None:
        expected = settings.model_admin_token
        if expected and authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid model administration token")

    @app.exception_handler(SetupFailure)
    async def setup_failure(_, error: SetupFailure):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.get("/api/v1/setup", dependencies=[Depends(require_model_token)])
    async def setup_state():
        return await setup_service.state()

    @app.post("/api/v1/setup/discover", dependencies=[Depends(require_model_token)])
    async def discover_nodes(request: DiscoveryRequest):
        return await setup_service.discover(request)

    @app.post("/api/v1/setup/check", dependencies=[Depends(require_model_token)])
    async def setup_check(request: SetupRequest):
        checks = await setup_service.preflight(request)
        return {"checks": checks, "nodes": request.nodes}

    @app.post("/api/v1/setup/jobs", status_code=202, dependencies=[Depends(require_model_token)])
    async def start_setup(request: SetupRequest):
        return await setup_service.launch(request)

    @app.post(
        "/api/v1/setup/jobs/{job_id}/resume",
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def resume_setup(job_id: str):
        try:
            return await setup_service.resume(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Setup job not found") from error

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/healthz")
    async def health() -> dict:
        return {
            "status": "ok",
            "mode": settings.mode,
            "logs_mode": log_service.mode,
            "runtime_control_mode": deployment_service.mode,
            "has_snapshot": runtime.latest is not None,
        }

    @app.get("/api/v1/snapshot", response_model=ClusterSnapshot)
    async def snapshot() -> ClusterSnapshot:
        if runtime.latest is None:
            raise HTTPException(status_code=503, detail="Waiting for the first cluster sample")
        return runtime.latest

    @app.get("/api/v1/history")
    async def history(minutes: int = Query(default=30, ge=1, le=1440)) -> list[dict]:
        return await asyncio.to_thread(runtime.store.history, minutes)

    @app.post(
        "/api/v1/snapshot", response_model=ClusterSnapshot, dependencies=[Depends(require_token)]
    )
    async def ingest(snapshot: ClusterSnapshot) -> ClusterSnapshot:
        await runtime.publish(snapshot)
        return snapshot

    @app.post(
        "/api/v1/nodes", response_model=ClusterSnapshot, dependencies=[Depends(require_token)]
    )
    async def ingest_node(node: NodeSnapshot) -> ClusterSnapshot:
        if runtime.latest is None:
            updated = ClusterSnapshot(
                cluster_name=settings.cluster_name,
                nodes=[node],
                cache_tiers=[],
                runtime=ModelRuntime(
                    model=settings.runtime_model,
                    quantization=settings.runtime_quantization,
                    context_length=settings.runtime_context_length,
                    tensor_parallel=settings.runtime_tensor_parallel,
                    pipeline_parallel=settings.runtime_pipeline_parallel,
                ),
                total_input_tokens_per_second=node.input_tokens_per_second,
                total_output_tokens_per_second=node.output_tokens_per_second,
            )
        else:
            nodes = [current for current in runtime.latest.nodes if current.id != node.id] + [node]
            updated = runtime.latest.model_copy(
                update={
                    "timestamp": datetime.now(UTC),
                    "nodes": nodes,
                    "total_input_tokens_per_second": sum(
                        current.input_tokens_per_second for current in nodes
                    ),
                    "total_output_tokens_per_second": sum(
                        current.output_tokens_per_second for current in nodes
                    ),
                }
            )
        await runtime.publish(updated)
        return updated

    @app.post("/api/v1/capacity", response_model=CapacityPlan)
    async def capacity(request: CapacityRequest) -> CapacityPlan:
        return calculate_capacity(request)

    @app.get("/api/v1/runtime", response_model=RuntimeEndpoint)
    async def runtime_endpoint() -> RuntimeEndpoint:
        return await model_manager.runtime_endpoint()

    @app.get("/api/v1/models", response_model=list[LocalModel])
    async def local_models() -> list[LocalModel]:
        return await model_manager.local_models()

    @app.post("/api/v1/models/fit", response_model=ModelFitPlan)
    async def model_fit(request: ModelFitRequest) -> ModelFitPlan:
        try:
            return await model_manager.fit(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail=f"Hugging Face metadata lookup failed: {model_manager.safe_error(error)}",
            ) from error

    @app.post("/api/v1/models/fit-local", response_model=ModelFitPlan)
    async def local_model_fit(request: ModelFitRequest) -> ModelFitPlan:
        try:
            return await model_manager.fit_for_local(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail=f"Local model metadata is unavailable: {model_manager.safe_error(error)}",
            ) from error

    @app.post(
        "/api/v1/models/download",
        response_model=DownloadJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def download_model(request: ModelDownloadRequest) -> DownloadJob:
        try:
            return await model_manager.enqueue(request)
        except ModelDoesNotFit as error:
            raise HTTPException(
                status_code=409,
                detail={"message": str(error), "fit": error.fit.model_dump(mode="json")},
            ) from error
        except ModelAlreadyInstalled as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail=f"Hugging Face metadata lookup failed: {model_manager.safe_error(error)}",
            ) from error

    @app.get("/api/v1/downloads", response_model=list[DownloadJob])
    async def downloads() -> list[DownloadJob]:
        return await model_manager.downloads()

    @app.post(
        "/api/v1/downloads/{job_id}/retry",
        response_model=DownloadJob,
        dependencies=[Depends(require_model_token)],
    )
    async def retry_download(job_id: str) -> DownloadJob:
        try:
            return await model_manager.retry(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Download job not found") from error

    @app.get("/api/v1/logs", response_model=list[RuntimeLog])
    async def runtime_logs(
        limit: int = Query(default=500, ge=1, le=5000),
        pod: str | None = None,
        level: str | None = None,
    ) -> list[RuntimeLog]:
        return await log_service.history(limit=limit, pod=pod, level=level)

    @app.get("/api/v1/logs/instances", response_model=list[str])
    async def log_instances() -> list[str]:
        return await log_service.instances()

    @app.get("/api/v1/launches", response_model=list[ModelLaunchJob])
    async def launches() -> list[ModelLaunchJob]:
        return await deployment_service.launches()

    @app.get("/api/v1/control")
    async def control_state() -> dict:
        endpoint, models, downloads_value, launches_value = await asyncio.gather(
            model_manager.runtime_endpoint(),
            model_manager.local_models(),
            model_manager.downloads(),
            deployment_service.launches(),
        )
        return {
            "runtime": endpoint,
            "models": models,
            "downloads": downloads_value,
            "launches": launches_value,
            "logs_mode": log_service.mode,
            "runtime_control_mode": deployment_service.mode,
        }

    @app.post(
        "/api/v1/launches",
        response_model=ModelLaunchJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def launch_model(request: ModelLaunchRequest) -> ModelLaunchJob:
        try:
            return await deployment_service.launch(request)
        except ModelNotManaged as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ModelDoesNotFit as error:
            raise HTTPException(
                status_code=409,
                detail={"message": str(error), "fit": error.fit.model_dump(mode="json")},
            ) from error
        except InvalidParallelism as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/api/v1/launches/{job_id}/retry",
        response_model=ModelLaunchJob,
        dependencies=[Depends(require_model_token)],
    )
    async def retry_launch(job_id: str) -> ModelLaunchJob:
        try:
            return await deployment_service.retry(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Launch job not found") from error

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        if runtime.latest is None:
            return ""
        return render_metrics(runtime.latest)

    @app.websocket("/api/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[ClusterSnapshot] = asyncio.Queue(maxsize=1)
        runtime.subscribers.add(queue)
        try:
            if runtime.latest:
                await websocket.send_json(runtime.latest.model_dump(mode="json"))
            while True:
                value = await queue.get()
                await websocket.send_json(value.model_dump(mode="json"))
        except WebSocketDisconnect:
            pass
        finally:
            runtime.subscribers.discard(queue)

    @app.websocket("/api/v1/logs/stream")
    async def stream_logs(websocket: WebSocket) -> None:
        await websocket.accept()
        queue: asyncio.Queue[RuntimeLog] = asyncio.Queue(maxsize=200)
        log_service.subscribers.add(queue)
        try:
            while True:
                entry = await queue.get()
                await websocket.send_json(entry.model_dump(mode="json"))
        except WebSocketDisconnect:
            pass
        finally:
            log_service.subscribers.discard(queue)

    return app


app = create_app()
