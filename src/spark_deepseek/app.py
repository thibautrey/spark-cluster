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
    ModelRuntime,
    NodeSnapshot,
    RuntimeEndpoint,
)
from .prometheus import render_metrics
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

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        await model_manager.start()
        yield
        await model_manager.stop()
        await runtime.stop()

    app = FastAPI(title=settings.title, version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.model_manager = model_manager
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

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/healthz")
    async def health() -> dict:
        return {"status": "ok", "mode": settings.mode, "has_snapshot": runtime.latest is not None}

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

    return app


app = create_app()
