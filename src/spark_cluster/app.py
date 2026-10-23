import asyncio
import contextlib
import json
import shutil
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .capacity import calculate_capacity
from .catalog import (
    CatalogDetail,
    CatalogFitBatchRequest,
    CatalogFitBatchResponse,
    CatalogPage,
    CatalogSort,
    ModelBrowser,
)
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
    ModelDeletionBlocked,
    ModelDoesNotFit,
    ModelManager,
)
from .model_relay import ModelRelay
from .models import (
    CapacityPlan,
    CapacityRequest,
    ClusterSnapshot,
    ControllerModelImportRequest,
    DownloadJob,
    LocalModel,
    ModelDownloadRequest,
    ModelFitPlan,
    ModelFitRequest,
    ModelLaunchJob,
    ModelLaunchRequest,
    ModelMutationResult,
    ModelRuntime,
    NodeSnapshot,
    RuntimeEndpoint,
    RuntimeInstallJob,
    RuntimeLog,
    RuntimeReadiness,
    RuntimeReadinessRequest,
    RuntimeTerminalRequest,
    RuntimeTerminalTicket,
)
from .performance import PerformancePlan, PerformancePlanRequest, build_performance_plan
from .prometheus import render_metrics
from .remote_cluster import (
    RemoteClusterConnector,
    RemoteClusterProxyMiddleware,
    RemoteClusterUnavailable,
    load_active_connection,
    valid_controller_signature,
    validated_cluster_url,
)
from .runtime_prerequisites import RuntimePrerequisiteService
from .setup_models import ClusterConnectRequest, DiscoveryRequest, SetupRequest
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
    if settings.remote_cluster_url:
        settings.remote_cluster_url = validated_cluster_url(settings.remote_cluster_url)
    else:
        if active_connection := load_active_connection(settings):
            settings.remote_cluster_url = active_connection["url"]
    runtime = Runtime(settings)
    model_relay = ModelRelay(settings, runtime.store)
    model_manager = ModelManager(settings, runtime.store, runtime, model_relay)
    browser = ModelBrowser(settings)
    log_service = RuntimeLogService(settings, runtime.store)
    deployment_service = RuntimeDeploymentService(
        settings, runtime.store, model_manager, log_service
    )
    setup_service = ClusterSetupService(settings, runtime.store)
    runtime_prerequisites = RuntimePrerequisiteService(settings, runtime.store, setup_service)
    cluster_connector = RemoteClusterConnector(settings)

    async def activate_cluster(job) -> None:
        head = next(node for node in job.request.nodes if node.role == "head")
        await cluster_connector.attach(head.address, 30080)

    setup_service.on_cluster_ready = activate_cluster

    async def refresh_remote_capacity() -> None:
        if settings.remote_cluster_url:
            runtime.latest = await cluster_connector.snapshot()

    async def model_fit_plan(request: ModelFitRequest) -> ModelFitPlan:
        await refresh_remote_capacity()
        plan = await model_manager.fit(request)
        if not settings.remote_cluster_url:
            return plan
        cluster_available = await cluster_connector.model_storage_available()
        available_gib = min(plan.storage_available_gib, cluster_available / 2**30)
        storage_fits = plan.storage_fits and plan.repository_bytes <= cluster_available
        runtime_storage_bytes = plan.repository_bytes + plan.additional_artifacts_bytes
        runtime_storage_fits = (
            plan.runtime_storage_fits and runtime_storage_bytes <= cluster_available
        )
        notes = list(plan.notes)
        notes.append(
            "Storage preflight uses the lower free-space value from the controller and cluster."
        )
        return plan.model_copy(
            update={
                "storage_available_gib": available_gib,
                "storage_fits": storage_fits,
                "download_fits": storage_fits,
                "runtime_storage_fits": runtime_storage_fits,
                "fits": plan.runtime_fits and plan.launch_supported and runtime_storage_fits,
                "fit_status": ("insufficient-storage" if not storage_fits else plan.fit_status),
                "notes": notes,
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        await model_manager.start()
        await log_service.start()
        await deployment_service.start()
        await setup_service.start()
        yield
        await setup_service.stop()
        await runtime_prerequisites.stop()
        await deployment_service.stop()
        await log_service.stop()
        await model_manager.stop()
        browser.close()
        await runtime.stop()

    app = FastAPI(title=settings.title, version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.model_manager = model_manager
    app.state.model_browser = browser
    app.state.model_relay = model_relay
    app.state.log_service = log_service
    app.state.deployment_service = deployment_service
    app.state.setup_service = setup_service
    app.state.runtime_prerequisites = runtime_prerequisites
    app.state.settings = settings
    app.state.cluster_connector = cluster_connector
    app.add_middleware(RemoteClusterProxyMiddleware, settings=settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    static_dir = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        expected = settings.ingest_token
        if expected and authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid ingest token")

    async def require_model_token(
        request: Request,
        authorization: str | None = Header(default=None),
        x_spark_timestamp: str | None = Header(default=None),
        x_spark_signature: str | None = Header(default=None),
    ) -> None:
        expected = settings.model_admin_token
        if not expected:
            return
        if authorization == f"Bearer {expected}":
            return
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        if valid_controller_signature(
            expected,
            request.method,
            target,
            x_spark_timestamp,
            x_spark_signature,
            await request.body(),
        ):
            return
        raise HTTPException(status_code=401, detail="Invalid model administration token")

    @app.exception_handler(SetupFailure)
    async def setup_failure(_, error: SetupFailure):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(RemoteClusterUnavailable)
    async def remote_cluster_unavailable(_, error: RemoteClusterUnavailable):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.get("/api/v1/setup", dependencies=[Depends(require_model_token)])
    async def setup_state():
        return await setup_service.state()

    @app.post("/api/v1/setup/connect", dependencies=[Depends(require_model_token)])
    async def connect_cluster(request: ClusterConnectRequest):
        return await cluster_connector.attach(request.address, request.port)

    @app.get("/api/v1/controller/verify", dependencies=[Depends(require_model_token)])
    async def verify_controller():
        return {"status": "ready", "cluster_name": settings.cluster_name}

    @app.post("/api/v1/setup/discover", dependencies=[Depends(require_model_token)])
    async def discover_nodes(request: DiscoveryRequest):
        return await setup_service.discover(request)

    @app.post("/api/v1/setup/check", dependencies=[Depends(require_model_token)])
    async def setup_check(request: SetupRequest):
        return await setup_service.inspect_hardware(request)

    @app.post("/api/v1/setup/hardware", dependencies=[Depends(require_model_token)])
    async def setup_hardware(request: SetupRequest):
        return await setup_service.inspect_hardware(request)

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

    @app.post("/api/v1/performance/plan", response_model=PerformancePlan)
    async def performance_plan(request: PerformancePlanRequest) -> PerformancePlan:
        available_nodes = (
            len(runtime.latest.nodes)
            if runtime.latest and runtime.latest.nodes
            else settings.fallback_node_count
        )
        if request.topology == "auto":
            setup_jobs = await asyncio.to_thread(runtime.store.list_setup_jobs)
            configured = next((job for job in setup_jobs if job.status == "ready"), None)
            if configured:
                request = request.model_copy(update={"topology": configured.request.topology})
        return build_performance_plan(request, available_nodes)

    @app.get("/api/v1/runtime", response_model=RuntimeEndpoint)
    async def runtime_endpoint() -> RuntimeEndpoint:
        return await model_manager.runtime_endpoint()

    async def catalog_result(operation):
        try:
            return await operation
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail="Invalid model repository or metadata"
            ) from error
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            message = {
                401: "Authentication required. Set SPARK_CLUSTER_HF_TOKEN on the controller.",
                403: (
                    "Accept the model license on Hugging Face and configure "
                    "SPARK_CLUSTER_HF_TOKEN on the controller."
                ),
                404: "Model or revision not found.",
                429: "Hugging Face rate limit reached. Please try again shortly.",
            }.get(status, "Hugging Face is unavailable. Please retry.")
            raise HTTPException(
                status_code=status if status in (401, 403, 404, 429) else 502, detail=message
            ) from error
        except (httpx.HTTPError, KeyError, TypeError) as error:
            raise HTTPException(
                status_code=502, detail="Cannot reach the Hugging Face catalog. Retry shortly."
            ) from error

    @app.get("/api/v1/catalog/models", response_model=CatalogPage)
    async def catalog_models(
        query: str = Query("", max_length=200),
        sort: CatalogSort = "trendingScore",
        task: str = Query("", max_length=80),
        library: str = Query("", max_length=80),
        access: Literal["", "open", "gated"] = "",
        limit: int = Query(24, ge=1, le=48),
        cursor: str | None = Query(None, max_length=4000),
    ) -> CatalogPage:
        return await catalog_result(
            browser.search(query, sort, task, library, access, limit, cursor)
        )

    @app.post("/api/v1/catalog/fit", response_model=CatalogFitBatchResponse)
    async def catalog_fit(request: CatalogFitBatchRequest) -> CatalogFitBatchResponse:
        return await browser.fit_batch(request)

    @app.get("/api/v1/catalog/models/{repo_id:path}", response_model=CatalogDetail)
    async def catalog_model(repo_id: str, revision: str = Query("main", max_length=200)):
        return await catalog_result(browser.detail(repo_id, revision))

    @app.get("/api/v1/models", response_model=list[LocalModel])
    async def local_models() -> list[LocalModel]:
        return await model_manager.local_models()

    @app.post("/api/v1/models/fit", response_model=ModelFitPlan)
    async def model_fit(request: ModelFitRequest) -> ModelFitPlan:
        try:
            return await model_fit_plan(request)
        except RemoteClusterUnavailable:
            raise
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
    async def download_model(request: ModelDownloadRequest, http_request: Request) -> DownloadJob:
        try:
            fit_plan = await model_fit_plan(request)
            destination = "cluster" if settings.remote_cluster_url else "local"
            controller_url = (
                model_relay.controller_url(str(http_request.base_url).rstrip("/"))
                if destination == "cluster"
                else None
            )
            return await model_manager.enqueue(
                request,
                destination=destination,
                controller_url=controller_url,
                fit_plan=fit_plan,
            )
        except RemoteClusterUnavailable:
            raise
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

    @app.post(
        "/api/v1/models/import",
        response_model=DownloadJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def import_model(request: ControllerModelImportRequest) -> DownloadJob:
        try:
            validated_cluster_url(request.controller_url)
            return await model_manager.enqueue_from_controller(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get(
        "/api/v1/models/storage",
        dependencies=[Depends(require_model_token)],
    )
    async def model_storage() -> dict[str, int]:
        return {"available_bytes": shutil.disk_usage(settings.model_cache_path).free}

    @app.get("/api/v1/downloads", response_model=list[DownloadJob])
    async def downloads() -> list[DownloadJob]:
        return await model_manager.downloads()

    @app.get(
        "/api/v1/downloads/{job_id}",
        response_model=DownloadJob,
        dependencies=[Depends(require_model_token)],
    )
    async def download(job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(runtime.store.get_download, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Download job not found")
        return job

    @app.get(
        "/api/v1/controller/downloads/{job_id}/manifest",
        dependencies=[Depends(require_model_token)],
    )
    async def controller_model_manifest(job_id: str) -> dict:
        try:
            return model_relay.manifest(job_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Model transfer not found") from error

    @app.get(
        "/api/v1/controller/downloads/{job_id}/files/{file_path:path}",
        dependencies=[Depends(require_model_token)],
    )
    async def controller_model_file(job_id: str, file_path: str) -> FileResponse:
        try:
            path = model_relay.model_file(job_id, file_path)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Model transfer file not found") from error
        return FileResponse(path, filename=path.name)

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

    async def download_action(job_id: str, action: str) -> DownloadJob:
        try:
            handler = {
                "pause": model_manager.pause,
                "resume": model_manager.resume,
                "cancel": model_manager.cancel,
            }[action]
            return await handler(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Download job not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502,
                detail=f"Cluster download action failed: {model_manager.safe_error(error)}",
            ) from error

    @app.post(
        "/api/v1/downloads/{job_id}/pause",
        response_model=DownloadJob,
        dependencies=[Depends(require_model_token)],
    )
    async def pause_download(job_id: str) -> DownloadJob:
        return await download_action(job_id, "pause")

    @app.post(
        "/api/v1/downloads/{job_id}/resume",
        response_model=DownloadJob,
        dependencies=[Depends(require_model_token)],
    )
    async def resume_download(job_id: str) -> DownloadJob:
        return await download_action(job_id, "resume")

    @app.post(
        "/api/v1/downloads/{job_id}/cancel",
        response_model=DownloadJob,
        dependencies=[Depends(require_model_token)],
    )
    async def cancel_download(job_id: str) -> DownloadJob:
        return await download_action(job_id, "cancel")

    @app.delete(
        "/api/v1/model-management/models/{repo_id:path}",
        response_model=ModelMutationResult,
        dependencies=[Depends(require_model_token)],
    )
    async def delete_model(repo_id: str) -> ModelMutationResult:
        try:
            return await model_manager.delete_model(repo_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Managed model not found") from error
        except ModelDeletionBlocked as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ValueError, httpx.HTTPError) as error:
            raise HTTPException(
                status_code=502,
                detail=f"Cluster model deletion failed: {model_manager.safe_error(error)}",
            ) from error

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

    @app.post(
        "/api/v1/runtime-management/readiness",
        response_model=RuntimeReadiness,
        dependencies=[Depends(require_model_token)],
    )
    async def runtime_readiness(request: RuntimeReadinessRequest) -> RuntimeReadiness:
        return await runtime_prerequisites.inspect(request.model_id)

    @app.post(
        "/api/v1/runtime-management/install",
        response_model=RuntimeInstallJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def install_runtime(request: RuntimeReadinessRequest) -> RuntimeInstallJob:
        try:
            return await runtime_prerequisites.install(request.model_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get(
        "/api/v1/runtime-management/install/{job_id}",
        response_model=RuntimeInstallJob,
        dependencies=[Depends(require_model_token)],
    )
    async def runtime_install(job_id: str) -> RuntimeInstallJob:
        try:
            return await runtime_prerequisites.get_install(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Runtime installation not found") from error

    @app.post(
        "/api/v1/runtime-management/terminal-ticket",
        response_model=RuntimeTerminalTicket,
        dependencies=[Depends(require_model_token)],
    )
    async def terminal_ticket(request: RuntimeTerminalRequest) -> RuntimeTerminalTicket:
        try:
            return runtime_prerequisites.issue_terminal_ticket(request.address)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Managed cluster node not found") from error

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

    @app.websocket("/api/v1/runtime-management/terminal/{ticket}")
    async def runtime_terminal(websocket: WebSocket, ticket: str) -> None:
        try:
            terminal = await runtime_prerequisites.terminal_process(ticket)
        except (KeyError, AttributeError):
            await websocket.accept()
            await websocket.close(code=4403, reason="Terminal ticket is invalid or expired")
            return
        await websocket.accept()

        async def send_output() -> None:
            while terminal.stdout and (chunk := await terminal.stdout.read(4096)):
                await websocket.send_json(
                    {"type": "output", "data": chunk.decode(errors="replace")}
                )

        async def receive_input() -> None:
            while True:
                payload = json.loads(await websocket.receive_text())
                if payload.get("type") != "input":
                    continue
                data = str(payload.get("data", ""))[:4096]
                if terminal.stdin:
                    terminal.stdin.write(data.encode())
                    await terminal.stdin.drain()

        tasks = {asyncio.create_task(send_output()), asyncio.create_task(receive_input())}
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                with contextlib.suppress(WebSocketDisconnect, json.JSONDecodeError):
                    task.result()
            for task in pending:
                task.cancel()
        finally:
            for task in tasks:
                task.cancel()
            if terminal.returncode is None:
                terminal.terminate()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(terminal.wait(), timeout=3)
                if terminal.returncode is None:
                    terminal.kill()
                    await terminal.wait()

    return app


app = create_app()
