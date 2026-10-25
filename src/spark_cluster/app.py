import asyncio
import contextlib
import hmac
import json
import logging
import re
import shutil
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs

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
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles

from . import hf_credentials
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
    EngineNotExecutable,
    InvalidParallelism,
    ModelNotManaged,
    RuntimeDeploymentService,
    RuntimeLogService,
)
from .cluster_registry import (
    ClusterRegistry,
    LocalClusterRequest,
    RemoteClusterRequest,
)
from .cluster_setup import ClusterSetupService, SetupFailure
from .config import Settings
from .inference_tracking import InferenceTracker
from .inference_http import RequestTooLarge, inference_error, read_inference_body
from .local_docker import DockerEngineClient, detect_nvidia_gpus, local_gpu_snapshot
from .mcp_server import SparkClusterMCP
from .model_manager import (
    ModelAlreadyInstalled,
    ModelDeletionBlocked,
    ModelDoesNotFit,
    ModelManager,
)
from .model_pool import (
    ModelPool,
    ModelPreferences,
    ModelRetentionRequest,
    NodeLoadRequest,
    PoolLoadRequest,
    PoolNode,
    PoolResponse,
    PoolScaleRequest,
)
from .model_relay import ModelRelay
from .models import (
    BenchmarkRequest,
    BenchmarkResult,
    CachePrefixRecord,
    CachePrefixRequest,
    CapacityPlan,
    CapacityRequest,
    ClusterSnapshot,
    ControllerModelImportRequest,
    DownloadJob,
    InferenceTelemetry,
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
    RuntimeEngineCollection,
    RuntimeGuide,
    RuntimeInstallJob,
    RuntimeLog,
    RuntimePrepareRequest,
    RuntimeReadiness,
    RuntimeReadinessRequest,
    RuntimeTerminalRequest,
    RuntimeTerminalTicket,
    RuntimeWorkflowJob,
    RuntimeWorkflowRequest,
)
from .on_demand import OnDemandInference, load_profiles
from .optimization_lab import OptimizationLab
from .performance import PerformancePlan, PerformancePlanRequest, build_performance_plan
from .performance_history import RUNTIME_FIELDS
from .performance_history import scopes as performance_scopes
from .prometheus import render_metrics
from .remote_cluster import (
    RemoteClusterConnector,
    RemoteClusterProxyMiddleware,
    RemoteClusterUnavailable,
    valid_controller_signature,
    validated_cluster_url,
)
from .runtime_library import RuntimeHostRequest, RuntimeLibrary
from .runtime_library_api import register_runtime_library
from .runtime_prerequisites import RuntimePrerequisiteService
from .runtime_workflows import RuntimeWorkflowService
from .setup_models import (
    BootstrapConfirmRequest,
    BootstrapCreateRequest,
    ClusterConnectRequest,
    DiscoveryRequest,
    SetupRequest,
)
from .simulator import ClusterSimulator
from .store import SnapshotStore
from .telemetry import InferenceTelemetryService


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
    hf_credentials.load_token(settings)
    if settings.remote_cluster_url:
        settings.remote_cluster_url = validated_cluster_url(settings.remote_cluster_url)
    cluster_registry = ClusterRegistry(settings)
    runtime = Runtime(settings)
    model_relay = ModelRelay(settings, runtime.store)
    model_manager = ModelManager(settings, runtime.store, runtime, model_relay)
    browser = ModelBrowser(settings)
    docker_engine = DockerEngineClient(settings)
    log_service = RuntimeLogService(settings, runtime.store, docker_engine)
    deployment_service = RuntimeDeploymentService(
        settings,
        runtime.store,
        model_manager,
        log_service,
        registry=cluster_registry,
        docker=docker_engine,
    )
    on_demand = OnDemandInference(
        docker_engine,
        load_profiles(settings.on_demand_profiles_path) if settings.on_demand_profiles_path else [],
        state_path=settings.setup_state_path / "on-demand-state.json",
    )
    pool_node = PoolNode(settings, cluster_registry, model_manager, deployment_service, on_demand)
    model_pool = ModelPool(settings, cluster_registry, pool_node)
    on_demand.pool = model_pool
    setup_service = ClusterSetupService(settings, runtime.store)
    runtime_prerequisites = RuntimePrerequisiteService(
        settings,
        runtime.store,
        setup_service,
        registry=cluster_registry,
        docker=docker_engine,
    )
    runtime_library = RuntimeLibrary(
        settings, runtime.store, runtime_prerequisites, cluster_registry, docker_engine
    )
    runtime_prerequisites.attach_library(runtime_library)
    deployment_service.attach_prerequisites(runtime_prerequisites)
    runtime_workflows = RuntimeWorkflowService(
        settings,
        runtime.store,
        cluster_registry,
        runtime_prerequisites,
        runtime_library,
        deployment_service,
        model_pool,
    )
    inference_tracker = InferenceTracker(settings)
    model_pool.tracker = pool_node.tracker = inference_tracker

    def resource_sample(cluster_id):
        cache_age = time.monotonic() - model_pool.cache_time
        if cluster_id and model_pool.cache and cache_age < 15:
            pool = next((p for p in model_pool.cache["clusters"] if p["id"] == cluster_id), None)
            machines = pool.get("machines", []) if pool and pool["available"] else []
            if machines and all(m.get("available", True) for m in machines):
                return dict(
                    sampled_at=time.time() - cache_age,
                    used_gib=round(sum(m["memory_used_gib"] for m in machines), 3),
                    total_gib=round(sum(m["memory_total_gib"] for m in machines), 3),
                )
        snapshot = runtime.latest
        active = cluster_registry.active()
        if not snapshot or (cluster_id and active and cluster_id != active.id):
            return None
        age = (datetime.now(UTC) - snapshot.timestamp).total_seconds()
        if (
            age > max(15, settings.sample_interval_seconds * 3)
            or not snapshot.nodes
            or any(n.status == "offline" for n in snapshot.nodes)
        ):
            return None
        return dict(
            sampled_at=snapshot.timestamp.timestamp(),
            used_gib=round(sum(n.memory.used_gib for n in snapshot.nodes), 3),
            total_gib=round(sum(n.memory.total_gib for n in snapshot.nodes), 3),
            gpu_utilization=round(
                sum(n.gpu_utilization for n in snapshot.nodes) / max(1, len(snapshot.nodes)), 2
            ),
        )

    def pool_usage():
        cached = model_pool.cache
        if not cached:
            return []
        return [
            dict(
                id=c["id"],
                name=c["name"],
                available=c["available"],
                active_model=c.get("active_model"),
                resident_models=c.get("resident_models", []),
                busy=c.get("busy", False),
                machines=c.get("machines", []),
                age_seconds=round(max(0, time.monotonic() - model_pool.cache_time), 1),
            )
            for c in cached["clusters"]
        ]

    def metrics_target():
        # The physical engine owns counters, not the legacy listener or fixed default port.
        ready = [r for r in on_demand.residents.values() if r.status == "ready"]
        if ready:
            return {r.profile.id: r.profile.upstream + "/metrics" for r in ready}
        if pool_node.native.get("model_id"):
            return settings.direct_runtime_base_url.removesuffix("/v1") + "/metrics"
        return (
            None
            if on_demand.profiles
            else settings.direct_runtime_base_url.removesuffix("/v1") + "/metrics"
        )

    inference_tracker.resources = resource_sample
    inference_tracker.pool_state = pool_usage
    inference_tracker.local_pool = lambda: c.id if (c := pool_node.local_cluster()) else None
    telemetry_service = InferenceTelemetryService(settings, runtime.store)
    telemetry_service.target_provider = metrics_target

    def observe_telemetry(samples):
        on_demand.concurrency.telemetry(samples)
        local = pool_node.local_cluster()
        # Only known local runtime placements receive model/pool/GPU attribution.
        measured = []
        grouped = {}
        for key, metrics in samples.items():
            if not metrics.get("available"):
                continue
            resident = on_demand.residents.get(key)
            if resident and local:
                entities = performance_scopes(
                    resident.profile.model_id or resident.profile.model,
                    local.id,
                    resident.gpu_ids,
                )[1:]
                for entity in entities:
                    grouped.setdefault(entity, []).append(metrics)
            measured.append(metrics)
        for entity, metrics_list in grouped.items():
            combined = {}
            for field in RUNTIME_FIELDS:
                values = [m[field] for m in metrics_list if m.get(field) is not None]
                if values:
                    combined[field] = (
                        sum(values)
                        if field.endswith("tokens_per_second")
                        else (sum(values) / len(values))
                    )
            inference_tracker.observe_runtime(
                [entity],
                telemetry_service.latest.sampled_at.timestamp(),
                combined,
            )
        if measured:
            # Use the scraper's combined sample once; never add runtime and request tokens.
            metrics = telemetry_service.latest.model_dump(mode="json")
            inference_tracker.observe_runtime(
                [("overall", "all")],
                telemetry_service.latest.sampled_at.timestamp(),
                metrics,
            )

    telemetry_service.observer = observe_telemetry
    optimization_lab = OptimizationLab(settings, runtime.store, telemetry_service)
    cluster_connector = RemoteClusterConnector(settings, cluster_registry)

    async def activate_cluster(job) -> None:
        head = next(node for node in job.request.nodes if node.role == "head")
        await cluster_connector.attach(head.address, 30080)

    setup_service.on_cluster_ready = activate_cluster

    async def activate_native_host(name, node):
        return await runtime_library.register_host(
            RuntimeHostRequest(
                name=name,
                address=node.address,
                username=node.username,
                fingerprint=node.fingerprint,
            )
        )

    setup_service.on_native_host_ready = activate_native_host

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
        runtime_storage_bytes = max(
            plan.repository_bytes + plan.additional_artifacts_bytes,
            int(plan.runtime_storage_required_gib * 2**30),
        )
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

    async def model_performance_plan(request: PerformancePlanRequest) -> PerformancePlan:
        plan_settings = settings.model_copy(deep=True)
        active = cluster_registry.active()
        if active:
            for engine_id, image in runtime_prerequisites.image_overrides(active.id).items():
                attribute = f"runtime_{engine_id.replace('-', '_')}_image"
                if hasattr(plan_settings, attribute):
                    setattr(plan_settings, attribute, image)
        available_nodes = (
            sum(node.status != "offline" for node in runtime.latest.nodes)
            if runtime.latest and runtime.latest.nodes
            else settings.fallback_node_count
        )
        if request.topology == "auto" and plan_settings.active_hardware == "dgx-spark":
            setup_jobs = await asyncio.to_thread(runtime.store.list_setup_jobs)
            configured = next((job for job in setup_jobs if job.status == "ready"), None)
            if configured:
                request = request.model_copy(update={"topology": configured.request.topology})
        return build_performance_plan(request, available_nodes, plan_settings)

    async def enqueue_model_download(
        request: ModelDownloadRequest, request_base_url: str
    ) -> DownloadJob:
        fit_plan = await model_fit_plan(request)
        target = cluster_registry.active()
        destination = "cluster" if target and target.kind == "remote-bridge" else "local"
        controller_url = (
            model_relay.controller_url(request_base_url) if destination == "cluster" else None
        )
        return await model_manager.enqueue(
            request,
            destination=destination,
            controller_url=controller_url,
            fit_plan=fit_plan,
            target_cluster_id=target.id if target else None,
            target_cluster_name=target.name if target else None,
            target_cluster_url=target.endpoint if target else None,
        )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        local_monitor: asyncio.Task | None = None
        if docker_engine.available and await docker_engine.ping():
            gpus = await asyncio.to_thread(detect_nvidia_gpus)
            if gpus:
                cluster_registry.add_local(
                    name="Unraid GPUs",
                    host_address=settings.docker_runtime_host,
                    runtime_port=settings.docker_runtime_port_start,
                    gpus=gpus,
                    model_storage_path=str(settings.docker_host_model_path),
                    activate=cluster_registry.active() is None,
                )

            if any(c.kind == "local-docker" for c in cluster_registry.list()):

                async def monitor_local_cluster() -> None:
                    while True:
                        active = cluster_registry.active()
                        if active and active.kind == "local-docker":
                            try:
                                loaded_models = await docker_engine.loaded_models_by_gpu(active)
                                snapshot = await asyncio.to_thread(
                                    local_gpu_snapshot, active, settings, loaded_models
                                )
                                snapshot.runtime.model = (
                                    (
                                        ", ".join(
                                            r.profile.model
                                            for r in on_demand.residents.values()
                                            if r.status == "ready"
                                        )
                                        or "No model loaded"
                                    )
                                    if on_demand.profiles
                                    else settings.runtime_model
                                )
                                snapshot.runtime.quantization = settings.runtime_quantization
                                snapshot.runtime.context_length = settings.runtime_context_length
                                snapshot.runtime.tensor_parallel = settings.runtime_tensor_parallel
                                snapshot.runtime.pipeline_parallel = (
                                    settings.runtime_pipeline_parallel
                                )
                                await runtime.publish(snapshot)
                            except (OSError, RuntimeError, ValueError):
                                logging.getLogger(__name__).warning(
                                    "Local telemetry sample failed; retrying"
                                )
                        await asyncio.sleep(max(2, settings.sample_interval_seconds))

                local_monitor = asyncio.create_task(monitor_local_cluster())
        await runtime.start()
        await model_manager.start()
        await log_service.start()
        await inference_tracker.start()
        await telemetry_service.start()
        await deployment_service.start()
        await setup_service.start()
        await runtime_library.start()
        await runtime_workflows.start()
        await on_demand.start()
        await pool_node.start()
        try:
            yield
        finally:
            await model_pool.close()
            await pool_node.close()
            await on_demand.stop()
            await runtime_workflows.stop()
        if local_monitor:
            local_monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await local_monitor
        await setup_service.stop()
        await runtime_prerequisites.stop()
        await runtime_library.stop()
        await deployment_service.stop()
        await telemetry_service.stop()
        await inference_tracker.stop()
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
    app.state.runtime_library = runtime_library
    app.state.runtime_workflows = runtime_workflows
    app.state.telemetry_service = telemetry_service
    app.state.inference_tracker = inference_tracker
    app.state.optimization_lab = optimization_lab
    app.state.settings = settings
    app.state.cluster_connector = cluster_connector
    app.state.cluster_registry = cluster_registry
    app.state.docker_engine = docker_engine
    app.state.on_demand = on_demand
    app.state.model_pool = model_pool
    app.state.pool_node = pool_node
    mcp_server = SparkClusterMCP(
        browser=browser,
        runtime=runtime,
        model_manager=model_manager,
        deployment_service=deployment_service,
        model_fit_plan=model_fit_plan,
        performance_plan=model_performance_plan,
        download_model=enqueue_model_download,
        cluster_registry=cluster_registry,
    )
    app.state.mcp_server = mcp_server
    app.add_middleware(RemoteClusterProxyMiddleware, settings=settings, registry=cluster_registry)
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

    register_runtime_library(app, runtime_library, require_model_token, settings)

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

    @app.get("/api/v1/clusters")
    async def clusters():
        return {
            "clusters": [item.model_dump(mode="json") for item in cluster_registry.list()],
            "active_cluster_id": (
                cluster_registry.active().id if cluster_registry.active() else None
            ),
        }

    @app.post(
        "/api/v1/clusters/remote",
        dependencies=[Depends(require_model_token)],
    )
    async def add_remote_cluster(request: RemoteClusterRequest):
        connection = await cluster_connector.attach(request.address, request.port)
        cluster = cluster_registry.get(connection["id"])
        if cluster.name != request.name or cluster.hardware != request.hardware:
            cluster = cluster_registry.add_remote(
                name=request.name,
                endpoint=cluster.endpoint or "",
                head_address=cluster.host_address or request.address,
                hardware=request.hardware,
                activate=True,
            )
        return cluster

    @app.post(
        "/api/v1/clusters/local-docker",
        dependencies=[Depends(require_model_token)],
    )
    async def add_local_cluster(request: LocalClusterRequest):
        if not docker_engine.available or not await docker_engine.ping():
            raise HTTPException(
                status_code=409,
                detail="Mount the Docker socket and enable the local Docker runtime first.",
            )
        gpus = await asyncio.to_thread(detect_nvidia_gpus)
        return cluster_registry.add_local(
            name=request.name,
            host_address=settings.docker_runtime_host,
            runtime_port=settings.docker_runtime_port_start,
            gpus=gpus,
            model_storage_path=str(settings.docker_host_model_path),
            activate=True,
        )

    @app.post(
        "/api/v1/clusters/{cluster_id}/select",
        dependencies=[Depends(require_model_token)],
    )
    async def select_cluster(cluster_id: str):
        try:
            cluster = cluster_registry.select(cluster_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Cluster not found") from error
        if cluster.kind == "local-docker":
            loaded_models = await docker_engine.loaded_models_by_gpu(cluster)
            runtime.latest = await asyncio.to_thread(
                local_gpu_snapshot, cluster, settings, loaded_models
            )
        else:
            runtime.latest = None
            with contextlib.suppress(RemoteClusterUnavailable):
                runtime.latest = await cluster_connector.snapshot()
        return cluster

    @app.delete(
        "/api/v1/clusters/{cluster_id}",
        dependencies=[Depends(require_model_token)],
    )
    async def delete_cluster(cluster_id: str):
        try:
            return cluster_registry.delete(cluster_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Cluster not found") from error

    @app.post("/api/v1/setup/connect", dependencies=[Depends(require_model_token)])
    async def connect_cluster(request: ClusterConnectRequest):
        return await cluster_connector.attach(request.address, request.port)

    @app.get("/api/v1/controller/verify", dependencies=[Depends(require_model_token)])
    async def verify_controller():
        return {"status": "ready", "cluster_name": settings.cluster_name}

    @app.post("/api/v1/setup/discover", dependencies=[Depends(require_model_token)])
    async def discover_nodes(request: DiscoveryRequest):
        return await setup_service.discover(request)

    @app.post("/api/v1/setup/bootstrap", dependencies=[Depends(require_model_token)])
    async def create_setup_bootstrap(payload: BootstrapCreateRequest, request: Request):
        callback_base_url = settings.controller_public_url or str(request.base_url)
        return await setup_service.create_bootstrap(payload.address, callback_base_url)

    @app.get(
        "/api/v1/setup/bootstrap/{enrollment_id}",
        dependencies=[Depends(require_model_token)],
    )
    async def setup_bootstrap_status(enrollment_id: str):
        try:
            return await setup_service.bootstrap_status(enrollment_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Bootstrap command not found") from error

    @app.post("/api/v1/setup/bootstrap/confirm")
    async def confirm_setup_bootstrap(
        x_spark_username: str = Header(),
        x_spark_host_key: str = Header(),
        x_spark_bootstrap: str = Header(),
    ):
        try:
            confirmation = BootstrapConfirmRequest(
                username=x_spark_username,
                host_key=x_spark_host_key,
            )
            return await setup_service.confirm_bootstrap(
                x_spark_bootstrap, confirmation.username, confirmation.host_key
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Bootstrap command not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

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

    @app.api_route("/mcp", methods=["GET", "POST", "DELETE"], include_in_schema=False)
    async def mcp_endpoint(request: Request):
        if request.method != "POST":
            return JSONResponse(
                status_code=405,
                headers={"Allow": "POST"},
                content={
                    "name": "spark-cluster",
                    "transport": "MCP Streamable HTTP",
                    "endpoint": str(request.base_url).rstrip("/") + "/mcp",
                    "message": "Connect an MCP client with HTTP POST.",
                },
            )
        origin = request.headers.get("origin")
        controller_origin = str(request.base_url).rstrip("/")
        allowed_origins = {*settings.cors_origin_list, controller_origin}
        if origin and origin not in allowed_origins:
            return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(
                status_code=400,
                content=SparkClusterMCP.error(None, -32700, "Parse error"),
            )
        expected = settings.model_admin_token
        authorization = request.headers.get("authorization", "")
        admin = not expected or hmac.compare_digest(authorization, f"Bearer {expected}")
        messages = payload if isinstance(payload, list) else [payload]
        if not messages or any(not isinstance(item, dict) for item in messages):
            return JSONResponse(
                status_code=400,
                content=SparkClusterMCP.error(None, -32600, "Invalid Request"),
            )
        responses = []
        base_url = str(request.base_url).rstrip("/")
        for message in messages:
            response = await mcp_server.request(message, admin=admin, base_url=base_url)
            if response is not None:
                responses.append(response)
        if not responses:
            return Response(status_code=202)
        content = responses if isinstance(payload, list) else responses[0]
        return JSONResponse(content=content)

    @app.get("/healthz")
    async def health() -> dict:
        return {
            "status": "ok",
            "mode": settings.mode,
            "logs_mode": log_service.mode,
            "runtime_control_mode": deployment_service.mode,
            "has_snapshot": runtime.latest is not None,
        }

    @app.get("/v1", include_in_schema=False)
    async def openai_api_root() -> dict:
        return {
            "object": "spark-cluster-pool",
            "models": "/v1/models",
            "message": "Use an OpenAI-compatible endpoint and specify model in the request.",
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
        return await model_performance_plan(request)

    @app.get("/api/v1/inference/activity")
    async def inference_activity(limit: int = Query(default=20, ge=0, le=100)) -> dict:
        return {
            **inference_tracker.summary(limit),
            "runtime_metrics": telemetry_service.latest.model_dump(mode="json"),
            "retained_models": sorted(
                model_id
                for model_id, preferences in model_pool.preferences.items()
                if preferences.keep_loaded
            ),
        }

    @app.get("/api/v1/inference/performance")
    async def inference_performance(
        scope: str = Query(default="overall", pattern="^(overall|model|pool|gpu)$"),
        entity: str = Query(default="all", min_length=1, max_length=1024),
        hours: int = Query(default=24, ge=1, le=24 * 730),
    ) -> dict:
        result = await asyncio.to_thread(
            inference_tracker.journal.performance_summary,
            scope,
            "all" if scope == "overall" else entity,
            hours,
        )
        result["persistence"] = dict(
            healthy=not inference_tracker.persistence_error,
            pending=len(inference_tracker.pending) + len(inference_tracker.runtime_pending),
            dropped=inference_tracker.dropped,
        )
        return result

    @app.get("/api/v1/telemetry", response_model=InferenceTelemetry)
    async def inference_telemetry() -> InferenceTelemetry:
        return telemetry_service.latest

    @app.get("/api/v1/telemetry/history", response_model=list[InferenceTelemetry])
    async def inference_telemetry_history(
        limit: int = Query(default=120, ge=1, le=5000),
    ) -> list[InferenceTelemetry]:
        return await asyncio.to_thread(runtime.store.telemetry_history, limit)

    @app.post(
        "/api/v1/optimization/benchmarks",
        response_model=BenchmarkResult,
        dependencies=[Depends(require_model_token)],
    )
    async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResult:
        try:
            return await optimization_lab.benchmark(request)
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(status_code=502, detail="Runtime benchmark failed") from error

    @app.get("/api/v1/optimization/benchmarks", response_model=list[BenchmarkResult])
    async def benchmarks(limit: int = Query(default=50, ge=1, le=500)):
        return await asyncio.to_thread(runtime.store.list_benchmarks, limit)

    @app.post(
        "/api/v1/cache/prefixes/prewarm",
        response_model=CachePrefixRecord,
        dependencies=[Depends(require_model_token)],
    )
    async def prewarm_prefix(request: CachePrefixRequest) -> CachePrefixRecord:
        return await optimization_lab.prewarm(request)

    @app.get("/api/v1/cache/prefixes", response_model=list[CachePrefixRecord])
    async def cache_prefixes(limit: int = Query(default=100, ge=1, le=500)):
        return await asyncio.to_thread(runtime.store.list_cache_prefixes, limit)

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
        sort: CatalogSort = "spark",
        task: str = Query("", max_length=80),
        library: str = Query("", max_length=80),
        access: Literal["", "open", "gated"] = "",
        limit: int = Query(24, ge=1, le=48),
        cursor: str | None = Query(None, max_length=4000),
    ) -> CatalogPage:
        return await catalog_result(
            browser.search(query, sort, task, library, access, limit, cursor)
        )

    @app.get("/api/v1/catalog/profiles/{author}/avatar", include_in_schema=False)
    async def catalog_profile_avatar(author: str):
        try:
            url = await browser.author_avatar_url(author)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(
                status_code=502, detail="Cannot load this profile avatar."
            ) from error
        return RedirectResponse(url=url, status_code=307)

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
            return await enqueue_model_download(request, str(http_request.base_url).rstrip("/"))
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

    hf_token_lock = asyncio.Lock()

    @app.get("/api/v1/controller/huggingface")
    async def huggingface_status() -> dict:
        return {
            "configured": bool(settings.hf_token),
            "requires_admin": bool(settings.model_admin_token),
        }

    @app.put(
        "/api/v1/controller/huggingface",
        dependencies=[Depends(require_model_token)],
    )
    async def configure_huggingface(request: hf_credentials.HuggingFaceTokenRequest) -> dict:
        token = request.token.get_secret_value().strip()
        async with hf_token_lock:
            try:
                await hf_credentials.validate_token(token)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from None
            except RuntimeError as error:
                raise HTTPException(status_code=502, detail=str(error)) from None
            await asyncio.to_thread(hf_credentials.save_token, settings, token)
            browser.token = token
            restarted = await model_manager.refresh_huggingface_token(token)
        return {"configured": True, "restarted_downloads": restarted}

    @app.get("/api/v1/launches", response_model=list[ModelLaunchJob])
    async def launches() -> list[ModelLaunchJob]:
        return await deployment_service.launches()

    @app.get("/api/v1/pool")
    async def pool_inventory():
        return await model_pool.inventory()

    @app.put(
        "/api/v1/pool/models/{model_id:path}/settings", dependencies=[Depends(require_model_token)]
    )
    async def pool_preferences(model_id: str, preferences: ModelPreferences):
        try:
            return await model_pool.update_preferences(model_id, preferences)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.patch(
        "/api/v1/pool/models/{model_id:path}/retention",
        dependencies=[Depends(require_model_token)],
    )
    async def pool_retention(model_id: str, request: ModelRetentionRequest):
        try:
            return await model_pool.update_retention(model_id, request.keep_loaded)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/pool/load", status_code=202, dependencies=[Depends(require_model_token)])
    async def pool_load(request: PoolLoadRequest):
        try:
            return await model_pool.enqueue(request)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/pool/scale", status_code=202, dependencies=[Depends(require_model_token)])
    async def pool_scale(request: PoolScaleRequest):
        try:
            return await model_pool.scale_instances(request)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/pool/unload", dependencies=[Depends(require_model_token)])
    async def pool_unload(request: PoolLoadRequest):
        try:
            return await model_pool.unload(request.model_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/pool/node/status", dependencies=[Depends(require_model_token)])
    async def pool_node_status():
        return await pool_node.inventory()

    @app.post("/api/v1/pool/node/{action}", dependencies=[Depends(require_model_token)])
    async def pool_node_action(action: str, request: NodeLoadRequest):
        try:
            if action == "plan":
                return await pool_node.plan(request)
            if action == "load":
                return await pool_node.load(request)
            if action == "retain":
                await pool_node.retain(request.model_id, request.keep_loaded)
            elif action == "unload":
                await pool_node.unload(request.model_id)
            else:
                raise HTTPException(status_code=404, detail="Unknown node action")
            return {"status": "ok"}
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=model_manager.safe_error(error)) from error

    @app.api_route(
        "/api/v1/pool/node/inference/{path:path}",
        methods=["GET", "POST", "DELETE"],
        dependencies=[Depends(require_model_token)],
    )
    async def pool_node_inference(request: Request, path: str, model_id: str):
        from urllib.parse import parse_qsl, urlencode

        scope = dict(request.scope)
        scope["query_string"] = urlencode(
            [(k, v) for k, v in parse_qsl(request.url.query) if k != "model_id"]
        ).encode()

        async def handler(_, receive, send):
            await pool_node.infer(model_id, path, scope, receive, send)

        return PoolResponse(handler)

    @app.api_route("/api/v1/pool/serve/{key}/{path:path}", methods=["GET", "POST", "DELETE"])
    async def pool_inference(request: Request, key: str, path: str):
        request.scope["spark.arrival"] = (time.time(), time.perf_counter())
        inventory = await model_pool.inventory()
        model = next((m for m in inventory["models"] if m["key"] == key), None)
        if not model:
            raise HTTPException(status_code=404, detail="Unknown pool model")

        async def handler(scope, receive, send):
            await model_pool.infer(model["id"], path, scope, receive, send)

        return PoolResponse(handler)

    @app.get("/v1/models")
    async def global_openai_models():
        inventory = await model_pool.inventory(force=True)
        data = []
        for model in inventory["models"]:
            if not model["loadable"]:
                continue
            runtime_response = model.get("runtime_models")
            runtime_data = (
                runtime_response.get("data", []) if isinstance(runtime_response, dict) else []
            )
            if runtime_data:
                # Forward every field reported by the runtime without maintaining a shadow schema.
                data.extend(runtime_data)
            else:
                # The routing id is the only fact available before this runtime has run once.
                data.append({"id": model["id"], "object": "model"})
        return {"object": "list", "data": data}

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def global_openai_inference(request: Request, path: str):
        request.scope["spark.arrival"] = (time.time(), time.perf_counter())
        request.scope["spark.client_request"] = request
        try:
            body = await read_inference_body(request)
        except RequestTooLarge as error:
            return inference_error(str(error), status=413)
        content_type = request.headers.get("content-type", "")
        reference = request.headers.get("x-spark-model") or request.query_params.get("model")
        json_body = None
        if not reference and "application/json" in content_type and body:
            try:
                json_body = json.loads(body)
                if isinstance(json_body, dict):
                    reference = json_body.get("model")
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise HTTPException(status_code=400, detail="Invalid JSON request body") from None
        if not reference and "application/x-www-form-urlencoded" in content_type:
            reference = parse_qs(body.decode(errors="replace")).get("model", [None])[0]
        if not reference and "multipart/form-data" in content_type:
            match = re.search(
                rb'name="model"[^\r\n]*\r\n(?:[^\r\n]*\r\n)?\r\n([^\r\n]+)',
                body,
            )
            if match:
                reference = match.group(1).decode(errors="replace")
        if not reference and path.startswith("models/"):
            reference = path.removeprefix("models/").split("/", 1)[0]
        if not isinstance(reference, str) or not reference.strip():
            raise HTTPException(
                status_code=400,
                detail="Specify model in the request body, query, or X-Spark-Model header",
            )
        try:
            model = await model_pool.resolve_model(reference)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        forwarded = body
        if json_body is not None:
            served_name = model.get("served_model_name") or model["id"]
            json_body["model"] = served_name
            forwarded = json.dumps(json_body, separators=(",", ":")).encode()
        delivered = False

        async def replay_receive():
            nonlocal delivered
            if delivered:
                # StreamingResponse listens for the real client disconnect in parallel with
                # upstream bytes. A synthetic disconnect here cancels every valid SSE stream.
                return await request.receive()
            delivered = True
            return {"type": "http.request", "body": forwarded, "more_body": False}

        async def handler(scope, _, send):
            forwarded_scope = dict(scope)
            forwarded_scope["headers"] = [
                (name, value)
                for name, value in scope["headers"]
                if name.lower() != b"content-length"
            ]
            await model_pool.infer(model["id"], f"v1/{path}", forwarded_scope, replay_receive, send)

        return PoolResponse(handler)

    @app.get("/api/v1/inference/status")
    async def inference_status() -> dict:
        return on_demand.status()

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
        except EngineNotExecutable as error:
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
        except EngineNotExecutable as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/api/v1/runtime-management/readiness",
        response_model=RuntimeReadiness,
        dependencies=[Depends(require_model_token)],
    )
    async def runtime_readiness(request: RuntimeReadinessRequest) -> RuntimeReadiness:
        try:
            return await runtime_prerequisites.inspect(
                request.model_id, request.engine_id, request.cluster_id
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Compute pool not found") from error

    @app.get(
        "/api/v1/runtime-management/guide",
        response_model=RuntimeGuide,
        dependencies=[Depends(require_model_token)],
    )
    async def runtime_guide(
        cluster_id: str,
        model_id: str | None = None,
        engine_id: Literal["auto", "vllm", "sglang", "tensorfold", "deepinfer"] = "auto",
    ) -> RuntimeGuide:
        try:
            return await runtime_workflows.guide(cluster_id, model_id, engine_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="Compute pool or runtime not found"
            ) from error

    @app.post(
        "/api/v1/runtime-management/workflows",
        response_model=RuntimeWorkflowJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def create_runtime_workflow(request: RuntimeWorkflowRequest) -> RuntimeWorkflowJob:
        try:
            return await runtime_workflows.create(request)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="Compute pool or runtime not found"
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get(
        "/api/v1/runtime-management/workflows/{job_id}",
        response_model=RuntimeWorkflowJob,
        dependencies=[Depends(require_model_token)],
    )
    async def runtime_workflow(job_id: str) -> RuntimeWorkflowJob:
        try:
            return await runtime_workflows.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Runtime workflow not found") from error

    @app.post(
        "/api/v1/runtime-management/workflows/{job_id}/retry",
        response_model=RuntimeWorkflowJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def retry_runtime_workflow(job_id: str) -> RuntimeWorkflowJob:
        try:
            return await runtime_workflows.retry(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Runtime workflow not found") from error

    @app.get(
        "/api/v1/runtime-management/engines",
        response_model=RuntimeEngineCollection,
        dependencies=[Depends(require_model_token)],
    )
    async def runtime_engines(cluster_id: str) -> RuntimeEngineCollection:
        try:
            return await runtime_prerequisites.inspect_engines(cluster_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Compute pool not found") from error

    @app.post(
        "/api/v1/runtime-management/prepare",
        response_model=RuntimeInstallJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def prepare_runtime(request: RuntimePrepareRequest) -> RuntimeInstallJob:
        try:
            return await runtime_prerequisites.prepare(request.cluster_id, request.engine_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail="Compute pool or engine not found"
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post(
        "/api/v1/runtime-management/install",
        response_model=RuntimeInstallJob,
        status_code=202,
        dependencies=[Depends(require_model_token)],
    )
    async def install_runtime(request: RuntimeReadinessRequest) -> RuntimeInstallJob:
        try:
            return await runtime_prerequisites.install(
                request.model_id, request.engine_id, request.cluster_id
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Compute pool not found") from error
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
        base = render_metrics(runtime.latest, telemetry_service.latest) if runtime.latest else ""
        return base + inference_tracker.prometheus()

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
