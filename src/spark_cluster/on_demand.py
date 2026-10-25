"""Opt-in, controller-owned Docker inference with stable legacy HTTP endpoints.

Recipes are trusted administrator input, never client-provided Docker specifications.
Each runtime has a device reservation; request leases last through the last byte.
"""

import asyncio
import contextlib
import copy
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx
import uvicorn
from pydantic import BaseModel, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from .adaptive_concurrency import (
    AdaptiveConcurrency,
    RuntimeCapacityError,
    fingerprint,
    response_feedback,
    runtime_parallelism,
    startup_capacity_failure,
)
from .inference_tracking import mark_dispatch, request_stream_usage
from .local_docker import DockerEngineClient
from .pool_memory import (
    MemoryPlan,
    RuntimeResources,
    gpu_availability,
    imported_resources,
    plan_memory,
)

logger = logging.getLogger(__name__)
HOP_HEADERS = {
    b"host",
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


class InferenceProfile(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    model: str
    model_id: str | None = None
    launch_settings: dict = Field(default_factory=dict)
    model_revision: str | None = None
    listen_host: str = "127.0.0.1"
    listen_port: int = Field(ge=1024, le=65535)
    upstream_port: int = Field(ge=1024, le=65535)
    readiness_path: str = "/health"
    preload_path: str | None = None
    idle_seconds: float = Field(default=120, ge=1)
    startup_seconds: float = Field(default=900, ge=1)
    container: dict
    resources: RuntimeResources | None = None

    @model_validator(mode="after")
    def validate_ports(self):
        if self.listen_port == self.upstream_port:
            raise ValueError("Inference listener and runtime must use different ports")
        if not self.readiness_path.startswith("/"):
            raise ValueError("Readiness path must be relative")
        if self.preload_path is not None and (
            not self.preload_path.startswith("/") or self.preload_path.startswith("//")
        ):
            raise ValueError("Preload path must be relative to the runtime")
        return self

    @property
    def name(self) -> str:
        return f"spark-cluster-demand-{self.id}"

    @property
    def upstream(self) -> str:
        return f"http://127.0.0.1:{self.upstream_port}"


def load_profiles(path: Path) -> list[InferenceProfile]:
    if path.stat().st_mode & 0o077:
        raise ValueError("On-demand recipes must have private permissions (0600)")
    profiles = [InferenceProfile.model_validate(item) for item in json.loads(path.read_text())]
    ids = [p.id for p in profiles]
    listeners = [p.listen_port for p in profiles]
    backends = [p.upstream_port for p in profiles]
    if len(set(ids)) != len(ids) or len(set(listeners + backends)) != len(listeners + backends):
        raise ValueError("On-demand profile names and ports must be unique")
    return profiles


class ListenerServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        # The main controller owns process signal handling.
        yield


@dataclass
class ResidentRuntime:
    profile: InferenceProfile
    gpu_ids: list[str]
    budget_gib: dict[str, float]
    fraction: float | None = None
    exclusive: bool = False
    last_used: float = field(default_factory=time.monotonic)
    active_requests: int = 0
    status: str = "starting"
    runtime_ceiling: int = 64
    engine_image: str | None = None
    parallelism_prepared: bool = False

    def public(self):
        return {
            "profile_id": self.profile.id,
            "model_id": self.profile.model_id or self.profile.model,
            "gpu_ids": self.gpu_ids,
            "budget_gib": self.budget_gib,
            "gpu_memory_utilization": self.fraction,
            "active_requests": self.active_requests,
            "status": self.status,
            "exclusive": self.exclusive,
        }


class OnDemandInference:
    def __init__(
        self,
        docker: DockerEngineClient,
        profiles: list[InferenceProfile],
        state_path: Path | None = None,
    ):
        self.docker = docker
        self.profiles = profiles
        self.lock = asyncio.Lock()
        self.residents: dict[str, ResidentRuntime] = {}
        self.profile_locks: dict[str, asyncio.Lock] = {}
        self.resource_provider = None
        self.usable_fraction = getattr(
            getattr(docker, "settings", None), "model_usable_memory_fraction", 0.82
        )
        self._gpus = []
        self.servers: list[ListenerServer] = []
        self.tasks: list[asyncio.Task] = []
        self.sockets: list[socket.socket] = []
        self.pending = 0
        self.last_error: str | None = None
        self.state_path = state_path
        self.concurrency = AdaptiveConcurrency(
            state_path.with_name("adaptive-concurrency.json") if state_path else None
        )
        self.retained: set[str] = set()
        self.changed = asyncio.Event()
        self.dynamic: set[str] = set()
        self.pool = None
        if state_path and state_path.exists():
            state = json.loads(state_path.read_text())
            self.retained = set(state.get("retained", []))
            for item in state.get("profiles", []):
                if item.get("launch_settings"):
                    item["launch_settings"].setdefault("memory_budget_mode", "fixed")
                profile = InferenceProfile.model_validate(item)
                if not any(p.id == profile.id for p in self.profiles):
                    self.profiles.append(profile)
                    self.dynamic.add(profile.id)

    def persist(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.touch(mode=0o600, exist_ok=True)
        temporary.chmod(0o600)
        temporary.write_text(
            json.dumps(
                {
                    "retained": sorted(self.retained),
                    "profiles": [p.model_dump() for p in self.profiles if p.id in self.dynamic],
                }
            )
        )
        temporary.replace(self.state_path)

    def profile_for(self, model_id: str) -> InferenceProfile | None:
        return next((p for p in self.profiles if model_id in {p.id, p.model, p.model_id}), None)

    def retain(self, profile: InferenceProfile, retained: bool) -> None:
        if retained:
            self.retained.add(profile.id)
        else:
            self.retained.discard(profile.id)
        self.persist()
        self.changed.set()

    @property
    def active(self) -> InferenceProfile | None:
        ready = [r.profile for r in self.residents.values() if r.status == "ready"]
        return ready[0] if len(ready) == 1 else None

    @active.setter
    def active(self, profile):
        # Compatibility for integrations that seed a legacy exclusive runtime.
        self.residents.clear()
        if profile:
            self.residents[profile.id] = ResidentRuntime(
                profile,
                ["unknown"],
                {"unknown": self.usable_fraction},
                exclusive=True,
                status="ready",
            )

    @property
    def last_used(self):
        return max((r.last_used for r in self.residents.values()), default=0.0)

    @last_used.setter
    def last_used(self, value):
        for resident in self.residents.values():
            resident.last_used = value

    async def gpu_memory(self):
        if self.resource_provider:
            self._gpus = await self.resource_provider()
        elif hasattr(self.docker, "gpu_memory"):
            self._gpus = await self.docker.gpu_memory([r.profile for r in self.residents.values()])
        else:
            # Adapters without device telemetry retain the safe legacy exclusive slot.
            self._gpus = [
                {"id": "unknown", "index": 0, "total_gib": 1.0, "used_gib": 0.0, "managed_gib": {}}
            ]
        return self._gpus

    def resources_for(self, profile, gpus):
        return profile.resources or imported_resources(profile, gpus)

    async def plan(self, profile, gpus=None):
        if gpus is None:
            gpus = await self.gpu_memory()
        resident = self.residents.get(profile.id)
        if resident and resident.status == "ready":
            availability = gpu_availability(gpus, self.residents, self.usable_fraction)
            return MemoryPlan(
                resident.gpu_ids,
                resident.budget_gib,
                resident.fraction,
                available_gib=sum(
                    g["available_gib"] for g in availability if g["id"] in resident.gpu_ids
                ),
            )
        return plan_memory(
            self.resources_for(profile, gpus),
            gpus,
            self.residents,
            self.retained,
            self.usable_fraction,
        )

    def admission_controller(self, resident):
        profile = resident.profile
        identity = fingerprint(
            {
                "model": profile.model_id or profile.model,
                "revision": profile.model_revision,
                "image": resident.engine_image or profile.container.get("Image"),
                "command": profile.container.get("Cmd"),
                "settings": profile.launch_settings,
                "gpus": resident.gpu_ids,
                "budget": resident.budget_gib,
                "ceiling": resident.runtime_ceiling,
            }
        )
        return self.concurrency.controller(profile.id, identity, resident.runtime_ceiling)

    def resident_state(self, resident):
        return dict(
            resident.public(),
            keep_loaded=resident.profile.id in self.retained,
            concurrency=self.admission_controller(resident).public()
            if resident.status == "ready"
            else None,
        )

    @contextlib.asynccontextmanager
    async def lease(self, profile: InferenceProfile, feedback=None):
        await self.ensure(profile, wait=True)
        resident = self.residents.get(profile.id)
        if resident is None:
            raise RuntimeError("Runtime was unloaded before admission")
        controller = self.admission_controller(resident)
        async with controller.slot(feedback) as admission:
            while True:
                async with self.lock:
                    resident = self.residents.get(profile.id)
                    if resident and resident.status == "ready":
                        resident.active_requests += 1
                        self.admission_controller(resident)
                        break
                await self.ensure(profile, wait=True)
            admission.begin()
            try:
                yield admission
            finally:
                async with self.lock:
                    resident.active_requests -= 1
                    resident.last_used = time.monotonic()
                    self.changed.set()

    def register(self, profile: InferenceProfile) -> None:
        if self.profile_for(profile.model_id or profile.model):
            return
        self.profiles.append(profile)
        self.dynamic.add(profile.id)
        self.persist()

    def status(self) -> dict:
        return {
            "enabled": bool(self.profiles),
            "active_profile": self.active.id if self.active else None,
            "active_profiles": [
                r.profile.id for r in self.residents.values() if r.status == "ready"
            ],
            "resident_models": [self.resident_state(r) for r in self.residents.values()],
            "pending_requests": self.pending,
            "last_error": self.last_error,
            "profiles": [
                {
                    "id": p.id,
                    "model": p.model,
                    "port": p.listen_port,
                    "idle_seconds": p.idle_seconds,
                    "model_id": p.model_id or p.model,
                    "keep_loaded": p.id in self.retained,
                    "launch_settings": p.launch_settings,
                }
                for p in self.profiles
            ],
        }

    async def remove(self, profile: InferenceProfile) -> None:
        async with self.lock:
            await self._remove(profile)

    async def _remove(self, profile: InferenceProfile, force=False) -> None:
        resident = self.residents.get(profile.id)
        if resident and resident.status == "starting" and not force:
            raise RuntimeError("Wait for runtime startup before unloading this model")
        if resident and resident.active_requests and not force:
            raise RuntimeError("Wait for active requests before unloading this model")
        name = quote(profile.name, safe="")
        response = await self.docker.request("GET", f"/containers/{name}/json", allow={404})
        if response.status_code == 404:
            self.residents.pop(profile.id, None)
            self.changed.set()
            return
        if (
            response.json().get("Config", {}).get("Labels", {}).get("spark-cluster.on-demand")
            != profile.id
        ):
            raise RuntimeError("Refusing to remove an unmanaged container")
        await self.docker.request(
            "POST", f"/containers/{name}/stop", params={"t": 20}, allow={304, 404}, timeout=30
        )
        await self.docker.request("DELETE", f"/containers/{name}", allow={404}, timeout=30)
        self.residents.pop(profile.id, None)
        self.changed.set()

    async def start(self) -> None:
        if not self.profiles and not self.docker.available:
            return
        if not await self.docker.ping():
            raise RuntimeError("On-demand inference requires Docker access")
        # Bind every listener before accepting any requests; conflicts fail startup.
        try:
            for profile in self.profiles:
                if profile.id in self.dynamic:
                    continue
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((profile.listen_host, profile.listen_port))
                sock.listen(128)
                sock.setblocking(False)
            gpus = await self.gpu_memory()
            for profile in self.profiles:
                state = await self.docker.request(
                    "GET", f"/containers/{profile.name}/json", allow={404}
                )
                if state.status_code == 404:
                    continue
                details = state.json()
                labels = details.get("Config", {}).get("Labels", {})
                if labels.get("spark-cluster.on-demand") != profile.id:
                    raise RuntimeError("Refusing to adopt an unmanaged container")
                if not details.get("State", {}).get("Running"):
                    await self._remove(profile)
                    continue
                actual = profile.model_copy(
                    update={
                        "container": {
                            **details.get("Config", {}),
                            "HostConfig": details.get("HostConfig", {}),
                        }
                    }
                )
                resources = imported_resources(actual, gpus)
                allocation = json.loads(labels.get("spark-cluster.reservation", "null") or "null")
                if not allocation and not gpus:
                    raise RuntimeError("GPU telemetry is required to adopt a legacy runtime safely")
                if allocation:
                    budgets = allocation["budget_gib"]
                    if not all(
                        0 < float(budgets[device]) < float("inf")
                        for device in allocation["gpu_ids"]
                    ):
                        raise RuntimeError("Invalid persisted GPU reservation")
                    resident = ResidentRuntime(
                        profile,
                        allocation["gpu_ids"],
                        budgets,
                        allocation.get("fraction"),
                        allocation.get("exclusive", False),
                        status="ready",
                    )
                else:
                    selected = [g for g in gpus if g["id"] in resources.gpu_ids]
                    resident = ResidentRuntime(
                        profile,
                        [g["id"] for g in selected],
                        {
                            g["id"]: g["total_gib"]
                            * (
                                resources.memory_fraction
                                if resources.mode == "fixed"
                                else self.usable_fraction
                            )
                            for g in selected
                        },
                        resources.memory_fraction if resources.mode == "fixed" else None,
                        resources.mode == "exclusive",
                        status="ready",
                    )
                resident.runtime_ceiling = runtime_parallelism(actual.container)
                resident.engine_image = details.get("Image") or profile.container.get("Image")
                resident.parallelism_prepared = (
                    labels.get("spark-cluster.adaptive-batching") == "true"
                )
                self.residents[profile.id] = resident
            listeners = [p for p in self.profiles if p.id not in self.dynamic]
            for profile, sock in zip(listeners, self.sockets, strict=True):
                server = ListenerServer(
                    uvicorn.Config(
                        DemandProxy(self, profile),
                        log_level="warning",
                        access_log=False,
                        lifespan="off",
                        timeout_graceful_shutdown=30,
                    )
                )
                self.servers.append(server)
                self.tasks.append(asyncio.create_task(server.serve(sockets=[sock])))
            self.tasks.append(asyncio.create_task(self.reap()))
        except BaseException:
            for sock in self.sockets:
                sock.close()
            raise

    async def stop(self) -> None:
        for server in self.servers:
            server.should_exit = True
        if self.tasks:
            self.tasks[-1].cancel()  # idle reaper
            await asyncio.gather(*self.tasks, return_exceptions=True)
        async with self.lock:
            for resident in list(self.residents.values()):
                if resident.profile.id not in self.retained:
                    await self._remove(resident.profile)
        for sock in self.sockets:
            sock.close()
        await self.concurrency.flush(force=True)

    async def reap(self) -> None:
        while True:
            await asyncio.sleep(1)
            async with self.lock:
                for resident in list(self.residents.values()):
                    if (
                        resident.status == "ready"
                        and not resident.active_requests
                        and resident.profile.id not in self.retained
                        and time.monotonic() - resident.last_used >= resident.profile.idle_seconds
                    ):
                        try:
                            await self._remove(resident.profile)
                        except Exception:
                            self.last_error = "Idle runtime cleanup failed; inspect controller logs"
                            logger.exception("On-demand idle cleanup failed")
            await self.concurrency.flush()

    async def ensure(
        self, profile: InferenceProfile, *, wait=False, keep_loaded=None, _profile_locked=False
    ) -> None:
        gate = (
            contextlib.nullcontext()
            if _profile_locked
            else self.profile_locks.setdefault(profile.id, asyncio.Lock())
        )
        async with gate:
            profile = next((p for p in self.profiles if p.id == profile.id), profile)
            while True:
                async with self.lock:
                    existing = self.residents.get(profile.id)
                    if existing and existing.status == "ready":
                        state = await self.docker.request(
                            "GET", f"/containers/{profile.name}/json", allow={404}
                        )
                        if state.status_code == 200 and state.json().get("State", {}).get(
                            "Running"
                        ):
                            actual = state.json()
                            existing.runtime_ceiling = runtime_parallelism(actual.get("Config", {}))
                            existing.engine_image = actual.get("Image") or profile.container.get(
                                "Image"
                            )
                            desired = runtime_parallelism(
                                copy.deepcopy(profile.container), prepare=True
                            )
                            if (
                                existing.parallelism_prepared
                                or existing.runtime_ceiling >= desired
                                or existing.active_requests
                            ):
                                existing.last_used = time.monotonic()
                                if keep_loaded is not None:
                                    self.retain(profile, keep_loaded)
                                return
                            # Upgrade continuous-batching flags only after active streams drain.
                        await self._remove(profile)
                    gpus = await self.gpu_memory()
                    resources = self.resources_for(profile, gpus)
                    plan = await self.plan(profile, gpus)
                    if plan.gpu_ids:
                        for victim in plan.evict:
                            await self._remove(self.residents[victim].profile)
                        await self._remove(profile)
                        # The reservation is visible before the slow engine startup.
                        resident = ResidentRuntime(
                            profile,
                            plan.gpu_ids,
                            plan.budget_gib,
                            plan.fraction,
                            resources.mode == "exclusive",
                        )
                        self.residents[profile.id] = resident
                        break
                    if not plan.waiting or not wait:
                        raise RuntimeError(plan.reason)
                    self.changed.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.changed.wait(), timeout=1)
            try:
                while True:
                    try:
                        await self._start_runtime(profile, resident)
                        break
                    except RuntimeCapacityError:
                        if resident.runtime_ceiling <= 1:
                            raise
                        async with self.lock:
                            await self._remove(profile, force=True)
                            resident.runtime_ceiling = max(1, resident.runtime_ceiling // 2)
                            self.residents[profile.id] = resident
                async with self.lock:
                    if self.residents.get(profile.id) is not resident:
                        raise RuntimeError("Runtime residency changed during startup")
                    resident.status = "ready"
                    resident.parallelism_prepared = True
                    resident.last_used = time.monotonic()
                    if keep_loaded is not None:
                        self.retain(profile, keep_loaded)
                    self.changed.set()
            except BaseException:
                async with self.lock:
                    await self._remove(profile, force=True)
                raise

    async def _start_runtime(self, profile, resident):
        payload = copy.deepcopy(profile.container)
        if profile.resources:
            host = payload.setdefault("HostConfig", {})
            host["DeviceRequests"] = [
                {"Driver": "nvidia", "DeviceIDs": resident.gpu_ids, "Capabilities": [["gpu"]]}
            ]
            payload["Env"] = [
                e for e in payload.get("Env", []) if not e.startswith("NVIDIA_VISIBLE_DEVICES=")
            ]
            payload["Env"].append("NVIDIA_VISIBLE_DEVICES=" + ",".join(resident.gpu_ids))
            if resident.fraction is not None:
                command = payload.get("Cmd", [])
                for flag in ("--gpu-memory-utilization", "--mem-fraction-static"):
                    if flag in command:
                        command[command.index(flag) + 1] = str(resident.fraction)
        resident.runtime_ceiling = runtime_parallelism(
            payload, prepare=True, ceiling=resident.runtime_ceiling
        )
        labels = payload.setdefault("Labels", {})
        labels["spark-cluster.reservation"] = json.dumps(
            {
                "gpu_ids": resident.gpu_ids,
                "budget_gib": resident.budget_gib,
                "fraction": resident.fraction,
                "exclusive": resident.exclusive,
            }
        )
        labels["spark-cluster.managed"] = "true"
        labels["spark-cluster.adaptive-batching"] = "true"
        labels["spark-cluster.on-demand"] = profile.id
        labels["spark-cluster.model"] = profile.model
        labels["spark-cluster.model-id"] = profile.model_id or profile.model
        payload.setdefault("HostConfig", {})["RestartPolicy"] = {"Name": "no"}
        payload["HostConfig"]["AutoRemove"] = False
        # Docker health probes must not call the public listener and trigger recursion.
        payload["Healthcheck"] = {"Test": ["NONE"]}
        created = await self.docker.request(
            "POST", "/containers/create", payload=payload, params={"name": profile.name}, timeout=30
        )
        container_id = created.json()["Id"]
        await self.docker.request("POST", f"/containers/{container_id}/start", timeout=30)
        deadline = time.monotonic() + profile.startup_seconds
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            while time.monotonic() < deadline:
                state = await self.docker.request("GET", f"/containers/{container_id}/json")
                resident.engine_image = state.json().get("Image") or payload.get("Image")
                if not state.json().get("State", {}).get("Running"):
                    logs = b""
                    with contextlib.suppress(httpx.HTTPError, RuntimeError):
                        response = await self.docker.request(
                            "GET",
                            f"/containers/{container_id}/logs",
                            params={"stdout": "true", "stderr": "true", "tail": 64},
                            allow={404},
                            timeout=5,
                        )
                        logs = response.content
                    if startup_capacity_failure(state.json().get("State", {}), logs):
                        raise RuntimeCapacityError(
                            "Runtime batching exceeded startup memory capacity"
                        )
                    raise RuntimeError("Inference runtime exited during startup")
                try:
                    response = await client.get(profile.upstream + profile.readiness_path)
                    if response.status_code in {200, 401, 403}:
                        if profile.preload_path:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            loaded = await client.post(
                                profile.upstream + profile.preload_path, timeout=remaining
                            )
                            if not loaded.is_success:
                                raise RuntimeError(
                                    "Runtime model preload failed "
                                    f"(HTTP {loaded.status_code})"
                                )
                        self.last_error = None
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError("Inference runtime startup deadline exceeded")


class DemandProxy:
    def __init__(
        self, manager: OnDemandInference, profile: InferenceProfile, internal: bool = False
    ):
        self.manager = manager
        self.profile = profile
        self.internal = internal

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        request = Request(scope, receive)
        path = scope["path"]
        # Inventory and health monitoring never warm up a model or prolong its lease.
        if request.method in {"GET", "HEAD"} and path in {"/health", "/healthz", "/v1/models"}:
            body = (
                {
                    "object": "list",
                    "data": [
                        {"id": self.profile.model, "object": "model", "owned_by": "spark-cluster"}
                    ],
                }
                if path == "/v1/models"
                else {"status": "ok", "mode": "on-demand"}
            )
            await JSONResponse(body)(scope, receive, send)
            return
        if self.manager.pool and not self.internal:
            model_id = self.profile.model_id or self.profile.model
            await self.manager.pool.infer(model_id, path.lstrip("/"), scope, receive, send)
            return
        if self.manager.pending >= 64:
            await JSONResponse({"error": "Inference queue is full"}, status_code=503)(
                scope, receive, send
            )
            return
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 64 * 1024 * 1024:
                await JSONResponse({"error": "Request exceeds 64 MiB"}, status_code=413)(
                    scope, receive, send
                )
                return
        forwarded = request_stream_usage(body, request.headers.get("content-type", ""), path)
        scope, feedback, send = response_feedback(
            scope, self.profile.model_id or self.profile.model, path, send
        )
        self.manager.pending += 1
        response_started = False
        try:
            async with asyncio.timeout(1800):
                async with self.manager.lease(self.profile, feedback):
                    mark_dispatch(scope)
                    query = scope.get("query_string", b"").decode("ascii")
                    raw_path = scope.get("raw_path", path.encode()).decode("ascii")
                    url = self.profile.upstream + raw_path + ("?" + query if query else "")
                    connection_tokens = {
                        token.strip().lower().encode()
                        for token in request.headers.get("connection", "").split(",")
                    }
                    headers = [
                        (k, v)
                        for k, v in scope["headers"]
                        if k.lower() not in HOP_HEADERS | connection_tokens | {b"content-length"}
                    ]
                    async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
                        async with client.stream(
                            request.method, url, headers=headers, content=forwarded
                        ) as upstream:
                            response = StreamingResponse(
                                upstream.aiter_raw(), status_code=upstream.status_code
                            )
                            excluded = HOP_HEADERS | {
                                token.strip().lower().encode()
                                for token in upstream.headers.get("connection", "").split(",")
                            }
                            response.raw_headers = [
                                (k, v) for k, v in upstream.headers.raw if k.lower() not in excluded
                            ]
                            response_started = True
                            await response(scope, receive, send)
        except (httpx.HTTPError, RuntimeError, TimeoutError):
            self.manager.last_error = "Inference request failed; inspect controller logs"
            logger.exception("On-demand inference failed for %s", self.profile.id)
            if response_started:
                raise
            if not response_started:
                await JSONResponse({"error": self.manager.last_error}, status_code=503)(
                    scope, receive, send
                )
        finally:
            self.manager.pending -= 1
