"""Opt-in, controller-owned Docker inference with stable legacy HTTP endpoints.

Recipes are trusted administrator input, never client-provided Docker specifications.
Only one runtime owns the GPU pool at a time; the lease lasts through the last byte.
"""

import asyncio
import contextlib
import copy
import json
import logging
import socket
import time
from pathlib import Path
from urllib.parse import quote

import httpx
import uvicorn
from pydantic import BaseModel, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from .inference_tracking import mark_dispatch
from .local_docker import DockerEngineClient

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
    listen_host: str = "127.0.0.1"
    listen_port: int = Field(ge=1024, le=65535)
    upstream_port: int = Field(ge=1024, le=65535)
    readiness_path: str = "/health"
    idle_seconds: float = Field(default=120, ge=1)
    startup_seconds: float = Field(default=900, ge=1)
    container: dict

    @model_validator(mode="after")
    def validate_ports(self):
        if self.listen_port == self.upstream_port:
            raise ValueError("Inference listener and runtime must use different ports")
        if not self.readiness_path.startswith("/"):
            raise ValueError("Readiness path must be relative")
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
        self.active: InferenceProfile | None = None
        self.last_used = 0.0
        self.servers: list[ListenerServer] = []
        self.tasks: list[asyncio.Task] = []
        self.sockets: list[socket.socket] = []
        self.pending = 0
        self.last_error: str | None = None
        self.state_path = state_path
        self.retained: set[str] = set()
        self.changed = asyncio.Event()
        self.dynamic: set[str] = set()
        self.pool = None
        if state_path and state_path.exists():
            state = json.loads(state_path.read_text())
            self.retained = set(state.get("retained", []))
            for item in state.get("profiles", []):
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

    def blocked(self, profile: InferenceProfile) -> bool:
        return bool(
            self.active and self.active.id != profile.id and self.active.id in self.retained
        )

    @contextlib.asynccontextmanager
    async def lease(self, profile: InferenceProfile):
        while True:
            await self.lock.acquire()
            if not self.blocked(profile):
                break
            self.changed.clear()
            self.lock.release()
            await self.changed.wait()
        try:
            yield
        finally:
            self.lock.release()
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
        name = quote(profile.name, safe="")
        response = await self.docker.request("GET", f"/containers/{name}/json", allow={404})
        if response.status_code == 404:
            if self.active and self.active.id == profile.id:
                self.active = None
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
        if self.active and self.active.id == profile.id:
            self.active = None
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
            for profile in self.profiles:
                state = await self.docker.request(
                    "GET", f"/containers/{profile.name}/json", allow={404}
                )
                if (
                    profile.id in self.retained
                    and state.status_code == 200
                    and state.json().get("State", {}).get("Running")
                    and state.json()
                    .get("Config", {})
                    .get("Labels", {})
                    .get("spark-cluster.on-demand")
                    == profile.id
                ):
                    if self.active:
                        raise RuntimeError("Multiple retained runtimes cannot share one GPU slot")
                    self.active = profile
                    self.last_used = time.monotonic()
                else:
                    await self.remove(profile)
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
            if self.active and self.active.id not in self.retained:
                await self.remove(self.active)
        for sock in self.sockets:
            sock.close()

    async def reap(self) -> None:
        while True:
            await asyncio.sleep(1)
            if self.lock.locked() or not self.active:
                continue
            async with self.lock:
                if (
                    self.active
                    and self.active.id not in self.retained
                    and time.monotonic() - self.last_used >= self.active.idle_seconds
                ):
                    try:
                        await self.remove(self.active)
                    except Exception:
                        self.last_error = "Idle runtime cleanup failed; inspect controller logs"
                        logger.exception("On-demand idle cleanup failed")

    async def ensure(self, profile: InferenceProfile) -> None:
        if self.blocked(profile):
            raise RuntimeError("The GPU slot is reserved by a retained model")
        if self.active and self.active.id == profile.id:
            state = await self.docker.request(
                "GET", f"/containers/{profile.name}/json", allow={404}
            )
            if state.status_code == 200 and state.json().get("State", {}).get("Running"):
                return
        if self.active:
            await self.remove(self.active)
        await self.remove(profile)
        payload = copy.deepcopy(profile.container)
        payload.setdefault("Labels", {})["spark-cluster.on-demand"] = profile.id
        payload.setdefault("HostConfig", {})["RestartPolicy"] = {"Name": "no"}
        payload["HostConfig"]["AutoRemove"] = False
        # Docker health probes must not call the public listener and trigger recursion.
        payload["Healthcheck"] = {"Test": ["NONE"]}
        created = await self.docker.request(
            "POST", "/containers/create", payload=payload, params={"name": profile.name}, timeout=30
        )
        container_id = created.json()["Id"]
        try:
            await self.docker.request("POST", f"/containers/{container_id}/start", timeout=30)
            deadline = time.monotonic() + profile.startup_seconds
            async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
                while time.monotonic() < deadline:
                    state = await self.docker.request("GET", f"/containers/{container_id}/json")
                    if not state.json().get("State", {}).get("Running"):
                        raise RuntimeError("Inference runtime exited during startup")
                    try:
                        response = await client.get(profile.upstream + profile.readiness_path)
                        if response.status_code in {200, 401, 403}:
                            self.active = profile
                            self.last_used = time.monotonic()
                            self.last_error = None
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
            raise TimeoutError("Inference runtime startup deadline exceeded")
        except BaseException:
            await self.remove(profile)
            raise


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
        self.manager.pending += 1
        response_started = False
        try:
            async with asyncio.timeout(1800):
                async with self.manager.lease(self.profile):
                    try:
                        await self.manager.ensure(self.profile)
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
                            if k.lower() not in HOP_HEADERS | connection_tokens
                        ]
                        async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
                            async with client.stream(
                                request.method, url, headers=headers, content=bytes(body)
                            ) as upstream:
                                response = StreamingResponse(
                                    upstream.aiter_raw(), status_code=upstream.status_code
                                )
                                excluded = HOP_HEADERS | {
                                    token.strip().lower().encode()
                                    for token in upstream.headers.get("connection", "").split(",")
                                }
                                response.raw_headers = [
                                    (k, v)
                                    for k, v in upstream.headers.raw
                                    if k.lower() not in excluded
                                ]
                                response_started = True
                                await response(scope, receive, send)
                    finally:
                        self.manager.last_used = time.monotonic()
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
