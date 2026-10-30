"""Attach externally managed OpenAI-compatible runtimes to a compute pool.

Strata is a native, systemd-managed serving stack on the Spark host. Spark Cluster
does not containerize or relaunch it; it records the endpoint, keeps the bearer key
outside the cluster registry, probes the runtime, and can proxy OpenAI requests and
run an explicit inference check through the same endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from .cluster_registry import ClusterRegistry, ClusterTarget
from .config import Settings
from .models import RuntimeEndpoint

EXTERNAL_ENGINES = ("strata",)
PROBE_TTL_SECONDS = 2.0
MODELS_TTL_SECONDS = 5.0


class ExternalRuntimeRequest(BaseModel):
    engine_id: Literal["strata", "none"] = "strata"
    port: int = Field(default=8080, ge=1024, le=65535)
    api_key: str | None = Field(default=None, max_length=1024)
    clear_api_key: bool = False


class ExternalRuntimeStatus(BaseModel):
    cluster_id: str
    engine_id: str | None = None
    base_url: str | None = None
    status: Literal["ready", "unreachable", "not_configured"] = "not_configured"
    model_ids: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    api_key_configured: bool = False
    error: str | None = None
    checked_at: datetime | None = None


class ExternalRuntimeTestRequest(BaseModel):
    cluster_id: str | None = None
    prompt: str = Field(default="Reply with exactly SPARK_OK.", min_length=1, max_length=4000)
    model: str | None = Field(default=None, max_length=200)
    max_tokens: int = Field(default=128, ge=1, le=4096)


class ExternalRuntimeTestResult(BaseModel):
    ok: bool
    status_code: int | None = None
    model: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    elapsed_ms: float = 0.0
    detail: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    status: Literal["ready", "unreachable"]
    models: tuple[dict, ...]
    latency_ms: float
    error: str | None = None


class RuntimeCredentialStore:
    """Per-cluster bearer keys kept outside the world-readable cluster registry."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text())
            values = payload.get("clusters", {})
            return {
                str(cluster_id): str(key)
                for cluster_id, key in values.items()
                if isinstance(cluster_id, str) and isinstance(key, str) and key
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _save(self, values: dict[str, str]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"version": 1, "clusters": values}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def get(self, cluster_id: str) -> str | None:
        return self._load().get(cluster_id)

    def set(self, cluster_id: str, api_key: str) -> None:
        values = self._load()
        values[cluster_id] = api_key
        self._save(values)

    def remove(self, cluster_id: str) -> None:
        values = self._load()
        if values.pop(cluster_id, None) is not None:
            self._save(values)


class ExternalRuntimeService:
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, settings: Settings, registry: ClusterRegistry):
        self.settings = settings
        self.registry = registry
        self.credentials = RuntimeCredentialStore(
            settings.setup_state_path / "runtime-credentials.json"
        )
        self._probes: dict[str, tuple[float, ProbeResult]] = {}
        self._probe_lock = asyncio.Lock()

    def cluster(self, cluster_id: str | None = None) -> ClusterTarget:
        return self.registry.get(cluster_id) if cluster_id else self.registry.active() or _missing()

    def configured(self, cluster: ClusterTarget) -> bool:
        return bool(cluster.runtime_engine in EXTERNAL_ENGINES and cluster.host_address)

    def base_url(self, cluster: ClusterTarget) -> str:
        if not cluster.host_address:
            raise ValueError("The compute pool has no LAN address")
        return f"http://{cluster.host_address}:{cluster.runtime_port}/v1"

    def authorization(self, cluster: ClusterTarget) -> str | None:
        return self.credentials.get(cluster.id)

    async def configure(
        self, cluster_id: str, request: ExternalRuntimeRequest
    ) -> ExternalRuntimeStatus:
        cluster = self.registry.get(cluster_id)
        if cluster.kind != "remote-bridge":
            raise ValueError("Only a remote compute pool can attach an externally managed runtime")
        if request.engine_id == "none":
            self.credentials.remove(cluster.id)
            cluster = self.registry.set_external_runtime(
                cluster.id, engine_id=None, port=cluster.runtime_port, api_key_configured=False
            )
            self._probes.pop(cluster.id, None)
            return await self.status_for(cluster)
        api_key = (request.api_key or "").strip()
        if request.clear_api_key:
            self.credentials.remove(cluster.id)
        elif api_key:
            self.credentials.set(cluster.id, api_key)
        configured = bool(self.credentials.get(cluster.id))
        cluster = self.registry.set_external_runtime(
            cluster.id,
            engine_id=request.engine_id,
            port=request.port,
            api_key_configured=configured,
        )
        self._probes.pop(cluster.id, None)
        return await self.status_for(cluster)

    async def clear(self, cluster_id: str) -> ExternalRuntimeStatus:
        return await self.configure(
            cluster_id,
            ExternalRuntimeRequest(
                engine_id="none", port=self.registry.get(cluster_id).runtime_port
            ),
        )

    async def status(self, cluster_id: str | None = None) -> ExternalRuntimeStatus:
        return await self.status_for(self.cluster(cluster_id))

    async def status_for(self, cluster: ClusterTarget) -> ExternalRuntimeStatus:
        if not self.configured(cluster):
            return ExternalRuntimeStatus(
                cluster_id=cluster.id,
                engine_id=cluster.runtime_engine,
                base_url=None,
                status="not_configured",
                api_key_configured=cluster.runtime_api_key_configured,
                checked_at=datetime.now(UTC),
            )
        probe = await self.probe(cluster)
        return ExternalRuntimeStatus(
            cluster_id=cluster.id,
            engine_id=cluster.runtime_engine,
            base_url=self.base_url(cluster),
            status=probe.status,
            model_ids=[str(model.get("id")) for model in probe.models if model.get("id")],
            latency_ms=round(probe.latency_ms, 1),
            api_key_configured=cluster.runtime_api_key_configured,
            error=probe.error,
            checked_at=datetime.now(UTC),
        )

    async def probe(self, cluster: ClusterTarget) -> ProbeResult:
        cached = self._probes.get(cluster.id)
        if cached and time.monotonic() - cached[0] < PROBE_TTL_SECONDS:
            return cached[1]
        async with self._probe_lock:
            cached = self._probes.get(cluster.id)
            if cached and time.monotonic() - cached[0] < PROBE_TTL_SECONDS:
                return cached[1]
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(
                    timeout=3,
                    headers=self._headers(cluster),
                    transport=self.transport,
                    trust_env=False,
                ) as client:
                    response = await client.get(self.base_url(cluster) + "/models")
                    response.raise_for_status()
                    payload = response.json()
                models = tuple(
                    item
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and item.get("id")
                )
                if not models:
                    raise ValueError("The runtime returned no models")
                result = ProbeResult(
                    status="ready",
                    models=models,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as error:
                result = ProbeResult(
                    status="unreachable",
                    models=(),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=_safe_error(error),
                )
            self._probes[cluster.id] = (time.monotonic(), result)
            return result

    async def models(self, cluster: ClusterTarget) -> list[dict]:
        if not self.configured(cluster):
            return []
        return list((await self.probe(cluster)).models)

    async def matching_model(self, reference: str) -> tuple[ClusterTarget, dict] | None:
        cluster = self.registry.active()
        if cluster is None or not self.configured(cluster):
            return None
        wanted = reference.strip().casefold()
        for model in await self.models(cluster):
            if str(model.get("id", "")).casefold() == wanted:
                return cluster, model
        return None

    async def chat(
        self, cluster: ClusterTarget, request: ExternalRuntimeTestRequest
    ) -> ExternalRuntimeTestResult:
        cluster = self.registry.get(cluster.id)
        if not self.configured(cluster):
            raise ValueError("No external runtime is attached to this compute pool")
        probe = await self.probe(cluster)
        if probe.status != "ready":
            raise ValueError(probe.error or "The external runtime is unreachable")
        model = request.model or str(probe.models[0]["id"])
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=300, transport=self.transport, trust_env=False
            ) as client:
                response = await client.post(
                    self.base_url(cluster) + "/chat/completions",
                    headers=self._headers(cluster),
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": request.prompt}],
                        "max_tokens": request.max_tokens,
                        "temperature": 0,
                    },
                )
            elapsed = (time.perf_counter() - started) * 1000
            if response.status_code >= 400:
                return ExternalRuntimeTestResult(
                    ok=False,
                    status_code=response.status_code,
                    model=model,
                    elapsed_ms=round(elapsed, 1),
                    detail=_response_detail(response),
                )
            payload = response.json()
            message = payload["choices"][0]["message"]
            content = message.get("content")
            reasoning = message.get("reasoning_content")
            return ExternalRuntimeTestResult(
                ok=bool(content or reasoning),
                status_code=response.status_code,
                model=str(payload.get("model") or model),
                content=content,
                reasoning_content=reasoning,
                elapsed_ms=round(elapsed, 1),
                detail=None if (content or reasoning) else "The runtime returned an empty message",
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as error:
            return ExternalRuntimeTestResult(
                ok=False,
                model=model,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                detail=_safe_error(error),
            )

    async def active_endpoint(self) -> RuntimeEndpoint | None:
        cluster = self.registry.active()
        if cluster is None or not self.configured(cluster):
            return None
        status = await self.status_for(cluster)
        base = self.base_url(cluster)
        return RuntimeEndpoint(
            base_url=base,
            models_url=base + "/models",
            chat_completions_url=base + "/chat/completions",
            status="ready" if status.status == "ready" else "unreachable",
            loaded_model_ids=status.model_ids,
            engine_id=cluster.runtime_engine,
            auth_mode="bearer" if cluster.runtime_api_key_configured else "none",
        )

    def _headers(self, cluster: ClusterTarget) -> dict[str, str]:
        key = self.credentials.get(cluster.id)
        return {"Authorization": f"Bearer {key}"} if key else {}


def _missing() -> ClusterTarget:
    raise KeyError("No active compute pool")


def _safe_error(error: BaseException) -> str:
    text = str(error).strip().replace("\n", " ")
    if isinstance(error, httpx.HTTPStatusError):
        return f"Runtime returned HTTP {error.response.status_code}"
    if isinstance(error, httpx.ConnectError):
        return "The runtime refused the connection"
    if isinstance(error, httpx.TimeoutException):
        return "The runtime did not answer in time"
    return (text or error.__class__.__name__)[:300]


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("error") or payload.get("detail")
            if isinstance(detail, dict):
                detail = detail.get("message")
            if detail:
                return str(detail)[:300]
    except (ValueError, TypeError):
        pass
    return f"Runtime returned HTTP {response.status_code}"
