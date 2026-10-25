"""Model-centric placement across registered clusters without changing UI selection."""

import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .inference_tracking import mark_dispatch, request_stream_usage
from .local_docker import local_gpu_snapshot
from .models import ModelFitPlan, ModelFitRequest, ModelLaunchRequest
from .on_demand import HOP_HEADERS, DemandProxy, InferenceProfile
from .remote_cluster import signed_controller_headers


def profile_runtime(profile: InferenceProfile) -> dict:
    payload = profile.container
    image = str(payload.get("Image", "")).casefold()
    command = " ".join(
        str(item) for item in [payload.get("Entrypoint"), payload.get("Cmd")] if item
    ).casefold()
    evidence = f"{image} {command} {profile.model_id or profile.model}".casefold()
    if "speaches" in evidence or "whisper" in evidence:
        engine_id, name = "speaches", "Speaches · faster-whisper"
        reason = "Speech-to-text model detected; using the CUDA faster-whisper service."
    elif "llama" in evidence or "gguf" in evidence or "bonsai" in evidence:
        engine_id, name = "llama-cpp", "llama.cpp · Prism"
        reason = "GGUF model detected; using its validated llama.cpp runtime."
    elif "vllm" in evidence:
        engine_id, name = "vllm", "vLLM"
        if "embedding" in evidence or "bge" in evidence or "pooling" in evidence:
            reason = "Embedding model detected; using vLLM pooling mode."
        elif "awq" in evidence:
            reason = "AWQ transformer detected; using the validated vLLM quantized runtime."
        else:
            reason = "Transformer model detected; using the validated vLLM runtime."
    else:
        engine_id, name = "container", "Specialized container"
        reason = "Using the model's validated imported runtime recipe."
    return {
        "engine_id": engine_id,
        "name": name,
        "selection": "automatic",
        "source": "detected-profile",
        "reason": reason,
        "image": payload.get("Image"),
        "executable": True,
        "warnings": [],
        "blockers": [],
    }


def execution_runtime(plan, requested_engine: str) -> dict:
    return {
        "engine_id": plan.engine_id,
        "name": plan.engine_name,
        "selection": "automatic" if requested_engine == "auto" else "configured",
        "source": "execution-plan",
        "reason": (
            "Selected automatically from model format, workload settings, cluster hardware, "
            "and installed executable adapters."
            if requested_engine == "auto"
            else "Selected by the saved model configuration and validated on this cluster."
        ),
        "image": plan.image,
        "executable": plan.executable,
        "warnings": plan.warnings,
        "blockers": plan.blockers,
    }


class ModelPreferences(BaseModel):
    keep_loaded: bool = False
    cluster_id: str | None = None
    last_settings: ModelLaunchRequest | None = None


class ModelRetentionRequest(BaseModel):
    keep_loaded: bool


class PoolLoadRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    settings: ModelLaunchRequest | None = None
    cluster_id: str | None = None


class PoolScaleRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    delta: Literal[-1, 1]


class NodeLoadRequest(PoolLoadRequest):
    keep_loaded: bool = False


def write_private(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
    temporary = path.with_suffix(".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class PoolNode:
    """An authenticated physical cluster adapter, independent of selected dashboard cluster."""

    transport = None

    def __init__(self, settings, registry, models, deployments, demand):
        self.settings = settings
        self.registry = registry
        self.models = models
        self.deployments = deployments
        self.demand = demand
        self.path = settings.setup_state_path / "pool-node.json"
        self.metadata_path = settings.setup_state_path / "runtime-model-metadata.json"
        self.native = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.runtime_metadata = (
            json.loads(self.metadata_path.read_text()) if self.metadata_path.exists() else {}
        )
        self._metadata_checked = {key: time.monotonic() for key in self.runtime_metadata}
        self.tracker = None
        self.native_lock = asyncio.Lock()
        self.reaper = None

    async def capture_runtime_metadata(self, model_id: str, base_url: str) -> None:
        now = time.monotonic()
        if now - self._metadata_checked.get(model_id, -float("inf")) < 60:
            return
        if len(self._metadata_checked) >= 256:
            self._metadata_checked.clear()
        self._metadata_checked[model_id] = now
        try:
            async with httpx.AsyncClient(
                timeout=5, trust_env=False, transport=self.transport
            ) as client:
                response = await client.get(base_url.rstrip("/") + "/v1/models")
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                return
            if not all(isinstance(item, dict) for item in payload["data"]):
                return
            # Preserve the runtime response verbatim. Spark does not infer optional metadata.
            self.runtime_metadata[model_id] = payload
            write_private(self.metadata_path, self.runtime_metadata)
        except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
            return

    def metadata_for(self, model_id: str) -> dict | None:
        payload = self.runtime_metadata.get(model_id)
        return payload if isinstance(payload, dict) else None

    async def start(self):
        self.reaper = asyncio.create_task(self.reap_native())

    async def close(self):
        if self.reaper:
            self.reaper.cancel()
            await asyncio.gather(self.reaper, return_exceptions=True)

    async def reap_native(self):
        while True:
            await asyncio.sleep(2)
            if (
                self.native.get("model_id")
                and not self.native.get("keep_loaded")
                and not self.native_lock.locked()
                and time.time() - self.native.get("last_used", time.time()) >= 120
            ):
                await self.unload(self.native["model_id"])

    async def infer(self, model_id, path, scope, receive, send):
        if self.tracker:

            async def handler(scope, receive, send):
                await self._infer_local(model_id, path, scope, receive, send)

            return await self.tracker.track(handler, model_id, path, scope, receive, send)
        return await self._infer_local(model_id, path, scope, receive, send)

    async def _infer_local(self, model_id, path, scope, receive, send):
        cluster = self.local_cluster()
        if record := scope.get("spark.inference"):
            record.cluster_id = record.cluster_id or (cluster.id if cluster else "self")
        profile = self.demand.profile_for(model_id)
        scope = dict(scope, path="/" + path, raw_path=("/" + path).encode())
        if profile:
            try:
                await DemandProxy(self.demand, profile, internal=True)(scope, receive, send)
            finally:
                await self.capture_runtime_metadata(model_id, profile.upstream)
            return
        async with self.native_lock:
            if self.native.get("model_id") != model_id:
                await JSONResponse(
                    {"error": "Model residency changed; retry the request"}, status_code=503
                )(scope, receive, send)
                return
            try:
                mark_dispatch(scope)
                base = self.settings.direct_runtime_base_url.removesuffix("/v1")
                await forward_http(base + "/" + path, scope, receive, send)
            finally:
                await self.capture_runtime_metadata(model_id, base)
                self.native["last_used"] = time.time()
                write_private(self.path, self.native)

    def local_cluster(self):
        return next((c for c in self.registry.list() if c.kind == "local-docker"), None)

    @staticmethod
    def profile_context(profile: InferenceProfile | None, fallback: int) -> int:
        if not profile:
            return fallback
        env = profile.container.get("Env") or []
        for item in env:
            if isinstance(item, str) and item.startswith("CTX="):
                try:
                    return int(item.split("=", 1)[1])
                except ValueError:
                    pass
        command = profile.container.get("Cmd") or []
        for flag in ("--max-model-len", "--context-length", "-c", "--ctx-size"):
            if flag in command:
                try:
                    return int(command[command.index(flag) + 1])
                except (ValueError, IndexError, TypeError):
                    pass
        return fallback

    async def model_capacity(self, model: dict, available_gib: float) -> dict | None:
        model_path = model.get("model_path")
        if not model_path:
            return None
        manifest_fit = getattr(self.models, "recorded_fit", lambda _: None)(model["id"])
        if manifest_fit is None:
            try:
                payload = await asyncio.to_thread(
                    lambda: json.loads((Path(model_path) / ".spark-cluster.json").read_text())
                )
                manifest_fit = ModelFitPlan.model_validate(payload["fit"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        saved = model.get("last_settings")
        profile = model.get("profile")
        if saved:
            launch = ModelLaunchRequest.model_validate(saved)
            request = ModelFitRequest(
                repo_id=model["id"],
                revision=manifest_fit.revision if manifest_fit else "main",
                bits_per_weight=launch.bits_per_weight,
                context_length=launch.context_length,
                concurrent_sequences=launch.concurrent_sequences,
                kv_cache_dtype=launch.kv_cache_dtype,
            )
            fit = await self.models.fit_for_local(request)
        elif manifest_fit:
            context = self.profile_context(profile, manifest_fit.context_length)
            if context != manifest_fit.context_length:
                fit = await self.models.fit_for_local(
                    ModelFitRequest(
                        repo_id=model["id"],
                        revision=manifest_fit.revision,
                        bits_per_weight=(
                            manifest_fit.detected_bits_per_weight or manifest_fit.bits_per_weight
                        ),
                        context_length=context,
                        concurrent_sequences=manifest_fit.concurrent_sequences,
                        kv_cache_dtype=manifest_fit.kv_cache_dtype,
                    )
                )
            else:
                fit = manifest_fit
        else:
            return None
        required_gib = (
            fit.resident_weights_gib + fit.resident_kv_cache_gib + fit.runtime_overhead_gib
        )
        if required_gib <= 0:
            required_gib = fit.total_required_gib
        missing_gib = max(0.0, required_gib - available_gib)
        return {
            "fits": missing_gib <= 0.01,
            "required_gib": round(required_gib, 2),
            "available_gib": round(available_gib, 2),
            "missing_gib": round(missing_gib, 2),
            "context_length": fit.context_length,
            "confidence": fit.confidence,
        }

    def machine_inventory(self, snapshot=None) -> list[dict]:
        cluster = self.local_cluster()
        if snapshot is None:
            snapshot = (
                local_gpu_snapshot(cluster, self.settings)
                if cluster
                else self.models.runtime.latest
            )
        if not snapshot:
            return []
        grouped: dict[str, dict] = {}
        for node in snapshot.nodes:
            key = (node.address or snapshot.cluster_name) if cluster else node.id
            machine = grouped.setdefault(
                key,
                {
                    "id": key,
                    "name": snapshot.cluster_name if cluster else node.hostname,
                    "address": node.address,
                    "available": True,
                    "accelerators": 0,
                    "memory_used_gib": 0.0,
                    "memory_total_gib": 0.0,
                },
            )
            machine["available"] = machine["available"] and node.status != "offline"
            machine["accelerators"] += 1
            machine["memory_used_gib"] += node.memory.accelerator_used_gib or node.memory.used_gib
            machine["memory_total_gib"] += node.memory.total_gib
        for machine in grouped.values():
            machine["memory_used_gib"] = round(machine["memory_used_gib"], 2)
            machine["memory_total_gib"] = round(machine["memory_total_gib"], 2)
        return list(grouped.values())

    async def inventory(self) -> dict:
        active_profile = self.demand.active
        if active_profile:
            await self.capture_runtime_metadata(
                active_profile.model_id or active_profile.model, active_profile.upstream
            )
        elif self.native.get("model_id"):
            await self.capture_runtime_metadata(
                self.native["model_id"],
                self.settings.direct_runtime_base_url.removesuffix("/v1"),
            )
        models = {
            m.id: {
                "id": m.id,
                "size_bytes": m.size_bytes,
                "loadable": m.launch_supported is not False,
                "loaded": False,
                "keep_loaded": False,
                "editable_settings": True,
                "aliases": [m.id],
                "served_model_name": None,
                "model_path": m.path,
                "runtime": None,
            }
            for m in await self.models.local_models()
            if m.source == "managed"
        }
        for job in await self.deployments.launches():
            if job.status == "ready" and job.request.model_id in models:
                entry = models[job.request.model_id]
                entry.setdefault("last_settings", job.request.model_dump())
        for p in self.demand.profiles:
            key = p.model_id or p.model
            models[key] = {
                **models.get(key, {}),
                "id": key,
                "loadable": True,
                "loaded": bool(self.demand.active and self.demand.active.id == p.id),
                "keep_loaded": p.id in self.demand.retained,
                "last_settings": p.launch_settings or None,
                "editable_settings": p.id in self.demand.dynamic,
                "endpoint": f"http://{self.settings.docker_runtime_host}:{p.listen_port}"
                if p.id not in self.demand.dynamic
                else None,
                "aliases": sorted({p.id, p.model, key}),
                "served_model_name": p.model,
                "runtime": profile_runtime(p),
                "profile": p,
            }
        active = self.demand.active
        active_id = active.model_id or active.model if active else self.native.get("model_id")
        retained = (
            active.id in self.demand.retained if active else self.native.get("keep_loaded", False)
        )
        if active_id in models:
            models[active_id]["loaded"] = True
            models[active_id]["keep_loaded"] = bool(retained)
        cluster = self.local_cluster()
        snapshot = (
            await asyncio.to_thread(local_gpu_snapshot, cluster, self.settings) if cluster else None
        )
        if snapshot:
            usable_total = sum(
                node.memory.total_gib * self.settings.model_usable_memory_fraction
                for node in snapshot.nodes
            )
            external_used = (
                0.0
                if active_id
                else sum(
                    node.memory.accelerator_used_gib or node.memory.used_gib
                    for node in snapshot.nodes
                )
            )
            available_gib = max(0.0, usable_total - external_used)
        else:
            available_gib = (
                self.settings.fallback_node_count
                * self.settings.fallback_node_memory_gib
                * self.settings.model_usable_memory_fraction
            )
        for model in models.values():
            model["capacity"] = await self.model_capacity(model, available_gib)
            if model.get("runtime") or not model.get("model_path"):
                continue
            saved = model.get("last_settings")
            launch = (
                ModelLaunchRequest.model_validate(saved)
                if saved
                else ModelLaunchRequest(
                    model_id=model["id"],
                    served_model_name=model["id"].rsplit("/", 1)[-1],
                    context_length=8192,
                )
            )
            plan = self.deployments.execution_plan(
                launch, model["model_path"], self.local_cluster(), self.settings
            )
            model["runtime"] = execution_runtime(plan, launch.engine_id)
        for model in models.values():
            model["runtime_models"] = self.metadata_for(model["id"])
            model.pop("model_path", None)
            model.pop("profile", None)
        return {
            "models": list(models.values()),
            "active_model": active_id,
            "keep_loaded": bool(retained),
            "busy": self.demand.lock.locked() or self.native_lock.locked(),
            "machines": await asyncio.to_thread(self.machine_inventory, snapshot),
        }

    def model_context(self):
        cluster = self.local_cluster()
        if not cluster:
            return self.models, self.settings
        settings = self.settings.model_copy(deep=True)
        settings.remote_cluster_url = None
        settings.active_hardware = cluster.hardware
        settings.active_compute_capability = min(
            (float(g.compute_capability) for g in cluster.gpus if g.compute_capability),
            default=None,
        )
        settings.fallback_node_count = cluster.gpu_count
        settings.fallback_node_memory_gib = cluster.gpu_memory_gib
        models = copy.copy(self.models)
        models.settings = settings
        models.runtime = SimpleNamespace(latest=local_gpu_snapshot(cluster, settings))
        return models, settings

    async def prepare(self, request: PoolLoadRequest):
        if request.settings and request.settings.model_id != request.model_id:
            raise ValueError("Launch settings must belong to this model")
        profile = self.demand.profile_for(request.model_id)
        if profile and profile.id not in self.demand.dynamic:
            if request.settings:
                raise ValueError("This imported runtime uses its saved service configuration")
            return profile, None, None, None
        models, settings = self.model_context()
        local = await models.managed_model(request.model_id)
        if local is None or not local.path:
            raise ValueError("Model files are not installed on this cluster")
        saved = next(
            (
                j.request
                for j in await self.deployments.launches()
                if j.request.model_id == request.model_id and j.status == "ready"
            ),
            None,
        )
        launch = (
            request.settings
            or (
                ModelLaunchRequest.model_validate(profile.launch_settings)
                if profile and profile.launch_settings
                else saved
            )
            or ModelLaunchRequest(
                model_id=request.model_id,
                served_model_name=request.model_id.rsplit("/", 1)[-1],
                context_length=8192,
            )
        )
        fit = await models.fit_for_local(
            ModelFitRequest(
                repo_id=request.model_id,
                revision=local.revision or "main",
                bits_per_weight=launch.bits_per_weight,
                context_length=launch.context_length,
                concurrent_sequences=launch.concurrent_sequences,
                kv_cache_dtype=launch.kv_cache_dtype,
            )
        )
        if not fit.fits:
            raise ValueError("This model and its saved settings do not fit this cluster")
        plan = self.deployments.execution_plan(launch, local.path, self.local_cluster(), settings)
        if not plan.executable or not plan.image:
            raise ValueError("This cluster has no executable engine for these settings")
        cluster = self.local_cluster()
        if cluster:
            if not await self.demand.docker.image_present(plan.image):
                raise ValueError("Required runtime image is not installed on this cluster")
            key = "model-" + hashlib.sha256(request.model_id.encode()).hexdigest()[:12]
            # One physical runtime at a time per cluster; reserve a separate private port.
            port = profile.upstream_port if profile else 31000 + len(self.demand.dynamic)
            used_ports = {
                p.upstream_port for p in self.demand.profiles if not profile or p.id != profile.id
            }
            while port in used_ports:
                port += 1
            cluster = cluster.model_copy(update={"runtime_port": port})
            payload = self.demand.docker.runtime_payload(
                cluster=cluster,
                engine_id=plan.engine_id,
                image=plan.image,
                model_path=local.path,
                served_model_name=launch.served_model_name,
                context_length=launch.context_length,
                tensor_parallel=launch.tensor_parallel,
                pipeline_parallel=launch.pipeline_parallel,
                gpu_memory_utilization=launch.gpu_memory_utilization,
                kv_cache_dtype=launch.kv_cache_dtype,
                max_num_seqs=launch.max_num_seqs,
                max_num_batched_tokens=launch.max_num_batched_tokens,
                expert_parallel=launch.expert_parallel,
                chunked_prefill=launch.chunked_prefill,
            )
            payload["Labels"]["spark-cluster.model-id"] = request.model_id
            # Bind only loopback; inference is routed through the authenticated pool gateway.
            payload["HostConfig"]["NetworkMode"] = "bridge"
            payload["HostConfig"]["PortBindings"] = {
                f"{port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]
            }
            payload["ExposedPorts"] = {f"{port}/tcp": {}}
            profile = InferenceProfile(
                id=key,
                model=launch.served_model_name,
                model_id=request.model_id,
                listen_port=port + 1000,
                upstream_port=port,
                readiness_path="/v1/models",
                container=payload,
                launch_settings=launch.model_dump(),
            )
        return profile, launch, local, plan

    async def plan(self, request: PoolLoadRequest) -> dict:
        try:
            profile, launch, _, plan = await self.prepare(request)
            runtime = (
                profile_runtime(profile)
                if plan is None
                else execution_runtime(plan, launch.engine_id)
            )
            return {"compatible": True, "runtime": runtime}
        except (ValueError, RuntimeError) as error:
            return {"compatible": False, "reason": self.models.safe_error(error)}

    async def load(self, request: NodeLoadRequest) -> dict:
        profile, launch, _, _ = await self.prepare(request)
        if profile:
            async with self.demand.lock:
                if self.demand.blocked(profile):
                    raise ValueError("The cluster is reserved by a retained model")
                previous = self.demand.profile_for(request.model_id)
                if previous and previous.id in self.demand.dynamic:
                    if self.demand.active and self.demand.active.id == previous.id:
                        if request.settings and previous.launch_settings != profile.launch_settings:
                            if previous.id in self.demand.retained:
                                raise ValueError(
                                    "Unload the retained model before changing its runtime settings"
                                )
                            await self.demand.remove(previous)
                    self.demand.profiles[self.demand.profiles.index(previous)] = profile
                    self.demand.persist()
                else:
                    self.demand.register(profile)
                await self.demand.ensure(profile)
                await self.capture_runtime_metadata(request.model_id, profile.upstream)
                self.demand.last_used = time.monotonic()
                self.demand.retain(profile, request.keep_loaded)
                return {
                    "model_id": request.model_id,
                    "status": "ready",
                    "settings": profile.launch_settings or None,
                }
        # Physical Kubernetes bridge: the existing validated launcher owns the runtime.
        async with self.native_lock:
            if self.native.get("keep_loaded") and self.native.get("model_id") != request.model_id:
                raise ValueError("The cluster is reserved by a retained model")
            if self.native.get("model_id") == request.model_id:
                self.native.update(keep_loaded=request.keep_loaded, last_used=time.time())
            else:
                job = await self.deployments.launch(launch)
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    current = next(j for j in await self.deployments.launches() if j.id == job.id)
                    if current.status == "failed":
                        raise RuntimeError(current.error or "Runtime launch failed")
                    if current.status == "ready":
                        break
                    await asyncio.sleep(2)
                else:
                    raise TimeoutError("Runtime launch deadline exceeded")
                await self.capture_runtime_metadata(
                    request.model_id, self.settings.direct_runtime_base_url.removesuffix("/v1")
                )
                self.native = {
                    "model_id": request.model_id,
                    "keep_loaded": request.keep_loaded,
                    "last_used": time.time(),
                }
            write_private(self.path, self.native)
            return {
                "model_id": request.model_id,
                "status": "ready",
                "settings": launch.model_dump(),
            }

    async def retain(self, model_id: str, retained: bool):
        profile = self.demand.profile_for(model_id)
        if profile:
            async with self.demand.lock:
                if retained and (not self.demand.active or self.demand.active.id != profile.id):
                    raise ValueError("The model was unloaded; load it again before retaining it")
                self.demand.retain(profile, retained)
        async with self.native_lock:
            if self.native.get("model_id") == model_id:
                self.native["keep_loaded"] = retained
                write_private(self.path, self.native)

    async def unload(self, model_id: str):
        profile = self.demand.profile_for(model_id)
        if profile:
            async with self.demand.lock:
                if self.demand.active and self.demand.active.id == profile.id:
                    await self.demand.remove(profile)
                self.demand.retain(profile, False)
            return
        async with self.native_lock:
            if self.native.get("model_id") == model_id:
                if self.deployments.mode != "kubernetes":
                    raise ValueError("No managed Kubernetes runtime can be unloaded")
                k = self.deployments.kubernetes
                await k.patch_deployment(
                    self.settings.runtime_head_deployment, {"spec": {"replicas": 0}}
                )
                await k.patch_statefulset(
                    self.settings.runtime_worker_statefulset, {"spec": {"replicas": 0}}
                )
                self.native = {}
                write_private(self.path, self.native)


class ModelPool:
    transport = None

    def __init__(self, settings, registry, node):
        self.settings = settings
        self.registry = registry
        self.node = node
        self.path = settings.setup_state_path / "model-pool.json"
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.preferences = {
            k: ModelPreferences.model_validate(v) for k, v in saved.get("models", {}).items()
        }
        self.jobs = saved.get("jobs", [])[-100:]
        for job in self.jobs:
            if job["status"] not in {"ready", "failed", "cancelled"}:
                job.update(status="failed", message="Controller restarted; load again to retry")
        self.tasks = set()
        self.reserved = set()
        self.scheduling = asyncio.Lock()
        self.cache = None
        self.cache_time = 0.0
        self.pending_requests = 0
        self._routing_cursor = {}
        self._inventory_task = None
        self.tracker = None

    def save(self):
        self.jobs = self.jobs[-100:]
        write_private(
            self.path,
            {
                "models": {k: v.model_dump(mode="json") for k, v in self.preferences.items()},
                "jobs": self.jobs,
            },
        )

    async def close(self):
        if self._inventory_task and not self._inventory_task.done():
            self._inventory_task.cancel()
            await asyncio.gather(self._inventory_task, return_exceptions=True)
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.save()

    def local(self, cluster):
        return cluster.kind == "local-docker"

    async def call(self, cluster, action, payload=None):
        if self.local(cluster):
            if action == "status":
                return await self.node.inventory()
            if action == "plan":
                return await self.node.plan(PoolLoadRequest.model_validate(payload))
            if action == "load":
                return await self.node.load(NodeLoadRequest.model_validate(payload))
            if action == "retain":
                await self.node.retain(payload["model_id"], payload["keep_loaded"])
            if action == "unload":
                await self.node.unload(payload["model_id"])
            return {"status": "ok"}
        if not self.settings.model_admin_token:
            raise ValueError("Configure the shared administration token to use remote pool members")
        method = "GET" if action == "status" else "POST"
        path = "/api/v1/pool/node/" + action
        body = json.dumps(payload).encode() if payload is not None else b""
        headers = signed_controller_headers(self.settings.model_admin_token, method, path, body)
        headers["content-type"] = "application/json"
        async with httpx.AsyncClient(
            base_url=cluster.endpoint,
            transport=self.transport,
            timeout=1250 if action == "load" else 30,
            trust_env=False,
        ) as client:
            response = await client.request(method, path, headers=headers, content=body)
            response.raise_for_status()
            return response.json()

    async def inventory(self, force=False):
        if not force and self.cache is not None and time.monotonic() - self.cache_time < 3:
            return self.cache
        if self._inventory_task is None or self._inventory_task.done():
            self._inventory_task = asyncio.create_task(self._refresh_inventory())
        return await asyncio.shield(self._inventory_task)

    async def _refresh_inventory(self):
        clusters = self.registry.list()
        # A Kubernetes bridge with no local Docker registration is itself a pool member.
        if not clusters and self.node.deployments.mode in {"kubernetes", "demo"}:
            clusters = [
                SimpleNamespace(
                    id="self",
                    name=self.settings.cluster_name,
                    kind="local-docker",
                    status="ready",
                    model_dump=lambda **kw: {},
                )
            ]

        async def inspect(cluster):
            try:
                data = await self.call(cluster, "status")
                return {"id": cluster.id, "name": cluster.name, "available": True, **data}
            except (httpx.HTTPError, ValueError, RuntimeError):
                return {
                    "id": cluster.id,
                    "name": cluster.name,
                    "available": False,
                    "models": [],
                    "active_model": None,
                    "keep_loaded": False,
                    "busy": False,
                    "machines": [],
                }

        nodes = await asyncio.gather(*(inspect(c) for c in clusters))
        models = {}
        for node in nodes:
            for m in node["models"]:
                entry = models.setdefault(
                    m["id"],
                    {
                        "id": m["id"],
                        "size_bytes": m.get("size_bytes", 0),
                        "placements": [],
                        "available_clusters": [],
                        "loadable": False,
                        "last_settings": m.get("last_settings"),
                        "editable_settings": m.get("editable_settings", True),
                        "aliases": list(m.get("aliases", [m["id"]])),
                        "served_model_name": m.get("served_model_name"),
                        "runtime": m.get("runtime"),
                        "runtime_models": m.get("runtime_models"),
                        "cluster_capacity": [],
                    },
                )
                entry["aliases"] = sorted(set(entry["aliases"] + m.get("aliases", [])))
                entry["served_model_name"] = m.get("served_model_name") or entry.get(
                    "served_model_name"
                )
                if not entry.get("runtime") or m.get("runtime", {}).get("executable"):
                    entry["runtime"] = m.get("runtime") or entry.get("runtime")
                if m.get("runtime_models"):
                    entry["runtime_models"] = m["runtime_models"]
                capacity = m.get("capacity")
                entry["cluster_capacity"].append(
                    {
                        "cluster_id": node["id"],
                        "cluster_name": node["name"],
                        "available": node["available"],
                        **(capacity or {"fits": False, "missing_gib": None}),
                    }
                )
                if m["loadable"]:
                    entry["loadable"] = True
                    if capacity and capacity["fits"]:
                        entry["available_clusters"].append(node["id"])
                if m.get("loaded"):
                    entry["placements"].append(
                        {
                            "cluster_id": node["id"],
                            "cluster_name": node["name"],
                            "keep_loaded": m.get("keep_loaded", False),
                        }
                    )
        for m in models.values():
            prefs = self.preferences.get(m["id"], ModelPreferences())
            m["key"] = hashlib.sha256(m["id"].encode()).hexdigest()[:16]
            m["endpoint_path"] = "/api/v1/pool/serve/" + m["key"] + "/v1"
            m["preferences"] = prefs.model_dump(mode="json")
            m["last_settings"] = (
                prefs.last_settings.model_dump() if prefs.last_settings else m["last_settings"]
            )
            if m["last_settings"]:
                served = m["last_settings"].get("served_model_name")
                if served:
                    m["served_model_name"] = served
                    m["aliases"] = sorted(set(m["aliases"] + [served]))
        result = {"clusters": nodes, "models": list(models.values()), "jobs": self.jobs[-20:]}
        self.cache = result
        self.cache_time = time.monotonic()
        return result

    def cluster(self, cluster_id):
        if cluster_id == "self" and not self.registry.list():
            return SimpleNamespace(id="self", kind="local-docker")
        return self.registry.get(cluster_id)

    def resident_for(self, model_id, clusters, preferences):
        residents = [
            node
            for node in clusters
            if node["available"]
            and node["active_model"] == model_id
            and node["id"] not in self.reserved
            and (not preferences.cluster_id or node["id"] == preferences.cluster_id)
        ]
        if not residents:
            return None
        idle = sorted((node for node in residents if not node["busy"]), key=lambda node: node["id"])
        choices = idle or sorted(residents, key=lambda node: node["id"])
        cursor = self._routing_cursor.get(model_id, 0)
        selected = choices[cursor % len(choices)]
        self._routing_cursor[model_id] = cursor + 1
        return selected

    async def resolve_model(self, reference: str) -> dict:
        reference = reference.strip()
        if not reference:
            raise ValueError("A model is required")
        inventory = await self.inventory()
        matches = [
            model
            for model in inventory["models"]
            if reference == model["id"] or reference in model.get("aliases", [])
        ]
        if not matches:
            raise ValueError(f"Model {reference} is not available in the pool")
        if len(matches) > 1:
            exact = [model for model in matches if model["id"] == reference]
            if len(exact) == 1:
                return exact[0]
            raise ValueError(f"Model name {reference} is ambiguous; use its full model id")
        return matches[0]

    async def update_preferences(self, model_id, preferences):
        if preferences.cluster_id:
            self.cluster(preferences.cluster_id)
        if preferences.last_settings and preferences.last_settings.model_id != model_id:
            raise ValueError("Saved settings must belong to this model")
        state = await self.inventory(force=True)
        previous = self.preferences.get(model_id, ModelPreferences())
        applied = []
        try:
            for node in state["clusters"]:
                if node["active_model"] == model_id:
                    await self.call(
                        self.cluster(node["id"]),
                        "retain",
                        {"model_id": model_id, "keep_loaded": preferences.keep_loaded},
                    )
                    applied.append(node["id"])
        except Exception:
            for cid in applied:
                await self.call(
                    self.cluster(cid),
                    "retain",
                    {"model_id": model_id, "keep_loaded": previous.keep_loaded},
                )
            raise
        self.preferences[model_id] = preferences
        self.save()
        self.cache = None
        return preferences

    async def update_retention(self, model_id: str, keep_loaded: bool):
        preferences = self.preferences.get(model_id, ModelPreferences()).model_copy(deep=True)
        preferences.keep_loaded = keep_loaded
        return await self.update_preferences(model_id, preferences)

    async def enqueue(self, request, *, new_instance=False):
        for job in self.jobs:
            if job["model_id"] == request.model_id and job["status"] in {
                "queued",
                "waiting",
                "loading",
            }:
                return job
        if len(self.tasks) >= 64:
            raise ValueError("The pool load queue is full")
        if request.settings:
            if request.settings.model_id != request.model_id:
                raise ValueError("Launch settings must belong to this model")
            prefs = self.preferences.get(request.model_id, ModelPreferences()).model_copy(
                update={"last_settings": request.settings}
            )
            self.preferences[request.model_id] = prefs
        job = {
            "id": uuid4().hex,
            "model_id": request.model_id,
            "status": "queued",
            "message": "Waiting for a compatible cluster",
            "cluster_id": None,
            "stage": "queued",
            "progress": 2.0,
            "machines": [],
            "completed_at": None,
            "new_instance": new_instance,
        }
        self.jobs.append(job)
        self.save()
        task = asyncio.create_task(self.run(job, request))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return job

    async def animate_load(self, job: dict) -> None:
        while job.get("status") == "loading":
            await asyncio.sleep(1)
            if job.get("status") != "loading":
                return
            job["progress"] = min(90.0, float(job.get("progress", 20)) + 2.0)
            machines = job.get("machines", [])
            count = max(1, len(machines))
            for index, machine in enumerate(machines):
                offset = index * (12.0 / count)
                machine["progress"] = min(
                    88.0,
                    max(10.0, job["progress"] - offset),
                )
                machine["status"] = "loading-runtime"
            self.save()

    async def run(self, job, request):
        selected = None
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                async with self.scheduling:
                    state = await self.inventory(force=True)
                    prefs = self.preferences.get(request.model_id, ModelPreferences())
                    target_cluster_id = request.cluster_id or prefs.cluster_id
                    if target_cluster_id:
                        self.cluster(target_cluster_id)
                    settings = request.settings or prefs.last_settings
                    payload = {
                        "model_id": request.model_id,
                        "settings": settings.model_dump() if settings else None,
                        "keep_loaded": prefs.keep_loaded,
                    }
                    candidates = [
                        n
                        for n in state["clusters"]
                        if n["available"]
                        and (not target_cluster_id or n["id"] == target_cluster_id)
                        and (not job.get("new_instance") or n["active_model"] != request.model_id)
                        and any(m["id"] == request.model_id and m["loadable"] for m in n["models"])
                    ]
                    if not candidates:
                        raise ValueError(
                            "No reachable cluster has a runnable copy of this model"
                            + (" on the required cluster" if target_cluster_id else "")
                        )
                    # Reuse residency, then idle resources, then evict an unretained model.
                    candidates.sort(
                        key=lambda n: (
                            n["active_model"] != request.model_id,
                            n["active_model"] is not None,
                            n["id"],
                        )
                    )
                    reasons = []
                    compatible = False
                    for node in candidates:
                        if node["id"] in self.reserved or node["busy"]:
                            compatible = True
                            reasons.append(node["name"] + ": another request is active")
                            continue
                        if node["keep_loaded"] and node["active_model"] != request.model_id:
                            compatible = True
                            reasons.append(node["name"] + ": retained model reserves the resources")
                            continue
                        cluster = self.cluster(node["id"])
                        try:
                            plan = await self.call(cluster, "plan", payload)
                        except (httpx.HTTPError, ValueError, RuntimeError):
                            reasons.append(node["name"] + ": cluster preflight unavailable")
                            continue
                        if not plan["compatible"]:
                            reasons.append(
                                node["name"] + ": " + plan.get("reason", "Incompatible settings")
                            )
                            continue
                        compatible = True
                        selected = cluster
                        job["runtime"] = plan.get("runtime")
                        self.reserved.add(cluster.id)
                        machines = [
                            {
                                **machine,
                                "status": "starting",
                                "progress": 15.0,
                                "memory_before_gib": machine.get("memory_used_gib", 0.0),
                                "memory_after_gib": None,
                            }
                            for machine in node.get("machines", [])
                        ] or [
                            {
                                "id": cluster.id,
                                "name": node["name"],
                                "address": None,
                                "accelerators": None,
                                "memory_used_gib": 0.0,
                                "memory_total_gib": 0.0,
                                "memory_before_gib": 0.0,
                                "memory_after_gib": None,
                                "status": "starting",
                                "progress": 15.0,
                            }
                        ]
                        job.update(
                            status="loading",
                            stage="starting-runtime",
                            progress=20.0,
                            cluster_id=cluster.id,
                            machines=machines,
                            message="Starting runtime on " + node["name"],
                        )
                        break
                    if selected:
                        self.save()
                        break
                    if not compatible:
                        raise ValueError("; ".join(reasons))
                    job.update(
                        status="waiting",
                        stage="waiting-capacity",
                        progress=8.0,
                        message="; ".join(reasons),
                    )
                    self.save()
                await asyncio.sleep(2)
            if not selected:
                raise TimeoutError("Pool queue deadline exceeded (30 minutes)")
            progress_task = asyncio.create_task(self.animate_load(job))
            try:
                result = await self.call(selected, "load", payload)
            finally:
                progress_task.cancel()
                await asyncio.gather(progress_task, return_exceptions=True)
            refreshed = await self.inventory(force=True)
            refreshed_node = next(
                (node for node in refreshed["clusters"] if node["id"] == selected.id),
                None,
            )
            after_by_id = {
                machine["id"]: machine for machine in (refreshed_node or {}).get("machines", [])
            }
            for machine in job.get("machines", []):
                after = after_by_id.get(machine["id"], {})
                machine.update(
                    status="ready",
                    progress=100.0,
                    memory_after_gib=after.get(
                        "memory_used_gib", machine.get("memory_used_gib", 0.0)
                    ),
                    memory_total_gib=after.get(
                        "memory_total_gib", machine.get("memory_total_gib", 0.0)
                    ),
                )
            prefs = self.preferences.get(request.model_id, ModelPreferences())
            if prefs.keep_loaded != payload["keep_loaded"]:
                await self.call(
                    selected,
                    "retain",
                    {"model_id": request.model_id, "keep_loaded": prefs.keep_loaded},
                )
            if result.get("settings"):
                prefs.last_settings = ModelLaunchRequest.model_validate(result["settings"])
                self.preferences[request.model_id] = prefs
            job.update(
                status="ready",
                stage="ready",
                progress=100.0,
                completed_at=time.time(),
                message="Model ready",
            )
        except asyncio.CancelledError:
            job.update(status="cancelled", message="Load cancelled")
            raise
        except Exception as error:
            for machine in job.get("machines", []):
                machine["status"] = "failed"
            job.update(
                status="failed",
                stage="failed",
                message=self.node.models.safe_error(error),
                completed_at=time.time(),
            )
        finally:
            if selected:
                self.reserved.discard(selected.id)
            if job.get("status") != "ready":
                self.cache = None
            self.save()

    async def scale_instances(self, request: PoolScaleRequest):
        pending = next(
            (
                job
                for job in self.jobs
                if job["model_id"] == request.model_id
                and job["status"] in {"queued", "waiting", "loading"}
            ),
            None,
        )
        if pending:
            raise ValueError("Wait for the pending load to finish before changing instances")
        async with self.scheduling:
            state = await self.inventory(force=True)
            instances = [
                node for node in state["clusters"] if node["active_model"] == request.model_id
            ]
            if not instances:
                raise ValueError("Load the model before changing its instance count")
            if request.delta < 0:
                if len(instances) == 1:
                    raise ValueError("At least one model instance must remain loaded")
                idle = [node for node in instances if not node["busy"]]
                if not idle:
                    raise ValueError(
                        "Wait for an active request to finish before removing an instance"
                    )
                affinity = self.preferences.get(request.model_id, ModelPreferences()).cluster_id
                idle.sort(
                    key=lambda node: (
                        node["id"] == affinity,
                        node.get("keep_loaded", False),
                        node["id"],
                    )
                )
                selected = idle[0]
                await self.call(
                    self.cluster(selected["id"]), "unload", {"model_id": request.model_id}
                )
                self.cache = None
                await self.inventory(force=True)
                return {
                    "status": "ready",
                    "instances": len(instances) - 1,
                    "cluster_id": selected["id"],
                }
            affinity = self.preferences.get(request.model_id, ModelPreferences()).cluster_id
            additional = [
                node
                for node in state["clusters"]
                if node["available"]
                and (not affinity or node["id"] == affinity)
                and node["active_model"] != request.model_id
                and any(
                    model["id"] == request.model_id and model["loadable"]
                    for model in node["models"]
                )
            ]
            if not additional:
                raise ValueError("No additional pool can run another instance of this model")
            job = await self.enqueue(PoolLoadRequest(model_id=request.model_id), new_instance=True)
            job["instances"] = len(instances)
            return job

    async def unload(self, model_id):
        for job in self.jobs:
            if job["model_id"] == model_id and job["status"] in {"queued", "waiting", "loading"}:
                raise ValueError("Wait for the pending load to finish before unloading")
        state = await self.inventory(force=True)
        for node in state["clusters"]:
            if node["active_model"] == model_id:
                await self.call(self.cluster(node["id"]), "unload", {"model_id": model_id})
        prefs = self.preferences.get(model_id, ModelPreferences())
        prefs.keep_loaded = False
        self.preferences[model_id] = prefs
        self.cache = None
        self.save()
        return {"status": "unloaded"}

    async def infer(self, model_id, path, scope, receive, send):
        if self.tracker:

            async def handler(scope, receive, send):
                await self._infer_queued(model_id, path, scope, receive, send)

            return await self.tracker.track(handler, model_id, path, scope, receive, send)
        return await self._infer_queued(model_id, path, scope, receive, send)

    async def _infer_queued(self, model_id, path, scope, receive, send):
        if self.pending_requests >= 64:
            await JSONResponse({"error": "Pool inference queue is full"}, status_code=503)(
                scope, receive, send
            )
            return
        self.pending_requests += 1
        try:
            await self._infer(model_id, path, scope, receive, send)
        finally:
            self.pending_requests -= 1

    async def _infer(self, model_id, path, scope, receive, send):
        state = await self.inventory()
        prefs = self.preferences.get(model_id, ModelPreferences())
        resident = self.resident_for(model_id, state["clusters"], prefs)
        if resident:
            job = {"status": "ready", "cluster_id": resident["id"]}
        else:
            job = await self.enqueue(PoolLoadRequest(model_id=model_id))
        async with asyncio.timeout(1800):
            while job["status"] not in {"ready", "failed", "cancelled"}:
                client = scope.get("spark.client_request")
                if client and await client.is_disconnected():
                    raise ClientDisconnect
                await asyncio.sleep(0.2)
            if job["status"] != "ready":
                await JSONResponse({"error": job["message"]}, status_code=503)(scope, receive, send)
                return
            cluster = self.cluster(job["cluster_id"])
            if record := scope.get("spark.inference"):
                record.cluster_id = cluster.id
            if self.local(cluster):
                await self.node.infer(model_id, path, scope, receive, send)
            else:
                mark_dispatch(scope, cluster.id)
                target = "/api/v1/pool/node/inference/" + path
                query = scope.get("query_string", b"").decode()
                query += ("&" if query else "") + urlencode({"model_id": model_id})
                await forward_http(
                    cluster.endpoint + target,
                    dict(scope, query_string=query.encode()),
                    receive,
                    send,
                    token=self.settings.model_admin_token,
                    transport=self.transport,
                )


async def forward_http(url, scope, receive, send, token=None, transport=None):
    request = Request(scope, receive)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 64 * 1024 * 1024:
            await JSONResponse({"error": "Request exceeds 64 MiB"}, status_code=413)(
                scope, receive, send
            )
            return
    query = scope.get("query_string", b"").decode()
    if query:
        url += "?" + query
    body = request_stream_usage(body, request.headers.get("content-type", ""), request.url.path)
    headers = {
        k.decode(): v.decode()
        for k, v in scope["headers"]
        if k.lower() not in HOP_HEADERS | {b"content-length"}
    }
    if token:
        parsed = httpx.URL(url)
        target = parsed.raw_path.decode()
        headers.update(signed_controller_headers(token, request.method, target, bytes(body)))
    async with httpx.AsyncClient(timeout=1800, trust_env=False, transport=transport) as client:
        async with client.stream(
            request.method, url, content=bytes(body), headers=headers
        ) as upstream:
            response = StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code)
            response.raw_headers = [
                (k, v) for k, v in upstream.headers.raw if k.lower() not in HOP_HEADERS
            ]
            await response(scope, receive, send)


class PoolResponse(Response):
    def __init__(self, handler):
        super().__init__()
        self.handler = handler

    async def __call__(self, scope, receive, send):
        started = False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.handler(scope, receive, tracked_send)
        except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError):
            if started:
                raise
            await JSONResponse(
                {"error": "Pool inference unavailable; inspect the load queue"}, status_code=503
            )(scope, receive, send)
