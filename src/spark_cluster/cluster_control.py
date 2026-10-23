import asyncio
import hashlib
import math
import os
import random
import re
import ssl
from collections import deque
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from .config import Settings
from .model_manager import ModelDoesNotFit, ModelManager
from .models import (
    ModelFitRequest,
    ModelLaunchJob,
    ModelLaunchRequest,
    RuntimeLog,
)
from .store import SnapshotStore


class KubernetesClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        host = os.getenv("KUBERNETES_SERVICE_HOST")
        port = os.getenv("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        self.base_url = settings.kubernetes_api_url or (f"https://{host}:{port}" if host else None)
        self.token_path = settings.kubernetes_token_path
        namespace_path = self.token_path.parent / "namespace"
        self.namespace = (
            namespace_path.read_text().strip()
            if namespace_path.exists()
            else settings.kubernetes_namespace
        )

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.token_path.exists())

    def _ssl_context(self) -> ssl.SSLContext | bool:
        if self.settings.kubernetes_ca_path.exists():
            return ssl.create_default_context(cafile=str(self.settings.kubernetes_ca_path))
        return True

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        strategic: bool = False,
    ) -> Any:
        if not self.available:
            raise RuntimeError("Kubernetes service-account credentials are unavailable")
        headers = {"Authorization": f"Bearer {self.token_path.read_text().strip()}"}
        if strategic:
            headers["Content-Type"] = "application/strategic-merge-patch+json"
        async with httpx.AsyncClient(
            base_url=str(self.base_url),
            headers=headers,
            verify=self._ssl_context(),
            timeout=30,
        ) as client:
            response = await client.request(method, path, params=params, json=payload)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            return response.json() if "json" in content_type else response.text

    async def runtime_pods(self) -> list[dict[str, Any]]:
        result = await self.request(
            "GET",
            f"/api/v1/namespaces/{self.namespace}/pods",
            params={"labelSelector": self.settings.runtime_pod_selector},
        )
        return list(result.get("items", []))

    async def pod_logs(
        self, pod: str, container: str, since_seconds: int, tail_lines: int = 300
    ) -> str:
        return await self.request(
            "GET",
            f"/api/v1/namespaces/{self.namespace}/pods/{pod}/log",
            params={
                "container": container,
                "timestamps": "true",
                "sinceSeconds": since_seconds,
                "tailLines": tail_lines,
            },
        )

    async def patch_statefulset(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.request(
            "PATCH",
            f"/apis/apps/v1/namespaces/{self.namespace}/statefulsets/{name}",
            payload=payload,
            strategic=True,
        )

    async def patch_deployment(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.request(
            "PATCH",
            f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}",
            payload=payload,
            strategic=True,
        )

    async def get_statefulset(self, name: str) -> dict[str, Any]:
        return await self.request(
            "GET", f"/apis/apps/v1/namespaces/{self.namespace}/statefulsets/{name}"
        )

    async def get_deployment(self, name: str) -> dict[str, Any]:
        return await self.request(
            "GET", f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}"
        )


class RuntimeLogService:
    DEMO_MESSAGES = (
        ("info", "Scheduler admitted request batch; active=6 queued=1"),
        ("debug", "Prefix cache lookup completed; matched_tokens=114688"),
        ("info", "KV block promoted from NVMe to unified memory"),
        ("warning", "GPU KV utilization above 80%; evicting cold blocks"),
        ("info", "Generated 2048 tokens; throughput=23.4 tok/s"),
        ("debug", "Ray heartbeat received; worker latency=1.8ms"),
    )

    def __init__(self, settings: Settings, store: SnapshotStore):
        self.settings = settings
        self.store = store
        self.kubernetes = KubernetesClient(settings)
        configured = settings.runtime_logs_mode
        self.mode = configured if configured != "auto" else self._auto_mode()
        self.task: asyncio.Task | None = None
        self.subscribers: set[asyncio.Queue[RuntimeLog]] = set()
        self._seen: set[str] = set()
        self._seen_order: deque[str] = deque(maxlen=5000)
        self._demo_index = 0

    def _auto_mode(self) -> str:
        if self.kubernetes.available:
            return "kubernetes"
        return "demo" if self.settings.mode == "demo" else "unavailable"

    async def start(self) -> None:
        if self.mode == "demo":
            self.task = asyncio.create_task(self._demo_loop())
        elif self.mode == "kubernetes":
            self.task = asyncio.create_task(self._kubernetes_loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def emit(
        self, pod: str, container: str, level: str, message: str, timestamp: datetime | None = None
    ) -> RuntimeLog:
        event_time = timestamp or datetime.now(UTC)
        message = self._sanitize(message)
        value = f"{event_time.isoformat()}|{pod}|{container}|{message}"
        entry = RuntimeLog(
            id=hashlib.sha256(value.encode()).hexdigest(),
            timestamp=event_time,
            pod=pod,
            container=container,
            level=level,  # type: ignore[arg-type]
            message=message,
        )
        if entry.id in self._seen:
            return entry
        self._remember(entry.id)
        await asyncio.to_thread(self.store.append_runtime_log, entry)
        for queue in tuple(self.subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(entry)
        return entry

    async def history(
        self, limit: int = 500, pod: str | None = None, level: str | None = None
    ) -> list[RuntimeLog]:
        return await asyncio.to_thread(self.store.list_runtime_logs, limit, pod, level)

    async def instances(self) -> list[str]:
        if self.mode == "kubernetes":
            try:
                return sorted(
                    item["metadata"]["name"] for item in await self.kubernetes.runtime_pods()
                )
            except (httpx.HTTPError, KeyError, TypeError):
                pass
        return sorted({entry.pod for entry in await self.history(limit=2000)})

    def _remember(self, entry_id: str) -> None:
        if len(self._seen_order) == self._seen_order.maxlen:
            expired = self._seen_order.popleft()
            self._seen.discard(expired)
        self._seen_order.append(entry_id)
        self._seen.add(entry_id)

    async def _demo_loop(self) -> None:
        randomizer = random.Random(42)
        while True:
            level, message = self.DEMO_MESSAGES[self._demo_index % len(self.DEMO_MESSAGES)]
            pod_number = self._demo_index % 4 + 1
            container = "engine" if pod_number == 1 else "ray-worker"
            pod = (
                "spark-cluster-runtime-head"
                if pod_number == 1
                else f"spark-cluster-runtime-worker-{pod_number - 2}"
            )
            await self.emit(
                pod,
                container,
                level,
                f"{message}; trace={randomizer.randrange(16**8):08x}",
            )
            self._demo_index += 1
            if self._demo_index % 100 == 0:
                await asyncio.to_thread(
                    self.store.prune_runtime_logs, self.settings.runtime_log_retention
                )
            await asyncio.sleep(0.8)

    async def _kubernetes_loop(self) -> None:
        since_seconds = max(2, math.ceil(self.settings.runtime_log_poll_seconds * 2))
        while True:
            try:
                for pod in await self.kubernetes.runtime_pods():
                    pod_name = pod["metadata"]["name"]
                    for container_spec in pod.get("spec", {}).get("containers", []):
                        container = container_spec.get("name", "runtime")
                        text = await self.kubernetes.pod_logs(pod_name, container, since_seconds)
                        for line in text.splitlines():
                            timestamp, message = self._parse_line(line)
                            await self.emit(
                                pod_name,
                                container,
                                self._infer_level(message),
                                message,
                                timestamp,
                            )
            except (httpx.HTTPError, KeyError, TypeError, RuntimeError) as error:
                await self.emit(
                    "control-plane", "log-collector", "error", f"Log collection failed: {error}"
                )
            await asyncio.sleep(self.settings.runtime_log_poll_seconds)

    @staticmethod
    def _parse_line(line: str) -> tuple[datetime, str]:
        timestamp_text, separator, message = line.partition(" ")
        if separator:
            try:
                return datetime.fromisoformat(timestamp_text.replace("Z", "+00:00")), message
            except ValueError:
                pass
        return datetime.now(UTC), line

    @staticmethod
    def _infer_level(message: str) -> str:
        lowered = message.lower()
        if any(word in lowered for word in ("error", "failed", "exception", "fatal")):
            return "error"
        if any(word in lowered for word in ("warn", "evict", "retry")):
            return "warning"
        if any(word in lowered for word in ("debug", "trace", "heartbeat")):
            return "debug"
        return "info"

    @staticmethod
    def _sanitize(message: str) -> str:
        value = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[redacted]", message)
        value = re.sub(r"\b(?:hf|sk)-[A-Za-z0-9_-]{12,}\b", "[redacted]", value)
        return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", value)


class RuntimeDeploymentService:
    def __init__(
        self,
        settings: Settings,
        store: SnapshotStore,
        models: ModelManager,
        logs: RuntimeLogService,
    ):
        self.settings = settings
        self.store = store
        self.models = models
        self.logs = logs
        self.kubernetes = logs.kubernetes
        configured = settings.runtime_control_mode
        self.mode = configured if configured != "auto" else self._auto_mode()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker: asyncio.Task | None = None

    def _auto_mode(self) -> str:
        if self.kubernetes.available:
            return "kubernetes"
        return "demo" if self.settings.mode == "demo" else "unavailable"

    async def start(self) -> None:
        for job in await asyncio.to_thread(self.store.recover_launches):
            await self.queue.put(job.id)
        self.worker = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def launch(self, request: ModelLaunchRequest) -> ModelLaunchJob:
        local = await self.models.managed_model(request.model_id)
        if local is None or local.path is None:
            raise ModelNotManaged(request.model_id)
        fit = await self.models.fit_for_local(
            ModelFitRequest(
                repo_id=request.model_id,
                revision=local.revision or "main",
                bits_per_weight=request.bits_per_weight,
                kv_cache_dtype=request.kv_cache_dtype,
                context_length=request.context_length,
                concurrent_sequences=request.concurrent_sequences,
            )
        )
        if not fit.fits:
            raise ModelDoesNotFit(fit)
        requested_nodes = request.tensor_parallel * request.pipeline_parallel
        if requested_nodes < fit.minimum_nodes or requested_nodes > fit.cluster_nodes:
            raise InvalidParallelism(requested_nodes, fit.minimum_nodes, fit.cluster_nodes)
        job = ModelLaunchJob(
            id=uuid4().hex,
            request=request,
            model_path=local.path,
            fit=fit,
        )
        await asyncio.to_thread(self.store.save_launch, job)
        await self.queue.put(job.id)
        return job

    async def launches(self) -> list[ModelLaunchJob]:
        return await asyncio.to_thread(self.store.list_launches)

    async def retry(self, job_id: str) -> ModelLaunchJob:
        job = await asyncio.to_thread(self.store.get_launch, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "failed":
            return job
        job.status = "queued"
        job.progress = 0
        job.message = "Queued for retry"
        job.error = None
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_launch, job)
        await self.queue.put(job.id)
        return job

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._run(job_id)
            except asyncio.CancelledError:
                raise
            finally:
                self.queue.task_done()

    async def _run(self, job_id: str) -> None:
        job = await asyncio.to_thread(self.store.get_launch, job_id)
        if job is None:
            return
        job.attempts += 1
        try:
            if self.mode == "demo":
                await self._run_demo(job)
            elif self.mode == "kubernetes":
                await self._run_kubernetes(job)
            else:
                raise RuntimeError("Runtime control is unavailable outside a Kubernetes cluster")
        except Exception as error:
            job.status = "failed"
            job.message = "Launch failed"
            job.error = self.models.safe_error(error)
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_launch, job)
            await self.logs.emit(
                "control-plane",
                "model-launcher",
                "error",
                f"Launch {job.id[:8]} failed: {job.error}",
            )

    async def _run_demo(self, job: ModelLaunchJob) -> None:
        stages = (
            ("validating", 12, "Validated model manifest and memory plan"),
            ("scaling-workers", 35, "Scaled Ray workers for requested parallelism"),
            ("starting-runtime", 65, "Started vLLM head with the selected model"),
            ("waiting-ready", 88, "Waiting for OpenAI-compatible health endpoint"),
            ("ready", 100, "Model is ready to serve"),
        )
        for status, progress, message in stages:
            await self._update(job, status, progress, message)
            await asyncio.sleep(0.35)

    async def _run_kubernetes(self, job: ModelLaunchJob) -> None:
        request = job.request
        workers = max(0, request.tensor_parallel * request.pipeline_parallel - 1)
        await self._update(
            job, "scaling-workers", 25, f"Scaling runtime to {workers + 1} GB10 instances"
        )
        await self.kubernetes.patch_statefulset(
            self.settings.runtime_worker_statefulset, {"spec": {"replicas": workers}}
        )
        await self._update(job, "starting-runtime", 55, "Applying model configuration to vLLM head")
        environment = [
            {"name": "MODEL_ID", "value": job.model_path},
            {"name": "SERVED_MODEL_NAME", "value": request.served_model_name},
            {"name": "MAX_MODEL_LEN", "value": str(request.context_length)},
            {"name": "TENSOR_PARALLEL_SIZE", "value": str(request.tensor_parallel)},
            {"name": "PIPELINE_PARALLEL_SIZE", "value": str(request.pipeline_parallel)},
            {"name": "GPU_MEMORY_UTILIZATION", "value": str(request.gpu_memory_utilization)},
            {"name": "KV_CACHE_DTYPE", "value": request.kv_cache_dtype},
            {"name": "MAX_NUM_SEQS", "value": str(request.max_num_seqs)},
            {
                "name": "MAX_NUM_BATCHED_TOKENS",
                "value": str(request.max_num_batched_tokens),
            },
        ]
        await self.kubernetes.patch_deployment(
            self.settings.runtime_head_deployment,
            {
                "spec": {
                    "replicas": 1,
                    "template": {
                        "metadata": {
                            "annotations": {
                                "spark-cluster.io/launched-at": datetime.now(UTC).isoformat()
                            }
                        },
                        "spec": {"containers": [{"name": "engine", "env": environment}]},
                    },
                }
            },
        )
        await self._update(job, "waiting-ready", 75, "Waiting for runtime pods to become ready")
        deadline = asyncio.get_running_loop().time() + 900
        while asyncio.get_running_loop().time() < deadline:
            worker_resource = await self.kubernetes.get_statefulset(
                self.settings.runtime_worker_statefulset
            )
            head_resource = await self.kubernetes.get_deployment(
                self.settings.runtime_head_deployment
            )
            ready_workers = int(worker_resource.get("status", {}).get("readyReplicas", 0) or 0)
            ready_head = int(head_resource.get("status", {}).get("availableReplicas", 0) or 0)
            if ready_workers >= workers and ready_head >= 1:
                await self._update(job, "ready", 100, "Model is ready to serve")
                return
            await asyncio.sleep(3)
        raise TimeoutError("Runtime rollout did not become ready within 15 minutes")

    async def _update(
        self, job: ModelLaunchJob, status: str, progress: float, message: str
    ) -> None:
        job.status = status  # type: ignore[assignment]
        job.progress = progress
        job.message = message
        job.error = None
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_launch, job)
        await self.logs.emit(
            "control-plane", "model-launcher", "info", f"[{job.request.model_id}] {message}"
        )


class ModelNotManaged(Exception):
    def __init__(self, model_id: str):
        super().__init__(f"Download {model_id} into the managed model cache before launching it")


class InvalidParallelism(Exception):
    def __init__(self, requested: int, minimum: int, available: int):
        super().__init__(
            f"Requested parallelism uses {requested} nodes; this model needs at least {minimum} "
            f"and the cluster currently exposes {available}"
        )
