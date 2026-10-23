import asyncio
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .config import Settings
from .models import (
    ClusterSnapshot,
    DownloadJob,
    LocalModel,
    ModelDownloadRequest,
    ModelFitPlan,
    ModelFitRequest,
    RuntimeEndpoint,
)
from .store import SnapshotStore

GIB = 2**30
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
REPO_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RepositoryMetadata:
    repo_id: str
    revision: str
    repository_bytes: int
    weight_bytes: int
    parameter_count: int | None
    architecture: str | None
    config: dict[str, Any]


def safe_repo_parts(repo_id: str) -> tuple[str, ...]:
    parts = tuple(part for part in repo_id.strip().strip("/").split("/") if part)
    if not 1 <= len(parts) <= 2 or any(not REPO_PART.fullmatch(part) for part in parts):
        raise ValueError("Use a Hugging Face model id such as organization/model-name")
    return parts


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class HuggingFaceCatalog:
    def __init__(self, token: str | None = None, download_workers: int = 4):
        self.token = token
        self.download_workers = download_workers
        self.api = HfApi(token=token)

    def inspect(self, repo_id: str, revision: str) -> RepositoryMetadata:
        safe_repo_parts(repo_id)
        info = self.api.model_info(
            repo_id, revision=revision, files_metadata=True, token=self.token
        )
        siblings = info.siblings or []
        repository_bytes = sum(int(getattr(item, "size", 0) or 0) for item in siblings)
        weight_bytes = sum(
            int(getattr(item, "size", 0) or 0)
            for item in siblings
            if str(getattr(item, "rfilename", "")).lower().endswith(WEIGHT_SUFFIXES)
        )
        config = dict(info.config or {})
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                revision=revision,
                token=self.token,
            )
            config.update(json.loads(Path(config_path).read_text()))
        except Exception:
            # Some repositories omit config.json; size-based planning remains available.
            pass
        architecture = None
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            architecture = str(architectures[0])
        elif config.get("model_type"):
            architecture = str(config["model_type"])

        parameter_count = None
        safetensors = getattr(info, "safetensors", None)
        total = getattr(safetensors, "total", None)
        if total:
            parameter_count = int(total)
        elif isinstance(safetensors, dict) and safetensors.get("total"):
            parameter_count = int(safetensors["total"])

        return RepositoryMetadata(
            repo_id=repo_id,
            revision=revision,
            repository_bytes=repository_bytes,
            weight_bytes=weight_bytes,
            parameter_count=parameter_count,
            architecture=architecture,
            config=config,
        )

    def download(self, repo_id: str, revision: str, target: Path) -> None:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
            max_workers=self.download_workers,
            token=self.token,
        )


def _kv_bytes_per_token(config: dict[str, Any]) -> tuple[float, bool]:
    layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
    if not layers:
        return 0, False
    dtype_bytes = 2
    kv_lora_rank = int(config.get("kv_lora_rank") or 0)
    rope_dim = int(config.get("qk_rope_head_dim") or 0)
    if kv_lora_rank:
        return layers * (kv_lora_rank + rope_dim) * dtype_bytes, True

    attention_heads = int(config.get("num_attention_heads") or config.get("n_head") or 0)
    kv_heads = int(config.get("num_key_value_heads") or attention_heads)
    hidden_size = int(config.get("hidden_size") or config.get("n_embd") or 0)
    head_dim = int(
        config.get("head_dim") or (hidden_size // attention_heads if attention_heads else 0)
    )
    if not kv_heads or not head_dim:
        return 0, False
    return layers * 2 * kv_heads * head_dim * dtype_bytes, True


def build_fit_plan(
    request: ModelFitRequest,
    metadata: RepositoryMetadata,
    snapshot: ClusterSnapshot | None,
    settings: Settings,
) -> ModelFitPlan:
    if snapshot and snapshot.nodes:
        node_capacities = [
            node.memory.total_gib * settings.model_usable_memory_fraction for node in snapshot.nodes
        ]
    else:
        node_capacities = [
            settings.fallback_node_memory_gib * settings.model_usable_memory_fraction
        ] * settings.fallback_node_count
    node_capacities.sort(reverse=True)

    if metadata.parameter_count:
        weights_gib = metadata.parameter_count * request.bits_per_weight / 8 / GIB
        weight_confidence = "parameters"
    else:
        weights_gib = metadata.weight_bytes / GIB
        weight_confidence = "repository"

    bytes_per_token, architecture_known = _kv_bytes_per_token(metadata.config)
    notes: list[str] = []
    if architecture_known:
        kv_cache_gib = bytes_per_token * request.context_length * request.concurrent_sequences / GIB
        confidence = "architecture"
    else:
        kv_cache_gib = (
            0.00025 * 1_000_000_000 * request.context_length * request.concurrent_sequences / GIB
        )
        confidence = weight_confidence
        notes.append(
            "KV memory uses the conservative generic fallback because "
            "the architecture metadata is incomplete."
        )

    runtime_overhead_gib = weights_gib * 0.08 + 8
    total_required_gib = weights_gib + kv_cache_gib + runtime_overhead_gib
    running = 0.0
    minimum_nodes = len(node_capacities) + 1
    for index, capacity in enumerate(node_capacities, start=1):
        running += capacity
        if running >= total_required_gib:
            minimum_nodes = index
            break

    usable_cluster_gib = sum(node_capacities)
    memory_fits = total_required_gib <= usable_cluster_gib
    settings.model_cache_path.mkdir(parents=True, exist_ok=True)
    storage_available_gib = shutil.disk_usage(settings.model_cache_path).free / GIB
    storage_required_gib = metadata.repository_bytes / GIB
    storage_fits = storage_required_gib <= storage_available_gib
    if not memory_fits:
        notes.append(
            "The selected context and concurrency exceed the cluster's usable memory envelope."
        )
    if not storage_fits:
        notes.append("The model repository is larger than the free model-cache storage.")

    return ModelFitPlan(
        repo_id=request.repo_id,
        revision=request.revision,
        architecture=metadata.architecture,
        parameter_count=metadata.parameter_count,
        repository_bytes=metadata.repository_bytes,
        weight_bytes=metadata.weight_bytes,
        weights_gib=round(weights_gib, 2),
        kv_cache_gib=round(kv_cache_gib, 2),
        runtime_overhead_gib=round(runtime_overhead_gib, 2),
        total_required_gib=round(total_required_gib, 2),
        cluster_nodes=len(node_capacities),
        usable_per_node_gib=round(sum(node_capacities) / len(node_capacities), 2),
        usable_cluster_gib=round(usable_cluster_gib, 2),
        minimum_nodes=minimum_nodes,
        headroom_gib=round(usable_cluster_gib - total_required_gib, 2),
        memory_fits=memory_fits,
        storage_available_gib=round(storage_available_gib, 2),
        storage_fits=storage_fits,
        fits=memory_fits and storage_fits,
        confidence=confidence,
        notes=notes,
    )


class ModelManager:
    def __init__(self, settings: Settings, store: SnapshotStore, runtime: Any):
        self.settings = settings
        self.store = store
        self.runtime = runtime
        self.catalog = HuggingFaceCatalog(settings.hf_token, settings.hf_download_workers)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker: asyncio.Task | None = None
        self._inventory_cache: list[LocalModel] | None = None
        self._inventory_cached_at = 0.0
        self._runtime_cache: RuntimeEndpoint | None = None
        self._runtime_cached_at = 0.0
        self._runtime_lock = asyncio.Lock()
        self.settings.model_cache_path.mkdir(parents=True, exist_ok=True)
        (self.settings.model_cache_path / ".partial").mkdir(exist_ok=True)

    async def start(self) -> None:
        for job in await asyncio.to_thread(self.store.recover_downloads):
            await self.queue.put(job.id)
        self.worker = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def fit(self, request: ModelFitRequest) -> ModelFitPlan:
        metadata = await asyncio.to_thread(self.catalog.inspect, request.repo_id, request.revision)
        return await asyncio.to_thread(
            build_fit_plan, request, metadata, self.runtime.latest, self.settings
        )

    async def enqueue(self, request: ModelDownloadRequest) -> DownloadJob:
        fit = await self.fit(request)
        if not fit.fits and not request.allow_oversized:
            raise ModelDoesNotFit(fit)
        parts = safe_repo_parts(request.repo_id)
        target = self.settings.model_cache_path.joinpath(*parts)
        if (target / ".spark-deepseek.json").exists():
            raise ModelAlreadyInstalled(request.repo_id)
        active = [
            job
            for job in await asyncio.to_thread(self.store.list_downloads)
            if job.repo_id == request.repo_id
            and job.revision == request.revision
            and job.status in {"queued", "resolving", "downloading"}
        ]
        if active:
            return active[0]

        job_id = uuid4().hex
        job = DownloadJob(
            id=job_id,
            repo_id=request.repo_id,
            revision=request.revision,
            total_bytes=fit.repository_bytes,
            target_path=str(target),
            staging_path=str(self.settings.model_cache_path / ".partial" / job_id),
            fit=fit,
        )
        await asyncio.to_thread(self.store.save_download, job)
        await self.queue.put(job.id)
        return job

    async def retry(self, job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(self.store.get_download, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "failed":
            return job
        job.status = "queued"
        job.error = None
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)
        await self.queue.put(job.id)
        return job

    async def downloads(self) -> list[DownloadJob]:
        return await asyncio.to_thread(self.store.list_downloads)

    async def runtime_endpoint(self) -> RuntimeEndpoint:
        if self._runtime_cache and time.monotonic() - self._runtime_cached_at < 1:
            return self._runtime_cache
        async with self._runtime_lock:
            if self._runtime_cache and time.monotonic() - self._runtime_cached_at < 1:
                return self._runtime_cache
            base = self.settings.openai_api_base.rstrip("/")
            models_url = f"{base}/models"
            loaded: list[str] = []
            status = "unreachable"
            try:
                headers = (
                    {"Authorization": f"Bearer {self.settings.openai_api_key}"}
                    if self.settings.openai_api_key
                    else {}
                )
                async with httpx.AsyncClient(timeout=2, headers=headers) as client:
                    response = await client.get(models_url)
                    response.raise_for_status()
                    loaded = [str(item["id"]) for item in response.json().get("data", [])]
                    status = "ready"
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                pass
            endpoint = RuntimeEndpoint(
                base_url=base,
                models_url=models_url,
                chat_completions_url=f"{base}/chat/completions",
                status=status,
                loaded_model_ids=loaded,
            )
            self._runtime_cache = endpoint
            self._runtime_cached_at = time.monotonic()
            return endpoint

    async def local_models(self) -> list[LocalModel]:
        endpoint = await self.runtime_endpoint()
        loaded = set(endpoint.loaded_model_ids)
        if self._inventory_cache is None or time.monotonic() - self._inventory_cached_at > 10:
            models = await asyncio.to_thread(self._scan_managed_models, set())
            known = {model.id for model in models}
            models.extend(await asyncio.to_thread(self._scan_huggingface_cache, set(), known))
            self._inventory_cache = models
            self._inventory_cached_at = time.monotonic()
        models = [
            model.model_copy(update={"loaded": model.id in loaded})
            for model in self._inventory_cache
        ]
        known = {model.id for model in models}
        models.extend(
            LocalModel(id=model_id, source="runtime", loaded=True)
            for model_id in sorted(loaded - known)
        )
        return sorted(models, key=lambda model: (not model.loaded, model.id.lower()))

    def _scan_managed_models(self, loaded: set[str]) -> list[LocalModel]:
        models: list[LocalModel] = []
        for manifest_path in self.settings.model_cache_path.glob("**/.spark-deepseek.json"):
            if ".partial" in manifest_path.parts:
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                downloaded_at = datetime.fromisoformat(manifest["downloaded_at"])
                repo_id = str(manifest["repo_id"])
                models.append(
                    LocalModel(
                        id=repo_id,
                        revision=manifest.get("revision"),
                        source="managed",
                        path=str(manifest_path.parent),
                        size_bytes=directory_size(manifest_path.parent),
                        downloaded_at=downloaded_at,
                        loaded=repo_id in loaded,
                    )
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return models

    def _scan_huggingface_cache(self, loaded: set[str], known: set[str]) -> list[LocalModel]:
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = hf_home / "hub"
        models: list[LocalModel] = []
        if not hub.exists():
            return models
        for path in hub.glob("models--*"):
            repo_id = path.name.removeprefix("models--").replace("--", "/", 1)
            if repo_id in known:
                continue
            snapshots = path / "snapshots"
            weight_files = [
                file
                for file in snapshots.glob("**/*")
                if file.is_file() and file.name.lower().endswith(WEIGHT_SUFFIXES)
            ]
            if not weight_files:
                continue
            unique_weight_files = {file.resolve() for file in weight_files}
            models.append(
                LocalModel(
                    id=repo_id,
                    source="huggingface-cache",
                    path=str(path),
                    size_bytes=sum(file.stat().st_size for file in unique_weight_files),
                    loaded=repo_id in loaded,
                )
            )
        return models

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._run_job(job_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                job = await asyncio.to_thread(self.store.get_download, job_id)
                if job:
                    job.status = "failed"
                    job.error = self.safe_error(error)
                    job.updated_at = datetime.now(UTC)
                    await asyncio.to_thread(self.store.save_download, job)
            finally:
                self.queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = await asyncio.to_thread(self.store.get_download, job_id)
        if job is None:
            return
        existing_target = Path(job.target_path)
        if (existing_target / ".spark-deepseek.json").exists():
            job.status = "completed"
            job.downloaded_bytes = job.total_bytes or directory_size(existing_target)
            job.progress = 100
            job.error = None
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_download, job)
            return
        job.status = "downloading"
        job.attempts += 1
        job.error = None
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)
        staging = Path(job.staging_path)
        staging.mkdir(parents=True, exist_ok=True)
        outcome: dict[str, BaseException | None] = {"error": None}

        def download() -> None:
            try:
                self.catalog.download(job.repo_id, job.revision, staging)
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=download, daemon=True, name=f"hf-{job.id[:8]}")
        thread.start()
        while thread.is_alive():
            job.downloaded_bytes = min(job.total_bytes, self._staging_progress(staging))
            job.progress = (
                round(job.downloaded_bytes / job.total_bytes * 100, 2) if job.total_bytes else 0
            )
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_download, job)
            await asyncio.sleep(0.5)
        thread.join()

        if outcome["error"] is not None:
            job.status = "failed"
            job.error = self.safe_error(outcome["error"])
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_download, job)
            return

        target = Path(job.target_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Target model directory already exists: {target}")
            manifest = {
                "repo_id": job.repo_id,
                "revision": job.revision,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "fit": job.fit.model_dump(mode="json"),
            }
            (staging / ".spark-deepseek.json").write_text(json.dumps(manifest, indent=2))
            staging.replace(target)
            self._inventory_cache = None
            job.status = "completed"
            job.downloaded_bytes = job.total_bytes or directory_size(target)
            job.progress = 100
            job.error = None
        except (OSError, ValueError) as error:
            job.status = "failed"
            job.error = self.safe_error(error)
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)

    @staticmethod
    def _staging_progress(staging: Path) -> int:
        completed = sum(
            path.stat().st_size
            for path in staging.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        )
        partial = sum(
            path.stat().st_size
            for path in (staging / ".cache").rglob("*.incomplete")
            if path.is_file()
        )
        return completed + partial

    def safe_error(self, error: BaseException) -> str:
        message = str(error).strip() or error.__class__.__name__
        if self.settings.hf_token:
            message = message.replace(self.settings.hf_token, "[redacted]")
        message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", message)
        return message[:1000]


class ModelDoesNotFit(Exception):
    def __init__(self, fit: ModelFitPlan):
        super().__init__("Model does not fit the current cluster or model-cache storage")
        self.fit = fit


class ModelAlreadyInstalled(Exception):
    pass
