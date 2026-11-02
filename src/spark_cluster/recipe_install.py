"""Persistent, explicit recipe installation through existing model/runtime services."""

import asyncio
import time
from uuid import uuid4

from .model_pool import PoolLoadRequest
from .models import ModelDownloadRequest
from .recipes import SparkRecipe
from .runtime_library import RuntimeImportRequest


class RecipeInstaller:
    def __init__(self, catalog, registry, store, library, pool, download, verify, safe_error):
        self.catalog, self.registry, self.store = catalog, registry, store
        self.library, self.pool, self.download, self.verify = library, pool, download, verify
        self.safe_error = safe_error
        self.preflight = None
        self.remote = None
        self.tasks = {}
        self.lock = asyncio.Lock()

    async def jobs(self):
        return await asyncio.to_thread(self.store.runtime_objects, "recipe-install")

    async def save(self, job, **changes):
        job.update(**changes, updated_at=time.time())
        await asyncio.to_thread(self.store.save_runtime_object, "recipe-install", job)

    async def start(self):
        for job in await self.jobs():
            if job["status"] in {"queued", "running"}:
                if job.get("remote") and self.remote:
                    self.tasks[job["id"]] = asyncio.create_task(self.run(job))
                else:
                    await self.save(
                        job, status="failed", error="Controller restarted; retry the saved release"
                    )

    async def stop(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def create(self, digest, cluster_id, *, launch=True, request_id=None):
        if request_id:
            previous = next(
                (j for j in await self.jobs() if j.get("request_id") == request_id), None
            )
            if previous:
                if (
                    previous["recipe_hash"] != digest
                    or previous["cluster_id"] != cluster_id
                    or previous.get("launch", True) != launch
                ):
                    raise ValueError("Idempotency key belongs to another recipe request")
                return previous
        entry = await self.catalog.get(digest)
        recipe = SparkRecipe.model_validate(entry["recipe"])
        target = self.registry.get(cluster_id)
        if target.kind == "remote-bridge" and self.remote:
            target = await self.remote.refresh(target, force=True)
        blockers = await self.catalog.compatible(recipe, target)
        if blockers:
            raise ValueError("; ".join(blockers))
        if target.kind == "remote-bridge" and self.remote:
            await self.remote.prepare(target, recipe, launch=launch)
        elif self.preflight:
            await self.preflight(recipe, cluster_id)
        async with self.lock:
            existing_jobs = await self.jobs()
            if request_id:
                previous = next(
                    (j for j in existing_jobs if j.get("request_id") == request_id), None
                )
                if previous:
                    if (
                        previous["recipe_hash"] != digest
                        or previous["cluster_id"] != cluster_id
                        or previous.get("launch", True) != launch
                    ):
                        raise ValueError("Idempotency key belongs to another recipe request")
                    return previous
            for job in existing_jobs:
                if job["cluster_id"] == cluster_id and job["status"] in {"queued", "running"}:
                    if job["recipe_hash"] == digest and job.get("launch", True) == launch:
                        return job
                    raise ValueError("Another recipe installation is running on this Spark")
            job = {
                "id": uuid4().hex,
                "recipe_hash": digest,
                "cluster_id": cluster_id,
                "launch": launch,
                "request_id": request_id,
                "remote": target.kind == "remote-bridge",
                "recipe": entry["recipe"],
                "origin": entry["origin"],
                "status": "queued",
                "stage": "preflight",
                "progress": 0,
                "created_at": time.time(),
                "error": None,
                "download_job_id": None,
                "launch_job_id": None,
            }
            await self.save(job)
            self.tasks[job["id"]] = asyncio.create_task(self.run(job))
            return job

    async def _wait_library(self, job_id):
        deadline = time.monotonic() + 14400
        while time.monotonic() < deadline:
            result = await self.library.job(job_id)
            if result["status"] == "completed":
                return result
            if result["status"] in {"failed", "cancelled"}:
                raise ValueError(result.get("message") or "Runtime preparation failed")
            await asyncio.sleep(2)
        raise ValueError("Runtime preparation exceeded four hours")

    async def run(self, job):
        try:
            recipe = SparkRecipe.model_validate(job["recipe"])
            target = self.registry.get(job["cluster_id"])
            if target.kind == "remote-bridge" and self.remote:
                await self.run_remote(job, target)
                return
            blockers = await self.catalog.compatible(recipe, target)
            if blockers:
                raise ValueError("; ".join(blockers))
            await self.save(job, status="running")
            inventory = await self.pool.inventory(force=True)
            pool = next((c for c in inventory["clusters"] if c["id"] == target.id), None)
            if not pool or not pool["available"]:
                raise ValueError("Target Spark is unavailable")
            if job.get("launch", True) and (
                pool.get("resident_models") or pool.get("active_model")
            ):
                await self.launch_preflight(recipe, target)
            # Resolve and check architecture before starting multi-GB downloads.
            metadata = await self.library.remote.resolve_image(
                "vllm", recipe.runtime.image, recipe.requirements.platform
            )
            # Model service runs storage and memory preflight before any transfer.
            launch = recipe.launch_request()
            download = await self.download(
                ModelDownloadRequest(
                    repo_id=recipe.model.repository,
                    revision=recipe.model.revision,
                    bits_per_weight=launch.bits_per_weight,
                    context_length=launch.context_length,
                    concurrent_sequences=launch.max_num_seqs,
                    kv_cache_dtype=launch.kv_cache_dtype,
                ),
                target.id,
            )
            if download is not None:
                await self.save(
                    job, stage="download-model", progress=10, download_job_id=download.id
                )
                deadline = time.monotonic() + 86400
                while time.monotonic() < deadline:
                    result = await asyncio.to_thread(self.store.get_download, download.id)
                    if result and result.status == "completed":
                        break
                    if result and result.status in {"failed", "cancelled", "deleted"}:
                        raise ValueError(result.error or result.message)
                    await self.save(job, progress=10 + (result.progress if result else 0) * 0.45)
                    await asyncio.sleep(3)
                else:
                    raise ValueError("Model download exceeded 24 hours")
            await self.save(job, stage="prepare-runtime", progress=60)
            if target.kind == "local-docker":
                await self.library.ensure_local_image(metadata)
            else:
                cached = await self.library.verified_artifact_for_image(recipe.runtime.image)
                if not cached:
                    queued = await self.library.download_resolved(metadata)
                    done = await self._wait_library(queued["id"])
                    cached = await self.library.get_artifact(done["artifact_id"])
                queued = await self.library.install(
                    RuntimeImportRequest(artifact_id=cached["id"], cluster_id=target.id)
                )
                await self._wait_library(queued["id"])
            if not job.get("launch", True):
                await self.save(
                    job,
                    status="completed",
                    stage="installed",
                    progress=100,
                    artifacts_present=True,
                    inference_verified=False,
                )
                return
            # A long download must not turn into an implicit replacement of a newer workload.
            inventory = await self.pool.inventory(force=True)
            current = next((c for c in inventory["clusters"] if c["id"] == target.id), None)
            if not current:
                raise ValueError("Target Spark is unavailable")
            if current.get("resident_models") or current.get("active_model"):
                await self.launch_preflight(recipe, target)
            if any(
                j.get("model_id") == recipe.model.repository
                and j["status"] in {"queued", "waiting", "loading"}
                for j in self.pool.jobs
            ):
                raise ValueError("Another launch for this model is running; wait and retry")
            await self.save(job, stage="launch", progress=80)
            queued = await self.pool.enqueue(
                PoolLoadRequest(
                    model_id=recipe.model.repository, settings=launch, cluster_id=target.id
                )
            )
            await self.save(job, launch_job_id=queued["id"])
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                result = next((j for j in self.pool.jobs if j["id"] == queued["id"]), None)
                if result and result["status"] == "ready":
                    break
                if result and result["status"] in {"failed", "cancelled"}:
                    raise ValueError(result.get("message") or "Launch failed")
                await asyncio.sleep(2)
            else:
                raise ValueError("Recipe launch exceeded 30 minutes")
            await self.save(job, stage="verify", progress=95)
            await self.verify(recipe, target)
            # Snapshot includes exact file, source, weights, image digest, and launch settings.
            # Old releases and deployments stay available for explicit rollback/reinstall.
            await self.save(
                job,
                status="completed",
                stage="ready",
                progress=100,
                artifacts_present=True,
                inference_verified=True,
            )
        except asyncio.CancelledError:
            if job.get("remote"):
                await self.save(
                    job, status="running"
                )  # Node owns progress; reconnect idempotently.
            else:
                await self.save(
                    job, status="failed", error="Installation interrupted; retry this release"
                )
            raise
        except Exception as error:
            await self.save(
                job,
                status="failed",
                error=self.safe_error(error),
                storage=getattr(error, "storage", None),
            )
        finally:
            self.tasks.pop(job["id"], None)

    async def launch_preflight(self, recipe, target):
        node = getattr(self.pool, "node", None)
        if node is None:
            raise ValueError("Install without launching while another model is loaded")
        result = await node.plan(
            PoolLoadRequest(
                model_id=recipe.model.repository,
                settings=recipe.launch_request(),
                cluster_id=target.id,
            )
        )
        placement = result.get("placement") or {}
        if not result.get("compatible") or result.get("waiting") or placement.get("evict"):
            reason = (
                result.get("reason") or "Insufficient free memory without unloading another model"
            )
            raise ValueError("Recipe can remain installed on disk. Launch blocked: " + reason)

    async def run_remote(self, job, target):
        await self.save(job, status="running", stage="contact-node")
        remote = await self.remote.install(target, job)
        await self.save(job, remote_job_id=remote["id"])
        deadline = time.monotonic() + 86400
        while True:
            if remote.get("recipe_hash") != job["recipe_hash"]:
                raise ValueError("Remote installation returned another recipe release")
            fields = {
                k: remote[k]
                for k in [
                    "status",
                    "stage",
                    "progress",
                    "error",
                    "download_job_id",
                    "launch_job_id",
                    "artifacts_present",
                    "inference_verified",
                ]
                if k in remote
            }
            if remote.get("storage"):
                from .remote_recipes import remap_storage

                fields["storage"] = remap_storage(remote["storage"], target.id)
            await self.save(job, **fields)
            if remote["status"] in {"completed", "failed", "cancelled"}:
                return
            if time.monotonic() > deadline:
                raise ValueError("Remote preparation exceeded 24 hours; inspect the saved node job")
            await asyncio.sleep(2)
            remote = await self.remote.job(target, job["remote_job_id"])
