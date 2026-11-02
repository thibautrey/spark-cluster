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
                await self.save(
                    job, status="failed", error="Controller restarted; retry the saved release"
                )

    async def stop(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def create(self, digest, cluster_id):
        entry = await self.catalog.get(digest)
        recipe = SparkRecipe.model_validate(entry["recipe"])
        blockers = await self.catalog.compatible(recipe, self.registry.get(cluster_id))
        if blockers:
            raise ValueError("; ".join(blockers))
        async with self.lock:
            for job in await self.jobs():
                if job["cluster_id"] == cluster_id and job["status"] in {"queued", "running"}:
                    if job["recipe_hash"] == digest:
                        return job
                    raise ValueError("Another recipe installation is running on this Spark")
            job = {
                "id": uuid4().hex,
                "recipe_hash": digest,
                "cluster_id": cluster_id,
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
            blockers = await self.catalog.compatible(recipe, target)
            if blockers:
                raise ValueError("; ".join(blockers))
            await self.save(job, status="running")
            inventory = await self.pool.inventory(force=True)
            pool = next((c for c in inventory["clusters"] if c["id"] == target.id), None)
            if not pool or not pool["available"]:
                raise ValueError("Target Spark is unavailable")
            if pool.get("resident_models") or pool.get("active_model"):
                raise ValueError("Unload running models on this Spark before installing a recipe")
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
            # A long download must not turn into an implicit replacement of a newer workload.
            inventory = await self.pool.inventory(force=True)
            current = next((c for c in inventory["clusters"] if c["id"] == target.id), None)
            if not current or current.get("resident_models") or current.get("active_model"):
                raise ValueError("Spark workload changed during installation; unload it and retry")
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
            await self.save(job, status="completed", stage="ready", progress=100)
        except asyncio.CancelledError:
            await self.save(
                job, status="failed", error="Installation interrupted; retry this release"
            )
            raise
        except Exception as error:
            await self.save(job, status="failed", error=self.safe_error(error))
        finally:
            self.tasks.pop(job["id"], None)
