"""Goal-oriented runtime preparation, activation, launch, and verification."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import httpx

from .deepseek_mia import MANAGED_LAUNCH_BLOCKER as MIA_BLOCKER
from .deepseek_mia import MODEL_ID as MIA_MODEL_ID
from .model_pool import PoolLoadRequest
from .models import (
    RuntimeGuide,
    RuntimeGuideStep,
    RuntimeWorkflowJob,
    RuntimeWorkflowRequest,
    RuntimeWorkflowStep,
)
from .runtime_catalog import RegistryError, runtime_entry
from .runtime_engines import build_execution_plan, capabilities, recommended_engine
from .runtime_library import RuntimeDownloadRequest, RuntimeImportRequest

STEP_COPY = {
    "resolve": ("Choose a stable runtime", "Resolve the newest compatible release to a digest."),
    "cache": ("Cache the verified image", "Download the exact image once on the controller."),
    "install": ("Prepare the compute pool", "Import and verify the image on the selected target."),
    "activate": (
        "Activate the runtime",
        "Use this verified image for future launches on the pool.",
    ),
    "launch": ("Launch the model", "Apply the recommended runtime settings."),
    "verify": ("Verify a real response", "Check model discovery and generate a minimal response."),
}
PROGRESS = {"resolve": 5, "cache": 30, "install": 55, "activate": 70, "launch": 85, "verify": 100}


class RuntimeWorkflowService:
    def __init__(
        self, settings, store, registry, prerequisites, library, deployments, model_pool=None
    ):
        self.settings = settings
        self.store = store
        self.registry = registry
        self.prerequisites = prerequisites
        self.library = library
        self.deployments = deployments
        self.model_pool = model_pool
        self.tasks: dict[str, asyncio.Task] = {}
        self.http_transport: httpx.AsyncBaseTransport | None = None

    async def start(self) -> None:
        for value in await asyncio.to_thread(self.store.runtime_objects, "workflow"):
            if value.get("status") in {"queued", "running"}:
                value.update(
                    status="failed",
                    error="Controller restarted; resume the guided preparation.",
                    updated_at=datetime.now(UTC).isoformat(),
                )
                for step in value.get("steps", []):
                    if step.get("status") == "running":
                        step.update(status="failed", message="Interrupted by controller restart")
                await asyncio.to_thread(self.store.save_runtime_object, "workflow", value)

    async def stop(self) -> None:
        tasks = tuple(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _target(self, cluster_id: str):
        target = self.registry.get(cluster_id)
        if not target:
            raise KeyError(cluster_id)
        return target

    async def guide(
        self, cluster_id: str, model_id: str | None = None, engine_id: str = "auto"
    ) -> RuntimeGuide:
        target = self._target(cluster_id)
        if engine_id == "auto":
            engine_id = recommended_engine(model_id, self.settings)[0] if model_id else "vllm"
        platform = await self.prerequisites._target_platform(target)
        try:
            catalog = runtime_entry(engine_id)
        except KeyError:
            engine = capabilities(self.settings).get(engine_id)
            name = engine.name if engine else engine_id
            return RuntimeGuide(
                cluster_id=target.id,
                cluster_name=target.name,
                engine_id=engine_id,
                engine_name=name,
                model_id=model_id,
                compatible=False,
                platform=platform or "unknown",
                summary=f"{name} has no guided executable recipe on this controller.",
                blockers=list(engine.blockers) if engine else ["No managed launch adapter exists."],
                alternatives=["Choose a model supported by vLLM or SGLang."],
            )
        blockers: list[str] = []
        alternatives: list[str] = []
        if not platform or platform not in catalog.platforms:
            blockers.append(f"{catalog.name} is not published for this pool platform.")
        if not catalog.launch_adapter or engine_id not in {"vllm", "sglang"}:
            blockers.append(f"{catalog.name} has no managed launch adapter on this controller.")
        if model_id == MIA_MODEL_ID:
            blockers.append(MIA_BLOCKER)
        if engine_id == "sglang" and target.kind == "remote-bridge" and target.gpu_count > 1:
            blockers.append("Managed multi-node SGLang startup is not available yet.")

        metadata = None
        if not blockers:
            try:
                metadata = await self.library.remote.latest_stable(engine_id, platform)
            except RegistryError as error:
                activation = self.prerequisites.activation(cluster_id, engine_id)
                if not activation:
                    blockers.append(str(error))
                else:
                    metadata = {
                        **activation,
                        "runtime_id": engine_id,
                        "size_bytes": None,
                        "stale": True,
                    }

        activation = self.prerequisites.activation(cluster_id, engine_id)
        prepared = bool(
            activation and metadata and activation.get("digest") == metadata.get("digest")
        )
        if prepared:
            report = await self.prerequisites.inspect(
                model_id or f"engine:{engine_id}", engine_id, cluster_id
            )
            prepared = report.ready
        update_available = bool(
            activation and metadata and activation.get("digest") != metadata.get("digest")
        )
        if blockers:
            if engine_id != "vllm" and model_id != MIA_MODEL_ID:
                alternatives.append("Use the guided vLLM runtime on this compute pool.")
            alternatives.append("Choose a model supported by an executable runtime recipe.")
        steps = [
            RuntimeGuideStep(id=step, title=STEP_COPY[step][0], description=STEP_COPY[step][1])
            for step in ("resolve", "cache", "install", "activate")
        ]
        if model_id:
            steps.extend(
                [
                    RuntimeGuideStep(
                        id=step, title=STEP_COPY[step][0], description=STEP_COPY[step][1]
                    )
                    for step in ("launch", "verify")
                ]
            )
        summary = (
            f"{catalog.name} is ready to be prepared with one guided action."
            if not blockers
            else blockers[0]
        )
        if prepared:
            summary = f"{catalog.name} {activation['version']} is prepared on this pool."
        if update_available:
            summary = f"A newer stable {catalog.name} release is available."
        return RuntimeGuide(
            cluster_id=target.id,
            cluster_name=target.name,
            engine_id=engine_id,
            engine_name=catalog.name,
            model_id=model_id,
            compatible=not blockers,
            prepared=prepared,
            update_available=update_available,
            platform=platform or "unknown",
            version=metadata.get("version") if metadata else None,
            image=metadata.get("image") if metadata else None,
            digest=metadata.get("digest") if metadata else None,
            size_bytes=metadata.get("size_bytes") if metadata else None,
            summary=summary,
            blockers=blockers,
            alternatives=alternatives,
            steps=steps,
        )

    async def create(self, request: RuntimeWorkflowRequest) -> RuntimeWorkflowJob:
        guide = await self.guide(request.cluster_id, request.model_id, request.engine_id)
        if not guide.compatible or not guide.version or not guide.image or not guide.digest:
            raise ValueError(guide.summary)
        if request.launch:
            target = self._target(request.cluster_id)
            plan = build_execution_plan(
                request.launch,
                "/models/pending",
                self.settings,
                image_overrides={guide.engine_id: guide.image},
                cluster_kind=target.kind,
            )
            blockers = [item for item in plan.blockers if not item.startswith("No pinned ")]
            if blockers:
                raise ValueError(blockers[0])
        for value in await asyncio.to_thread(self.store.runtime_objects, "workflow"):
            if (
                value.get("status") in {"queued", "running"}
                and value.get("request", {}).get("cluster_id") == request.cluster_id
                and value.get("engine_id") == guide.engine_id
                and value.get("request", {}).get("model_id") == request.model_id
            ):
                return RuntimeWorkflowJob.model_validate(value)
        steps = [RuntimeWorkflowStep(id=step.id, title=step.title) for step in guide.steps]
        job = RuntimeWorkflowJob(
            id=uuid4().hex,
            request=request,
            engine_id=guide.engine_id,
            engine_name=guide.engine_name,
            version=guide.version,
            image=guide.image,
            digest=guide.digest,
            platform=guide.platform,
            steps=steps,
        )
        await self._save(job)
        self.tasks[job.id] = asyncio.create_task(self._run(job.id))
        return job

    async def get(self, job_id: str) -> RuntimeWorkflowJob:
        value = next(
            (
                item
                for item in await asyncio.to_thread(self.store.runtime_objects, "workflow")
                if item.get("id") == job_id
            ),
            None,
        )
        if not value:
            raise KeyError(job_id)
        return RuntimeWorkflowJob.model_validate(value)

    async def retry(self, job_id: str) -> RuntimeWorkflowJob:
        job = await self.get(job_id)
        if job.status != "failed":
            return job
        failed = next((index for index, step in enumerate(job.steps) if step.status == "failed"), 0)
        for step in job.steps[failed:]:
            step.status = "pending"
            step.message = "Waiting"
        job.status = "queued"
        job.error = None
        job.progress = PROGRESS.get(job.steps[failed - 1].id, 0) if failed else 0
        await self._save(job)
        self.tasks[job.id] = asyncio.create_task(self._run(job.id))
        return job

    async def _save(self, job: RuntimeWorkflowJob) -> None:
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(
            self.store.save_runtime_object, "workflow", job.model_dump(mode="json")
        )

    async def _step(self, job: RuntimeWorkflowJob, step_id: str, status: str, message: str) -> None:
        step = next(step for step in job.steps if step.id == step_id)
        step.status = status
        step.message = message
        job.stage = step_id
        if status == "completed":
            job.progress = PROGRESS[step_id]
        await self._save(job)

    async def _run(self, job_id: str) -> None:
        job = await self.get(job_id)
        job.status = "running"
        await self._save(job)
        current = job.stage
        try:
            current = "resolve"
            await self._step(job, current, "running", "Checking the immutable release identity")
            metadata = await self.library.remote.resolve(job.engine_id, job.version, job.platform)
            if metadata["digest"] != job.digest or metadata["image"] != job.image:
                raise ValueError("The stable tag changed; start a new guided preparation")
            await self._step(job, current, "completed", f"Resolved {job.version} to {job.digest}")

            activation = self.prerequisites.activation(job.request.cluster_id, job.engine_id)
            already_prepared = bool(activation and activation.get("digest") == job.digest)
            if already_prepared:
                report = await self.prerequisites.inspect(
                    job.request.model_id or f"engine:{job.engine_id}",
                    job.engine_id,
                    job.request.cluster_id,
                )
                already_prepared = report.ready
            if already_prepared:
                job.artifact_id = activation.get("artifact_id")
                await self._step(job, "cache", "completed", "Verified controller cache reused")
                await self._step(job, "install", "completed", "Exact image already present")
                await self._step(job, "activate", "completed", "Existing activation reused")
            else:
                current = "cache"
                await self._step(job, current, "running", "Looking for a verified controller copy")
                artifact = await self.library.verified_artifact_for_image(job.image)
                if not artifact:
                    download = await self.library.download(
                        RuntimeDownloadRequest(
                            runtime_id=job.engine_id,
                            version=job.version,
                            platform=job.platform,
                        )
                    )
                    job.library_job_id = download["id"]
                    await self._save(job)
                    task = self.library.tasks.get(download["id"])
                    if task:
                        await task
                    result = await self.library.job(download["id"])
                    if result["status"] != "completed":
                        raise ValueError(result["message"])
                    artifact = await self.library.get_artifact(result["artifact_id"])
                if artifact.get("digest") != job.digest:
                    raise ValueError("The cached image does not match the resolved digest")
                job.artifact_id = artifact["id"]
                await self._step(
                    job, current, "completed", "Verified image cached on the controller"
                )

                current = "install"
                await self._step(job, current, "running", "Importing the image on the compute pool")
                installed = await self.library.install(
                    RuntimeImportRequest(
                        artifact_id=artifact["id"], cluster_id=job.request.cluster_id
                    )
                )
                job.install_job_id = installed["id"]
                await self._save(job)
                task = self.library.tasks.get(installed["id"])
                if task:
                    await task
                result = await self.library.job(installed["id"])
                if result["status"] != "completed":
                    raise ValueError(result["message"])
                await self._step(job, current, "completed", "Exact image verified on the target")

                current = "activate"
                await self._step(
                    job, current, "running", "Activating this image for future launches"
                )
                await self.prerequisites.activate(
                    cluster_id=job.request.cluster_id,
                    engine_id=job.engine_id,
                    version=job.version,
                    image=job.image,
                    digest=job.digest,
                    platform=job.platform,
                    artifact_id=artifact["id"],
                )
                report = await self.prerequisites.inspect(
                    job.request.model_id or f"engine:{job.engine_id}",
                    job.engine_id,
                    job.request.cluster_id,
                )
                if not report.ready:
                    raise ValueError(report.message)
                await self._step(job, current, "completed", "Runtime prepared and activated")

            if job.request.launch:
                current = "launch"
                await self._step(job, current, "running", "Launching the selected model")
                if self.model_pool:
                    launch = await self.model_pool.enqueue(
                        PoolLoadRequest(
                            model_id=job.request.launch.model_id,
                            settings=job.request.launch,
                            cluster_id=job.request.cluster_id,
                        )
                    )
                    job.launch_job_id = launch["id"]
                    await self._save(job)
                    await self._wait_pool_launch(launch["id"])
                else:
                    active = self.registry.active()
                    if not active or active.id != job.request.cluster_id:
                        raise ValueError("Select this compute pool before launching the model")
                    launch = await self.deployments.launch(job.request.launch)
                    job.launch_job_id = launch.id
                    await self._save(job)
                    await self._wait_launch(launch.id)
                await self._step(job, current, "completed", "Model endpoint is ready")

                current = "verify"
                await self._step(job, current, "running", "Generating a minimal real response")
                job.response_preview = await self._smoke(job)
                await self._step(
                    job, current, "completed", "A real inference response was received"
                )

            job.status = "completed"
            job.progress = 100
            await self._save(job)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            job.status = "failed"
            job.error = (
                str(error)
                if isinstance(error, (ValueError, RegistryError))
                else (
                    "Runtime preparation failed; inspect the technical details and retry this step"
                )
            )
            step = next((item for item in job.steps if item.id == current), None)
            if step:
                step.status = "failed"
                step.message = job.error
            await self._save(job)
        finally:
            self.tasks.pop(job_id, None)

    async def _wait_launch(self, launch_id: str) -> None:
        deadline = asyncio.get_running_loop().time() + 900
        while asyncio.get_running_loop().time() < deadline:
            launch = await asyncio.to_thread(self.store.get_launch, launch_id)
            if launch and launch.status == "ready":
                return
            if launch and launch.status == "failed":
                raise ValueError(launch.error or "Model launch failed")
            await asyncio.sleep(1)
        raise ValueError("Model launch did not become ready within 15 minutes")

    async def _wait_pool_launch(self, launch_id: str) -> None:
        deadline = asyncio.get_running_loop().time() + 1800
        while asyncio.get_running_loop().time() < deadline:
            launch = next((item for item in self.model_pool.jobs if item["id"] == launch_id), None)
            if launch and launch["status"] == "ready":
                return
            if launch and launch["status"] in {"failed", "cancelled"}:
                raise ValueError(launch.get("message") or "Model launch failed")
            await asyncio.sleep(1)
        raise ValueError("Model launch did not become ready within 30 minutes")

    async def _smoke(self, job: RuntimeWorkflowJob) -> str:
        target = self._target(job.request.cluster_id)
        url = f"http://{target.host_address}:{target.runtime_port}/v1/completions"
        headers = {}
        if self.settings.openai_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"
        payload = {
            "model": job.request.launch.served_model_name,
            "prompt": "Reply with OK.",
            "max_tokens": 2,
            "temperature": 0,
        }
        try:
            async with httpx.AsyncClient(
                transport=self.http_transport, timeout=60, trust_env=False
            ) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ValueError(
                "The model endpoint did not return a valid inference response"
            ) from error
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("The model endpoint returned no inference choices")
        text = str(choices[0].get("text") or choices[0].get("message", {}).get("content") or "")
        return text[:200]
