"""Per-node runtime readiness, installation, and short-lived interactive SSH terminals."""

import asyncio
import contextlib
import secrets
import shlex
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import httpx

from .cluster_registry import ClusterRegistry, ClusterTarget
from .cluster_setup import ClusterSetupService, SetupFailure
from .config import Settings
from .local_docker import DockerEngineClient
from .models import (
    RuntimeEngineCollection,
    RuntimeEngineStatus,
    RuntimeInstallJob,
    RuntimeInstallLog,
    RuntimeNodeState,
    RuntimeReadiness,
    RuntimeTerminalTicket,
)
from .runtime_engines import MANAGED_ENGINE_IDS, capabilities, recommended_engine
from .setup_models import SetupNode
from .store import SnapshotStore


@dataclass(frozen=True)
class RuntimeSpec:
    id: str
    name: str
    image: str | None
    executable: bool
    capability: str
    blockers: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        if not self.image:
            return "not configured"
        return self.image.rsplit(":", 1)[-1] if ":" in self.image else "latest"

    @property
    def repository(self) -> str:
        if not self.image:
            return ""
        return self.image.rsplit(":", 1)[0] if ":" in self.image else self.image


class RuntimePrerequisiteService:
    def __init__(
        self,
        settings: Settings,
        store: SnapshotStore,
        setup: ClusterSetupService,
        registry: ClusterRegistry | None = None,
        docker: DockerEngineClient | None = None,
    ):
        self.settings = settings
        self.store = store
        self.setup = setup
        self.registry = registry
        self.docker = docker or DockerEngineClient(settings)
        self.library = None
        self.tasks: dict[str, asyncio.Task] = {}
        self.tickets: dict[str, tuple[float, SetupNode]] = {}

    async def stop(self) -> None:
        for task in tuple(self.tasks.values()):
            task.cancel()
        for task in tuple(self.tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def attach_library(self, library) -> None:
        self.library = library

    def activation(self, cluster_id: str, engine_id: str) -> dict | None:
        activation_id = f"{cluster_id}:{engine_id}"
        return next(
            (
                item
                for item in self.store.runtime_objects("activation")
                if item.get("id") == activation_id
            ),
            None,
        )

    def image_overrides(self, cluster_id: str | None) -> dict[str, str]:
        if not cluster_id:
            return {}
        return {
            item["engine_id"]: item["image"]
            for item in self.store.runtime_objects("activation")
            if item.get("cluster_id") == cluster_id and item.get("image")
        }

    async def activate(
        self,
        *,
        cluster_id: str,
        engine_id: str,
        version: str,
        image: str,
        digest: str,
        platform: str,
        artifact_id: str | None,
    ) -> dict:
        if not self.registry or not self.registry.get(cluster_id):
            raise KeyError(cluster_id)
        value = {
            "id": f"{cluster_id}:{engine_id}",
            "cluster_id": cluster_id,
            "engine_id": engine_id,
            "version": version,
            "image": image,
            "digest": digest,
            "platform": platform,
            "artifact_id": artifact_id,
            "source": "guided",
            "updated_at": datetime.now(UTC).isoformat(),
        }
        await asyncio.to_thread(self.store.save_runtime_object, "activation", value)
        return value

    def spec_for_engine(self, engine_id: str, cluster_id: str | None = None) -> RuntimeSpec:
        engine = capabilities(self.settings, self.image_overrides(cluster_id))[engine_id]
        has_managed_adapter = engine.id in MANAGED_ENGINE_IDS
        blockers = list(engine.blockers)
        if has_managed_adapter and not engine.image:
            blockers.insert(0, "No pinned runtime image is configured for this engine.")
        return RuntimeSpec(
            engine.id,
            engine.name,
            engine.image,
            bool(engine.executable),
            "managed_launch" if has_managed_adapter else "no_launch_adapter",
            tuple(blockers),
        )

    def spec_for(
        self, model_id: str, engine_id: str = "auto", cluster_id: str | None = None
    ) -> RuntimeSpec:
        model = model_id.lower()
        if "gguf" in model:
            return self.spec_for_engine("llama-cpp", cluster_id)
        if engine_id == "auto":
            engine_id, _ = recommended_engine(model_id, self.settings)
        return self.spec_for_engine(engine_id, cluster_id)

    def target(self, cluster_id: str | None = None) -> ClusterTarget | None:
        if not self.registry:
            if cluster_id:
                raise KeyError(cluster_id)
            return None
        return self.registry.get(cluster_id) if cluster_id else self.registry.active()

    def nodes(self, target: ClusterTarget | None = None) -> list[SetupNode]:
        for job in self.store.list_setup_jobs():
            if job.status != "ready":
                continue
            if target and target.host_address:
                if not any(
                    node.role == "head" and node.address == target.host_address
                    for node in job.request.nodes
                ):
                    continue
                return job.request.nodes
            if not target:
                return job.request.nodes
        return []

    async def inspect(
        self, model_id: str, engine_id: str = "auto", cluster_id: str | None = None
    ) -> RuntimeReadiness:
        target = self.target(cluster_id)
        spec = self.spec_for(model_id, engine_id, target.id if target else None)
        return await self._inspect_spec(model_id, spec, target)

    async def inspect_engines(self, cluster_id: str) -> RuntimeEngineCollection:
        target = self.target(cluster_id)
        if target is None:
            raise KeyError(cluster_id)
        specs = [
            self.spec_for_engine(engine_id, target.id)
            for engine_id in (*MANAGED_ENGINE_IDS, "tensorfold", "deepinfer")
        ]
        reports = await asyncio.gather(
            *(self._inspect_spec(f"engine:{spec.id}", spec, target) for spec in specs)
        )
        jobs = await asyncio.to_thread(self.store.list_runtime_installs)
        engines = [
            self._engine_status(spec, report, target, jobs)
            for spec, report in zip(specs, reports, strict=True)
        ]
        platform = await self._target_platform(target)
        return RuntimeEngineCollection(
            cluster_id=target.id,
            cluster_name=target.name,
            cluster_kind=target.kind,
            platform=platform,
            engines=engines,
        )

    async def _target_platform(self, target: ClusterTarget) -> str | None:
        if target.kind == "local-docker":
            platform = await self.docker.platform()
            if platform:
                return platform
            return "linux/arm64" if target.hardware == "dgx-spark" else "linux/amd64"
        nodes = self.nodes(target)
        if nodes:
            try:
                output = await self.setup.transport.run(nodes[0], "uname -s; uname -m", timeout=30)
                lines = [line.strip().lower() for line in output.splitlines() if line.strip()]
                systems = {"linux": "linux"}
                architectures = {
                    "x86_64": "amd64",
                    "amd64": "amd64",
                    "aarch64": "arm64",
                    "arm64": "arm64",
                }
                if len(lines) >= 2 and lines[0] in systems and lines[1] in architectures:
                    return f"{systems[lines[0]]}/{architectures[lines[1]]}"
            except (SetupFailure, OSError, TimeoutError):
                pass
        return "linux/arm64" if target.hardware == "dgx-spark" else "linux/amd64"

    def _engine_status(
        self,
        spec: RuntimeSpec,
        report: RuntimeReadiness,
        target: ClusterTarget,
        jobs: list[RuntimeInstallJob],
    ) -> RuntimeEngineStatus:
        statuses = [node.status for node in report.nodes]
        ready_nodes = statuses.count("ready")
        if not spec.executable or not spec.image:
            state = "blocked"
        elif report.ready:
            state = "ready"
        elif target.status == "offline" or (
            statuses and all(item == "unreachable" for item in statuses)
        ):
            state = "unreachable"
        elif not statuses:
            state = "blocked"
        elif ready_nodes or any(item in {"unreachable", "failed"} for item in statuses):
            state = "partial"
        else:
            state = "needs_preparation"
        active = next(
            (
                job
                for job in jobs
                if job.cluster_id == target.id
                and job.runtime_id == spec.id
                and job.status in {"queued", "running"}
            ),
            None,
        )
        return RuntimeEngineStatus(
            id=spec.id,
            name=spec.name,
            capability=spec.capability,
            image=spec.image,
            required_version=spec.version,
            state=state,
            ready=report.ready,
            prepare_supported=report.install_supported,
            nodes_ready=ready_nodes,
            nodes_total=len(report.nodes),
            nodes=report.nodes,
            message=report.message,
            blockers=list(spec.blockers),
            active_job_id=active.id if active else None,
            active_job_status=active.status if active else None,
            active_job_progress=active.progress if active else None,
        )

    async def _inspect_spec(
        self, model_id: str, spec: RuntimeSpec, target: ClusterTarget | None
    ) -> RuntimeReadiness:
        if not spec.executable:
            return RuntimeReadiness(
                model_id=model_id,
                runtime_id=spec.id,
                runtime_name=spec.name,
                required_version=spec.version,
                image=spec.image,
                ready=False,
                install_supported=False,
                message=(
                    spec.blockers[0] if spec.blockers else "No managed launch adapter is available"
                ),
            )
        if target and target.kind == "local-docker":
            reachable = self.docker.available
            try:
                present = bool(
                    reachable and spec.image and await self.docker.image_present(spec.image)
                )
            except (httpx.HTTPError, RuntimeError):
                reachable = False
                present = False
            states = [
                RuntimeNodeState(
                    address=f"{target.host_address}/gpu/{gpu.index}",
                    role="head" if index == 0 else "worker",
                    status="ready" if present else "missing" if reachable else "unreachable",
                    installed_version=spec.version if present else None,
                    required_version=spec.version,
                    message=(
                        "Pinned runtime image is present on the Docker host"
                        if present
                        else "Runtime image is missing from the Docker host"
                        if reachable
                        else "Docker host could not be inspected"
                    ),
                )
                for index, gpu in enumerate(target.gpus)
            ]
            return RuntimeReadiness(
                model_id=model_id,
                runtime_id=spec.id,
                runtime_name=spec.name,
                required_version=spec.version,
                image=spec.image,
                ready=present,
                install_supported=bool(spec.image and target.gpus and reachable),
                nodes=states,
                message=(
                    "Docker runtime image is ready for every assigned GPU"
                    if present
                    else "Pull the required runtime image on the Docker host"
                    if reachable
                    else "Docker host is unavailable"
                ),
            )
        nodes = self.nodes(target)
        if not nodes:
            return RuntimeReadiness(
                model_id=model_id,
                runtime_id=spec.id,
                runtime_name=spec.name,
                required_version=spec.version,
                image=spec.image,
                install_supported=False,
                message="No SSH-managed cluster nodes are available yet",
            )
        states = await asyncio.gather(*(self._inspect_node(node, spec) for node in nodes))
        ready = bool(states) and all(state.status == "ready" for state in states)
        return RuntimeReadiness(
            model_id=model_id,
            runtime_id=spec.id,
            runtime_name=spec.name,
            required_version=spec.version,
            image=spec.image,
            ready=ready,
            install_supported=bool(spec.image and spec.executable),
            nodes=list(states),
            message=(
                "Required runtime is ready on every node"
                if ready
                else "Install or update the required runtime on the highlighted nodes"
            ),
        )

    async def _inspect_node(self, node: SetupNode, spec: RuntimeSpec) -> RuntimeNodeState:
        if not spec.image:
            return RuntimeNodeState(
                address=node.address,
                role=node.role,
                status="missing",
                required_version=spec.version,
                message="No automated image is configured for this specialized runtime",
            )
        script = (
            "set -eu\n"
            "if command -v k3s >/dev/null 2>&1; then "
            "sudo -n k3s ctr images list -q 2>/dev/null || true; "
            "elif command -v crictl >/dev/null 2>&1; then "
            "sudo -n crictl images 2>/dev/null | awk 'NR > 1 {print $1 \":\" $2}'; "
            "elif command -v docker >/dev/null 2>&1; then "
            "docker image ls --format '{{.Repository}}:{{.Tag}}'; "
            "fi\n"
        )
        try:
            output = await self.setup.transport.run(node, script, timeout=30)
        except (SetupFailure, OSError, TimeoutError):
            return RuntimeNodeState(
                address=node.address,
                role=node.role,
                status="unreachable",
                required_version=spec.version,
                message="Node could not be inspected over trusted SSH",
            )
        images = [line.strip() for line in output.splitlines() if line.strip()]
        normalized_expected = self._normalized_image_reference(spec.image)
        normalized_repository = self._normalized_image_reference(spec.repository)
        exact = next(
            (
                image
                for image in images
                if self._normalized_image_reference(image) == normalized_expected
            ),
            None,
        )
        installed = next(
            (
                image
                for image in images
                if self._normalized_image_reference(image).startswith(
                    (normalized_repository + ":", normalized_repository + "@")
                )
            ),
            None,
        )
        if exact:
            return RuntimeNodeState(
                address=node.address,
                role=node.role,
                status="ready",
                installed_version=spec.version,
                required_version=spec.version,
                message="Pinned runtime image is present",
            )
        if installed:
            installed_version = installed.rsplit(":", 1)[-1] if ":" in installed else installed
            return RuntimeNodeState(
                address=node.address,
                role=node.role,
                status="outdated",
                installed_version=installed_version,
                required_version=spec.version,
                message="A different runtime version is installed",
            )
        return RuntimeNodeState(
            address=node.address,
            role=node.role,
            status="missing",
            required_version=spec.version,
            message="Runtime image is not present on this node",
        )

    @staticmethod
    def _normalized_image_reference(image: str | None) -> str:
        return (image or "").removeprefix("docker.io/")

    async def install(
        self, model_id: str, engine_id: str = "auto", cluster_id: str | None = None
    ) -> RuntimeInstallJob:
        target = self.target(cluster_id)
        spec = self.spec_for(model_id, engine_id, target.id if target else None)
        return await self._install_spec(model_id, spec, target)

    async def prepare(self, cluster_id: str, engine_id: str) -> RuntimeInstallJob:
        target = self.target(cluster_id)
        spec = self.spec_for_engine(engine_id, target.id if target else None)
        return await self._install_spec(f"engine:{engine_id}", spec, target)

    async def _install_spec(
        self, model_id: str, spec: RuntimeSpec, target: ClusterTarget | None
    ) -> RuntimeInstallJob:
        if not spec.executable:
            raise ValueError(
                spec.blockers[0] if spec.blockers else "No managed launch adapter is available"
            )
        if not spec.image:
            raise ValueError("Configure a pinned image before installing this specialized runtime")
        if self.registry and target is None:
            raise ValueError("Choose a compute pool before preparing a runtime")
        cluster_id = target.id if target else None
        cluster_name = target.name if target else None
        nodes = self.nodes(target)
        if target and target.kind == "local-docker":
            states = [
                RuntimeNodeState(
                    address=f"{target.host_address}/gpu/{gpu.index}",
                    role="head" if index == 0 else "worker",
                    status="installing",
                    required_version=spec.version,
                    message="Queued on Docker host",
                )
                for index, gpu in enumerate(target.gpus)
            ]
            if not states or not self.docker.available:
                raise ValueError("Docker host access is not configured for this compute pool")
        elif not nodes:
            raise ValueError("Complete cluster setup before installing runtimes")
        for existing in await asyncio.to_thread(self.store.list_runtime_installs):
            if (
                existing.runtime_id == spec.id
                and existing.cluster_id == cluster_id
                and existing.status in {"queued", "running"}
                and existing.nodes
                and existing.nodes[0].required_version == spec.version
            ):
                return existing
        job = RuntimeInstallJob(
            id=uuid4().hex,
            model_id=model_id,
            runtime_id=spec.id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            nodes=(
                states
                if target and target.kind == "local-docker"
                else [
                    RuntimeNodeState(
                        address=node.address,
                        role=node.role,
                        status="installing",
                        required_version=spec.version,
                        message="Queued for installation",
                    )
                    for node in nodes
                ]
            ),
        )
        await asyncio.to_thread(self.store.save_runtime_install, job)
        self.tasks[job.id] = asyncio.create_task(self._run_prepare(job, target, nodes, spec))
        return job.model_copy(deep=True)

    async def _run_prepare(
        self,
        job: RuntimeInstallJob,
        target: ClusterTarget | None,
        nodes: list[SetupNode],
        spec: RuntimeSpec,
    ) -> None:
        artifact = None
        if self.library and target:
            try:
                artifact = await self.library.verified_artifact_for_image(spec.image or "")
            except (KeyError, OSError, ValueError):
                artifact = None
        if artifact and target:
            await self._run_cached_install(job, target, artifact)
        elif target and target.kind == "local-docker":
            await self._run_docker_install(job, spec)
        else:
            await self._run_install(job, nodes, spec)

    async def _run_cached_install(
        self, job: RuntimeInstallJob, target: ClusterTarget, artifact: dict
    ) -> None:
        job.status = "running"
        self._log(job, "controller-cache", "info", "Importing verified controller archive")
        await asyncio.to_thread(self.store.save_runtime_install, job)
        try:
            library_job = await self.library.install_cached(artifact["id"], target.id)
            task = self.library.tasks.get(library_job["id"])
            if task:
                await task
            result = await self.library.job(library_job["id"])
            if result["status"] != "completed":
                raise ValueError(result["message"])
            outcomes = {item["address"]: item["status"] for item in result.get("nodes", [])}
            for index, node in enumerate(job.nodes):
                ready = result["status"] == "completed" and (
                    target.kind == "local-docker" or outcomes.get(node.address) == "ready"
                )
                job.nodes[index] = node.model_copy(
                    update={
                        "status": "ready" if ready else "failed",
                        "installed_version": node.required_version if ready else None,
                        "message": (
                            "Verified controller archive imported"
                            if ready
                            else "Controller archive import failed"
                        ),
                    }
                )
            job.progress = 100
            job.status = "completed"
            self._log(job, "controller-cache", "success", "Runtime image is ready")
        except (KeyError, ValueError):
            job.status = "failed"
            job.error = "Verified controller archive could not be imported"
            self._log(job, "controller-cache", "error", job.error)
        finally:
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_runtime_install, job)
            self.tasks.pop(job.id, None)

    async def _run_docker_install(self, job: RuntimeInstallJob, spec: RuntimeSpec) -> None:
        job.status = "running"
        await asyncio.to_thread(self.store.save_runtime_install, job)
        try:
            self._log(job, "docker-host", "info", f"Pulling {spec.name} {spec.version}")
            await self.docker.pull_image(spec.image or "")
            for index, node in enumerate(job.nodes):
                job.nodes[index] = node.model_copy(
                    update={
                        "status": "ready",
                        "installed_version": spec.version,
                        "message": "Runtime image is ready on the Docker host",
                    }
                )
            job.progress = 100
            job.status = "completed"
            self._log(job, "docker-host", "success", "Runtime image is ready")
        except Exception:
            job.status = "failed"
            job.error = "Docker runtime image installation failed"
            for index, node in enumerate(job.nodes):
                job.nodes[index] = node.model_copy(
                    update={"status": "failed", "message": job.error}
                )
        finally:
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_runtime_install, job)
            self.tasks.pop(job.id, None)

    async def _run_install(
        self, job: RuntimeInstallJob, nodes: list[SetupNode], spec: RuntimeSpec
    ) -> None:
        job.status = "running"
        await asyncio.to_thread(self.store.save_runtime_install, job)
        try:
            for index, node in enumerate(nodes):
                self._log(job, node.address, "info", f"Pulling {spec.name} {spec.version}")
                script = (
                    "set -eu\n"
                    f"image={shlex.quote(spec.image or '')}\n"
                    "if command -v k3s >/dev/null 2>&1; then "
                    'sudo -n k3s ctr images pull "$image"; '
                    'elif command -v crictl >/dev/null 2>&1; then sudo -n crictl pull "$image"; '
                    'elif command -v docker >/dev/null 2>&1; then docker pull "$image"; '
                    "else echo 'No supported container runtime found' >&2; exit 1; fi\n"
                )
                try:
                    await self.setup.transport.run(node, script, timeout=1800)
                    job.nodes[index] = RuntimeNodeState(
                        address=node.address,
                        role=node.role,
                        status="ready",
                        installed_version=spec.version,
                        required_version=spec.version,
                        message="Runtime installed and verified",
                    )
                    self._log(job, node.address, "success", "Runtime image is ready")
                except (SetupFailure, OSError, TimeoutError):
                    job.nodes[index] = RuntimeNodeState(
                        address=node.address,
                        role=node.role,
                        status="failed",
                        required_version=spec.version,
                        message="Installation failed; open terminal logs for diagnostics",
                    )
                    self._log(job, node.address, "error", "Runtime installation failed")
                job.progress = round((index + 1) / len(nodes) * 100, 2)
                job.updated_at = datetime.now(UTC)
                await asyncio.to_thread(self.store.save_runtime_install, job)
            failed = [node for node in job.nodes if node.status != "ready"]
            job.status = "failed" if failed else "completed"
            job.error = "One or more nodes need attention" if failed else None
        except asyncio.CancelledError:
            job.status = "failed"
            job.error = "Runtime installation was interrupted"
            raise
        finally:
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_runtime_install, job)
            self.tasks.pop(job.id, None)

    def _log(self, job: RuntimeInstallJob, node: str, level: str, message: str) -> None:
        job.logs.append(RuntimeInstallLog(node=node, level=level, message=message))
        job.logs = job.logs[-1000:]

    async def get_install(self, job_id: str) -> RuntimeInstallJob:
        job = await asyncio.to_thread(self.store.get_runtime_install, job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def issue_terminal_ticket(self, address: str) -> RuntimeTerminalTicket:
        node = next((node for node in self.nodes(self.target()) if node.address == address), None)
        if not node:
            raise KeyError(address)
        ticket = secrets.token_urlsafe(32)
        self.tickets[ticket] = (time.monotonic() + 60, node)
        return RuntimeTerminalTicket(ticket=ticket, address=address)

    def issue_setup_terminal_ticket(self, job_id: str, address: str) -> RuntimeTerminalTicket:
        job = self.store.get_setup_job(job_id)
        if not job or job.status not in {"failed", "blocked"}:
            raise KeyError(job_id)
        node = next((node for node in job.request.nodes if node.address == address), None)
        if not node or not self.setup.transport.identity_for(node):
            raise KeyError(address)
        ticket = secrets.token_urlsafe(32)
        self.tickets[ticket] = (time.monotonic() + 60, node)
        return RuntimeTerminalTicket(ticket=ticket, address=address)

    def consume_terminal_ticket(self, ticket: str) -> SetupNode:
        value = self.tickets.pop(ticket, None)
        if not value or value[0] < time.monotonic():
            raise KeyError(ticket)
        return value[1]

    async def terminal_process(self, ticket: str) -> asyncio.subprocess.Process:
        node = self.consume_terminal_ticket(ticket)
        command = self.setup.transport.command(node, ["bash", "-l"], tty=True)
        return await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
