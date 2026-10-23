"""Per-node runtime readiness, installation, and short-lived interactive SSH terminals."""

import asyncio
import contextlib
import secrets
import shlex
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from .cluster_registry import ClusterRegistry
from .cluster_setup import ClusterSetupService, SetupFailure
from .config import Settings
from .local_docker import DockerEngineClient
from .models import (
    RuntimeInstallJob,
    RuntimeInstallLog,
    RuntimeNodeState,
    RuntimeReadiness,
    RuntimeTerminalTicket,
)
from .runtime_engines import capabilities, recommended_engine
from .setup_models import SetupNode
from .store import SnapshotStore


@dataclass(frozen=True)
class RuntimeSpec:
    id: str
    name: str
    image: str | None

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
        self.tasks: dict[str, asyncio.Task] = {}
        self.tickets: dict[str, tuple[float, SetupNode]] = {}

    async def stop(self) -> None:
        for task in tuple(self.tasks.values()):
            task.cancel()
        for task in tuple(self.tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def spec_for(self, model_id: str, engine_id: str = "auto") -> RuntimeSpec:
        model = model_id.lower()
        if "gguf" in model:
            return RuntimeSpec(
                "llama-cpp", "llama.cpp CUDA server", self.settings.runtime_llama_cpp_image
            )
        if engine_id == "auto":
            engine_id, _ = recommended_engine(model_id, self.settings)
        engine = capabilities(self.settings)[engine_id]
        return RuntimeSpec(engine.id, engine.name, engine.image if engine.executable else None)

    def nodes(self) -> list[SetupNode]:
        for job in self.store.list_setup_jobs():
            if job.status == "ready":
                return job.request.nodes
        return []

    async def inspect(self, model_id: str, engine_id: str = "auto") -> RuntimeReadiness:
        spec = self.spec_for(model_id, engine_id)
        active = self.registry.active() if self.registry else None
        if active and active.kind == "local-docker":
            present = bool(spec.image and await self.docker.image_present(spec.image))
            states = [
                RuntimeNodeState(
                    address=f"{active.host_address}/gpu/{gpu.index}",
                    role="head" if index == 0 else "worker",
                    status="ready" if present else "missing",
                    installed_version=spec.version if present else None,
                    required_version=spec.version,
                    message=(
                        "Pinned runtime image is present on the Docker host"
                        if present
                        else "Runtime image is missing from the Docker host"
                    ),
                )
                for index, gpu in enumerate(active.gpus)
            ]
            return RuntimeReadiness(
                model_id=model_id,
                runtime_id=spec.id,
                runtime_name=spec.name,
                required_version=spec.version,
                image=spec.image,
                ready=present,
                install_supported=bool(spec.image),
                nodes=states,
                message=(
                    "Docker runtime image is ready for every assigned GPU"
                    if present
                    else "Pull the required runtime image on the Docker host"
                ),
            )
        nodes = self.nodes()
        if not nodes:
            return RuntimeReadiness(
                model_id=model_id,
                runtime_id=spec.id,
                runtime_name=spec.name,
                required_version=spec.version,
                image=spec.image,
                install_supported=bool(spec.image),
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
            install_supported=bool(spec.image),
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
        installed = next((image for image in images if spec.repository in image), None)
        exact = next((image for image in images if spec.image in image), None)
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

    async def install(self, model_id: str, engine_id: str = "auto") -> RuntimeInstallJob:
        spec = self.spec_for(model_id, engine_id)
        if not spec.image:
            raise ValueError("Configure a pinned image before installing this specialized runtime")
        active = self.registry.active() if self.registry else None
        if active and active.kind == "local-docker":
            states = [
                RuntimeNodeState(
                    address=f"{active.host_address}/gpu/{gpu.index}",
                    role="head" if index == 0 else "worker",
                    status="installing",
                    required_version=spec.version,
                    message="Queued on Docker host",
                )
                for index, gpu in enumerate(active.gpus)
            ]
            job = RuntimeInstallJob(
                id=uuid4().hex,
                model_id=model_id,
                runtime_id=spec.id,
                nodes=states,
            )
            await asyncio.to_thread(self.store.save_runtime_install, job)
            self.tasks[job.id] = asyncio.create_task(self._run_docker_install(job, spec))
            return job.model_copy(deep=True)
        nodes = self.nodes()
        if not nodes:
            raise ValueError("Complete cluster setup before installing runtimes")
        for existing in await asyncio.to_thread(self.store.list_runtime_installs):
            if existing.runtime_id == spec.id and existing.status in {"queued", "running"}:
                return existing
        job = RuntimeInstallJob(
            id=uuid4().hex,
            model_id=model_id,
            runtime_id=spec.id,
            nodes=[
                RuntimeNodeState(
                    address=node.address,
                    role=node.role,
                    status="installing",
                    required_version=spec.version,
                    message="Queued for installation",
                )
                for node in nodes
            ],
        )
        await asyncio.to_thread(self.store.save_runtime_install, job)
        self.tasks[job.id] = asyncio.create_task(self._run_install(job, nodes, spec))
        return job.model_copy(deep=True)

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
        node = next((node for node in self.nodes() if node.address == address), None)
        if not node:
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
