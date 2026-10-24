"""Controller-owned image archives and explicit, checksum-verified node imports."""

import asyncio
import contextlib
import hashlib
import os
import shlex
import shutil
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .cluster_setup import SetupFailure
from .runtime_catalog import TAG, RuntimeRegistry, catalog_entries, runtime_entry


class RuntimeDownloadRequest(BaseModel):
    runtime_id: str = Field(min_length=1, max_length=50)
    version: str = Field(pattern=f"^{TAG}$")
    platform: Literal["linux/amd64", "linux/arm64"]


class RuntimeImportRequest(BaseModel):
    artifact_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    cluster_id: str = Field(min_length=1, max_length=80)
    addresses: list[str] = Field(default_factory=list, max_length=64)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class RuntimeLibrary:
    def __init__(self, settings, store, prerequisites, registry, docker):
        self.settings = settings
        self.store = store
        self.prerequisites = prerequisites
        self.registry = registry
        self.docker = docker
        self.remote = RuntimeRegistry()
        self.root = settings.runtime_cache_path or settings.model_cache_path / ".runtime-images"
        self.tasks: dict[str, asyncio.Task] = {}
        self.download_lock = asyncio.Lock()
        self.install_lock = asyncio.Lock()
        self.queue_lock = asyncio.Lock()

    async def start(self):
        for job in await asyncio.to_thread(self.store.runtime_objects, "job"):
            if job["status"] in {"queued", "running"}:
                job.update(status="failed", message="Controller restarted; retry explicitly")
                await self._save("job", job)
        # Partial files have no published metadata and can never be installed.
        if self.root.exists():
            for partial in self.root.glob("*.partial"):
                partial.unlink(missing_ok=True)

    async def stop(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _save(self, kind, value):
        value["updated_at"] = datetime.now(UTC).isoformat()
        await asyncio.to_thread(self.store.save_runtime_object, kind, value)

    async def state(self):
        artifacts, jobs = await asyncio.gather(
            asyncio.to_thread(self.store.runtime_objects, "artifact"),
            asyncio.to_thread(self.store.runtime_objects, "job"),
        )
        return {
            "runtimes": catalog_entries(),
            "artifacts": artifacts,
            "jobs": jobs[:50],
            "download_available": bool(shutil.which("skopeo")),
            "message": "Image installation does not validate kernels or enable launch adapters.",
        }

    async def get_artifact(self, artifact_id):
        artifact = next(
            (
                item
                for item in await asyncio.to_thread(self.store.runtime_objects, "artifact")
                if item["id"] == artifact_id
            ),
            None,
        )
        if not artifact:
            raise KeyError("Controller artifact not found")
        return artifact

    async def job(self, job_id):
        job = next(
            (
                item
                for item in await asyncio.to_thread(self.store.runtime_objects, "job")
                if item["id"] == job_id
            ),
            None,
        )
        if not job:
            raise KeyError("Runtime job not found")
        return job

    async def _enqueue(self, operation, request, runner, **extra):
        async with self.queue_lock:
            jobs = await asyncio.to_thread(self.store.runtime_objects, "job")
            active = [job for job in jobs if job["status"] in {"queued", "running"}]
            for job in active:
                if job["operation"] == operation and job["request"] == request:
                    return job
            if len(active) >= 20:
                raise ValueError("Runtime queue is full; wait for a job to complete")
            job = {
                "id": uuid4().hex,
                "operation": operation,
                "request": request,
                "status": "queued",
                "message": "Queued",
                "progress": 0,
                "nodes": [],
                "created_at": datetime.now(UTC).isoformat(),
                **extra,
            }
            await self._save("job", job)
            self.tasks[job["id"]] = asyncio.create_task(self._run(job, runner))
            return dict(job)

    async def _run(self, job, runner):
        try:
            job["status"] = "running"
            await self._save("job", job)
            if job["operation"] == "install":
                async with self.install_lock:
                    await runner(job)
            else:
                await runner(job)
            job.update(status="completed", progress=100)
        except asyncio.CancelledError:
            job.update(status="failed", message="Operation interrupted; retry explicitly")
            for node in job["nodes"]:
                if node["status"] != "ready":
                    node["status"] = "failed"
            raise
        except Exception as error:
            # Process/registry errors are sanitized by the boundaries below.
            message = (
                str(error)
                if isinstance(error, (ValueError, SetupFailure))
                else (
                    "Runtime operation failed; check controller storage, registry and node access"
                )
            )
            job.update(status="failed", message=message)
        finally:
            await self._save("job", job)
            self.tasks.pop(job["id"], None)

    async def download(self, request: RuntimeDownloadRequest):
        runtime_entry(request.runtime_id)
        if not shutil.which("skopeo"):
            raise ValueError("Install skopeo on the controller (included in the Spark image)")
        return await self._enqueue("download", request.model_dump(), self._download)

    async def _download(self, job):
        async with self.download_lock:
            job["message"] = "Resolving immutable image and target architecture"
            await self._save("job", job)
            request = job["request"]
            metadata = await self.remote.resolve(**request)
            artifacts = await asyncio.to_thread(self.store.runtime_objects, "artifact")
            for artifact in artifacts:
                if all(
                    artifact[key] == metadata[key]
                    for key in ("runtime_id", "version", "digest", "platform")
                ):
                    path = self.root / f"{artifact['id']}.tar"
                    if (
                        path.is_file()
                        and await asyncio.to_thread(file_sha256, path) == artifact["sha256"]
                    ):
                        job.update(
                            artifact_id=artifact["id"], message="Verified controller cache reused"
                        )
                        return
            self.root.mkdir(parents=True, exist_ok=True)
            # Compressed layer size is a lower bound; leave room for archive expansion.
            needed = metadata["size_bytes"] * 3 + 2 * 1024**3
            if shutil.disk_usage(self.root).free < needed:
                raise ValueError("Insufficient controller disk space for this runtime archive")
            artifact_id = uuid4().hex
            partial = self.root / f"{artifact_id}.partial"
            destination = self.root / f"{artifact_id}.tar"
            try:
                job.update(message="Downloading image to controller", progress=10)
                await self._save("job", job)
                await self._command(
                    [
                        "skopeo",
                        "copy",
                        "--override-os",
                        "linux",
                        "--override-arch",
                        request["platform"].split("/")[1],
                        f"docker://{metadata['repository']}@{metadata['digest']}",
                        f"docker-archive:{partial}:{metadata['image']}",
                    ]
                )
                checksum = await asyncio.to_thread(file_sha256, partial)
                artifact = {
                    **metadata,
                    "id": artifact_id,
                    "sha256": checksum,
                    "archive_bytes": partial.stat().st_size,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                partial.replace(destination)
                await self._save("artifact", artifact)
                job.update(artifact_id=artifact_id, message="Verified archive cached on controller")
            finally:
                partial.unlink(missing_ok=True)

    async def _command(self, args, source: Path | None = None, timeout=7200):
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if source else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )

        async def finish():
            if source:
                with source.open("rb") as stream:
                    while chunk := await asyncio.to_thread(stream.read, 1024 * 1024):
                        proc.stdin.write(chunk)
                        await proc.stdin.drain()
                proc.stdin.close()
            await proc.wait()
            if proc.returncode:
                raise ValueError(
                    "Image transfer/import failed; check registry, disk and container services"
                )

        try:
            await asyncio.wait_for(finish(), timeout)
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()

    def targets(self, cluster_id, addresses):
        cluster = self.registry.get(cluster_id)
        if cluster.kind == "local-docker":
            if addresses:
                raise ValueError(
                    "Local Docker imports target the host, not individual GPU addresses"
                )
            if not self.docker.available:
                raise ValueError("Docker host access is not configured")
            return cluster, []
        jobs = self.prerequisites.store.list_setup_jobs()
        setup = next(
            (
                job
                for job in jobs
                if job.status == "ready"
                and any(
                    node.role == "head" and node.address == cluster.host_address
                    for node in job.request.nodes
                )
            ),
            None,
        )
        if not setup:
            raise ValueError(
                "This compute pool has no trusted SSH-managed nodes on this controller"
            )
        allowed = {node.address: node for node in setup.request.nodes}
        if set(addresses) - allowed.keys():
            raise ValueError("Choose only SSH-managed nodes belonging to this compute pool")
        return cluster, [
            node for node in setup.request.nodes if not addresses or node.address in addresses
        ]

    async def install(self, request: RuntimeImportRequest):
        artifact = await self.get_artifact(request.artifact_id)
        cluster, nodes = self.targets(request.cluster_id, request.addresses)

        # Freeze the target now. Changing the UI's active pool never redirects queued work.
        async def execute(job):
            await self._install(job, artifact, cluster, nodes)

        return await self._enqueue(
            "install",
            request.model_dump(),
            execute,
            runtime_id=artifact["runtime_id"],
            cluster_name=cluster.name,
        )

    async def _install(self, job, artifact, cluster, nodes):
        path = self.root / f"{artifact['id']}.tar"
        if not path.is_file() or await asyncio.to_thread(file_sha256, path) != artifact["sha256"]:
            raise ValueError("Controller archive is missing or corrupt; download it again")
        job.update(
            message="Verifying target architecture and importing controller archive", progress=5
        )
        await self._save("job", job)
        if cluster.kind == "local-docker":
            await self.docker.load_runtime_archive(
                path, artifact["platform"], artifact["image"], artifact["config_digest"]
            )
            job["nodes"] = [{"address": cluster.host_address, "status": "ready"}]
        else:
            # Preflight every selected machine before the first import.
            for node in nodes:
                facts = await self.prerequisites.setup.transport.run(
                    node, "uname -s; uname -m", timeout=30
                )
                expected = "aarch64" if artifact["platform"] == "linux/arm64" else "x86_64"
                if facts.strip().splitlines() != ["Linux", expected]:
                    raise ValueError(
                        f"Architecture mismatch on {node.address}; choose another artifact"
                    )
            job["nodes"] = [{"address": node.address, "status": "queued"} for node in nodes]
            await self._save("job", job)
            for index, node in enumerate(nodes):
                job["nodes"][index]["status"] = "installing"
                await self._save("job", job)
                try:
                    await self._import_node(node, path, artifact, job["id"])
                    job["nodes"][index]["status"] = "ready"
                except (ValueError, SetupFailure, OSError, TimeoutError):
                    job["nodes"][index]["status"] = "failed"
                job["progress"] = round((index + 1) / len(nodes) * 100)
                await self._save("job", job)
            if any(node["status"] != "ready" for node in job["nodes"]):
                raise ValueError("One or more node imports failed; inspect node status and retry")
        job["message"] = (
            "Image imported and verified; launch compatibility remains a separate check"
        )

    async def _import_node(self, node, path, artifact, job_id):
        remote = f"/var/tmp/spark-runtime-{job_id}.tar"
        image = shlex.quote(artifact["image"])
        script = (
            "set -eu\numask 077\n"
            f"archive={shlex.quote(remote)}\nimage={image}\n"
            "trap 'rm -f -- \"$archive\"' EXIT HUP INT TERM\n"
            "test $(df -Pk /var/tmp | awk 'NR==2 {print $4}') -gt "
            f"{artifact['archive_bytes'] // 1024 + 1}\n"
            'cat > "$archive"\n'
            f"printf '%s  %s\\n' {shlex.quote(artifact['sha256'])} \"$archive\" | sha256sum -c -\n"
            "if command -v k3s >/dev/null 2>&1; then\n"
            ' sudo -n k3s ctr --namespace k8s.io images import "$archive"\n'
            ' sudo -n k3s ctr --namespace k8s.io images list -q | grep -Fx -- "$image"\n'
            "elif command -v ctr >/dev/null 2>&1; then\n"
            ' sudo -n ctr --namespace k8s.io images import "$archive"\n'
            ' sudo -n ctr --namespace k8s.io images list -q | grep -Fx -- "$image"\n'
            "elif command -v docker >/dev/null 2>&1; then\n"
            ' docker load -i "$archive"\n'
            ' test "$(docker image inspect --format \'{{.Id}}\' "$image")" = '
            f"{shlex.quote(artifact['config_digest'])}\n"
            "else exit 1; fi\n"
        )
        args = self.prerequisites.setup.transport.command(node, ["bash -c " + shlex.quote(script)])
        try:
            await self._command(args, path)
        finally:
            # SSH disconnects may prevent the remote shell's trap from executing.
            with contextlib.suppress(Exception):
                await self.prerequisites.setup.transport.run(
                    node, f"rm -f -- {shlex.quote(remote)}", timeout=15
                )
