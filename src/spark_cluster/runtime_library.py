"""Controller-owned image archives and explicit, checksum-verified node imports."""

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

from .cluster_setup import SetupFailure
from .runtime_catalog import (
    SUPPORTED_PLATFORMS,
    TAG,
    RegistryError,
    RuntimeRegistry,
    catalog_entries,
    runtime_entry,
)
from .setup_models import private_address

Platform = Literal[
    "linux/amd64",
    "linux/arm64",
    "darwin/arm64",
    "darwin/amd64",
    "windows/amd64",
    "windows/arm64",
]


class RuntimeDownloadRequest(BaseModel):
    runtime_id: str = Field(min_length=1, max_length=50)
    version: str = Field(pattern=f"^{TAG}$")
    platform: Platform


class RuntimeImportRequest(BaseModel):
    artifact_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    cluster_id: str | None = Field(default=None, min_length=1, max_length=80)
    host_id: str | None = Field(default=None, pattern=r"^host-[a-f0-9]{12}$")
    addresses: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def one_destination(self):
        if bool(self.cluster_id) == bool(self.host_id):
            raise ValueError("Choose exactly one compute pool or native host")
        if self.host_id and self.addresses:
            raise ValueError("Native hosts do not accept compute-pool address filters")
        return self


class RuntimeHostRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    address: str
    username: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    fingerprint: str = Field(pattern=r"^SHA256:[A-Za-z0-9+/]{43}$")

    _address = field_validator("address")(private_address)


class RuntimeHost(BaseModel):
    id: str
    name: str
    address: str
    username: str
    fingerprint: str
    platform: Platform
    role: Literal["head", "worker"] = "worker"
    created_at: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(root: Path, artifact: dict) -> Path:
    return root / artifact.get("filename", f"{artifact['id']}.tar")


class RuntimeLibrary:
    def __init__(self, settings, store, prerequisites, registry, docker):
        self.settings = settings
        self.store = store
        self.prerequisites = prerequisites
        self.registry = registry
        self.docker = docker
        self.remote = RuntimeRegistry()
        self.native_transport: httpx.AsyncBaseTransport | None = None
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
        artifacts, jobs, hosts = await asyncio.gather(
            asyncio.to_thread(self.store.runtime_objects, "artifact"),
            asyncio.to_thread(self.store.runtime_objects, "job"),
            asyncio.to_thread(self.store.runtime_objects, "host"),
        )
        return {
            "runtimes": catalog_entries(),
            "artifacts": artifacts,
            "jobs": jobs[:50],
            "hosts": hosts,
            "download_available": bool(shutil.which("skopeo")),
            "message": (
                "Packages are installed in a managed user directory. Installation does not "
                "validate model or accelerator compatibility."
            ),
        }

    async def register_host(self, request: RuntimeHostRequest) -> dict:
        hosts = await asyncio.to_thread(self.store.runtime_objects, "host")
        existing = next((item for item in hosts if item.get("address") == request.address), None)
        host = RuntimeHost(
            id=existing["id"] if existing else f"host-{uuid4().hex[:12]}",
            name=request.name.strip(),
            address=request.address,
            username=request.username,
            fingerprint=request.fingerprint,
            platform="linux/amd64",  # Replaced by the authenticated platform probe below.
            created_at=existing["created_at"] if existing else datetime.now(UTC).isoformat(),
        )
        await self.prerequisites.setup.transport.trust([host])
        host.platform = await self._detect_host_platform(host)
        value = host.model_dump()
        await self._save("host", value)
        return value

    async def remove_host(self, host_id: str) -> None:
        hosts = await asyncio.to_thread(self.store.runtime_objects, "host")
        if not any(host["id"] == host_id for host in hosts):
            raise KeyError(host_id)
        await asyncio.to_thread(self.store.delete_runtime_object, "host", host_id)

    async def verified_artifact_for_image(self, image: str) -> dict | None:
        """Return a healthy cache entry only after proving its immutable registry identity."""
        artifacts = await asyncio.to_thread(self.store.runtime_objects, "artifact")
        digest_reference = None
        repository_reference = image
        if "@sha256:" in image:
            repository_reference, digest_value = image.rsplit("@", 1)
            digest_reference = digest_value
        for artifact in artifacts:
            if artifact.get("artifact_kind", "oci") == "native":
                continue
            exact_reference = artifact.get("image") == image
            exact_digest = (
                digest_reference
                and artifact.get("repository") == repository_reference
                and artifact.get("digest") == digest_reference
            )
            if not exact_reference and not exact_digest:
                continue
            path = artifact_path(self.root, artifact)
            valid = path.is_file() and (
                await asyncio.to_thread(file_sha256, path) == artifact["sha256"]
            )
            if not valid:
                continue
            if exact_digest:
                return artifact
            try:
                resolved = await self.remote.resolve(
                    artifact["runtime_id"], artifact["version"], artifact["platform"]
                )
            except (KeyError, RegistryError):
                continue
            if (
                resolved.get("image") == image
                and resolved.get("digest") == artifact.get("digest")
                and resolved.get("config_digest") == artifact.get("config_digest")
            ):
                return artifact
        return None

    async def install_cached(self, artifact_id: str, cluster_id: str) -> dict:
        return await self.install(
            RuntimeImportRequest(artifact_id=artifact_id, cluster_id=cluster_id)
        )

    async def delete_artifact(self, artifact_id: str) -> None:
        artifact = await self.get_artifact(artifact_id)
        jobs = await asyncio.to_thread(self.store.runtime_objects, "job")
        if any(
            job.get("status") in {"queued", "running"}
            and job.get("request", {}).get("artifact_id") == artifact_id
            for job in jobs
        ):
            raise ValueError("This controller archive is used by an active runtime job")
        path = artifact_path(self.root, artifact)
        tombstone = path.with_name(path.name + ".deleting")
        if path.exists():
            await asyncio.to_thread(path.replace, tombstone)
        try:
            await asyncio.to_thread(self.store.delete_runtime_object, "artifact", artifact_id)
        except Exception:
            if tombstone.exists():
                await asyncio.to_thread(tombstone.replace, path)
            raise
        if tombstone.exists():
            await asyncio.to_thread(tombstone.unlink)

    async def _detect_host_platform(self, host: RuntimeHost) -> Platform:
        transport = self.prerequisites.setup.transport
        try:
            output = await transport.run_command(host, ["uname -s; uname -m"], timeout=30)
            lines = [line.strip().lower() for line in output.splitlines() if line.strip()]
            if len(lines) >= 2 and lines[0] in {"linux", "darwin"}:
                platform = self._normalized_platform(lines[0], lines[1])
                if platform:
                    return platform
        except (SetupFailure, OSError, TimeoutError):
            pass
        source = (
            "$value = @{system='windows'; machine=$env:PROCESSOR_ARCHITECTURE} | "
            "ConvertTo-Json -Compress; [Console]::Out.Write($value)"
        )
        encoded = base64.b64encode(source.encode("utf-16-le")).decode()
        try:
            output = await transport.run_command(
                host,
                [f"powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded}"],
                timeout=30,
            )
            value = json.loads(output.strip())
            platform = self._normalized_platform(value.get("system", ""), value.get("machine", ""))
            if platform:
                return platform
        except (SetupFailure, OSError, TimeoutError, ValueError, TypeError, json.JSONDecodeError):
            pass
        raise ValueError("The trusted SSH host is not a supported macOS, Windows or Linux machine")

    @staticmethod
    def _normalized_platform(system: str, machine: str) -> Platform | None:
        systems = {"macos": "darwin", "win32": "windows", "windows_nt": "windows"}
        architectures = {
            "x86_64": "amd64",
            "amd64": "amd64",
            "x64": "amd64",
            "aarch64": "arm64",
            "arm64": "arm64",
        }
        target_system = systems.get(system.lower(), system.lower())
        target_architecture = architectures.get(machine.lower(), "")
        value = f"{target_system}/{target_architecture}"
        return value if value in SUPPORTED_PLATFORMS else None  # type: ignore[return-value]

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
        if request.platform.startswith("linux/") and not shutil.which("skopeo"):
            raise ValueError("Install skopeo on the controller (included in the Spark image)")
        return await self._enqueue("download", request.model_dump(), self._download)

    async def ensure_local_image(self, metadata: dict) -> dict:
        cluster_platform = metadata.get("platform")
        if not cluster_platform or not cluster_platform.startswith("linux/"):
            raise ValueError("Local Docker runtimes require a Linux image")
        return await self.docker.ensure_runtime_image(
            metadata["repository"],
            metadata["digest"],
            cluster_platform,
            metadata["config_digest"],
        )

    async def _download(self, job):
        async with self.download_lock:
            job["message"] = "Resolving immutable package and target platform"
            await self._save("job", job)
            request = job["request"]
            metadata = await self.remote.resolve(**request)
            artifacts = await asyncio.to_thread(self.store.runtime_objects, "artifact")
            for artifact in artifacts:
                if all(
                    artifact[key] == metadata[key]
                    for key in ("runtime_id", "version", "digest", "platform")
                ):
                    path = artifact_path(self.root, artifact)
                    if (
                        path.is_file()
                        and await asyncio.to_thread(file_sha256, path) == artifact["sha256"]
                    ):
                        job.update(
                            artifact_id=artifact["id"], message="Verified controller cache reused"
                        )
                        return
            self.root.mkdir(parents=True, exist_ok=True)
            # Image layers and native packages both need temporary expansion room.
            multiplier = 3 if metadata.get("artifact_kind", "oci") == "oci" else 2
            needed = metadata["size_bytes"] * multiplier + 1024**3
            if shutil.disk_usage(self.root).free < needed:
                raise ValueError("Insufficient controller disk space for this runtime archive")
            artifact_id = uuid4().hex
            partial = self.root / f"{artifact_id}.partial"
            suffix = ".tar" if metadata.get("artifact_kind", "oci") == "oci" else ".package"
            filename = f"{artifact_id}{suffix}"
            destination = self.root / filename
            try:
                job.update(message="Downloading runtime package to controller", progress=10)
                await self._save("job", job)
                if metadata.get("artifact_kind") == "native":
                    await self._download_native(metadata, partial)
                else:
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
                expected = str(metadata.get("digest", "")).removeprefix("sha256:")
                if metadata.get("artifact_kind") == "native" and checksum != expected:
                    raise ValueError("Downloaded native package checksum does not match GitHub")
                artifact = {
                    **metadata,
                    "id": artifact_id,
                    "sha256": checksum,
                    "archive_bytes": partial.stat().st_size,
                    "filename": filename,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                partial.replace(destination)
                await self._save("artifact", artifact)
                job.update(artifact_id=artifact_id, message="Verified archive cached on controller")
            finally:
                partial.unlink(missing_ok=True)

    async def _download_native(self, metadata: dict, destination: Path) -> None:
        allowed_hosts = {
            "github.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
        }
        maximum = max(int(metadata["size_bytes"]) + 1024 * 1024, 1024 * 1024)
        written = 0
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=1800,
                trust_env=False,
                transport=self.native_transport,
            ) as client:
                async with client.stream("GET", metadata["download_url"]) as response:
                    response.raise_for_status()
                    hosts = [item.request.url.host for item in response.history] + [
                        response.url.host
                    ]
                    if any(host not in allowed_hosts for host in hosts):
                        raise ValueError(
                            "GitHub release redirected to an unsupported download host"
                        )
                    with destination.open("wb") as target:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            written += len(chunk)
                            if written > maximum:
                                raise ValueError("Native package exceeded its published size")
                            target.write(chunk)
            if written != int(metadata["size_bytes"]):
                raise ValueError("Native package size does not match the GitHub release")
        except httpx.HTTPError as error:
            raise ValueError("Native package download failed") from error

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
        if request.host_id:
            hosts = await asyncio.to_thread(self.store.runtime_objects, "host")
            value = next((item for item in hosts if item["id"] == request.host_id), None)
            if not value:
                raise KeyError(request.host_id)
            host = RuntimeHost.model_validate(value)
            if artifact.get("artifact_kind") != "native":
                raise ValueError("OCI images install on Linux compute pools, not native hosts")
            if artifact["platform"] != host.platform:
                raise ValueError(
                    f"Package targets {artifact['platform']}; {host.name} is {host.platform}"
                )

            async def execute_native(job):
                await self._install_native(job, artifact, host)

            return await self._enqueue(
                "install",
                request.model_dump(),
                execute_native,
                runtime_id=artifact["runtime_id"],
                cluster_name=host.name,
            )

        cluster, nodes = self.targets(request.cluster_id, request.addresses)
        if artifact.get("artifact_kind") == "native":
            raise ValueError("Native macOS and Windows packages install on registered native hosts")

        # Freeze the target now. Changing the UI's active pool never redirects queued work.
        async def execute(job):
            await self._install_oci(job, artifact, cluster, nodes)

        return await self._enqueue(
            "install",
            request.model_dump(),
            execute,
            runtime_id=artifact["runtime_id"],
            cluster_name=cluster.name,
        )

    async def _install_oci(self, job, artifact, cluster, nodes):
        path = artifact_path(self.root, artifact)
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
                actual = await self._detect_host_platform(node)
                if actual != artifact["platform"]:
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

    async def _install_native(self, job: dict, artifact: dict, host: RuntimeHost) -> None:
        path = artifact_path(self.root, artifact)
        if not path.is_file() or await asyncio.to_thread(file_sha256, path) != artifact["sha256"]:
            raise ValueError("Controller package is missing or corrupt; download it again")
        job.update(
            message=f"Installing managed package on {host.name}",
            progress=10,
            nodes=[{"address": host.address, "status": "installing"}],
        )
        await self._save("job", job)
        try:
            if host.platform.startswith("darwin/"):
                await self._import_macos(host, path, artifact, job["id"])
            elif host.platform.startswith("windows/"):
                await self._import_windows(host, path, artifact, job["id"])
            else:
                raise ValueError("Native packages require a registered macOS or Windows host")
            job["nodes"][0]["status"] = "ready"
            job["message"] = (
                "Native runtime installed in the managed user directory; "
                "launch validation remains separate"
            )
        except Exception:
            job["nodes"][0]["status"] = "failed"
            raise

    async def _import_macos(
        self, host: RuntimeHost, path: Path, artifact: dict, job_id: str
    ) -> None:
        remote = f"/private/tmp/spark-runtime-{job_id}.package"
        runtime_id = artifact["runtime_id"]
        version = artifact["version"]
        checksum = artifact["sha256"]
        install = {
            "tar.gz": 'tar -xzf "$archive" -C "$stage"',
            "zip": 'ditto -x -k "$archive" "$stage"',
            "binary": (
                f'cp "$archive" "$stage/{shlex.quote(runtime_id)}"; '
                f'chmod 755 "$stage/{shlex.quote(runtime_id)}"'
            ),
        }[artifact["archive_format"]]
        script = (
            "set -eu\numask 077\n"
            f"archive={shlex.quote(remote)}\n"
            f'base="$HOME/Library/Application Support/Spark Cluster/runtimes/{runtime_id}"\n'
            f'dest="$base/{version}"\nstage="$base/.partial-{job_id}"\n'
            'trap \'rm -f -- "$archive"; rm -rf -- "$stage"\' EXIT HUP INT TERM\n'
            "test $(df -Pk /private/tmp | awk 'NR==2 {print $4}') -gt "
            f"{artifact['archive_bytes'] // 1024 + 1}\n"
            'cat > "$archive"\n'
            f'test "$(shasum -a 256 "$archive" | awk \'{{print $1}}\')" = {shlex.quote(checksum)}\n'
            'if test -f "$dest/.spark-runtime-sha256" && grep -Fxq '
            f'{shlex.quote(checksum)} "$dest/.spark-runtime-sha256"; then exit 0; fi\n'
            'test ! -e "$dest"\nmkdir -p "$stage"\n'
            f"{install}\n"
            f"printf '%s\\n' {shlex.quote(checksum)} > \"$stage/.spark-runtime-sha256\"\n"
            'mkdir -p "$base"\nmv "$stage" "$dest"\nln -sfn "$dest" "$base/current"\n'
        )
        args = self.prerequisites.setup.transport.command(
            host, ["/bin/sh -c " + shlex.quote(script)]
        )
        try:
            await self._command(args, path)
        finally:
            with contextlib.suppress(Exception):
                await self.prerequisites.setup.transport.run_command(
                    host, [f"rm -f -- {shlex.quote(remote)}"], timeout=15
                )

    async def _import_windows(
        self, host: RuntimeHost, path: Path, artifact: dict, job_id: str
    ) -> None:
        runtime_id = artifact["runtime_id"]
        version = artifact["version"]
        checksum = artifact["sha256"].upper()
        if artifact["archive_format"] != "zip":
            raise ValueError("This Windows package format is not supported")
        script = f"""
$ErrorActionPreference = 'Stop'
$archive = Join-Path $env:TEMP 'spark-runtime-{job_id}.package'
$root = Join-Path $env:LOCALAPPDATA 'SparkCluster\\runtimes\\{runtime_id}'
$dest = Join-Path $root '{version}'
$stage = Join-Path $root '.partial-{job_id}'
try {{
  $inputStream = [Console]::OpenStandardInput()
  $outputStream = [IO.File]::Open($archive, [IO.FileMode]::Create, [IO.FileAccess]::Write)
  try {{ $inputStream.CopyTo($outputStream) }} finally {{ $outputStream.Dispose() }}
  $actualHash = (Get-FileHash -Algorithm SHA256 $archive).Hash
  if ($actualHash -ne '{checksum}') {{ throw 'Checksum mismatch' }}
  $marker = Join-Path $dest '.spark-runtime-sha256'
  $alreadyInstalled = (Test-Path $marker) -and `
    ((Get-Content -Raw $marker).Trim() -eq '{artifact["sha256"]}')
  if ($alreadyInstalled) {{ return }}
  if (Test-Path $dest) {{ throw 'A different package already exists at this version path' }}
  New-Item -ItemType Directory -Force -Path $stage | Out-Null
  Expand-Archive -LiteralPath $archive -DestinationPath $stage -Force
  $stageMarker = Join-Path $stage '.spark-runtime-sha256'
  Set-Content -NoNewline -LiteralPath $stageMarker -Value '{artifact["sha256"]}'
  Move-Item -LiteralPath $stage -Destination $dest
  Set-Content -NoNewline -LiteralPath (Join-Path $root 'current.txt') -Value $dest
}} finally {{
  Remove-Item -Force -ErrorAction SilentlyContinue $archive
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $stage
}}
"""
        encoded = base64.b64encode(script.encode("utf-16-le")).decode()
        args = self.prerequisites.setup.transport.command(
            host,
            [f"powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded}"],
        )
        await self._command(args, path)

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
