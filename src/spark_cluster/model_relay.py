"""Authenticated, resumable model transfer from an external controller to the cluster."""

import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from .config import Settings
from .models import ControllerModelImportRequest, DownloadJob, ModelMutationResult
from .remote_cluster import signed_controller_headers, validated_cluster_url
from .store import SnapshotStore

CHUNK_SIZE = 4 * 1024 * 1024


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Invalid model file path")
    return Path(*path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRelay:
    """Build signed file manifests and copy them over the trusted management LAN."""

    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, settings: Settings, store: SnapshotStore):
        self.settings = settings
        self.store = store
        self.state_path = settings.model_cache_path / ".relay"
        self.state_path.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _token(self) -> str:
        if not self.settings.model_admin_token:
            raise RuntimeError("Configure MODEL_ADMIN_TOKEN before relaying models")
        return self.settings.model_admin_token

    def controller_url(self, request_base_url: str) -> str:
        value = self.settings.controller_public_url or request_base_url
        try:
            return validated_cluster_url(value)
        except ValueError as error:
            raise ValueError(
                "Configure CONTROLLER_PUBLIC_URL with this controller's reachable LAN IPv4 URL"
            ) from error

    def _manifest_path(self, job_id: str) -> Path:
        if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
            raise ValueError("Invalid transfer job id")
        return self.state_path / f"{job_id}.json"

    def _build_manifest(self, job: DownloadJob) -> dict:
        root = Path(job.target_path).resolve()
        if not (root / ".spark-cluster.json").is_file():
            raise FileNotFoundError("Managed model manifest is unavailable")
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink() or ".cache" in path.relative_to(root).parts:
                continue
            resolved = path.resolve()
            if root not in resolved.parents:
                raise ValueError("Model file resolves outside its managed directory")
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        manifest = {
            "version": 1,
            "job_id": job.id,
            "repo_id": job.repo_id,
            "revision": job.revision,
            "total_bytes": sum(item["size"] for item in files),
            "files": files,
        }
        manifest["signature"] = hmac.new(
            self._token().encode(), _canonical(manifest), hashlib.sha256
        ).hexdigest()
        return manifest

    async def prepare_manifest(self, job: DownloadJob) -> dict:
        manifest = await asyncio.to_thread(self._build_manifest, job)
        path = self._manifest_path(job.id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(path)
        return manifest

    def manifest(self, job_id: str) -> dict:
        path = self._manifest_path(job_id)
        if not path.exists():
            raise FileNotFoundError(job_id)
        return json.loads(path.read_text())

    def model_file(self, job_id: str, relative_path: str) -> Path:
        manifest = self.manifest(job_id)
        relative = _safe_relative(relative_path)
        if relative.as_posix() not in {item["path"] for item in manifest["files"]}:
            raise FileNotFoundError(relative_path)
        job = self.store.get_download(job_id)
        if not job:
            raise FileNotFoundError(job_id)
        root = Path(job.target_path).resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(relative_path)
        return path

    def remove_manifest(self, job_id: str) -> None:
        self._manifest_path(job_id).unlink(missing_ok=True)

    async def _remote_job_action(self, job: DownloadJob, action: str) -> bool:
        if not self.settings.remote_cluster_url:
            return False
        path = f"/api/v1/downloads/{job.id}/{action}"
        async with httpx.AsyncClient(
            base_url=self.settings.remote_cluster_url,
            transport=self.transport,
            timeout=30,
            trust_env=False,
        ) as client:
            response = await client.post(
                path,
                headers=signed_controller_headers(self._token(), "POST", path),
            )
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True

    async def pause_remote(self, job: DownloadJob) -> bool:
        return await self._remote_job_action(job, "pause")

    async def resume_remote(self, job: DownloadJob) -> bool:
        return await self._remote_job_action(job, "resume")

    async def cancel_remote(self, job: DownloadJob) -> bool:
        return await self._remote_job_action(job, "cancel")

    async def delete_remote_model(self, repo_id: str) -> ModelMutationResult:
        if not self.settings.remote_cluster_url:
            return ModelMutationResult(
                repo_id=repo_id,
                status="not-found",
                message="No attached cluster",
            )
        encoded = quote(repo_id, safe="/")
        path = f"/api/v1/model-management/models/{encoded}"
        async with httpx.AsyncClient(
            base_url=self.settings.remote_cluster_url,
            transport=self.transport,
            timeout=60,
            trust_env=False,
        ) as client:
            response = await client.delete(
                path,
                headers=signed_controller_headers(self._token(), "DELETE", path),
            )
            if response.status_code == 404:
                return ModelMutationResult(
                    repo_id=repo_id,
                    status="not-found",
                    message="Model was not present in cluster storage",
                )
            response.raise_for_status()
            result = ModelMutationResult.model_validate(response.json())
            return result.model_copy(update={"cluster_deleted": result.controller_deleted or result.cluster_deleted})

    def _verify_manifest(self, manifest: dict, job: DownloadJob) -> None:
        signature = str(manifest.pop("signature", ""))
        expected = hmac.new(
            self._token().encode(), _canonical(manifest), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise RuntimeError("Controller model manifest signature is invalid")
        if (
            manifest.get("job_id") != job.controller_job_id
            or manifest.get("repo_id") != job.repo_id
            or manifest.get("revision") != job.revision
        ):
            raise RuntimeError("Controller model manifest does not match this transfer")

    async def distribute(self, job: DownloadJob) -> bool:
        if not self.settings.remote_cluster_url or not job.controller_url:
            raise RuntimeError("No attached cluster or reachable controller URL for model relay")
        manifest = await self.prepare_manifest(job)
        job.total_bytes = manifest["total_bytes"]
        job.status = "distributing"
        job.message = "Cluster is copying the model from this controller over the LAN"
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)

        request = ControllerModelImportRequest(
            job_id=job.id,
            repo_id=job.repo_id,
            revision=job.revision,
            controller_url=job.controller_url,
            total_bytes=manifest["total_bytes"],
            fit=job.fit,
        )
        body = request.model_dump_json().encode()
        path = "/api/v1/models/import"
        timeout = httpx.Timeout(connect=10, read=60, write=60, pool=10)
        async with httpx.AsyncClient(
            base_url=self.settings.remote_cluster_url,
            transport=self.transport,
            timeout=timeout,
            trust_env=False,
        ) as client:
            response = await client.post(
                path,
                content=body,
                headers={
                    "content-type": "application/json",
                    **signed_controller_headers(self._token(), "POST", path, body),
                },
            )
            response.raise_for_status()
            while True:
                status_path = f"/api/v1/downloads/{job.id}"
                response = await client.get(
                    status_path,
                    headers=signed_controller_headers(self._token(), "GET", status_path),
                )
                response.raise_for_status()
                remote = DownloadJob.model_validate(response.json())
                job.status = "paused" if remote.status == "paused" else "distributing"
                job.transfer_progress = remote.progress
                job.transferred_bytes = remote.downloaded_bytes
                job.message = remote.message
                job.updated_at = datetime.now(UTC)
                await asyncio.to_thread(self.store.save_download, job)
                if remote.status == "completed":
                    return True
                if remote.status in {"cancelled", "deleted"}:
                    job.status = "cancelled"
                    job.message = "Cluster model transfer was cancelled"
                    job.updated_at = datetime.now(UTC)
                    await asyncio.to_thread(self.store.save_download, job)
                    return False
                if remote.status == "failed":
                    raise RuntimeError(remote.error or "Cluster model copy failed")
                await asyncio.sleep(1)

    async def pull(self, job: DownloadJob) -> None:
        if not job.controller_url or not job.controller_job_id:
            raise RuntimeError("Controller transfer details are missing")
        controller_url = validated_cluster_url(job.controller_url)
        manifest_path = f"/api/v1/controller/downloads/{job.controller_job_id}/manifest"
        timeout = httpx.Timeout(connect=10, read=None, write=60, pool=10)
        async with httpx.AsyncClient(
            base_url=controller_url,
            transport=self.transport,
            timeout=timeout,
            trust_env=False,
        ) as client:
            response = await client.get(
                manifest_path,
                headers=signed_controller_headers(self._token(), "GET", manifest_path),
            )
            response.raise_for_status()
            manifest = response.json()
            self._verify_manifest(manifest, job)
            files = manifest.get("files", [])
            job.total_bytes = int(manifest.get("total_bytes", 0))
            staging = Path(job.staging_path)
            staging.mkdir(parents=True, exist_ok=True)
            completed = 0
            for item in files:
                relative = _safe_relative(str(item["path"]))
                expected_size = int(item["size"])
                expected_hash = str(item["sha256"])
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size > expected_size:
                    target.unlink()
                offset = target.stat().st_size if target.exists() else 0
                digest = hashlib.sha256()
                if offset:
                    with target.open("rb") as existing:
                        while chunk := existing.read(CHUNK_SIZE):
                            digest.update(chunk)
                if offset == expected_size and digest.hexdigest() != expected_hash:
                    target.unlink()
                    offset = 0
                    digest = hashlib.sha256()
                if offset < expected_size:
                    encoded = quote(relative.as_posix(), safe="/")
                    file_path = (
                        f"/api/v1/controller/downloads/{job.controller_job_id}/files/{encoded}"
                    )
                    signature_path = (
                        f"/api/v1/controller/downloads/{job.controller_job_id}/files/"
                        f"{relative.as_posix()}"
                    )
                    headers = signed_controller_headers(self._token(), "GET", signature_path)
                    if offset:
                        headers["range"] = f"bytes={offset}-"
                    async with client.stream("GET", file_path, headers=headers) as stream:
                        stream.raise_for_status()
                        mode = "ab" if offset and stream.status_code == 206 else "wb"
                        if mode == "wb":
                            offset = 0
                            digest = hashlib.sha256()
                        with target.open(mode) as output:
                            async for chunk in stream.aiter_bytes(CHUNK_SIZE):
                                output.write(chunk)
                                digest.update(chunk)
                                offset += len(chunk)
                                job.downloaded_bytes = completed + offset
                                job.progress = (
                                    round(job.downloaded_bytes / job.total_bytes * 100, 2)
                                    if job.total_bytes
                                    else 0
                                )
                                job.message = "Copying model files from the controller over the LAN"
                                job.updated_at = datetime.now(UTC)
                                await asyncio.to_thread(self.store.save_download, job)
                if target.stat().st_size != expected_size or digest.hexdigest() != expected_hash:
                    target.unlink(missing_ok=True)
                    raise RuntimeError(f"Integrity verification failed for {relative.as_posix()}")
                completed += expected_size
            job.downloaded_bytes = completed
            job.progress = 100
