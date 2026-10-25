import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.cluster_registry import ClusterRegistry
from spark_cluster.cluster_setup import ClusterSetupService
from spark_cluster.config import Settings
from spark_cluster.local_docker import DockerEngineClient
from spark_cluster.runtime_library import (
    RuntimeDownloadRequest,
    RuntimeHost,
    RuntimeHostRequest,
    RuntimeImportRequest,
    RuntimeLibrary,
)
from spark_cluster.runtime_prerequisites import RuntimePrerequisiteService
from spark_cluster.setup_models import SetupJob, SetupNode, SetupRequest
from spark_cluster.store import SnapshotStore


@pytest.fixture
def library(tmp_path):
    settings = Settings(
        database_path=tmp_path / "state.db",
        setup_state_path=tmp_path / "setup",
        model_cache_path=tmp_path / "models",
        runtime_cache_path=tmp_path / "runtimes",
        model_admin_token="test-admin",
    )
    store = SnapshotStore(settings.database_path)
    registry = ClusterRegistry(settings)
    setup = ClusterSetupService(settings, store)
    prerequisites = RuntimePrerequisiteService(settings, store, setup, registry)
    result = RuntimeLibrary(settings, store, prerequisites, registry, DockerEngineClient(settings))
    prerequisites.attach_library(result)
    yield result
    store.close()


async def add_artifact(library, content=b"test-archive"):
    library.root.mkdir(exist_ok=True)
    artifact = {
        "id": "a" * 32,
        "runtime_id": "vllm",
        "version": "v1",
        "platform": "linux/arm64",
        "digest": "sha256:" + "b" * 64,
        "config_digest": "sha256:" + "c" * 64,
        "image": "docker.io/vllm/vllm-openai:v1",
        "repository": "docker.io/vllm/vllm-openai",
        "sha256": hashlib.sha256(content).hexdigest(),
        "archive_bytes": len(content),
    }
    (library.root / f"{artifact['id']}.tar").write_bytes(content)
    await library._save("artifact", artifact)
    return artifact


async def add_native_artifact(
    library, platform="darwin/arm64", content=b"native-package", runtime_id="llama-cpp"
):
    artifact = {
        "id": "b" * 32,
        "runtime_id": runtime_id,
        "version": "b1234",
        "platform": platform,
        "artifact_kind": "native",
        "archive_format": "zip" if platform.startswith("windows/") else "tar.gz",
        "asset_name": "runtime-package.zip",
        "repository": "example/runtime",
        "digest": "sha256:" + hashlib.sha256(content).hexdigest(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "archive_bytes": len(content),
        "filename": "b" * 32 + ".package",
    }
    (library.root / artifact["filename"]).parent.mkdir(exist_ok=True)
    (library.root / artifact["filename"]).write_bytes(content)
    await library._save("artifact", artifact)
    return artifact


def add_pool(library, address="192.168.50.10", ready=True):
    cluster = library.registry.add_remote(
        name="Spark", endpoint=f"http://{address}:30080", head_address=address, hardware="dgx-spark"
    )
    if ready:
        request = SetupRequest(
            name="spark",
            topology="direct",
            nodes=[
                SetupNode(
                    address=address,
                    username="nvidia",
                    fingerprint="SHA256:" + "A" * 43,
                    role="head",
                ),
                SetupNode(
                    address="192.168.50.11",
                    username="nvidia",
                    fingerprint="SHA256:" + "A" * 43,
                    role="worker",
                ),
            ],
        )
        library.store.save_setup_job(
            SetupJob(request=request, mode="ssh", status="ready", steps=[])
        )
    return cluster


@pytest.mark.asyncio
async def test_download_is_digest_pinned_deduplicated_and_cached(library, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.runtime_library.shutil.which", lambda name: "/usr/bin/skopeo"
    )
    artifact = await add_artifact(library)
    metadata = {
        key: value
        for key, value in artifact.items()
        if key not in {"id", "sha256", "archive_bytes", "updated_at"}
    }
    metadata["size_bytes"] = 10
    library.remote.resolve = AsyncMock(return_value=metadata)
    library._command = AsyncMock()
    request = RuntimeDownloadRequest(runtime_id="vllm", version="v1", platform="linux/arm64")
    first = await library.download(request)
    second = await library.download(request)
    assert first["id"] == second["id"]
    await library.tasks[first["id"]]
    job = await library.job(first["id"])
    assert job["status"] == "completed" and job["artifact_id"] == artifact["id"]
    library._command.assert_not_awaited()

    # Corruption forces a fresh archive and the registry reference is immutable.
    (library.root / f"{artifact['id']}.tar").write_bytes(b"corrupt")
    calls = []

    async def copy(args):
        calls.append(args)
        destination = args[-1].split(":", 2)[1]
        Path(destination).write_bytes(b"fresh-archive")

    library._command = copy
    third = await library.download(request)
    await library.tasks[third["id"]]
    assert (await library.job(third["id"]))["status"] == "completed"
    assert "docker://docker.io/vllm/vllm-openai@" + artifact["digest"] in calls[0]
    assert not list(library.root.glob("*.partial"))


@pytest.mark.asyncio
async def test_prepare_reuses_only_registry_verified_matching_archive(library):
    artifact = await add_artifact(library)
    library.settings.runtime_vllm_image = artifact["image"]
    library.remote.resolve = AsyncMock(
        return_value={
            "image": artifact["image"],
            "digest": artifact["digest"],
            "config_digest": artifact["config_digest"],
        }
    )
    pool = add_pool(library)
    library.prerequisites.setup.transport.run_command = AsyncMock(return_value="Linux\naarch64\n")
    library._import_node = AsyncMock()

    job = await library.prerequisites.prepare(pool.id, "vllm")
    await library.prerequisites.tasks[job.id]
    result = await library.prerequisites.get_install(job.id)
    assert result.status == "completed"
    assert any(log.node == "controller-cache" for log in result.logs)
    assert library._import_node.await_count == 2
    library.remote.resolve.assert_awaited_once_with("vllm", "v1", "linux/arm64")


@pytest.mark.asyncio
async def test_cached_artifact_deletion_refuses_active_job_and_removes_file(library):
    artifact = await add_artifact(library)
    active = {
        "id": "c" * 32,
        "operation": "install",
        "status": "running",
        "request": {"artifact_id": artifact["id"]},
    }
    await library._save("job", active)
    with pytest.raises(ValueError, match="active runtime job"):
        await library.delete_artifact(artifact["id"])
    assert (library.root / f"{artifact['id']}.tar").exists()

    active["status"] = "completed"
    await library._save("job", active)
    await library.delete_artifact(artifact["id"])
    assert not (library.root / f"{artifact['id']}.tar").exists()
    assert not library.store.runtime_objects("artifact")


@pytest.mark.asyncio
async def test_download_failure_cleans_partial_and_persists_error(library, monkeypatch):
    monkeypatch.setattr("spark_cluster.runtime_library.shutil.which", lambda name: "skopeo")
    library.remote.resolve = AsyncMock(
        return_value={
            "size_bytes": 1,
            "repository": "docker.io/a/b",
            "digest": "sha256:" + "a" * 64,
            "image": "a/b:v1",
        }
    )

    async def fail(args):
        Path(args[-1].split(":", 2)[1]).write_bytes(b"partial")
        raise ValueError("Registry copy failed")

    library._command = fail
    job = await library.download(
        RuntimeDownloadRequest(runtime_id="vllm", version="v1", platform="linux/arm64")
    )
    await library.tasks[job["id"]]
    result = await library.job(job["id"])
    assert result["status"] == "failed" and result["message"] == "Registry copy failed"
    assert not list(library.root.iterdir())
    assert not library.store.runtime_objects("artifact")


@pytest.mark.asyncio
async def test_native_download_uses_release_digest_without_skopeo(library, monkeypatch):
    content = b"verified-native-package"
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr("spark_cluster.runtime_library.shutil.which", lambda name: None)
    library.remote.resolve = AsyncMock(
        return_value={
            "runtime_id": "ollama",
            "version": "v1.2.3",
            "platform": "darwin/arm64",
            "artifact_kind": "native",
            "repository": "ollama/ollama",
            "asset_name": "ollama-darwin.tgz",
            "archive_format": "tar.gz",
            "download_url": "https://github.com/ollama/ollama/releases/download/v1.2.3/x",
            "digest": "sha256:" + checksum,
            "size_bytes": len(content),
        }
    )
    library.native_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=content)
    )
    job = await library.download(
        RuntimeDownloadRequest(runtime_id="ollama", version="v1.2.3", platform="darwin/arm64")
    )
    await library.tasks[job["id"]]
    result = await library.job(job["id"])
    artifact = await library.get_artifact(result["artifact_id"])
    assert result["status"] == "completed"
    assert artifact["artifact_kind"] == "native"
    assert (library.root / artifact["filename"]).read_bytes() == content


@pytest.mark.asyncio
async def test_native_download_rejects_untrusted_redirect(library, tmp_path):
    def handle(request):
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"location": "https://evil.example/package"})
        return httpx.Response(200, content=b"x")

    library.native_transport = httpx.MockTransport(handle)
    with pytest.raises(ValueError, match="unsupported download host"):
        await library._download_native(
            {"download_url": "https://github.com/x/y", "size_bytes": 1},
            tmp_path / "package",
        )


@pytest.mark.asyncio
async def test_install_freezes_pool_and_reports_each_node(library):
    artifact = await add_artifact(library)
    pool = add_pool(library)
    transport = library.prerequisites.setup.transport
    transport.run_command = AsyncMock(return_value="Linux\naarch64\n")
    imported = []

    async def import_node(node, path, metadata, job_id):
        imported.append(node.address)
        if node.role == "worker":
            raise ValueError("No disk space")

    library._import_node = import_node
    job = await library.install(
        RuntimeImportRequest(artifact_id=artifact["id"], cluster_id=pool.id)
    )
    other = library.registry.add_remote(
        name="Other",
        endpoint="http://192.168.60.10:30080",
        head_address="192.168.60.10",
        hardware="other",
    )
    library.registry.select(other.id)
    await library.tasks[job["id"]]
    result = await library.job(job["id"])
    assert result["status"] == "failed"
    assert imported == ["192.168.50.10", "192.168.50.11"]
    assert [node["status"] for node in result["nodes"]] == ["ready", "failed"]


@pytest.mark.asyncio
async def test_install_subset_checks_architecture_and_archive_before_any_import(library):
    artifact = await add_artifact(library)
    pool = add_pool(library)
    library.prerequisites.setup.transport.run_command = AsyncMock(return_value="Linux\nx86_64\n")
    library._import_node = AsyncMock()
    request = RuntimeImportRequest(
        artifact_id=artifact["id"], cluster_id=pool.id, addresses=["192.168.50.11"]
    )
    job = await library.install(request)
    await library.tasks[job["id"]]
    assert "Architecture mismatch" in (await library.job(job["id"]))["message"]
    library._import_node.assert_not_awaited()
    library.prerequisites.setup.transport.run_command.return_value = "Linux\naarch64\n"
    job = await library.install(request)
    await library.tasks[job["id"]]
    assert (await library.job(job["id"]))["status"] == "completed"
    assert library._import_node.call_args.args[0].address == "192.168.50.11"
    library._import_node.reset_mock()
    (library.root / f"{artifact['id']}.tar").write_bytes(b"corrupt")
    job = await library.install(request)
    await library.tasks[job["id"]]
    assert "corrupt" in (await library.job(job["id"]))["message"]
    library._import_node.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejects_unmanaged_pool_and_cross_pool_addresses(library):
    artifact = await add_artifact(library)
    pool = add_pool(library)
    with pytest.raises(ValueError, match="belonging"):
        await library.install(
            RuntimeImportRequest(
                artifact_id=artifact["id"], cluster_id=pool.id, addresses=["192.168.70.10"]
            )
        )
    unmanaged = add_pool(library, "192.168.60.10", ready=False)
    with pytest.raises(ValueError, match="no trusted"):
        await library.install(
            RuntimeImportRequest(artifact_id=artifact["id"], cluster_id=unmanaged.id)
        )


@pytest.mark.asyncio
async def test_registers_fingerprint_pinned_native_host_and_detects_macos(library):
    class NativeTransport:
        def __init__(self):
            self.trusted = []

        async def trust(self, nodes):
            self.trusted.extend(nodes)

        async def run_command(self, node, command, timeout=60):
            return "Darwin\narm64\n"

    transport = NativeTransport()
    library.prerequisites.setup.transport = transport
    request = RuntimeHostRequest(
        name="Studio Mac",
        address="192.168.50.30",
        username="thibaut",
        fingerprint="SHA256:" + "A" * 43,
    )
    host = await library.register_host(request)
    assert host["platform"] == "darwin/arm64"
    assert transport.trusted[0].fingerprint == "SHA256:" + "A" * 43
    assert (await library.state())["hosts"][0]["id"] == host["id"]
    repeated = await library.register_host(request.model_copy(update={"name": "Mac Studio"}))
    assert repeated["id"] == host["id"]
    assert repeated["name"] == "Mac Studio"
    assert len((await library.state())["hosts"]) == 1
    await library.remove_host(host["id"])
    assert (await library.state())["hosts"] == []


@pytest.mark.asyncio
async def test_native_install_requires_matching_registered_host(library):
    artifact = await add_native_artifact(library)
    host = RuntimeHost(
        id="host-" + "c" * 12,
        name="Studio Mac",
        address="192.168.50.30",
        username="thibaut",
        fingerprint="SHA256:" + "A" * 43,
        platform="darwin/arm64",
        created_at="2026-10-24T12:00:00+00:00",
    )
    library.store.save_runtime_object("host", host.model_dump())
    library._import_macos = AsyncMock()
    job = await library.install(RuntimeImportRequest(artifact_id=artifact["id"], host_id=host.id))
    await library.tasks[job["id"]]
    assert (await library.job(job["id"]))["status"] == "completed"
    library._import_macos.assert_awaited_once()

    host.platform = "windows/amd64"
    library.store.save_runtime_object("host", host.model_dump())
    with pytest.raises(ValueError, match="Package targets"):
        await library.install(RuntimeImportRequest(artifact_id=artifact["id"], host_id=host.id))


@pytest.mark.asyncio
async def test_macos_and_windows_installers_are_user_scoped_and_checksum_verified(library):
    mac_artifact = await add_native_artifact(library)
    mac = RuntimeHost(
        id="host-" + "d" * 12,
        name="Mac",
        address="192.168.50.31",
        username="user",
        fingerprint="SHA256:" + "A" * 43,
        platform="darwin/arm64",
        created_at="2026-10-24T12:00:00+00:00",
    )
    commands = []

    class CaptureTransport:
        def command(self, node, remote):
            commands.append(remote[0])
            return ["ssh"]

        async def run_command(self, node, remote, timeout=60):
            return ""

    library.prerequisites.setup.transport = CaptureTransport()
    library._command = AsyncMock()
    await library._import_macos(
        mac, library.root / mac_artifact["filename"], mac_artifact, "e" * 32
    )
    assert "shasum -a 256" in commands[0]
    assert "Library/Application Support/Spark Cluster/runtimes" in commands[0]
    assert "tar -xzf" in commands[0]

    windows_artifact = await add_native_artifact(
        library, platform="windows/amd64", runtime_id="ollama"
    )
    windows = mac.model_copy(update={"platform": "windows/amd64"})
    library._command.reset_mock()
    await library._import_windows(
        windows,
        library.root / windows_artifact["filename"],
        windows_artifact,
        "f" * 32,
    )
    encoded = commands[-1].rsplit(" ", 1)[-1]
    powershell = base64.b64decode(encoded).decode("utf-16-le")
    assert "Get-FileHash -Algorithm SHA256" in powershell
    assert "Expand-Archive" in powershell
    assert "$env:LOCALAPPDATA" in powershell


@pytest.mark.asyncio
async def test_ssh_stream_import_verifies_checksum_and_image_and_cleans_up(library):
    artifact = await add_artifact(library)
    pool = add_pool(library)
    _, nodes = library.targets(pool.id, [])
    transport = library.prerequisites.setup.transport
    command = transport.command
    captured = []

    def capture(node, remote):
        captured.extend(remote)
        return command(node, remote)

    transport.command = capture
    transport.run = AsyncMock()
    library._command = AsyncMock(side_effect=ValueError("Import failed"))
    with pytest.raises(ValueError):
        await library._import_node(
            nodes[0], library.root / f"{artifact['id']}.tar", artifact, "a" * 32
        )
    script = captured[0]
    assert "sha256sum -c" in script and artifact["sha256"] in script
    assert "--namespace k8s.io images import" in script
    assert "docker image inspect" in script and artifact["config_digest"] in script
    transport.run.assert_awaited_once()
    assert "rm -f -- /var/tmp/spark-runtime-" in transport.run.call_args.args[1]


@pytest.mark.asyncio
async def test_restart_marks_inflight_jobs_failed_without_replaying(library):
    library.store.save_runtime_object("job", {"id": "old", "status": "running"})
    library.root.mkdir()
    (library.root / "old.partial").write_bytes(b"partial")
    await library.start()
    assert (await library.job("old"))["status"] == "failed"
    assert not list(library.root.iterdir())
    assert not library.tasks


@pytest.mark.asyncio
async def test_docker_archive_stream_checks_host_identity_and_embedded_errors(tmp_path):
    settings = Settings(docker_runtime_enabled=True, docker_socket_path=tmp_path / "docker.sock")
    settings.docker_socket_path.touch()
    docker = DockerEngineClient(settings)
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"image-data")
    error = False
    architecture = "aarch64"
    received = []

    async def handle(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={"OSType": "linux", "Architecture": architecture})
        if request.url.path == "/images/load":
            received.append(await request.aread())
            return httpx.Response(
                200,
                text=json.dumps({"error": "bad archive"} if error else {"stream": "Loaded image"})
                + "\n",
            )
        return httpx.Response(
            200,
            json={
                "Id": "sha256:" + "a" * 64,
                "Os": "linux",
                "Architecture": architecture,
            },
        )

    docker.transport = httpx.MockTransport(handle)
    await docker.load_runtime_archive(archive, "linux/arm64", "a/b:v1", "sha256:" + "a" * 64)
    assert received == [b"image-data"]
    error = True
    with pytest.raises(ValueError, match="rejected"):
        await docker.load_runtime_archive(archive, "linux/arm64", "a/b:v1", "sha256:" + "a" * 64)
    architecture = "x86_64"
    with pytest.raises(ValueError, match="architecture"):
        await docker.load_runtime_archive(archive, "linux/arm64", "a/b:v1", "sha256:" + "a" * 64)


@pytest.mark.asyncio
async def test_local_docker_pulls_and_verifies_exact_runtime_digest(tmp_path):
    settings = Settings(docker_runtime_enabled=True, docker_socket_path=tmp_path / "docker.sock")
    settings.docker_socket_path.touch()
    docker = DockerEngineClient(settings)
    exact = "registry.example/runtime@sha256:" + "a" * 64
    config = "sha256:" + "b" * 64
    installed = False
    pulls = []

    async def handle(request):
        nonlocal installed
        if request.url.path == "/images/create":
            pulls.append(request.url.params["fromImage"])
            installed = True
            return httpx.Response(200, text='{"status":"Downloaded"}\n')
        if request.url.path.endswith("/json"):
            if not installed:
                return httpx.Response(404, json={"message": "missing"})
            return httpx.Response(
                200,
                json={"Id": config, "Os": "linux", "Architecture": "amd64"},
            )
        raise AssertionError(request.url)

    docker.transport = httpx.MockTransport(handle)
    value = await docker.ensure_runtime_image(
        "registry.example/runtime", "sha256:" + "a" * 64, "linux/amd64", config
    )
    assert value["Id"] == config
    assert pulls == [exact]
    await docker.ensure_runtime_image(
        "registry.example/runtime", "sha256:" + "a" * 64, "linux/amd64", config
    )
    assert pulls == [exact]


@pytest.mark.asyncio
async def test_local_docker_rejects_pull_errors_wrong_platform_and_wrong_config(tmp_path):
    settings = Settings(docker_runtime_enabled=True, docker_socket_path=tmp_path / "docker.sock")
    settings.docker_socket_path.touch()
    docker = DockerEngineClient(settings)

    async def pull_error(request):
        if request.url.path.endswith("/json"):
            return httpx.Response(404, json={"message": "missing"})
        return httpx.Response(200, text='{"error":"denied"}\n')

    docker.transport = httpx.MockTransport(pull_error)
    with pytest.raises(ValueError, match="failed to pull"):
        await docker.ensure_runtime_image("a/b", "sha256:" + "a" * 64, "linux/amd64", "x")

    async def wrong(request):
        return httpx.Response(
            200,
            json={"Id": "sha256:wrong", "Os": "linux", "Architecture": "arm64"},
        )

    docker.transport = httpx.MockTransport(wrong)
    with pytest.raises(ValueError, match="architecture"):
        await docker.ensure_runtime_image("a/b", "sha256:" + "a" * 64, "linux/amd64", "x")

    async def wrong_config(request):
        return httpx.Response(
            200,
            json={"Id": "sha256:wrong", "Os": "linux", "Architecture": "amd64"},
        )

    docker.transport = httpx.MockTransport(wrong_config)
    with pytest.raises(ValueError, match="configuration"):
        await docker.ensure_runtime_image("a/b", "sha256:" + "a" * 64, "linux/amd64", "x")


def test_runtime_api_authorization_validation_and_controller_routing(tmp_path):
    settings = Settings(
        database_path=tmp_path / "db",
        model_cache_path=tmp_path / "models",
        setup_state_path=tmp_path / "setup",
        model_admin_token="test-admin",
        remote_cluster_url="http://192.168.50.10:30080",
    )
    app = create_app(settings)
    headers = {"Authorization": "Bearer test-admin"}
    # No lifespan needed: validates controller routing without starting remote probes.
    with TestClient(app) as client:
        assert client.get("/api/v1/runtime-library").status_code == 401
        response = client.get("/api/v1/runtime-library", headers=headers)
        assert response.status_code == 200 and len(response.json()["runtimes"]) == 10
        llama = next(item for item in response.json()["runtimes"] if item["id"] == "llama-cpp")
        assert "darwin/arm64" in llama["platforms"]
        assert "windows/amd64" in llama["platforms"]
        page = client.get("/").text
        assert 'id="runtime-host-form"' in page
        assert "Runtime readiness" in page
        assert "Standalone macOS &amp; Windows hosts" in page
        cluster_id = client.get("/api/v1/clusters").json()["clusters"][0]["id"]
        assert (
            client.get(f"/api/v1/runtime-management/engines?cluster_id={cluster_id}").status_code
            == 401
        )
        engines = client.get(
            f"/api/v1/runtime-management/engines?cluster_id={cluster_id}", headers=headers
        )
        assert engines.status_code == 200
        assert engines.json()["cluster_id"] == cluster_id
        assert {item["id"] for item in engines.json()["engines"]} == {
            "vllm",
            "sglang",
            "llama-cpp",
            "tensorfold",
            "deepinfer",
        }
        assert (
            client.post(
                "/api/v1/runtime-management/prepare",
                headers=headers,
                json={"cluster_id": cluster_id, "engine_id": "vllm"},
            ).status_code
            == 409
        )
        assert (
            client.get("/api/v1/runtime-library/jobs/missing", headers=headers).status_code == 404
        )
        assert (
            client.post(
                "/api/v1/runtime-library/downloads",
                headers=headers,
                json={"runtime_id": "vllm", "version": "$(id)", "platform": "linux/arm64"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/runtime-library/installs",
                headers=headers,
                json={"artifact_id": "../bad", "cluster_id": "x"},
            ).status_code
            == 422
        )
        app.state.runtime_library.remote.versions = AsyncMock(return_value={"versions": []})
        assert client.get(
            "/api/v1/runtime-library/runtimes/vllm/versions", headers=headers
        ).json() == {"versions": []}
        app.state.runtime_library.register_host = AsyncMock(
            return_value={"id": "host-" + "a" * 12, "platform": "darwin/arm64"}
        )
        registered = client.post(
            "/api/v1/runtime-library/hosts",
            headers=headers,
            json={
                "name": "Studio Mac",
                "address": "192.168.50.30",
                "username": "thibaut",
                "fingerprint": "SHA256:" + "A" * 43,
            },
        )
        assert registered.status_code == 201
        app.state.runtime_library.remove_host = AsyncMock()
        removed = client.delete("/api/v1/runtime-library/hosts/host-" + "a" * 12, headers=headers)
        assert removed.status_code == 204
