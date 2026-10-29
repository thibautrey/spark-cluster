import json
import os
from hashlib import sha256
from unittest.mock import AsyncMock

import httpx
import pytest

from spark_cluster.app import create_app
from spark_cluster.config import Settings
from spark_cluster.models import DownloadJob, ModelFitPlan
from spark_cluster.remote_cluster import signed_controller_headers
from spark_cluster.simulator import ClusterSimulator


def fit(repo_id="acme/model"):
    return ModelFitPlan(
        repo_id=repo_id,
        revision="main",
        architecture="TestModel",
        parameter_count=1_000_000,
        repository_bytes=12,
        weight_bytes=12,
        weights_gib=0.01,
        kv_cache_gib=0.01,
        runtime_overhead_gib=1,
        total_required_gib=1.02,
        cluster_nodes=2,
        usable_per_node_gib=100,
        usable_cluster_gib=200,
        minimum_nodes=1,
        headroom_gib=198.98,
        memory_fits=True,
        storage_available_gib=100,
        storage_fits=True,
        fits=True,
        confidence="architecture",
    )


def settings(tmp_path, name, **kwargs):
    return Settings(
        mode="live",
        database_path=tmp_path / f"{name}.db",
        model_cache_path=tmp_path / name / "models",
        setup_state_path=tmp_path / name / "state",
        model_admin_token="shared-secret",
        openai_api_base="http://127.0.0.1:9/v1",
        **kwargs,
    )


def staged_job(app, payload=b"model-bytes"):
    target = app.state.settings.model_cache_path / "acme" / "model"
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(payload)
    (target / ".spark-cluster.json").write_text(
        json.dumps(
            {
                "repo_id": "acme/model",
                "revision": "main",
                "downloaded_at": "2026-10-23T12:00:00+00:00",
                "fit": fit().model_dump(mode="json"),
            }
        )
    )
    job = DownloadJob(
        id="a" * 32,
        repo_id="acme/model",
        revision="main",
        status="preparing-transfer",
        total_bytes=len(payload),
        downloaded_bytes=len(payload),
        progress=100,
        target_path=str(target),
        staging_path=str(app.state.settings.model_cache_path / ".partial" / ("a" * 32)),
        destination="cluster",
        controller_url="http://192.168.50.5:8080",
        fit=fit(),
    )
    app.state.runtime.store.save_download(job)
    return job


@pytest.mark.asyncio
async def test_relay_metadata_uses_private_state_and_reads_legacy_manifests(tmp_path, monkeypatch):
    from pathlib import Path

    app = create_app(settings(tmp_path, "controller"))
    job = staged_job(app)
    model_root = app.state.settings.model_cache_path
    original = Path.chmod

    def nas_chmod(path, *args, **kwargs):
        if model_root in path.parents:
            raise PermissionError("NAS does not support chmod")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", nas_chmod)
    relay = app.state.model_relay
    manifest = await relay.prepare_manifest(job)
    path = relay._manifest_path(job.id)
    assert app.state.settings.setup_state_path in path.parents
    assert path.stat().st_mode & 0o777 == 0o600
    assert relay.manifest(job.id) == manifest
    legacy = model_root / ".relay" / path.name
    legacy.parent.mkdir()
    path.replace(legacy)
    assert relay.manifest(job.id) == manifest
    app.state.runtime.store.close()


@pytest.mark.asyncio
async def test_selective_gguf_transfer_never_hashes_or_serves_other_variants(tmp_path, monkeypatch):
    app = create_app(settings(tmp_path, "controller"))
    job = staged_job(app)
    root = app.state.settings.model_cache_path / "acme" / "model"
    (root / "model-Q2.gguf").write_bytes(b"small")
    (root / "model-Q8.gguf").write_bytes(b"larger-checkpoint")
    (root / "model-Engram.gguf").write_bytes(b"shared")
    job.smallest_gguf_transfer_only = True
    app.state.runtime.store.save_download(job)
    from spark_cluster import model_relay

    original = model_relay._sha256

    def selected_hash(path):
        assert path.name != "model-Q8.gguf"
        return original(path)

    monkeypatch.setattr(model_relay, "_sha256", selected_hash)
    manifest = await app.state.model_relay.prepare_manifest(job)
    names = {item["path"] for item in manifest["files"]}
    assert {"model-Q2.gguf", "model-Engram.gguf"} <= names
    assert "model-Q8.gguf" not in names
    with pytest.raises(FileNotFoundError):
        app.state.model_relay.model_file(job.id, "model-Q8.gguf")
    app.state.runtime.store.close()


@pytest.mark.asyncio
async def test_recorded_hub_hash_skips_source_reread_but_modified_files_are_rehashed(
    tmp_path, monkeypatch
):
    from spark_cluster import model_relay

    app = create_app(settings(tmp_path, "controller"))
    job = staged_job(app)
    job.fit.checkpoint_commit = "c" * 40
    root = app.state.settings.model_cache_path / "acme" / "model"
    weights = root / "weights.bin"
    digest = sha256(weights.read_bytes()).hexdigest()
    verified_at = weights.stat().st_mtime + 1
    sidecar = root / ".cache" / "huggingface" / "download" / "weights.bin.metadata"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(f"{'c' * 40}\n{digest}\n{verified_at}\n")
    original = model_relay._sha256
    reread = []

    def tracked_hash(path):
        if path == weights:
            reread.append(path)
        return original(path)

    monkeypatch.setattr(model_relay, "_sha256", tracked_hash)
    manifest = await app.state.model_relay.prepare_manifest(job)
    assert not reread
    assert (
        next(item for item in manifest["files"] if item["path"] == "weights.bin")["sha256"]
        == digest
    )
    job.fit.checkpoint_commit = "d" * 40
    await app.state.model_relay.prepare_manifest(job)
    assert reread == [weights]
    job.fit.checkpoint_commit = "c" * 40
    reread.clear()
    weights.write_bytes(b"modified")
    os.utime(weights, (verified_at + 2, verified_at + 2))
    manifest = await app.state.model_relay.prepare_manifest(job)
    assert reread == [weights]
    assert (
        next(item for item in manifest["files"] if item["path"] == "weights.bin")["sha256"]
        == sha256(b"modified").hexdigest()
    )
    app.state.runtime.store.close()


@pytest.mark.asyncio
async def test_controller_transfer_endpoints_require_signatures_and_support_ranges(tmp_path):
    app = create_app(settings(tmp_path, "controller"))
    job = staged_job(app)
    manifest = await app.state.model_relay.prepare_manifest(job)
    assert {item["path"] for item in manifest["files"]} == {
        ".spark-cluster.json",
        "weights.bin",
    }
    assert (
        next(item for item in manifest["files"] if item["path"] == "weights.bin")["sha256"]
        == sha256(b"model-bytes").hexdigest()
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://192.168.50.5:8080"
    ) as client:
        manifest_path = f"/api/v1/controller/downloads/{job.id}/manifest"
        assert (await client.get(manifest_path)).status_code == 401
        response = await client.get(
            manifest_path,
            headers=signed_controller_headers("shared-secret", "GET", manifest_path),
        )
        assert response.status_code == 200

        file_path = f"/api/v1/controller/downloads/{job.id}/files/weights.bin"
        response = await client.get(
            file_path,
            headers={
                **signed_controller_headers("shared-secret", "GET", file_path),
                "range": "bytes=6-",
            },
        )
        assert response.status_code == 206
        assert response.content == b"bytes"


@pytest.mark.asyncio
async def test_resumed_copy_verifies_prefix_and_discards_streaming_page_cache(tmp_path, monkeypatch):
    from pathlib import Path

    from spark_cluster import model_relay

    controller = create_app(settings(tmp_path, "controller"))
    bridge = create_app(settings(tmp_path, "bridge"))
    job = staged_job(controller, b"0123456789abcdef")
    await controller.state.model_relay.prepare_manifest(job)
    remote = job.model_copy(
        update={
            "target_path": str(bridge.state.settings.model_cache_path / "acme" / "model"),
            "staging_path": str(bridge.state.settings.model_cache_path / ".partial" / job.id),
            "controller_job_id": job.id,
        }
    )
    staging = Path(remote.staging_path)
    staging.mkdir(parents=True)
    (staging / "weights.bin").write_bytes(b"01234567")
    discarded = []
    monkeypatch.setattr(model_relay, "CHUNK_SIZE", 4)
    monkeypatch.setattr(model_relay, "CACHE_FLUSH_BYTES", 4)
    monkeypatch.setattr(model_relay.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(
        model_relay.os, "posix_fadvise",
        lambda fd, start, size, advice: discarded.append(size), raising=False,
    )
    bridge.state.model_relay.transport = httpx.ASGITransport(app=controller)
    await bridge.state.model_relay.pull(remote)
    assert (staging / "weights.bin").read_bytes() == b"0123456789abcdef"
    assert remote.progress == 100
    assert 4 in discarded and 8 in discarded and 16 in discarded
    controller.state.runtime.store.close()
    bridge.state.runtime.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", [200, 401])
async def test_relay_status_recovers_transport_outages_but_not_auth_errors(
    tmp_path, monkeypatch, terminal_status
):
    from spark_cluster import model_relay

    app = create_app(settings(tmp_path, "controller"))
    job = staged_job(app)
    job.target_cluster_url = "http://192.168.50.10:30080"
    polls = 0

    def handle(request):
        nonlocal polls
        if request.method == "POST":
            return httpx.Response(200, json={})
        polls += 1
        if polls == 1:
            raise httpx.ReadError("Bridge restarting")
        if polls == 2:
            return httpx.Response(503)
        return httpx.Response(
            terminal_status, json=job.model_copy(update={"status": "completed"}).model_dump(mode="json")
        )

    app.state.model_relay.transport = httpx.MockTransport(handle)
    sleep = AsyncMock()
    monkeypatch.setattr(model_relay.asyncio, "sleep", sleep)
    if terminal_status == 200:
        assert await app.state.model_relay.distribute(job)
    else:
        with pytest.raises(httpx.HTTPStatusError):
            await app.state.model_relay.distribute(job)
    assert polls == 3
    assert sleep.await_count == 2
    app.state.runtime.store.close()


@pytest.mark.asyncio
async def test_controller_download_is_delivered_once_to_shared_cluster_storage(tmp_path):
    controller = create_app(
        settings(
            tmp_path,
            "controller",
            remote_cluster_url="http://192.168.50.10:30080",
            controller_public_url="http://192.168.50.5:8080",
        )
    )
    bridge = create_app(settings(tmp_path, "bridge"))
    job = staged_job(controller, b"one-controller-copy")

    controller.state.model_relay.transport = httpx.ASGITransport(app=bridge)
    bridge.state.model_relay.transport = httpx.ASGITransport(app=controller)
    await bridge.state.model_manager.start()
    try:
        await controller.state.model_relay.distribute(job)
        copied = bridge.state.settings.model_cache_path / "acme" / "model" / "weights.bin"
        assert copied.read_bytes() == b"one-controller-copy"
        remote_job = bridge.state.runtime.store.get_download(job.id)
        assert remote_job.status == "completed"
        assert remote_job.source == "controller"
        assert remote_job.message == "Copied from the controller into shared cluster storage"
        local_job = controller.state.runtime.store.get_download(job.id)
        assert local_job.transfer_progress == 100
        assert local_job.transferred_bytes > 0
    finally:
        await bridge.state.model_manager.stop()


@pytest.mark.asyncio
async def test_attached_cluster_download_is_queued_on_the_controller(tmp_path):
    app = create_app(
        settings(
            tmp_path,
            "controller",
            remote_cluster_url="http://192.168.50.10:30080",
            controller_public_url="http://192.168.50.5:8080",
        )
    )
    app.state.cluster_connector.snapshot = AsyncMock(
        return_value=ClusterSimulator("relay-test").snapshot()
    )
    app.state.cluster_connector.model_storage_available = AsyncMock(return_value=10 * 2**30)
    app.state.model_manager.fit = AsyncMock(return_value=fit())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://192.168.50.5:8080"
    ) as client:
        response = await client.post(
            "/api/v1/models/download",
            headers={"Authorization": "Bearer shared-secret"},
            json={"repo_id": "acme/model"},
        )
    assert response.status_code == 202
    assert response.json()["destination"] == "cluster"
    assert response.json()["controller_url"] == "http://192.168.50.5:8080"
    assert app.state.runtime.latest.cluster_name == "relay-test"

    app.state.cluster_connector.model_storage_available = AsyncMock(return_value=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://192.168.50.5:8080"
    ) as client:
        response = await client.post("/api/v1/models/fit", json={"repo_id": "acme/model"})
    assert response.status_code == 200
    assert response.json()["storage_fits"] is False
    assert response.json()["fits"] is False


@pytest.mark.asyncio
async def test_model_deletion_propagates_to_controller_and_cluster_storage(tmp_path):
    controller = create_app(
        settings(
            tmp_path,
            "controller-delete",
            remote_cluster_url="http://192.168.50.10:30080",
        )
    )
    bridge = create_app(settings(tmp_path, "bridge-delete"))
    controller_job = staged_job(controller, b"controller-copy")
    cluster_job = staged_job(bridge, b"cluster-copy")
    controller_job.status = "completed"
    cluster_job.status = "completed"
    controller.state.runtime.store.save_download(controller_job)
    bridge.state.runtime.store.save_download(cluster_job)
    controller.state.model_relay.transport = httpx.ASGITransport(app=bridge)

    result = await controller.state.model_manager.delete_model("acme/model")

    assert result.controller_deleted is True
    assert result.cluster_deleted is True
    assert not (controller.state.settings.model_cache_path / "acme" / "model").exists()
    assert not (bridge.state.settings.model_cache_path / "acme" / "model").exists()
    assert controller.state.runtime.store.get_download(controller_job.id).status == "deleted"
    assert bridge.state.runtime.store.get_download(cluster_job.id).status == "deleted"
