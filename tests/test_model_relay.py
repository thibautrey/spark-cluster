import json
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
    controller.state.model_relay.transport = httpx.ASGITransport(app=bridge)

    result = await controller.state.model_manager.delete_model("acme/model")

    assert result.controller_deleted is True
    assert result.cluster_deleted is True
    assert not (controller.state.settings.model_cache_path / "acme" / "model").exists()
    assert not (bridge.state.settings.model_cache_path / "acme" / "model").exists()
    assert controller.state.runtime.store.get_download(controller_job.id).status == "deleted"
    assert bridge.state.runtime.store.get_download(cluster_job.id).status == "deleted"
