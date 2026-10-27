import asyncio
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from test_model_manager import metadata

from spark_cluster import hf_credentials
from spark_cluster.app import create_app
from spark_cluster.config import Settings
from spark_cluster.model_manager import ModelManager, build_fit_plan
from spark_cluster.models import DownloadJob, ModelFitRequest
from spark_cluster.simulator import ClusterSimulator
from spark_cluster.store import SnapshotStore


def config(tmp_path, **values):
    return Settings(
        mode="live",
        database_path=tmp_path / "test.db",
        model_cache_path=tmp_path / "models",
        setup_state_path=tmp_path / "state",
        model_admin_token="admin",
        **values,
    )


def test_token_api_auth_persistence_and_no_secret_response(tmp_path, monkeypatch):
    async def validate(token):
        assert token == "hf_test_valid"

    monkeypatch.setattr(hf_credentials, "validate_token", validate)
    settings = config(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        page = client.get("/").text
        assert 'id="root"' in page
        assert '/assets/index-' in page
        path = "/api/v1/controller/huggingface"
        assert client.get(path).json() == {"configured": False, "requires_admin": True}
        assert client.put(path, json={"token": "hf_test_valid"}).status_code == 401
        response = client.put(
            path, json={"token": "hf_test_valid"}, headers={"Authorization": "Bearer admin"}
        )
        assert response.json() == {"configured": True, "restarted_downloads": []}
        assert "hf_test_valid" not in response.text
        assert "hf_test_valid" not in client.get(path).text
        assert settings.hf_token == app.state.model_manager.catalog.token == "hf_test_valid"
        assert app.state.model_browser.token == "hf_test_valid"
    stored = hf_credentials.token_path(settings)
    assert stored.stat().st_mode & 0o777 == 0o600
    restored = config(tmp_path)
    hf_credentials.load_token(restored)
    assert restored.hf_token == "hf_test_valid"


def test_rejected_token_does_not_replace_existing_credentials(tmp_path, monkeypatch):
    async def reject(token):
        raise ValueError("Hugging Face rejected this token.")

    monkeypatch.setattr(hf_credentials, "validate_token", reject)
    settings = config(tmp_path, hf_token="hf_old")
    with TestClient(create_app(settings)) as client:
        response = client.put(
            "/api/v1/controller/huggingface",
            json={"token": "hf_invalid"},
            headers={"Authorization": "Bearer admin"},
        )
        assert response.status_code == 400
        assert "hf_invalid" not in response.text
        assert settings.hf_token == "hf_old"
        assert not hf_credentials.token_path(settings).exists()


@pytest.mark.asyncio
async def test_active_worker_restarts_with_new_token_and_keeps_partial_files(tmp_path, monkeypatch):
    settings = config(tmp_path)
    store = SnapshotStore(settings.database_path)
    runtime = SimpleNamespace(latest=ClusterSimulator("test").snapshot())
    manager = ModelManager(settings, store, runtime)
    plan = build_fit_plan(
        ModelFitRequest(repo_id="acme/model"), metadata(), runtime.latest, settings
    )
    staging = settings.model_cache_path / ".partial" / "active"
    staging.mkdir()
    partial = staging / "weights.partial"
    partial.write_bytes(b"preserved")
    store.save_download(
        DownloadJob(
            id="active",
            repo_id="acme/model",
            revision="main",
            target_path=str(settings.model_cache_path / "acme/model"),
            staging_path=str(staging),
            fit=plan,
        )
    )
    paused = DownloadJob(
        id="paused",
        repo_id="acme/paused",
        revision="main",
        status="paused",
        target_path=str(settings.model_cache_path / "acme/paused"),
        staging_path=str(settings.model_cache_path / ".partial/paused"),
        fit=plan,
    )
    store.save_download(paused)
    environments = []
    original_spawn = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        environments.append(kwargs["env"].get("SPARK_CLUSTER_HF_TOKEN"))
        return await original_spawn(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await manager.start()
    try:
        async with asyncio.timeout(5):
            while not environments:
                await asyncio.sleep(0.02)
        old = manager._active_processes["active"]
        restarted = await manager.refresh_huggingface_token("hf_new")
        assert restarted == ["active"]
        async with asyncio.timeout(5):
            while len(environments) < 2:
                await asyncio.sleep(0.02)
        assert environments == [None, "hf_new"]
        assert old.returncode is not None
        assert partial.read_bytes() == b"preserved"
        assert store.get_download("paused").status == "paused"
        assert store.get_download("active").attempts == 2
    finally:
        await manager.stop()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,exception", [(200, None), (401, ValueError), (429, RuntimeError)])
async def test_validation_checks_huggingface_before_saving(monkeypatch, status, exception):
    real_client = httpx.AsyncClient

    def handle(request):
        assert str(request.url) == "https://huggingface.co/api/whoami-v2"
        assert request.headers["authorization"] == "Bearer hf_test"
        return httpx.Response(status)

    monkeypatch.setattr(
        hf_credentials.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    )
    if exception:
        with pytest.raises(exception):
            await hf_credentials.validate_token("hf_test")
    else:
        await hf_credentials.validate_token("hf_test")
