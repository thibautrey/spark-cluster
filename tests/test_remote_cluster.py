import json
import stat
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.config import Settings
from spark_cluster.remote_cluster import (
    RemoteClusterConnector,
    RemoteClusterProxyMiddleware,
    connection_path,
    controller_signature,
    load_active_connection,
    save_active_connection,
    signed_controller_headers,
    valid_controller_signature,
    validated_cluster_url,
)
from spark_cluster.setup_models import SetupJob, SetupRequest, SetupStep


def controller_settings(tmp_path, **values):
    return Settings(
        mode="live",
        database_path=tmp_path / "controller.db",
        model_cache_path=tmp_path / "models",
        setup_state_path=tmp_path / "setup",
        model_admin_token="controller-secret",
        openai_api_base="http://127.0.0.1:9/v1",
        **values,
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:30080",
        "http://127.0.0.1:30080",
        "http://169.254.169.254:30080",
        "file:///tmp/cluster",
        "http://user@192.168.1.2:30080",
        "http://192.168.1.2:30080/path",
    ],
)
def test_remote_cluster_url_is_lan_only(url):
    with pytest.raises(ValueError):
        validated_cluster_url(url)
    assert validated_cluster_url("http://192.168.1.2:30080") == "http://192.168.1.2:30080"


def test_active_connection_is_private_persistent_and_contains_no_token(tmp_path):
    settings = controller_settings(tmp_path)
    saved = save_active_connection(
        settings,
        {
            "url": "http://192.168.50.10:30080",
            "cluster_name": "spark-lab",
            "head_address": "192.168.50.10",
        },
    )
    path = connection_path(settings)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert "controller-secret" not in path.read_text()
    assert load_active_connection(settings) == saved


def test_controller_signature_covers_target_body_and_expiry():
    timestamp = str(int(time.time()))
    body = b'{"model":"one"}'
    signature = controller_signature("secret", "POST", "/api/v1/launches?x=1", timestamp, body)
    assert valid_controller_signature(
        "secret", "POST", "/api/v1/launches?x=1", timestamp, signature, body
    )
    assert not valid_controller_signature(
        "secret", "POST", "/api/v1/launches?x=2", timestamp, signature, body
    )
    assert not valid_controller_signature(
        "secret", "POST", "/api/v1/launches?x=1", str(int(timestamp) - 60), signature, body
    )
    headers = signed_controller_headers("secret", "GET", "/api/v1/controller/verify")
    assert "authorization" not in headers


def test_external_controller_attaches_and_forwards_http_without_bearer_token(tmp_path, monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        body = request.read()
        calls.append((request.url.path, request.headers, body))
        assert "authorization" not in request.headers
        target = request.url.raw_path.decode()
        assert valid_controller_signature(
            "controller-secret",
            request.method,
            target,
            request.headers.get("x-spark-timestamp"),
            request.headers.get("x-spark-signature"),
            body,
        )
        if request.url.path == "/api/v1/controller/verify":
            return httpx.Response(200, json={"status": "ready", "cluster_name": "remote-lab"})
        if request.url.path == "/api/v1/launches":
            return httpx.Response(202, json={"id": "remote-job", "status": "queued"})
        return httpx.Response(404, json={"detail": "unknown"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(RemoteClusterConnector, "transport", transport)
    monkeypatch.setattr(RemoteClusterProxyMiddleware, "transport", transport)
    app = create_app(controller_settings(tmp_path))
    headers = {"Authorization": "Bearer controller-secret"}
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/setup/connect",
            headers=headers,
            json={"address": "192.168.50.10", "port": 30080},
        )
        assert response.status_code == 200
        assert response.json()["cluster_name"] == "remote-lab"
        state = client.get("/api/v1/setup", headers=headers).json()
        assert state["connected_cluster"]["url"] == "http://192.168.50.10:30080"
        forwarded = client.post(
            "/api/v1/launches",
            headers={"Authorization": "Bearer browser-value-must-not-be-forwarded"},
            json={"model_id": "acme/model"},
        )
        assert forwarded.status_code == 202
        assert forwarded.json()["id"] == "remote-job"
        inference = client.post("/v1/chat/completions", json={"model": "acme/model"})
        assert inference.status_code == 404
        assert "not available in the pool" in inference.json()["detail"]
        assert not any(path.startswith("/v1/") for path, _, _ in calls)
    assert [call[0] for call in calls] == [
        "/api/v1/controller/verify",
        "/api/v1/launches",
    ]
    assert json.loads(calls[1][2]) == {"model_id": "acme/model"}


def test_remote_bridge_failure_is_reported_as_bad_gateway(tmp_path, monkeypatch):
    def handler(request: httpx.Request):
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(RemoteClusterProxyMiddleware, "transport", httpx.MockTransport(handler))
    app = create_app(controller_settings(tmp_path, remote_cluster_url="http://192.168.50.10:30080"))
    with TestClient(app) as client:
        response = client.get("/api/v1/snapshot")
    assert response.status_code == 502
    assert response.json()["detail"].startswith("Cluster bridge is unreachable")


def test_saved_connection_is_loaded_at_controller_restart(tmp_path):
    first = controller_settings(tmp_path)
    save_active_connection(
        first,
        {
            "url": "http://192.168.50.10:30080",
            "cluster_name": "persistent-lab",
            "head_address": "192.168.50.10",
        },
    )
    restarted = controller_settings(tmp_path)
    app = create_app(restarted)
    assert app.state.settings.remote_cluster_url == "http://192.168.50.10:30080"


def test_bridge_verification_requires_admin_authentication(tmp_path):
    app = create_app(controller_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/v1/controller/verify").status_code == 401
        response = client.get(
            "/api/v1/controller/verify",
            headers={"Authorization": "Bearer controller-secret"},
        )
        signed = client.get(
            "/api/v1/controller/verify",
            headers=signed_controller_headers(
                "controller-secret", "GET", "/api/v1/controller/verify"
            ),
        )
    assert response.json() == {"status": "ready", "cluster_name": "gb10-lab"}
    assert signed.status_code == 200


@pytest.mark.asyncio
async def test_successful_live_setup_automatically_attaches_controller(tmp_path, monkeypatch):
    app = create_app(controller_settings(tmp_path))
    attached = []

    async def attach(address, port):
        attached.append((address, port))
        return {"cluster_name": "created-lab"}

    monkeypatch.setattr(app.state.cluster_connector, "attach", attach)
    request = SetupRequest(
        name="created-lab",
        topology="single",
        nodes=[
            {
                "address": "192.168.50.10",
                "username": "nvidia",
                "fingerprint": "SHA256:" + "A" * 43,
                "role": "head",
            }
        ],
    )
    job = SetupJob(request=request, mode="ssh", steps=[SetupStep(id="verify", label="Verify")])
    await app.state.setup_service.on_cluster_ready(job)
    assert attached == [("192.168.50.10", 30080)]
