import json
import stat

import httpx
import pytest
from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.cluster_registry import ClusterRegistry
from spark_cluster.config import Settings
from spark_cluster.external_runtime import (
    ExternalRuntimeRequest,
    ExternalRuntimeService,
    ExternalRuntimeTestRequest,
)
from spark_cluster.models import ModelLaunchRequest
from spark_cluster.remote_cluster import RemoteClusterProxyMiddleware
from spark_cluster.runtime_engines import build_execution_plan

MODELS = {
    "object": "list",
    "data": [
        {
            "id": "deepseek-v4.1-flash",
            "object": "model",
            "status": {"value": "loaded"},
            "meta": {"n_ctx": 1_048_576},
        }
    ],
}
CHAT = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "deepseek-v4.1-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "SPARK_OK", "reasoning_content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
}


def settings(tmp_path, **values):
    return Settings(
        mode="demo",
        database_path=tmp_path / "controller.db",
        setup_state_path=tmp_path / "setup",
        model_cache_path=tmp_path / "models",
        model_admin_token="admin",
        openai_api_base="http://127.0.0.1:9/v1",
        **values,
    )


def remote_registry(tmp_path):
    config = settings(tmp_path)
    registry = ClusterRegistry(config)
    cluster = registry.add_remote(
        name="Spark",
        endpoint="http://192.168.1.59:30080",
        head_address="192.168.1.59",
        hardware="dgx-spark",
    )
    return config, registry, cluster


def json_response(payload):
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        stream=httpx.ByteStream(json.dumps(payload).encode()),
    )


def runtime_handler(seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "method": request.method,
                "path": request.url.path,
                "authorization": request.headers.get("authorization"),
                "body": request.content,
            }
        )
        if request.url.path == "/v1/models":
            return json_response(MODELS)
        if request.url.path == "/health":
            return json_response({"ok": True})
        if request.url.path == "/v1/chat/completions":
            return json_response(CHAT)
        return httpx.Response(404)

    return handler


async def test_strata_attachment_keeps_the_key_outside_the_cluster_registry(tmp_path):
    config, registry, cluster = remote_registry(tmp_path)
    seen = []
    service = ExternalRuntimeService(config, registry)
    service.transport = httpx.MockTransport(runtime_handler(seen))

    status = await service.configure(
        cluster.id,
        request=ExternalRuntimeRequest(engine_id="strata", port=8080, api_key="secret-key"),
    )

    assert status.status == "ready"
    assert status.model_ids == ["deepseek-v4.1-flash"]
    assert status.context_length == 1_048_576
    assert status.api_key_configured is True
    assert seen[0]["authorization"] == "Bearer secret-key"
    registry_text = (config.setup_state_path / "clusters.json").read_text()
    assert "secret-key" not in registry_text
    credentials = config.setup_state_path / "runtime-credentials.json"
    assert stat.S_IMODE(credentials.stat().st_mode) == 0o600
    assert json.loads(credentials.read_text())["clusters"][cluster.id] == "secret-key"

    cleared = await service.clear(cluster.id)
    assert cleared.status == "not_configured"
    assert registry.get(cluster.id).runtime_engine is None
    assert not json.loads(credentials.read_text())["clusters"]


async def test_chat_accepts_reasoning_only_and_rejects_missing_runtime(tmp_path):
    config, registry, cluster = remote_registry(tmp_path)
    service = ExternalRuntimeService(config, registry)
    service.transport = httpx.MockTransport(runtime_handler([]))

    with pytest.raises(ValueError, match="No external runtime"):
        await service.chat(
            cluster,
            ExternalRuntimeTestRequest(),
        )

    await service.configure(
        cluster.id,
        request=ExternalRuntimeRequest(engine_id="strata", port=8080, api_key="secret-key"),
    )

    def reasoning_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return json_response(MODELS)
        if request.url.path == "/health":
            return json_response({"ok": True})
        return json_response(
            {
                "model": "deepseek-v4.1-flash",
                "choices": [{"message": {"content": None, "reasoning_content": "SPARK_OK"}}],
            }
        )

    service.transport = httpx.MockTransport(reasoning_handler)
    result = await service.chat(
        cluster,
        ExternalRuntimeTestRequest(),
    )
    assert result.ok is True
    assert result.reasoning_content == "SPARK_OK"


def test_strata_is_visible_but_not_a_managed_container_launch():
    plan = build_execution_plan(
        ModelLaunchRequest(
            model_id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
            served_model_name="strata",
            engine_id="strata",
        ),
        "/models/deepseek",
        Settings(),
    )
    assert plan.executable is False
    assert any("Attach its OpenAI endpoint" in blocker for blocker in plan.blockers)


async def test_fatal_health_overrides_http_200_and_preserves_routing_identity(tmp_path):
    config, registry, cluster = remote_registry(tmp_path)
    service = ExternalRuntimeService(config, registry)
    service.transport = httpx.MockTransport(runtime_handler([]))
    await service.configure(cluster.id, ExternalRuntimeRequest(api_key="secret-key"))

    def failed(request):
        if request.url.path == "/v1/models":
            return json_response(MODELS)
        return json_response({"ok": False, "fatal": "CudaFailed"})

    service.transport = httpx.MockTransport(failed)
    service._probes.clear()
    assert (await service.status(cluster.id)).status == "unreachable"
    assert await service.models(registry.get(cluster.id)) == []
    assert (await service.matching_model("deepseek-v4.1-flash"))[0].id == cluster.id
    with pytest.raises(ValueError, match="unhealthy"):
        await service.chat(registry.get(cluster.id), ExternalRuntimeTestRequest())

    def disconnected(request):
        raise httpx.ConnectError("offline", request=request)

    service.transport = httpx.MockTransport(disconnected)
    service._probes.clear()
    assert (await service.matching_model("deepseek-v4.1-flash"))[0].id == cluster.id
    restarted = ExternalRuntimeService(config, registry)
    restarted.transport = httpx.MockTransport(disconnected)
    assert (await restarted.matching_model("deepseek-v4.1-flash"))[0].id == cluster.id


def test_attached_fatal_runtime_returns_503_without_forwarding_inference(tmp_path):
    app = create_app(settings(tmp_path))
    seen = []
    app.state.external_runtime.transport = httpx.MockTransport(runtime_handler(seen))
    cluster = app.state.cluster_registry.add_remote(
        name="Spark",
        endpoint="http://192.168.1.59:30080",
        head_address="192.168.1.59",
        hardware="dgx-spark",
    )
    with TestClient(app) as client:
        attached = client.put(
            f"/api/v1/clusters/{cluster.id}/runtime",
            headers={"Authorization": "Bearer admin"},
            json={"engine_id": "strata", "api_key": "secret-key"},
        )
        assert attached.status_code == 200
        app.state.external_runtime._probes.clear()

        def failed(request):
            seen.append({"path": request.url.path})
            return json_response(
                MODELS if request.url.path == "/v1/models" else {"ok": False, "fatal": "CudaFailed"}
            )

        app.state.external_runtime.transport = httpx.MockTransport(failed)
        reply = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4.1-flash",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert reply.status_code == 503
        assert reply.headers["retry-after"] == "5"
        assert not any(item["path"] == "/v1/chat/completions" for item in seen)


def test_live_runtime_routes_stay_on_the_controller(tmp_path):
    config = settings(tmp_path)
    middleware = RemoteClusterProxyMiddleware(lambda s, r, x: None, config, None)
    config.remote_cluster_url = "http://192.168.1.59:30080"
    assert middleware.should_forward({"path": "/api/v1/runtime"}) is False
    assert middleware.should_forward({"path": "/api/v1/runtime/test"}) is False
    assert middleware.should_forward({"path": "/api/v1/inference/activity"}) is False
    assert middleware.should_forward({"path": "/api/v1/telemetry"}) is False
    assert middleware.should_forward({"path": "/api/v1/snapshot"}) is False
    assert middleware.should_forward({"path": "/metrics"}) is False


def test_controller_attaches_strata_and_infers_through_it(tmp_path):
    app = create_app(settings(tmp_path))
    seen = []
    app.state.external_runtime.transport = httpx.MockTransport(runtime_handler(seen))
    cluster = app.state.cluster_registry.add_remote(
        name="Spark",
        endpoint="http://192.168.1.59:30080",
        head_address="192.168.1.59",
        hardware="dgx-spark",
    )
    headers = {"Authorization": "Bearer admin"}
    with TestClient(app) as client:
        assert client.get(f"/api/v1/clusters/{cluster.id}/runtime").status_code == 200
        attached = client.put(
            f"/api/v1/clusters/{cluster.id}/runtime",
            headers=headers,
            json={"engine_id": "strata", "port": 8080, "api_key": "secret-key"},
        )
        assert attached.status_code == 200
        assert attached.json()["status"] == "ready"
        assert attached.json()["api_key_configured"] is True

        runtime = client.get("/api/v1/runtime").json()
        assert runtime["base_url"] == "http://192.168.1.59:8080/v1"
        assert runtime["status"] == "ready"
        assert runtime["engine_id"] == "strata"
        assert runtime["auth_mode"] == "bearer"

        models = client.get("/v1/models").json()
        assert models["data"][0]["id"] == "deepseek-v4.1-flash"

        reply = client.post(
            "/v1/chat/completions",
            json={
                "model": "deepseek-v4.1-flash",
                "messages": [{"role": "user", "content": "Reply with SPARK_OK"}],
                "max_tokens": 32,
            },
        )
        assert reply.status_code == 200
        assert reply.json()["choices"][0]["message"]["content"] == "SPARK_OK"
        assert seen[-1]["authorization"] == "Bearer secret-key"

        activity = client.get("/api/v1/inference/activity").json()
        assert activity["totals"]["requests"] == 1
        assert activity["recent"][-1]["model_id"] == "deepseek-v4.1-flash"
        assert activity["recent"][-1]["cluster_id"] == cluster.id
        assert activity["recent"][-1]["input_tokens"] == 8
        assert activity["recent"][-1]["output_tokens"] == 1

        tested = client.post(
            "/api/v1/runtime/test",
            headers=headers,
            json={"prompt": "Reply with SPARK_OK", "max_tokens": 32},
        )
        assert tested.status_code == 200
        assert tested.json()["ok"] is True
        assert tested.json()["content"] == "SPARK_OK"
