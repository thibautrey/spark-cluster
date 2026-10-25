import asyncio
import json
import time

from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.config import Settings
from spark_cluster.store import SnapshotStore


def make_client(tmp_path):
    app = create_app(
        Settings(
            mode="demo",
            database_path=tmp_path / "test.db",
            sample_interval_seconds=0.05,
            cluster_name="test-cluster",
            model_cache_path=tmp_path / "models",
            openai_api_base="http://127.0.0.1:9/v1",
        )
    )
    return TestClient(app)


def test_dashboard_and_live_snapshot(tmp_path):
    with make_client(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert '<details class="launch-options" id="launch-options">' in page.text
        assert '<div class="log-terminal empty" id="log-terminal">' in page.text
        assert "Newest entries follow automatically" not in page.text
        response = client.get("/api/v1/snapshot")
        assert response.status_code == 200
        payload = response.json()
        assert payload["cluster_name"] == "test-cluster"
        assert len(payload["nodes"]) == 4
        assert payload["runtime"]["context_length"] == 524288
        control = client.get("/api/v1/control")
        assert control.status_code == 200
        assert set(control.json()) == {
            "runtime",
            "models",
            "downloads",
            "launches",
            "logs_mode",
            "runtime_control_mode",
        }
        dashboard_js = client.get("/assets/app.js").text
        assert "api('/api/v1/runtime')" in dashboard_js
        assert "api('/api/v1/models')" in dashboard_js
        assert "api('/api/v1/launches')" in dashboard_js
        assert "if (modelInventoryIsNeeded()) void refreshModelInventory();" in dashboard_js
        assert "await api('/api/v1/control')" not in dashboard_js
        assert "function renderPoolEndpoint(pool)" in dashboard_js
        assert "api('/api/v1/pool').then(renderPoolEndpoint)" in dashboard_js
        assert "Direct to cluster runtime" not in page.text
        assert "OpenAI pool endpoint" in page.text
        assert "`${window.location.origin}/v1`" in dashboard_js


def test_history_metrics_and_capacity(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/api/v1/history?minutes=5").status_code == 200
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "spark_cluster_cache_hit_ratio" in metrics.text
        result = client.post(
            "/api/v1/capacity",
            json={"parameter_count_billions": 70, "context_length": 131072},
        )
        assert result.status_code == 200
        assert result.json()["minimum_nodes"] >= 1

        endpoint = client.get("/api/v1/runtime")
        assert endpoint.status_code == 200
        assert endpoint.json()["base_url"] == "http://127.0.0.1:9/v1"
        assert endpoint.json()["chat_completions_url"].endswith("/chat/completions")

        performance = client.post(
            "/api/v1/performance/plan",
            json={
                "model_id": "zai-org/GLM-5.3-Flash",
                "node_count": 4,
                "topology": "switch",
                "workload": "throughput",
            },
        )
        assert performance.status_code == 200
        assert performance.json()["kv_cache_dtype"] == "fp8_e4m3"
        assert performance.json()["tensor_parallel"] == 4


def test_local_gpu_performance_plan_skips_spark_setup_history(tmp_path, monkeypatch):
    app = create_app(
        Settings(
            mode="demo",
            database_path=tmp_path / "local-plan.db",
            model_cache_path=tmp_path / "models",
            active_hardware="nvidia-gpu",
            active_compute_capability=8.6,
        )
    )

    def fail_setup_history(*_args, **_kwargs):
        raise AssertionError("Local PCIe planning must not read Spark setup history")

    with TestClient(app) as client:
        monkeypatch.setattr(SnapshotStore, "list_setup_jobs", fail_setup_history)
        response = client.post(
            "/api/v1/performance/plan",
            json={"model_id": "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ"},
        )

    assert response.status_code == 200
    assert response.json()["collective_backend"] == "NCCL over local PCIe"


def test_global_v1_routes_model_requests_through_the_pool(tmp_path):
    with make_client(tmp_path) as client:
        root = client.get("/v1")
        assert root.status_code == 200
        assert root.json()["models"] == "/v1/models"
        response = client.post(
            "/v1/chat/completions",
            json={"model": "anything", "messages": []},
        )
        assert response.status_code == 404
        assert "not available in the pool" in response.json()["detail"]


def test_ingest_token_is_enforced(tmp_path):
    app = create_app(
        Settings(
            mode="live",
            database_path=tmp_path / "secure.db",
            model_cache_path=tmp_path / "models",
            ingest_token="secret",
        )
    )
    with TestClient(app) as client:
        response = client.post("/api/v1/nodes", json={})
        assert response.status_code == 401


def test_first_agent_seeds_live_cluster(tmp_path):
    app = create_app(
        Settings(
            mode="live",
            database_path=tmp_path / "live.db",
            model_cache_path=tmp_path / "models",
        )
    )
    node = {
        "id": "spark-1",
        "hostname": "spark-1",
        "role": "head",
        "address": "10.0.0.1",
        "memory": {"total_gib": 128, "used_gib": 64},
        "gpu_utilization": 50,
        "temperature_c": 60,
    }
    with TestClient(app) as client:
        response = client.post("/api/v1/nodes", json=node)
        assert response.status_code == 200
        assert response.json()["nodes"][0]["id"] == "spark-1"


def test_model_mutations_can_require_separate_admin_token(tmp_path):
    app = create_app(
        Settings(
            mode="live",
            database_path=tmp_path / "models.db",
            model_cache_path=tmp_path / "models",
            model_admin_token="model-secret",
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/models/download",
            json={"repo_id": "acme/model"},
        )
        assert response.status_code == 401


def test_model_launch_and_runtime_logs_end_to_end(tmp_path):
    model_path = tmp_path / "models" / "acme" / "tiny-model"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 32,
            }
        )
    )
    fit = {
        "repo_id": "acme/tiny-model",
        "revision": "main",
        "architecture": "GPT2LMHeadModel",
        "parameter_count": 1000000,
        "repository_bytes": 1000000,
        "weight_bytes": 500000,
        "weights_gib": 0.01,
        "kv_cache_gib": 0.01,
        "runtime_overhead_gib": 8,
        "total_required_gib": 8.02,
        "cluster_nodes": 4,
        "usable_per_node_gib": 104.96,
        "usable_cluster_gib": 419.84,
        "minimum_nodes": 1,
        "headroom_gib": 411.82,
        "memory_fits": True,
        "storage_available_gib": 100,
        "storage_fits": True,
        "fits": True,
        "confidence": "architecture",
        "notes": [],
        "bits_per_weight": 4,
        "context_length": 131072,
        "concurrent_sequences": 1,
    }
    (model_path / ".spark-cluster.json").write_text(
        json.dumps(
            {
                "repo_id": "acme/tiny-model",
                "revision": "main",
                "downloaded_at": "2026-10-23T10:00:00+00:00",
                "fit": fit,
            }
        )
    )
    app = create_app(
        Settings(
            mode="demo",
            database_path=tmp_path / "launch.db",
            model_cache_path=tmp_path / "models",
            openai_api_base="http://127.0.0.1:9/v1",
            sample_interval_seconds=0.05,
        )
    )
    with TestClient(app) as client:
        invalid = client.post(
            "/api/v1/launches",
            json={
                "model_id": "acme/tiny-model",
                "served_model_name": "too-wide",
                "context_length": 131072,
                "tensor_parallel": 5,
                "pipeline_parallel": 1,
            },
        )
        assert invalid.status_code == 409
        assert "cluster currently exposes 4" in invalid.json()["detail"]
        response = client.post(
            "/api/v1/launches",
            json={
                "model_id": "acme/tiny-model",
                "served_model_name": "tiny-model",
                "context_length": 131072,
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            launches = client.get("/api/v1/launches").json()
            job = next(item for item in launches if item["id"] == job_id)
            if job["status"] == "ready":
                break
            time.sleep(0.1)
        assert job["status"] == "ready"
        assert job["progress"] == 100
        logs = client.get("/api/v1/logs?pod=control-plane").json()
        assert any("Model is ready to serve" in entry["message"] for entry in logs)


def test_pool_mutations_require_admin_and_reads_expose_no_recipes(tmp_path):
    app = create_app(
        Settings(
            mode="demo",
            model_admin_token="test-admin",
            setup_state_path=tmp_path / "setup",
            database_path=tmp_path / "db",
            model_cache_path=tmp_path / "models",
        )
    )
    with TestClient(app) as client:
        state = client.get("/api/v1/pool")
        assert state.status_code == 200
        assert "container" not in state.text
        assert "test-admin" not in state.text
        assert client.get("/api/v1/pool/node/status").status_code == 401
        assert client.post("/api/v1/pool/load", json={"model_id": "org/model"}).status_code == 401
        result = client.put(
            "/api/v1/pool/models/org/model/settings",
            json={"keep_loaded": True, "cluster_id": None},
            headers={"Authorization": "Bearer test-admin"},
        )
        assert result.status_code == 200
        assert result.json()["keep_loaded"] is True
        assert "org/model" in client.get("/api/v1/inference/activity").json()["retained_models"]
        assert (
            client.patch(
                "/api/v1/pool/models/org/model/retention", json={"keep_loaded": False}
            ).status_code
            == 401
        )
        retention = client.patch(
            "/api/v1/pool/models/org/model/retention",
            json={"keep_loaded": False},
            headers={"Authorization": "Bearer test-admin"},
        )
        assert retention.status_code == 200
        assert retention.json()["keep_loaded"] is False
        assert "org/model" not in client.get("/api/v1/inference/activity").json()["retained_models"]
        invalid = client.put(
            "/api/v1/pool/models/org/model/settings",
            json={"last_settings": {"model_id": "org/other", "served_model_name": "wrong"}},
            headers={"Authorization": "Bearer test-admin"},
        )
        assert invalid.status_code == 409


def test_global_openai_models_and_json_inference_route_through_pool(tmp_path, monkeypatch):
    app = create_app(
        Settings(
            mode="demo",
            setup_state_path=tmp_path / "setup",
            database_path=tmp_path / "db",
            model_cache_path=tmp_path / "models",
        )
    )
    model = {
        "id": "org/model",
        "object": "model",
        "loadable": True,
        "placements": [],
        "aliases": ["org/model", "friendly"],
        "served_model_name": "runtime-alias",
        "runtime_models": {
            "object": "list",
            "data": [
                {
                    "id": "runtime-alias",
                    "object": "model",
                    "created": 123,
                    "owned_by": "runtime-vendor",
                    "context_length": 32768,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"input_modalities": ["text"]},
                    "supported_parameters": ["tools", "structured_outputs"],
                    "model_info": {"supports_function_calling": True},
                }
            ],
        },
    }

    async def inventory(force=False):
        return {"clusters": [], "models": [model], "jobs": []}

    async def resolve(reference):
        if reference not in model["aliases"]:
            raise ValueError("not found")
        return model

    captured = {}

    async def infer(model_id, path, scope, receive, send):
        captured["model_id"] = model_id
        captured["path"] = path
        captured["body"] = (await receive())["body"]
        captured["headers"] = dict(scope["headers"])
        disconnect = asyncio.create_task(receive())
        await asyncio.sleep(0)
        captured["premature_disconnect"] = disconnect.done()
        disconnect.cancel()
        await asyncio.gather(disconnect, return_exceptions=True)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"ok":true}'})

    monkeypatch.setattr(app.state.model_pool, "inventory", inventory)
    monkeypatch.setattr(app.state.model_pool, "resolve_model", resolve)
    monkeypatch.setattr(app.state.model_pool, "infer", infer)
    with TestClient(app) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        runtime_model = models.json()["data"][0]
        assert runtime_model == model["runtime_models"]["data"][0]
        assert runtime_model["context_length"] == 32768
        assert runtime_model["pricing"] == {"prompt": "0", "completion": "0"}
        assert runtime_model["model_info"]["supports_function_calling"] is True
        assert captured == {}
        response = client.post(
            "/v1/chat/completions",
            json={"model": "friendly", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True}
    assert captured["model_id"] == "org/model"
    assert captured["path"] == "v1/chat/completions"
    assert json.loads(captured["body"])["model"] == "runtime-alias"
    assert b"content-length" not in captured["headers"]
    assert captured["premature_disconnect"] is False


def test_global_v1_supports_header_query_and_multipart_model_selection(tmp_path, monkeypatch):
    app = create_app(
        Settings(
            mode="demo",
            setup_state_path=tmp_path / "setup",
            database_path=tmp_path / "db",
            model_cache_path=tmp_path / "models",
        )
    )
    model = {
        "id": "org/model",
        "loadable": True,
        "placements": [],
        "aliases": ["org/model"],
        "served_model_name": "org/model",
    }

    async def resolve(reference):
        assert reference == "org/model"
        return model

    calls = []

    async def infer(model_id, path, scope, receive, send):
        body = (await receive())["body"]
        calls.append((model_id, path, body))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    monkeypatch.setattr(app.state.model_pool, "resolve_model", resolve)
    monkeypatch.setattr(app.state.model_pool, "infer", infer)
    with TestClient(app) as client:
        assert client.post("/v1/embeddings?model=org/model", content=b"raw").status_code == 204
        assert (
            client.post(
                "/v1/responses", content=b"raw", headers={"X-Spark-Model": "org/model"}
            ).status_code
            == 204
        )
        audio = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", b"RIFFtest", "audio/wav")},
            data={"model": "org/model"},
        )
        assert audio.status_code == 204
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert [call[1] for call in calls] == [
        "v1/embeddings",
        "v1/responses",
        "v1/audio/transcriptions",
    ]
    assert b"RIFFtest" in calls[2][2]


def test_admin_token_is_origin_scoped_and_requested_in_a_sheet(tmp_path):
    with make_client(tmp_path) as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        pool = client.get("/assets/pool.js").text
        setup = client.get("/assets/setup.js").text
    assert 'id="admin-token-sheet"' in page
    assert 'id="admin-token-value"' in page
    assert "Stored locally in this browser after validation" in page
    for removed_id in (
        "pool-admin-token",
        "launch-admin-token",
        "cache-admin-token",
        "action-admin-token",
        "hf-token-admin",
        "setup-token",
    ):
        assert f'id="{removed_id}"' not in page
    assert "localStorage.setItem(adminTokenStorageKey, token)" in script
    assert "localStorage.removeItem(adminTokenStorageKey)" in script
    assert "if (response.status === 401)" in script
    assert "requestAdminToken(" in script
    assert "window.adminAuthHeaders = adminAuthHeaders" in script
    assert "window.adminAuthHeaders()" in pool
    assert "window.adminAuthHeaders()" in setup
    assert "loadSetup(true, false)" in setup
    assert "promptForAdmin = true" in script


def test_cluster_dropdown_exposes_global_capacity_context(tmp_path):
    with make_client(tmp_path) as client:
        page = client.get("/").text
        app_script = client.get("/assets/app.js").text
        pool_script = client.get("/assets/pool.js").text
    assert "All compute pools" in app_script
    assert "spark-cluster.cluster-context" in app_script
    assert "window.selectedClusterContext" in app_script
    assert "cluster-context-changed" in app_script
    assert "capacity.missing_gib == null" in pool_script
    assert "capacity.capacity?.missing_gib == null" in pool_script
    assert "Memory preflight unavailable for this compute pool" in pool_script
    assert "Needs ${capacity.missing_gib.toFixed(1)} GiB more" in pool_script
    assert "Fits on ${escapeHtml(alternative.cluster_name)}" in pool_script
    assert "cluster_id: window.selectedClusterContext" in pool_script
    assert "pool-model-capacity" in page or "pool-model-capacity" in pool_script


def test_pool_ui_renders_global_and_per_machine_load_progress(tmp_path):
    with make_client(tmp_path) as client:
        page = client.get("/").text
        pool_script = client.get("/assets/pool.js").text
    assert "pool-load-progress" in pool_script
    assert "pool-machine-progress" in pool_script
    assert "memory_after_gib" in pool_script
    assert "now - job.completed_at < 12" in pool_script
    assert "adaptive-pools-v1" in page
