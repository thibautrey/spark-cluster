import json
import time

from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.config import Settings


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
        assert client.get("/").status_code == 200
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
        assert "await api('/api/v1/control')" not in dashboard_js


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


def test_controller_refuses_to_proxy_inference_and_advertises_direct_runtime(tmp_path):
    with make_client(tmp_path) as client:
        endpoint = client.get("/api/v1/runtime").json()
        assert endpoint["routing_mode"] == "direct-runtime"
        assert endpoint["controller_required"] is False
        assert endpoint["chat_completions_url"] == "http://127.0.0.1:9/v1/chat/completions"
        response = client.post(
            "/v1/chat/completions",
            json={"model": "anything", "messages": []},
        )
        assert response.status_code == 421
        assert "never routed" in response.json()["detail"]
        assert endpoint["base_url"] in response.json()["detail"]


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
