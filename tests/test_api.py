from fastapi.testclient import TestClient

from spark_deepseek.app import create_app
from spark_deepseek.config import Settings


def make_client(tmp_path):
    app = create_app(
        Settings(
            mode="demo",
            database_path=tmp_path / "test.db",
            sample_interval_seconds=0.05,
            cluster_name="test-cluster",
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


def test_history_metrics_and_capacity(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/api/v1/history?minutes=5").status_code == 200
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "spark_deepseek_cache_hit_ratio" in metrics.text
        result = client.post(
            "/api/v1/capacity",
            json={"parameter_count_billions": 70, "context_length": 131072},
        )
        assert result.status_code == 200
        assert result.json()["minimum_nodes"] >= 1


def test_ingest_token_is_enforced(tmp_path):
    app = create_app(
        Settings(mode="live", database_path=tmp_path / "secure.db", ingest_token="secret")
    )
    with TestClient(app) as client:
        response = client.post("/api/v1/nodes", json={})
        assert response.status_code == 401


def test_first_agent_seeds_live_cluster(tmp_path):
    app = create_app(Settings(mode="live", database_path=tmp_path / "live.db"))
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
