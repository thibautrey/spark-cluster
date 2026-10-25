import json
import sqlite3
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.config import Settings
from spark_cluster.inference_tracking import (
    SCOPE_KEY,
    InferenceTracker,
    RequestRecord,
    mark_dispatch,
)
from spark_cluster.performance_history import DAY, gpu_key


def event(timestamp=None, **changes):
    return dict(
        id=uuid4().hex,
        model_id="org/model",
        cluster_id="pool-a",
        gpu_ids=["0", "1"],
        finished_at=timestamp or time.time(),
        outcome="successful",
        duration_ms=1200,
        queue_ms=100,
        ttft_ms=200,
        prefill_ms=100,
        input_tokens=100,
        output_tokens=11,
        cached_tokens=25,
        decode_tokens_per_second=10,
        **changes,
    )


def history(tmp_path, **settings):
    config = Settings(database_path=tmp_path / "controller.db", **settings)
    tracker = InferenceTracker(config)
    return tracker, tracker.journal.performance


def test_weighted_rates_partial_usage_and_gpu_pool_identity(tmp_path):
    t, h = history(tmp_path)
    first = event()
    second = event()
    second.update(
        output_tokens=101,
        decode_tokens_per_second=20,
        input_tokens=300,
        prefill_ms=300,
        cached_tokens=None,
    )
    third = event()
    third.update(
        input_tokens=None,
        output_tokens=None,
        cached_tokens=None,
        prefill_ms=None,
        decode_tokens_per_second=None,
    )
    other_pool = event()
    other_pool.update(cluster_id="pool-b", gpu_ids=["0"])
    t.journal.save([first, second, third, other_pool], t.state)
    result = h.read("pool", "pool-a", 24)["summary"]
    assert result["requests"] == 3
    assert result["decode_tokens_per_second"] == pytest.approx(110 / 6)
    assert result["effective_prefill_tokens_per_second"] == pytest.approx(400 / 0.4)
    assert result["cached_input_ratio"] == 0.25  # exclude missing cache observations
    assert result["cached_tokens_known"] == 1
    assert result["input_tokens_known"] == 2
    assert h.read("gpu", gpu_key("pool-a", "0"), 24)["summary"]["requests"] == 3
    assert h.read("gpu", gpu_key("pool-b", "0"), 24)["summary"]["requests"] == 1
    assert h.read("overall", "all", 24)["summary"]["requests"] == 4
    # Journal retries or duplicate event delivery never add a rollup twice.
    t.journal.save([first], t.state)
    assert h.read("overall", "all", 24)["summary"]["requests"] == 4
    t.journal.db.close()


def test_failed_measurements_and_unknowns_are_not_zero(tmp_path):
    t, h = history(tmp_path)
    failed = event()
    failed.update(outcome="failed", input_tokens=None, output_tokens=None, cached_tokens=None)
    t.journal.save([failed], t.state)
    s = h.read("overall", "all", 1)["summary"]
    assert s["failed"] == 1 and s["requests"] == 1
    for field in (
        "decode_tokens_per_second",
        "effective_prefill_tokens_per_second",
        "input_tokens",
        "cached_input_ratio",
        "ttft_ms_mean",
    ):
        assert s[field] is None
    t.journal.db.close()


def test_rollups_age_progressively_expire_and_survive_restart(tmp_path):
    t, h = history(tmp_path)
    now = time.time()
    initial = event(now - 60)
    t.journal.save([initial], t.state)
    with h.db:
        h.maintain(now + 2 * DAY)
    assert h.db.execute("SELECT DISTINCT resolution FROM performance_buckets").fetchall() == [
        (3600,)
    ]
    with h.db:
        h.maintain(now + 35 * DAY)
    assert h.db.execute("SELECT DISTINCT resolution FROM performance_buckets").fetchall() == [
        (DAY,)
    ]
    assert h.read("overall", "all", 24 * 180, now + 35 * DAY)["summary"]["requests"] == 1
    t.journal.db.close()
    restored = InferenceTracker(Settings(database_path=tmp_path / "controller.db"))
    assert restored.journal.performance.read("overall", "all", 24 * 180)["summary"]["requests"] == 1
    with restored.journal.db:
        restored.journal.performance.maintain(now + 181 * DAY)
    assert (
        restored.journal.db.execute("SELECT COUNT(*) FROM performance_buckets").fetchone()[0] == 0
    )
    restored.journal.db.close()


def test_raw_age_retention_and_idle_maintenance_leave_rollups(tmp_path, monkeypatch):
    t, h = history(tmp_path)
    now = time.time()
    old = event(now - 8 * DAY)
    t.journal.save([old], t.state)
    assert t.journal.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    assert h.read("overall", "all", 24 * 30)["summary"]["requests"] == 1
    monkeypatch.setattr("spark_cluster.inference_tracking.time.time", lambda: now + 181 * DAY)
    t.journal.save([], t.state)
    assert t.journal.db.execute("SELECT COUNT(*) FROM performance_buckets").fetchone()[0] == 0
    t.journal.db.close()


def test_transaction_failure_rolls_back_requests_and_rollups(tmp_path, monkeypatch):
    t, h = history(tmp_path)
    original = h.maintain
    monkeypatch.setattr(h, "maintain", lambda _: (_ for _ in ()).throw(sqlite3.OperationalError()))
    e = event()
    with pytest.raises(sqlite3.OperationalError):
        t.journal.save([e], t.state)
    assert h.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 0
    assert h.read("overall", "all", 1)["summary"]["requests"] == 0
    monkeypatch.setattr(h, "maintain", original)
    t.journal.save([e], t.state)
    assert h.read("overall", "all", 1)["summary"]["requests"] == 1
    t.journal.db.close()


def test_runtime_samples_remain_separate_and_are_bounded(tmp_path):
    t, h = history(tmp_path, performance_max_rows=1000)
    now = time.time()
    samples = [
        dict(
            scopes=[("model", f"model-{i}")],
            timestamp=now,
            metrics={
                "decode_tokens_per_second": 12,
                "gpu_kv_cache_usage": 0.7,
            },
        )
        for i in range(1010)
    ]
    t.journal.save([], t.state, samples)
    assert h.db.execute("SELECT COUNT(*) FROM performance_buckets").fetchone()[0] == 1000
    s = h.read("model", "model-1009", 1)["summary"]
    assert s["runtime_decode_tokens_per_second"] == 12
    assert s["runtime_gpu_kv_cache_usage"] == 0.7
    assert s["requests"] == 0 and s["output_tokens"] is None
    t.journal.db.close()


@pytest.mark.asyncio
async def test_runtime_samples_retry_and_expire_without_inference(tmp_path, monkeypatch):
    t, h = history(tmp_path)
    t.observe_runtime([("overall", "all")], time.time(), {"prefill_tokens_per_second": 500})
    original = t.journal.save
    monkeypatch.setattr(
        t.journal, "save", lambda *a: (_ for _ in ()).throw(sqlite3.OperationalError())
    )
    await t.flush()
    assert t.persistence_error and len(t.runtime_pending) == 1
    monkeypatch.setattr(t.journal, "save", original)
    await t.flush()
    assert not t.persistence_error and not t.runtime_pending
    assert h.read("overall", "all", 1)["summary"]["runtime_prefill_tokens_per_second"] == 500
    await t.stop()


def test_dispatch_keeps_pool_and_exact_gpu_assignment(tmp_path):
    t, h = history(tmp_path)
    t.local_pool = lambda: "local-pool"
    r = RequestRecord(t, "org/model", "v1/chat/completions")
    mark_dispatch({SCOPE_KEY: r}, gpu_ids=["GPU-uuid"])
    assert r.cluster_id == "local-pool"
    assert r.finish()["gpu_ids"] == ["GPU-uuid"]
    # A controller-routed pool identity is preserved by nested local dispatch.
    r.cluster_id = "chosen-pool"
    mark_dispatch({SCOPE_KEY: r}, gpu_ids=["GPU-uuid"])
    assert r.cluster_id == "chosen-pool"
    t.journal.db.close()


def test_performance_api_validates_scope_and_serves_durable_history(tmp_path):
    app = create_app(
        Settings(
            database_path=tmp_path / "api.db",
            setup_state_path=tmp_path / "setup",
            model_cache_path=tmp_path / "models",
            openai_api_base="http://127.0.0.1:9/v1",
        )
    )
    with TestClient(app) as client:
        t = app.state.inference_tracker
        t.journal.save([event()], t.state)
        response = client.get(
            "/api/v1/inference/performance", params={"scope": "model", "entity": "org/model"}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["requests"] == 1
        assert payload["retention"]["day_days"] == 180
        assert len(payload["history"]) == 1
        assert client.get("/api/v1/inference/performance?scope=invalid").status_code == 422
        assert client.get("/api/v1/inference/performance?hours=0").status_code == 422
        assert client.get("/api/v1/inference/performance?hours=999999").status_code == 422
        assert (
            client.get("/api/v1/inference/performance?scope=gpu&entity=unknown").json()["summary"][
                "input_tokens"
            ]
            is None
        )
        assert "secret prompt" not in json.dumps(payload)
