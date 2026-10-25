import asyncio
import json
import sqlite3

import httpx
import pytest

from spark_cluster.config import Settings
from spark_cluster.inference_tracking import (
    BODY_LIMIT,
    FRAME_LIMIT,
    SCOPE_KEY,
    InferenceTracker,
    RequestRecord,
    mark_dispatch,
)
from spark_cluster.store import SnapshotStore
from spark_cluster.telemetry import InferenceTelemetryService


def tracker(tmp_path, **kwargs):
    return InferenceTracker(Settings(database_path=tmp_path / "test.db", **kwargs))


async def emit(tracking, chunks, content_type="application/json", status=200, nested=False):
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"secret prompt"}

    async def handler(scope, receive, send):
        mark_dispatch(scope, "pool-a")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", content_type.encode())],
            }
        )
        for chunk in chunks:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def outer(scope, receive, send):
        await tracking.track(handler, "org/model", "v1/chat/completions", scope, receive, send)

    await tracking.track(
        outer if nested else handler,
        "org/model",
        "v1/chat/completions",
        {"method": "POST"},
        receive,
        send,
    )
    return sent


@pytest.mark.asyncio
async def test_json_usage_nested_routes_and_durable_aggregate_without_content(tmp_path):
    t = tracker(tmp_path)
    t.resources = lambda _: {"used_gib": 4, "total_gib": 8}
    body = json.dumps(
        {
            "choices": [{"message": {"content": "secret answer"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
    ).encode()
    sent = await emit(t, [body[:30], body[30:]], nested=True)
    assert b"".join(message.get("body", b"") for message in sent) == body
    assert t.summary()["totals"]["requests"] == 1
    event = t.summary()["recent"][0]
    assert event["input_tokens"] == 12
    assert event["output_tokens"] == 4
    assert event["cached_tokens"] == 3
    assert event["ttft_ms"] is None  # a complete JSON body cannot measure token generation latency
    assert event["cluster_id"] == "pool-a"
    assert event["memory_end"]["used_gib"] == 4
    assert event["service_ms"] >= 0
    assert not t.active
    assert "secret" not in json.dumps(t.summary())
    await t.stop()
    restored = tracker(tmp_path)
    assert restored.summary()["totals"]["input_tokens"] == 12
    assert restored.summary()["recent"][0]["id"] == event["id"]
    await restored.stop()


@pytest.mark.asyncio
async def test_fragmented_sse_usage_and_first_actual_output(tmp_path):
    t = tracker(tmp_path)
    parts = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n',
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b'data: {"usage":{"prompt_tokens":20,"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]
    wire = b"".join(parts)
    sent = await emit(t, [wire[i : i + 3] for i in range(0, len(wire), 3)], "text/event-stream")
    assert b"".join(m.get("body", b"") for m in sent) == wire
    event = t.recent[0]
    assert event["outcome"] == "successful"
    assert event["input_tokens"] == 20
    assert event["output_tokens"] == 2
    assert event["ttft_ms"] is not None
    assert event["decode_tokens_per_second"] > 0
    await t.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,status,outcome",
    [
        (b'{"error":{"message":"secret"}}', 200, "failed"),
        (b"{}", 503, "failed"),
        (b"{}", 200, "successful"),
    ],
)
async def test_failures_and_unknown_tokens_are_not_fabricated(tmp_path, body, status, outcome):
    t = tracker(tmp_path)
    await emit(t, [body], status=status)
    assert t.recent[0]["outcome"] == outcome
    assert t.recent[0]["input_tokens"] is None
    assert t.summary()["totals"]["unknown_usage"] == 1
    await t.stop()


@pytest.mark.asyncio
async def test_cancellation_releases_active_tracking_and_keeps_event(tmp_path):
    t = tracker(tmp_path)

    async def cancelled(scope, receive, send):
        assert scope[SCOPE_KEY].id in t.active
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await t.track(cancelled, "model", "v1/embeddings", {"method": "POST"}, None, None)
    assert not t.active
    assert t.recent[0]["outcome"] == "cancelled"
    await t.stop()


@pytest.mark.asyncio
async def test_truncated_sse_is_failed_and_health_probes_are_not_counted(tmp_path):
    t = tracker(tmp_path)
    await emit(t, [b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'], "text/event-stream")
    assert t.recent[0]["outcome"] == "failed"

    async def health(*args):
        pass

    await t.track(health, "model", "v1/models", {"method": "GET"}, None, None)
    assert t.summary()["totals"]["requests"] == 1
    await t.stop()


@pytest.mark.asyncio
async def test_large_json_tail_usage_and_bounded_sse_buffer(tmp_path):
    t = tracker(tmp_path)
    body = b'{"data":["' + b"x" * (BODY_LIMIT + 1) + b'"],"usage":{"prompt_tokens":99}}'
    await emit(t, [body])
    assert t.recent[0]["input_tokens"] == 99
    record = RequestRecord(t, "model", "v1/completions")
    record.content_type = "text/event-stream"
    record.observe(b"data: " + b"x" * (FRAME_LIMIT * 10))
    assert len(record.buffer) <= FRAME_LIMIT
    record.observe(b'\ndata: {"usage":{"input_tokens":7}}\n\n')
    assert record.input_tokens == 7
    await t.stop()


@pytest.mark.asyncio
async def test_journal_failure_retry_and_retention_preserve_totals(tmp_path, monkeypatch):
    t = tracker(tmp_path, inference_retention_requests=100)
    for _ in range(105):
        await emit(t, [b'{"usage":{"prompt_tokens":1}}'])
    original = t.journal.save
    monkeypatch.setattr(
        t.journal, "save", lambda *args: (_ for _ in ()).throw(sqlite3.OperationalError())
    )
    await t.flush()
    assert t.persistence_error and len(t.pending) == 105
    monkeypatch.setattr(t.journal, "save", original)
    await t.flush()
    assert not t.persistence_error and not t.pending
    assert t.journal.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 100
    assert t.summary()["totals"]["requests"] == 105
    await t.stop()


@pytest.mark.asyncio
async def test_metrics_switch_reset_missing_and_nonfinite_values(tmp_path):
    settings = Settings(database_path=tmp_path / "state.db")
    store = SnapshotStore(settings.database_path)
    service = InferenceTelemetryService(settings, store)
    target = "http://engine-a/metrics"
    service.target_provider = lambda: target
    service.transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200, text="vllm:prompt_tokens_total 10\nvllm:num_requests_running 1"
        )
    )
    first = await service.sample()
    assert first.input_tokens_total == 10 and first.decode_tokens_per_second is None
    target = "http://engine-b/metrics"
    assert (await service.sample()).prefill_tokens_per_second is None
    missing = service.from_metrics("vllm:num_requests_running 0\nvllm:prompt_tokens_total NaN")
    assert missing.input_tokens_total is None and missing.prefill_tokens_per_second is None
    target = None
    assert not (await service.sample()).available
    assert service._previous is None
    store.close()
