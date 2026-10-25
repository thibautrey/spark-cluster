import asyncio
import json
import time

import httpx
import pytest
from starlette.requests import ClientDisconnect, Request

from spark_cluster import inference_http
from spark_cluster.inference_http import RequestTooLarge, RuntimeUnavailable, read_inference_body
from spark_cluster.model_pool import PoolResponse, forward_http
from spark_cluster.on_demand import DemandProxy, OnDemandInference
from test_on_demand import Docker, profile


def scope(headers=(), spec="2.4"):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec},
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": list(headers),
    }


def receiver(body=b'{"model":"test-model","stream":true}', disconnected=None):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        if disconnected is not None:
            await disconnected.wait()
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    return receive


def mock_http(monkeypatch, handler):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))


async def health_check(manager, resident):
    resident.health_checked_at = 0
    await manager.check_health(resident)


@pytest.mark.asyncio
async def test_health_threshold_resets_and_never_retires_busy_runtime(monkeypatch):
    p = profile()
    manager = OnDemandInference(Docker(), [p])
    manager.active = p
    resident = manager.residents[p.id]
    healthy = False
    calls = []

    def probe(request):
        calls.append(request.url.path)
        assert not manager.lock.locked()
        return httpx.Response(200 if healthy else 503)

    mock_http(monkeypatch, probe)
    await health_check(manager, resident)
    await manager.check_health(resident)  # Interval prevents repeated probes.
    assert calls == ["/health"]
    assert resident.health_failures == 1 and resident.status == "ready"
    healthy = True
    await health_check(manager, resident)
    assert resident.health_failures == 0
    healthy = False
    resident.active_requests = 1
    await health_check(manager, resident)
    assert len(calls) == 2 and resident.health_failures == 0
    resident.active_requests = 0
    for _ in range(3):
        await health_check(manager, resident)
    assert resident.status == "unhealthy"
    assert manager.status()["active_profiles"] == []
    assert 0 < manager.recovery_state(p)["retry_after_seconds"] <= 5
    assert not manager.docker.calls  # Probing itself never destroys a runtime.


@pytest.mark.asyncio
async def test_probe_that_overlaps_new_lease_cannot_mark_runtime_unhealthy(monkeypatch):
    p = profile(health_failure_threshold=1)
    manager = OnDemandInference(Docker(), [p])
    manager.active = p
    resident = manager.residents[p.id]
    entered, finish = asyncio.Event(), asyncio.Event()

    async def probe(request):
        entered.set()
        await finish.wait()
        return httpx.Response(503)

    mock_http(monkeypatch, probe)
    task = asyncio.create_task(manager.check_health(resident))
    await asyncio.wait_for(entered.wait(), 1)
    async with manager.lock:
        resident.active_requests += 1
    finish.set()
    await task
    assert resident.status == "ready" and not manager.recovery


@pytest.mark.asyncio
async def test_startup_failure_backs_off_and_stable_health_resets_budget(monkeypatch):
    p = profile()
    manager = OnDemandInference(Docker(), [p])
    attempts = 0

    async def start(p, resident):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("engine exited")

    monkeypatch.setattr(manager, "_start_runtime", start)
    with pytest.raises(RuntimeError, match="engine exited"):
        await manager.ensure(p)
    for _ in range(5):
        with pytest.raises(RuntimeUnavailable):
            await manager.ensure(p)
    assert attempts == 1 and not manager.residents
    for _ in range(20):
        manager.failed_runtime(p)
    assert 299 < manager.recovery_state(p)["retry_after_seconds"] <= 300
    manager.active = p
    resident = manager.residents[p.id]
    resident.ready_since = time.monotonic() - 61
    mock_http(monkeypatch, lambda request: httpx.Response(200))
    await health_check(manager, resident)
    assert not manager.recovery


@pytest.mark.asyncio
async def test_retained_recovery_is_background_and_does_not_repin_model(monkeypatch):
    p = profile()
    manager = OnDemandInference(Docker(), [p])
    manager.retain(p, True)
    manager.failed_runtime(p)
    manager.recovery[p.id] = (1, 0)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def start(p, resident):
        entered.set()
        await finish.wait()

    monkeypatch.setattr(manager, "_start_runtime", start)
    await manager.reconcile_health()
    await asyncio.wait_for(entered.wait(), 1)
    task = manager.recovery_tasks[p.id]
    await manager.reconcile_health()
    assert manager.recovery_tasks[p.id] is task
    manager.retain(p, False)
    finish.set()
    await task
    assert manager.residents[p.id].status == "ready"
    assert not manager.retained and not manager.recovery_tasks
    assert manager.recovery_state(p)["failures"] == 1  # Reset only after stability.


@pytest.mark.asyncio
async def test_startup_cancellation_cleans_reservation_without_teaching_failure(monkeypatch):
    p = profile()
    manager = OnDemandInference(Docker(), [p])
    entered = asyncio.Event()

    async def start(p, resident):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_start_runtime", start)
    task = asyncio.create_task(manager.ensure(p))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not manager.residents and not manager.recovery


@pytest.mark.asyncio
async def test_disconnect_while_admission_queued_releases_all_counters(monkeypatch):
    p = profile()
    manager = OnDemandInference(Docker(), [p])
    manager.active = p
    resident = manager.residents[p.id]

    async def ensure(*args, **kwargs):
        pass

    monkeypatch.setattr(manager, "ensure", ensure)
    controller = manager.admission_controller(resident)
    controller.limit = 1
    disconnected = asyncio.Event()
    async with manager.lease(p):
        task = asyncio.create_task(DemandProxy(manager, p)(
            scope(), receiver(disconnected=disconnected), lambda message: None
        ))
        async with asyncio.timeout(1):
            while not controller.queue:
                await asyncio.sleep(0)
        disconnected.set()
        with pytest.raises(ClientDisconnect):
            await task
        assert not controller.queue and controller.running == 1
        assert resident.active_requests == 1 and manager.pending == 0
    assert resident.active_requests == controller.running == 0


@pytest.mark.asyncio
async def test_body_bound_is_enforced_before_extending_buffer(monkeypatch):
    monkeypatch.setattr(inference_http, "MAX_REQUEST_BYTES", 4)
    with pytest.raises(RequestTooLarge):
        await read_inference_body(Request(scope(), receiver(b"12345")))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,status", [(httpx.ConnectError, 503), (httpx.ReadTimeout, 504)])
async def test_gateway_errors_are_openai_shaped_and_never_replay(failure, status):
    calls = []

    def upstream(request):
        calls.append(request)
        assert request.extensions["timeout"] == {
            "connect": 5, "read": 900, "write": 30, "pool": 5
        }
        raise failure("upstream unavailable", request=request)

    async def handler(scope, receive, send):
        await forward_http("http://runtime/v1/chat/completions", scope, receive, send,
                           transport=httpx.MockTransport(upstream))

    messages = []

    async def send(message):
        messages.append(message)

    await PoolResponse(handler)(scope(), receiver(), send)
    assert len(calls) == 1
    assert messages[0]["status"] == status
    error = json.loads(messages[-1]["body"])["error"]
    assert error["type"] == "server_error" and "upstream unavailable" not in error["message"]


@pytest.mark.asyncio
async def test_sse_preserves_bytes_and_headers_filters_both_connection_lists():
    chunks = [b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n", b"data: [DONE]\n\n"]
    closed = False

    class Events(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks:
                yield chunk

        async def aclose(self):
            nonlocal closed
            closed = True

    def upstream(request):
        assert request.headers["authorization"] == "Bearer fixture"
        assert "x-private-hop" not in request.headers
        return httpx.Response(200, stream=Events(), headers={
            "content-type": "text/event-stream", "connection": "x-backend-hop",
            "x-backend-hop": "private", "x-request-id": "upstream-id"
        })

    async def handler(scope, receive, send):
        await forward_http("http://runtime/v1/chat/completions", scope, receive, send,
                           transport=httpx.MockTransport(upstream))

    messages = []

    async def send(message):
        messages.append(message)

    await PoolResponse(handler)(scope([
        (b"authorization", b"Bearer fixture"), (b"connection", b"x-private-hop"),
        (b"x-private-hop", b"private")
    ]), receiver(), send)
    headers = dict(messages[0]["headers"])
    assert b"x-backend-hop" not in headers and headers[b"x-request-id"] == b"upstream-id"
    assert b"".join(m.get("body", b"") for m in messages) == b"".join(chunks)
    assert closed


@pytest.mark.asyncio
async def test_midstream_failure_closes_backend_without_second_status_or_done():
    closed, calls = False, 0

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: partial\n\n"
            raise httpx.ReadError("engine died")

        async def aclose(self):
            nonlocal closed
            closed = True

    def upstream(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, stream=Broken())

    async def handler(scope, receive, send):
        await forward_http("http://runtime/v1/chat/completions", scope, receive, send,
                           transport=httpx.MockTransport(upstream))

    messages = []

    async def send(message):
        messages.append(message)

    with pytest.raises(httpx.ReadError):
        await PoolResponse(handler)(scope(), receiver(), send)
    assert closed and calls == 1
    assert len([m for m in messages if m["type"] == "http.response.start"]) == 1
    assert not any(m.get("more_body") is False for m in messages)
