import asyncio
import copy
from types import SimpleNamespace

import httpx
import pytest
from test_model_pool import FakePool
from test_pool_memory import gpu, manager_for, profile

from spark_cluster.adaptive_concurrency import (
    AdaptiveConcurrency,
    ConcurrencyController,
    fingerprint,
    runtime_parallelism,
)
from spark_cluster.model_pool import ModelPreferences, PoolNode
from spark_cluster.on_demand import DemandProxy


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def feedback(status=200, output=100, input_tokens=10):
    return SimpleNamespace(
        status=status,
        outcome=None,
        complete=True,
        input_tokens=input_tokens,
        output_tokens=output,
        path="v1/chat/completions",
        first_output=None,
    )


def window(controller, capacity, latency=0.05, metric="output_tokens/s", workload=None):
    count = max(4, controller.limit * 2)
    rate = 100 * min(controller.limit, capacity)
    for _ in range(count):
        controller.busy_seconds += 100 / rate
        controller.demand = True
        controller.observe(100, metric, latency, workload or ("v1/chat/completions", 5, 7))


def test_learns_the_throughput_plateau_and_retests_when_capacity_changes():
    c = ConcurrencyController("model", ceiling=16)
    for _ in range(100):
        window(c, capacity=4)
    assert c.best_limit == 4
    assert 3 <= c.limit <= 5
    assert c.best_score == pytest.approx(400)
    for _ in range(150):
        window(c, capacity=8)
    assert c.best_limit == 8
    assert c.best_score == pytest.approx(800)


def test_serial_runtime_converges_to_one_instead_of_keeping_wasteful_parallelism():
    c = ConcurrencyController("serial", ceiling=8)
    for _ in range(100):
        window(c, capacity=1)
    assert c.best_limit == 1
    assert c.limit in {1, 2}


def test_no_parallel_demand_does_not_claim_a_discovered_higher_limit():
    c = ConcurrencyController("model", ceiling=16)
    for _ in range(16):
        c.busy_seconds += 1
        c.observe(100, "output_tokens/s", 0.05, ("generation", 5, 7))
    assert c.limit == 2
    assert c.state == "awaiting-demand"


def test_latency_regression_and_overload_reduce_limit_without_killing_in_flight_requests():
    clock = Clock()
    c = ConcurrencyController("model", ceiling=16, clock=clock)
    window(c, capacity=8)
    assert c.limit == 3
    window(c, capacity=8, latency=0.5)
    assert c.limit == 1
    assert c.state == "backoff"
    assert "latency" in c.reason
    c.limit, c.running = 8, 8
    clock.now += 3
    c.backoff("OOM")
    assert c.limit == 4
    assert c.running == 8  # Existing requests drain; they are not cancelled.
    c.backoff("another simultaneous failure")
    assert c.limit == 4


def test_kv_cache_pressure_requires_sustained_active_pressure_and_missing_metrics_stay_unknown():
    c = ConcurrencyController("model", ceiling=16)
    c.limit = 8
    c.telemetry({"gpu_kv_cache_usage": 0.99})
    c.telemetry({"gpu_kv_cache_usage": 0.99})
    assert c.limit == 8  # Idle cached prefixes are not active memory pressure.
    c.running = 8
    c.telemetry({"gpu_kv_cache_usage": 0.99, "queued_requests": 2})
    assert c.limit == 8
    c.telemetry({"gpu_kv_cache_usage": 0.99, "queued_requests": 2})
    assert c.limit == 4
    c.telemetry({"gpu_kv_cache_usage": None})
    assert c.pressure == 0.99
    fresh = ConcurrencyController("unreported")
    fresh.telemetry({})
    assert fresh.pressure is None


def test_workload_or_measurement_unit_changes_are_not_compared_as_a_throughput_gain():
    c = ConcurrencyController("model", ceiling=8)
    window(c, capacity=4)
    assert c.metric[0] == "output_tokens/s"
    window(c, capacity=1, metric="requests/s", workload=("embedding", 10, 0))
    assert c.metric == ("requests/s", ("embedding", 10, 0))
    assert c.best_limit == 3
    assert c.best_score == pytest.approx(100)


@pytest.mark.asyncio
async def test_overlapping_requests_use_busy_wall_time_for_aggregate_tokens_per_second():
    clock = Clock()
    c = ConcurrencyController("model", ceiling=4, clock=clock)
    for _ in range(2):
        async with c.slot(feedback()) as one, c.slot(feedback()) as two:
            one.begin()
            two.begin()
            clock.now += 1
    assert c.best_score == pytest.approx(200)  # 400 tokens / 2 s, not / 4 request-seconds.
    assert c.public()["throughput_unit"] == "output_tokens/s"
    assert c.running == c.serving == 0


@pytest.mark.asyncio
async def test_fifo_queue_releases_cancelled_waiters_and_respects_limit():
    c = ConcurrencyController("model", ceiling=2)
    c.limit = 1
    entered = []
    release = asyncio.Event()

    async def run(index):
        async with c.slot() as admission:
            admission.begin()
            entered.append(index)
            assert c.running <= c.limit
            await release.wait()

    first = asyncio.create_task(run(0))
    await asyncio.sleep(0)
    queued = [asyncio.create_task(run(i)) for i in range(1, 5)]
    await asyncio.sleep(0)
    queued[1].cancel()
    await asyncio.gather(queued[1], return_exceptions=True)
    release.set()
    await asyncio.wait_for(asyncio.gather(first, queued[0], queued[2], queued[3]), 1)
    assert entered == [0, 1, 3, 4]
    assert c.running == 0 and not c.queue
    assert c.state != "backoff"


@pytest.mark.asyncio
async def test_client_errors_and_cancellations_do_not_teach_capacity_failures():
    c = ConcurrencyController("model", ceiling=8)
    invalid = feedback(status=400)
    invalid.outcome = "failed"  # A normal JSON validation error includes an error object.
    async with c.slot(invalid) as admission:
        admission.begin()
    assert c.samples == 0 and c.limit == 2
    with pytest.raises(asyncio.CancelledError):
        async with c.slot(feedback()) as admission:
            admission.begin()
            raise asyncio.CancelledError
    assert c.limit == 2 and c.running == 0
    async with c.slot(feedback(status=503)) as admission:
        admission.begin()
    assert c.limit == 1 and c.state == "backoff"


@pytest.mark.asyncio
async def test_learning_persists_privately_and_resets_for_new_runtime_identity(tmp_path):
    path = tmp_path / "concurrency.json"
    manager = AdaptiveConcurrency(path)
    key = fingerprint({"model": "org/model", "gpu": "gpu-0", "context": 8192})
    controller = manager.controller("model", key, 16)
    controller.best_limit, controller.changed = 7, True
    await manager.flush(force=True)
    assert path.stat().st_mode & 0o777 == 0o600
    assert "org/model" not in path.read_text()
    restored = AdaptiveConcurrency(path)
    assert restored.controller("model", key, 4).limit == 4
    assert restored.controller("model", fingerprint({"context": 32768}), 16).limit == 2
    path.write_text("not json")
    assert AdaptiveConcurrency(path).controller("model", key).limit == 2


def test_reconfiguration_retains_in_flight_counters_and_bounds_new_admissions():
    c = ConcurrencyController("old", ceiling=16)
    c.limit, c.running = 8, 7
    c.reconfigure("new-image-and-gpu-budget", 4)
    assert c.limit == 2 and c.running == 7
    assert c.best_score is None


@pytest.mark.parametrize(
    "engine,flag", [("vllm", "--max-num-seqs"), ("sglang", "--max-running-requests")]
)
def test_batching_adapters_enable_capacity_without_reducing_context_or_memory(engine, flag):
    payload = {
        "Image": engine,
        "Cmd": ["--context-length", "8192", "--gpu-memory-utilization", "0.4", flag, "1"],
    }
    original = copy.deepcopy(payload)
    assert runtime_parallelism(payload, prepare=True) == 64
    assert payload["Cmd"][:-2] == original["Cmd"][:-2]
    assert runtime_parallelism(payload) == 64
    llama = {"Image": "llama.cpp", "Cmd": ["--ctx-size", "8192", "--parallel", "2"]}
    before = copy.deepcopy(llama)
    assert runtime_parallelism(llama, prepare=True) == 2
    assert llama == before


def test_routing_prefers_the_instance_with_available_adaptive_capacity(tmp_path):
    pool = FakePool(tmp_path)
    nodes = []
    for index, (running, limit) in enumerate([(2, 8), (1, 1)]):
        nodes.append(
            {
                "id": str(index),
                "available": True,
                "active_model": None,
                "resident_models": [
                    {
                        "model_id": "org/model",
                        "status": "ready",
                        "active_requests": running,
                        "concurrency": {"running": running, "waiting": 0, "limit": limit},
                    }
                ],
            }
        )
    assert pool.resident_for("org/model", nodes, ModelPreferences())["id"] == "0"


@pytest.mark.asyncio
async def test_two_requests_to_the_same_model_stream_in_parallel_and_report_usage(monkeypatch):
    p = profile("same-model")
    manager = manager_for([p], [gpu()])
    original = httpx.AsyncClient
    started = 0
    both = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"token"}}]}\n\n'
            await both.wait()
            yield b'data: {"usage":{"prompt_tokens":10,"completion_tokens":100}}\n\n'
            yield b"data: [DONE]\n\n"

    async def backend(request):
        nonlocal started
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={})
        started += 1
        if started == 2:
            both.set()
        await both.wait()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

    transport = httpx.MockTransport(backend)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport))
    async with original(
        transport=httpx.ASGITransport(app=DemandProxy(manager, p)), base_url="http://test"
    ) as client:
        responses = await asyncio.wait_for(
            asyncio.gather(
                *[client.post("/v1/chat/completions", json={"stream": True}) for _ in range(2)]
            ),
            2,
        )
    assert all(b"[DONE]" in r.content for r in responses)
    state = manager.status()["resident_models"][0]["concurrency"]
    assert state["mode"] == "automatic"
    assert state["samples"] == 2
    assert state["running"] == state["waiting"] == 0


@pytest.mark.asyncio
async def test_native_requests_overlap_and_model_mutation_waits_for_all_streams(monkeypatch):
    original = httpx.AsyncClient
    node = PoolNode.__new__(PoolNode)
    node.registry = SimpleNamespace(list=lambda: [])
    node.demand = manager_for([], [])
    node.native = {"model_id": "org/model", "runtime_ceiling": 4}
    node.native_lock = asyncio.Lock()
    node.native_requests = 0
    node.native_changed = asyncio.Event()
    node.settings = SimpleNamespace(direct_runtime_base_url="http://engine/v1")
    both = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def metadata(*args):
        pass

    node.capture_runtime_metadata = metadata

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            assert not node.native_lock.locked()
            yield b'data: {"choices":[{"delta":{"content":"token"}}]}\n\n'
            await release.wait()
            yield b"data: [DONE]\n\n"

    async def backend(request):
        nonlocal started
        started += 1
        if started == 2:
            both.set()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

    transport = httpx.MockTransport(backend)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport))

    async def app(scope, receive, send):
        await node._infer_local("org/model", "v1/chat/completions", scope, receive, send)

    mutated = asyncio.Event()

    async def mutate():
        async with node.native_mutation():
            assert node.native_requests == 0
            mutated.set()

    async with original(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        requests = [
            asyncio.create_task(client.post("/v1/chat/completions", json={"stream": True}))
            for _ in range(2)
        ]
        try:
            await asyncio.wait_for(both.wait(), 1)
            mutation = asyncio.create_task(mutate())
            await asyncio.sleep(0)
            assert not mutated.is_set()
            assert node.native_requests == 2
        finally:
            release.set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), 2)
        await asyncio.wait_for(mutation, 1)
    assert all(b"[DONE]" in r.content for r in responses)
    assert node.native_requests == 0 and mutated.is_set()
