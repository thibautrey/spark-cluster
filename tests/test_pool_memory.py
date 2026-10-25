import asyncio
import copy
import time

import httpx
import pytest
from test_on_demand import Docker

from spark_cluster.cluster_registry import ClusterTarget, GPUDevice
from spark_cluster.config import Settings
from spark_cluster.local_docker import DockerEngineClient
from spark_cluster.model_pool import NodeLoadRequest, PoolNode
from spark_cluster.models import LocalModel, ModelLaunchRequest
from spark_cluster.on_demand import (
    DemandProxy,
    InferenceProfile,
    OnDemandInference,
    ResidentRuntime,
)
from spark_cluster.pool_memory import (
    RuntimeResources,
    gpu_availability,
    imported_resources,
    plan_memory,
)


def gpu(index=0, total=8.0, used=0.0, managed=None):
    return {
        "id": f"gpu-{index}",
        "index": index,
        "total_gib": total,
        "used_gib": used,
        "managed_gib": managed or {},
        "accounting_complete": True,
    }


def profile(name, required=2.0, devices=None, ranks=1):
    return InferenceProfile(
        id=name,
        model=name,
        model_id=f"org/{name}",
        listen_port=18000,
        upstream_port=28000,
        container={"Image": "test", "Cmd": ["--gpu-memory-utilization", "0.82"]},
        resources=RuntimeResources(
            gpu_count=ranks, gpu_ids=devices or [], required_gib_per_gpu=required, mode="auto"
        ),
    )


def resident(p, device="gpu-0", amount=2.0, **kwargs):
    return ResidentRuntime(p, [device], {device: amount}, status="ready", **kwargs)


def test_prefers_unused_gpu_then_shares_a_device_with_room():
    first = profile("first")
    residents = {first.id: resident(first)}
    plan = plan_memory(profile("second").resources, [gpu(), gpu(1)], residents, set())
    assert plan.gpu_ids == ["gpu-1"]
    shared = plan_memory(profile("second").resources, [gpu()], residents, set())
    assert shared.gpu_ids == ["gpu-0"]
    assert shared.evict == []
    assert shared.budget_gib["gpu-0"] == pytest.approx(2.0)


def test_budget_counts_external_use_without_counting_managed_memory_twice():
    first = profile("first")
    devices = [gpu(used=3.0, managed={"first": 2.0})]
    state = gpu_availability(devices, {"first": resident(first, amount=2.5)}, 0.82)[0]
    assert state["external_gib"] == 1.0
    assert state["reserved_gib"] == 2.5
    assert state["available_gib"] == pytest.approx(3.06)
    devices[0]["managed_gib"]["first"] = 3.0
    assert (
        gpu_availability(devices, {"first": resident(first, amount=2.5)}, 0.82)[0]["reserved_gib"]
        == 3.0
    )


def test_sum_of_free_memory_does_not_make_a_single_gpu_model_fit():
    plan = plan_memory(profile("large", required=7).resources, [gpu(), gpu(1)], {}, set())
    assert not plan.gpu_ids and not plan.waiting
    assert "capacity" in plan.reason


def test_heterogeneous_parallel_ranks_use_the_same_fraction_without_overbooking():
    plan = plan_memory(
        profile("parallel", required=3, ranks=2).resources,
        [gpu(total=8), gpu(1, total=16)],
        {},
        set(),
    )
    assert set(plan.gpu_ids) == {"gpu-0", "gpu-1"}
    assert plan.fraction == 0.375
    assert plan.budget_gib == {"gpu-1": 6.0, "gpu-0": 3.0}


def test_lru_eviction_protects_retained_busy_and_starting_runtimes():
    items = {name: resident(profile(name), amount=2) for name in ("old", "recent", "pinned")}
    items["old"].last_used = 1
    items["recent"].last_used = 2
    items["pinned"].last_used = 0
    plan = plan_memory(profile("new", required=2).resources, [gpu()], items, {"pinned"})
    assert plan.evict == ["old"]
    items["old"].active_requests = 1
    items["recent"].status = "starting"
    waiting = plan_memory(profile("new").resources, [gpu()], items, {"pinned"})
    assert waiting.waiting and not waiting.gpu_ids


def test_imported_recipe_keeps_affinity_and_declared_budget():
    p = profile("imported").model_copy(
        update={
            "resources": None,
            "container": {
                "Cmd": ["--gpu-memory-utilization=0.4"],
                "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["1"]}]},
            },
        }
    )
    resources = imported_resources(p, [gpu(), gpu(1)])
    assert resources.gpu_ids == ["gpu-1"]
    assert resources.mode == "fixed"
    plan = plan_memory(resources, [gpu(), gpu(1)], {}, set())
    assert plan.budget_gib == {"gpu-1": 3.2}
    p.container["Cmd"] = ["unknown-engine"]
    assert imported_resources(p, [gpu(), gpu(1)]).mode == "exclusive"


def test_unknown_budget_only_reserves_its_assigned_gpu():
    p = profile("unknown")
    items = {p.id: resident(p, exclusive=True, amount=6.56)}
    plan = plan_memory(profile("other").resources, [gpu(), gpu(1)], items, {p.id})
    assert plan.gpu_ids == ["gpu-1"]
    unavailable = plan_memory(
        profile("other", devices=["gpu-0"]).resources, [gpu(), gpu(1)], items, {p.id}
    )
    assert unavailable.waiting


@pytest.fixture
def readiness(monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport))


def manager_for(profiles, devices, path=None):
    manager = OnDemandInference(Docker(), profiles, state_path=path)

    async def resources():
        return copy.deepcopy(devices)

    manager.resource_provider = resources
    return manager


@pytest.mark.asyncio
async def test_two_residents_survive_with_actual_engine_caps_and_targeted_unload(readiness):
    first, second = profile("first"), profile("second")
    manager = manager_for([first, second], [gpu()])
    await manager.ensure(first, keep_loaded=True)
    await manager.ensure(second, keep_loaded=True)
    assert set(manager.residents) == {"first", "second"}
    assert manager.active is None
    assert manager.status()["active_profiles"] == ["first", "second"]
    reuse = await manager.plan(first)
    assert reuse.available_gib == pytest.approx(2.56)
    for p in (first, second):
        payload = manager.docker.containers[p.name]["Config"]
        assert payload["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] == ["gpu-0"]
        assert payload["Cmd"] == ["--gpu-memory-utilization", "0.25"]
    node = PoolNode.__new__(PoolNode)
    node.demand = manager
    await node.unload(first.model_id)
    assert manager.active is second
    assert second.name in manager.docker.containers
    assert first.id not in manager.retained
    assert second.id in manager.retained


@pytest.mark.asyncio
async def test_startup_reservations_prevent_concurrent_overbooking(monkeypatch):
    first, second, third = [profile(name, required=3) for name in ("first", "second", "third")]
    manager = manager_for([first, second, third], [gpu()])
    starting = asyncio.Event()
    finish = asyncio.Event()

    async def start(p, reservation):
        if len(manager.residents) == 2:
            starting.set()
        await finish.wait()

    monkeypatch.setattr(manager, "_start_runtime", start)
    tasks = [asyncio.create_task(manager.ensure(p)) for p in (first, second)]
    try:
        await asyncio.wait_for(starting.wait(), 1)
        assert sum(r.budget_gib["gpu-0"] for r in manager.residents.values()) == 6
        with pytest.raises(RuntimeError, match="starting runtimes"):
            await manager.ensure(third)
        with pytest.raises(RuntimeError, match="startup"):
            await manager.remove(first)
        assert set(manager.residents) == {first.id, second.id}
    finally:
        finish.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_failed_start_releases_only_its_own_reservation(readiness, monkeypatch):
    first, second = profile("first"), profile("second")
    manager = manager_for([first, second], [gpu()])
    await manager.ensure(first)

    async def fail(p, reservation):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(manager, "_start_runtime", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        await manager.ensure(second)
    assert set(manager.residents) == {first.id}
    assert first.name in manager.docker.containers


@pytest.mark.asyncio
async def test_two_streams_run_concurrently_and_pin_their_own_models(monkeypatch):
    first = profile("first").model_copy(update={"upstream_port": 28001})
    second = profile("second").model_copy(update={"upstream_port": 28002})
    manager = manager_for([first, second], [gpu()])
    original = httpx.AsyncClient
    entered = set()
    both = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            assert not manager.lock.locked()
            assert any(r.active_requests == 1 for r in manager.residents.values())
            yield b"data: token\n\n"
            await both.wait()
            yield b"data: [DONE]\n\n"

    async def upstream(request):
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={})
        entered.add(request.url.port)
        if len(entered) == 2:
            both.set()
        await both.wait()
        return httpx.Response(200, stream=Stream())

    transport = httpx.MockTransport(upstream)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport))
    await manager.ensure(first)
    await manager.ensure(second)

    async def infer(p):
        async with original(
            transport=httpx.ASGITransport(app=DemandProxy(manager, p)), base_url="http://test"
        ) as client:
            return await client.post("/v1/chat/completions", json={"stream": True})

    responses = await asyncio.wait_for(asyncio.gather(infer(first), infer(second)), 2)
    assert all(b"[DONE]" in response.content for response in responses)
    assert all(r.active_requests == 0 for r in manager.residents.values())


@pytest.mark.asyncio
async def test_idle_expiration_is_per_model_and_skips_active_stream(readiness):
    first, second = profile("first"), profile("second")
    first.idle_seconds = second.idle_seconds = 1
    manager = manager_for([first, second], [gpu()])
    await manager.ensure(first)
    await manager.ensure(second)
    manager.residents[first.id].last_used = time.monotonic() - 10
    manager.residents[second.id].last_used = time.monotonic() - 10
    manager.residents[second.id].active_requests = 1
    task = asyncio.create_task(manager.reap())
    try:
        await asyncio.sleep(1.1)
        assert set(manager.residents) == {second.id}
        assert second.name in manager.docker.containers
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_restart_adopts_multiple_retained_residents_without_recreating(readiness, tmp_path):
    profiles = [profile("first"), profile("second")]
    path = tmp_path / "state.json"
    manager = manager_for([], [gpu()], path)
    for p in profiles:
        manager.register(p)
        await manager.ensure(p, keep_loaded=True)
    restored = manager_for([], [gpu()], path)
    restored.docker = manager.docker
    restored.docker.available = True

    async def ping():
        return True

    restored.docker.ping = ping
    count = len(restored.docker.calls)
    try:
        await restored.start()
        assert set(restored.residents) == {p.id for p in profiles}
        assert restored.status()["active_profile"] is None
        assert not any(
            method in {"POST", "DELETE"} for method, _, _ in restored.docker.calls[count:]
        )
    finally:
        await restored.stop()
    assert len(restored.docker.containers) == 2


@pytest.mark.asyncio
async def test_docker_process_attribution_separates_external_gpu_use(tmp_path, monkeypatch):
    import spark_cluster.local_docker as local_docker

    socket = tmp_path / "docker.sock"
    socket.touch()
    settings = Settings(docker_runtime_enabled=True, docker_socket_path=socket)
    docker = DockerEngineClient(settings)
    measured = [{**gpu(used=3.0), "processes": {101: 2.0, 999: 1.0}}]
    monkeypatch.setattr(local_docker, "gpu_process_memory", lambda devices: copy.deepcopy(measured))
    first = profile("first")

    def engine(request):
        if request.url.path.endswith("/top"):
            return httpx.Response(200, json={"Titles": ["PID"], "Processes": [["101"]]})
        return httpx.Response(
            200,
            json={
                "State": {"Running": True},
                "Config": {"Labels": {"spark-cluster.on-demand": "first"}},
            },
        )

    docker.transport = httpx.MockTransport(engine)
    devices = await docker.gpu_memory(
        [first], ClusterTarget(id="local", name="local", kind="local-docker")
    )
    assert devices[0]["managed_gib"] == {"first": 2.0}
    assert "processes" not in devices[0]
    available = gpu_availability(devices, {"first": resident(first, amount=2.5)}, 0.82)[0]
    assert available["external_gib"] == 1.0
    assert available["available_gib"] == pytest.approx(3.06)


@pytest.mark.asyncio
async def test_node_loads_assign_distinct_gpus_and_ports_without_lowering_concurrency(
    tmp_path, readiness
):
    from types import SimpleNamespace

    cache = tmp_path / "models"
    cache.mkdir()
    settings = Settings(
        model_cache_path=cache, setup_state_path=tmp_path / "state", active_hardware="nvidia-gpu"
    )
    cluster = ClusterTarget(
        id="local",
        name="local",
        kind="local-docker",
        hardware="nvidia-gpu",
        gpu_count=2,
        gpu_memory_gib=8,
        gpus=[GPUDevice(index=i, uuid=f"gpu-{i}", name="test", memory_gib=8) for i in range(2)],
    )
    manager = manager_for([], [gpu(), gpu(1)])
    manager.docker.runtime_payload = DockerEngineClient(settings).runtime_payload

    async def image_present(image):
        return True

    manager.docker.image_present = image_present
    requests = []

    async def managed_model(model_id):
        return LocalModel(
            id=model_id, source="managed", path=str(cache / model_id), revision="main"
        )

    async def fit_for_local(request):
        requests.append(request)
        await asyncio.sleep(0)
        return SimpleNamespace(fits=True, resident_weights_gib=1.0, resident_kv_cache_gib=0.5)

    async def launches():
        return []

    plan = SimpleNamespace(
        executable=True,
        image="test",
        engine_id="vllm",
        engine_name="vLLM",
        warnings=[],
        blockers=[],
    )
    models = SimpleNamespace(
        settings=settings, managed_model=managed_model, fit_for_local=fit_for_local
    )
    deployments = SimpleNamespace(launches=launches, execution_plan=lambda *args: plan)
    registry = SimpleNamespace(list=lambda: [cluster])
    node = PoolNode(settings, registry, models, deployments, manager)
    jobs = [
        NodeLoadRequest(
            model_id=f"org/{name}",
            settings=ModelLaunchRequest(
                model_id=f"org/{name}",
                served_model_name=name,
                max_num_seqs=7,
                context_length=4096,
            ),
        )
        for name in ("first", "second")
    ]
    results = await asyncio.gather(*(node.load(job) for job in jobs))
    assert all(result["status"] == "ready" for result in results)
    assert [r.concurrent_sequences for r in requests] == [7, 7]
    assert len(manager.residents) == 2
    assert {tuple(r.gpu_ids) for r in manager.residents.values()} == {("gpu-0",), ("gpu-1",)}
    ports = set()
    for r in manager.residents.values():
        ports.update([r.profile.upstream_port, r.profile.listen_port])
        command = manager.docker.containers[r.profile.name]["Config"]["Cmd"]
        assert command[command.index("--max-num-seqs") + 1] == "64"
        assert command[command.index("--max-model-len") + 1] == "4096"
    assert len(ports) == 4


@pytest.mark.asyncio
async def test_legacy_adoption_without_gpu_telemetry_preserves_running_container():
    first = profile("legacy")
    manager = manager_for([], [])
    manager.register(first)
    manager.docker.available = True
    manager.docker.containers[first.name] = {
        "State": {"Running": True},
        "Config": {"Labels": {"spark-cluster.on-demand": first.id}},
    }

    async def ping():
        return True

    manager.docker.ping = ping
    with pytest.raises(RuntimeError, match="telemetry"):
        await manager.start()
    assert first.name in manager.docker.containers
    assert not any(method in {"POST", "DELETE"} for method, _, _ in manager.docker.calls)
