import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from spark_cluster.config import Settings
from spark_cluster.model_pool import (
    ModelPool,
    ModelPreferences,
    PoolLoadRequest,
    PoolNode,
    PoolScaleRequest,
)
from spark_cluster.models import ModelLaunchRequest


class Registry:
    def __init__(self):
        self.selected = "first"
        self.members = [
            SimpleNamespace(id=name, name=name, kind="local-docker", status="ready")
            for name in ["first", "second"]
        ]

    def list(self):
        return self.members

    def get(self, key):
        return next(c for c in self.members if c.id == key)


class FakePool(ModelPool):
    def __init__(self, tmp_path):
        settings = Settings(setup_state_path=tmp_path)
        models = SimpleNamespace(safe_error=lambda e: str(e))
        node = SimpleNamespace(models=models)
        super().__init__(settings, Registry(), node)
        self.nodes = {
            key: {
                "models": [{"id": "org/model", "loadable": True, "loaded": False}],
                "active_model": None,
                "keep_loaded": False,
                "busy": False,
                "machines": [
                    {
                        "id": f"{key}-node-1",
                        "name": f"{key} node 1",
                        "address": f"10.0.0.{1 if key == 'first' else 2}",
                        "accelerators": 1,
                        "memory_used_gib": 1.0,
                        "memory_total_gib": 16.0,
                    }
                ],
            }
            for key in ["first", "second"]
        }
        self.calls = []
        self.compatible = {"first": True, "second": True}
        self.gate = None

    async def call(self, cluster, action, payload=None):
        self.calls.append((cluster.id, action, payload))
        node = self.nodes[cluster.id]
        if action == "status":
            return node
        if action == "plan":
            return {"compatible": self.compatible[cluster.id], "reason": "Insufficient VRAM"}
        if action == "load":
            if self.gate:
                await self.gate.wait()
            node.update(active_model=payload["model_id"], keep_loaded=payload["keep_loaded"])
            node["models"][0]["loaded"] = True
            return {"settings": payload.get("settings"), "status": "ready"}
        if action == "retain":
            node["keep_loaded"] = payload["keep_loaded"]
        if action == "unload":
            node.update(active_model=None, keep_loaded=False)
            node["models"][0]["loaded"] = False
        return {"status": "ok"}


async def finished(pool):
    await asyncio.wait_for(asyncio.gather(*list(pool.tasks)), 3)


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_automatic_pool_avoids_retained_model_and_keeps_ui_selection(tmp_path):
    pool = FakePool(tmp_path)
    pool.nodes["first"].update(active_model="other/model", keep_loaded=True)
    job = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await finished(pool)
    assert job["status"] == "ready"
    assert job["cluster_id"] == "second"
    assert pool.registry.selected == "first"
    assert pool.nodes["first"]["active_model"] == "other/model"


@pytest.mark.asyncio
async def test_automatic_pool_skips_incompatible_cluster(tmp_path):
    pool = FakePool(tmp_path)
    pool.compatible["first"] = False
    job = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await finished(pool)
    assert job["cluster_id"] == "second"
    assert job["status"] == "ready"


@pytest.mark.asyncio
async def test_existing_residency_preferred_to_empty_cluster(tmp_path):
    pool = FakePool(tmp_path)
    pool.nodes["second"]["active_model"] = "org/model"
    job = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await finished(pool)
    assert job["cluster_id"] == "second"


@pytest.mark.asyncio
async def test_scale_up_loads_an_additional_instance(tmp_path):
    pool = FakePool(tmp_path)
    pool.nodes["first"]["active_model"] = "org/model"
    pool.nodes["first"]["models"][0]["loaded"] = True
    job = await pool.scale_instances(PoolScaleRequest(model_id="org/model", delta=1))
    await finished(pool)
    assert job["status"] == "ready"
    assert job["cluster_id"] == "second"
    assert pool.nodes["first"]["active_model"] == "org/model"
    assert pool.nodes["second"]["active_model"] == "org/model"
    assert pool.cache is not None


@pytest.mark.asyncio
async def test_scale_down_removes_one_instance_and_preserves_minimum(tmp_path):
    pool = FakePool(tmp_path)
    for cluster_id, node in pool.nodes.items():
        node.update(id=cluster_id, available=True, active_model="org/model")
        node["models"][0]["loaded"] = True
    pool.preferences["org/model"] = ModelPreferences(cluster_id="first", keep_loaded=True)
    result = await pool.scale_instances(PoolScaleRequest(model_id="org/model", delta=-1))
    assert result == {"status": "ready", "instances": 1, "cluster_id": "second"}
    assert pool.nodes["first"]["active_model"] == "org/model"
    assert pool.nodes["second"]["active_model"] is None
    assert pool.preferences["org/model"].keep_loaded is True
    assert pool.cache is not None
    with pytest.raises(ValueError, match="At least one"):
        await pool.scale_instances(PoolScaleRequest(model_id="org/model", delta=-1))


def test_resident_routing_uses_idle_instances_round_robin(tmp_path):
    pool = FakePool(tmp_path)
    for cluster_id, node in pool.nodes.items():
        node.update(id=cluster_id, available=True, active_model="org/model")
    preferences = ModelPreferences()
    assert pool.resident_for("org/model", list(pool.nodes.values()), preferences)["id"] == "first"
    assert pool.resident_for("org/model", list(pool.nodes.values()), preferences)["id"] == "second"
    pool.nodes["first"]["busy"] = True
    assert pool.resident_for("org/model", list(pool.nodes.values()), preferences)["id"] == "second"


@pytest.mark.asyncio
async def test_strict_affinity_does_not_fall_back_when_retained(tmp_path):
    pool = FakePool(tmp_path)
    pool.preferences["org/model"] = ModelPreferences(cluster_id="first")
    pool.nodes["first"].update(active_model="other/model", keep_loaded=True)
    job = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await eventually(lambda: job["status"] == "waiting")
    assert "retained" in job["message"]
    assert not any(c[1] == "load" for c in pool.calls)
    pool.nodes["first"]["keep_loaded"] = False
    await finished(pool)
    assert job["cluster_id"] == "first"


@pytest.mark.asyncio
async def test_concurrent_models_cannot_reserve_the_same_cluster(tmp_path):
    pool = FakePool(tmp_path)
    for n in pool.nodes.values():
        n["models"].append({"id": "org/second", "loadable": True})
    pool.gate = asyncio.Event()
    first = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await eventually(lambda: first["status"] == "loading")
    second = await pool.enqueue(PoolLoadRequest(model_id="org/second"))
    await eventually(lambda: second["status"] == "loading")
    assert first["cluster_id"] != second["cluster_id"]
    pool.gate.set()
    await finished(pool)


@pytest.mark.asyncio
async def test_saved_settings_are_reused_after_restart(tmp_path):
    pool = FakePool(tmp_path)
    settings = ModelLaunchRequest(
        model_id="org/model", served_model_name="saved", context_length=32768, tensor_parallel=2
    )
    await pool.update_preferences(
        "org/model", ModelPreferences(cluster_id="second", keep_loaded=True, last_settings=settings)
    )
    updated = await pool.update_retention("org/model", False)
    assert updated.cluster_id == "second"
    assert updated.last_settings.context_length == 32768
    assert updated.keep_loaded is False
    await pool.update_retention("org/model", True)
    restarted = FakePool(tmp_path)
    job = await restarted.enqueue(PoolLoadRequest(model_id="org/model"))
    await finished(restarted)
    launch = next(call for call in restarted.calls if call[1] == "load")
    assert launch[2]["settings"]["context_length"] == 32768
    assert launch[2]["settings"]["tensor_parallel"] == 2
    assert launch[2]["keep_loaded"] is True
    assert job["cluster_id"] == "second"
    assert restarted.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_deduplicates_load_and_does_not_unload_in_flight(tmp_path):
    pool = FakePool(tmp_path)
    pool.gate = asyncio.Event()
    first = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    second = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    assert first is second
    with pytest.raises(ValueError, match="pending load"):
        await pool.unload("org/model")
    pool.gate.set()
    await finished(pool)
    await pool.unload("org/model")
    assert not pool.nodes[first["cluster_id"]]["keep_loaded"]
    assert pool.nodes[first["cluster_id"]]["active_model"] is None


@pytest.mark.asyncio
async def test_affinity_rejects_mismatched_settings_and_unknown_cluster(tmp_path):
    pool = FakePool(tmp_path)
    settings = ModelLaunchRequest(model_id="org/other", served_model_name="other")
    with pytest.raises(ValueError, match="belong"):
        await pool.update_preferences("org/model", ModelPreferences(last_settings=settings))
    with pytest.raises((KeyError, StopIteration, RuntimeError)):
        await pool.update_preferences("org/model", ModelPreferences(cluster_id="missing"))


@pytest.mark.asyncio
async def test_incompatible_affinity_fails_without_using_other_cluster(tmp_path):
    pool = FakePool(tmp_path)
    pool.preferences["org/model"] = ModelPreferences(cluster_id="first")
    pool.compatible["first"] = False
    job = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await finished(pool)
    assert job["status"] == "failed"
    assert "VRAM" in job["message"]
    assert not any(c[1] == "load" for c in pool.calls)


@pytest.mark.asyncio
async def test_interrupted_load_is_never_replayed_after_restart(tmp_path):
    pool = FakePool(tmp_path)
    pool.jobs.append({"id": "interrupted", "model_id": "org/model", "status": "loading"})
    pool.save()
    restarted = FakePool(tmp_path)
    assert restarted.jobs[0]["status"] == "failed"
    assert not restarted.tasks


@pytest.mark.asyncio
async def test_remote_node_calls_are_signed_and_never_select_a_cluster(tmp_path):
    import httpx

    from spark_cluster.remote_cluster import valid_controller_signature

    pool = FakePool(tmp_path)
    pool.settings.model_admin_token = "test-shared-key"
    member = SimpleNamespace(
        id="remote", kind="remote-bridge", endpoint="http://192.168.1.42:30080"
    )

    def remote(request):
        assert valid_controller_signature(
            "test-shared-key",
            request.method,
            request.url.raw_path.decode(),
            request.headers.get("x-spark-timestamp"),
            request.headers.get("x-spark-signature"),
            request.content,
        )
        return httpx.Response(200, json={"compatible": True})

    pool.transport = httpx.MockTransport(remote)
    result = await ModelPool.call(pool, member, "plan", {"model_id": "org/model"})
    assert result["compatible"]
    assert pool.registry.selected == "first"


def test_pool_routes_stay_local_when_another_cluster_is_selected():
    from spark_cluster.remote_cluster import RemoteClusterProxyMiddleware

    middleware = RemoteClusterProxyMiddleware(
        None, Settings(remote_cluster_url="http://192.168.1.42:30080")
    )
    assert not middleware.should_forward({"type": "http", "path": "/api/v1/pool"})
    assert not middleware.should_forward({"type": "http", "path": "/api/v1/pool/load"})
    assert not middleware.should_forward({"type": "http", "path": "/api/v1/pool/node/status"})


def test_runtime_detection_classifies_specialized_model_profiles():
    from spark_cluster.model_pool import profile_runtime
    from spark_cluster.on_demand import InferenceProfile

    cases = [
        (
            {
                "Image": "vllm/vllm-openai:latest",
                "Cmd": ["BAAI/bge-small-en-v1.5", "--runner", "pooling"],
            },
            "BAAI/bge-small-en-v1.5",
            "vllm",
            "pooling",
        ),
        (
            {"Image": "ghcr.io/speaches-ai/speaches:latest-cuda"},
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            "speaches",
            "faster-whisper",
        ),
        (
            {"Image": "bonsai-llama:prism", "Entrypoint": ["/opt/bonsai/run-server.sh"]},
            "prism-ml/Ternary-Bonsai-2-27B-gguf",
            "llama-cpp",
            "llama.cpp",
        ),
        (
            {"Image": "vllm/vllm-openai:latest", "Cmd": ["Qwen/Qwen2.5-Coder-3B-Instruct-AWQ"]},
            "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ",
            "vllm",
            "AWQ",
        ),
    ]
    for index, (container, model_id, engine_id, evidence) in enumerate(cases):
        runtime = profile_runtime(
            InferenceProfile(
                id=f"profile-{index}",
                model=model_id,
                model_id=model_id,
                listen_port=18000 + index,
                upstream_port=28000 + index,
                container=container,
            )
        )
        assert runtime["engine_id"] == engine_id
        assert runtime["selection"] == "automatic"
        assert runtime["executable"] is True
        assert evidence.casefold() in (runtime["name"] + runtime["reason"]).casefold()


def test_execution_runtime_reports_automatic_engine_evidence():
    from spark_cluster.model_pool import execution_runtime
    from spark_cluster.runtime_engines import build_execution_plan

    request = ModelLaunchRequest(
        model_id="org/transformer",
        served_model_name="transformer",
        engine_id="auto",
        context_length=8192,
    )
    plan = build_execution_plan(request, "/models/org/transformer", Settings())
    runtime = execution_runtime(plan, request.engine_id)
    assert runtime["engine_id"] == "vllm"
    assert runtime["selection"] == "automatic"
    assert runtime["executable"] is True
    assert "model format" in runtime["reason"]


@pytest.mark.asyncio
async def test_runtime_model_metadata_is_persisted_verbatim(tmp_path, monkeypatch):
    payload = {
        "object": "list",
        "data": [
            {
                "id": "runtime-name",
                "object": "model",
                "created": 42,
                "owned_by": "runtime-owner",
                "context_length": 65536,
                "pricing": {"prompt": "0.1", "completion": "0.2"},
                "architecture": {"input_modalities": ["text", "image"]},
                "top_provider": {"max_completion_tokens": 4096},
                "supported_parameters": ["tools", "reasoning"],
                "capabilities": {"vision": True},
                "max_model_len": 65536,
                "model_info": {"supports_vision": True},
            }
        ],
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    monkeypatch.setattr(PoolNode, "transport", transport)
    node = PoolNode.__new__(PoolNode)
    node.metadata_path = tmp_path / "runtime-model-metadata.json"
    node.runtime_metadata = {}
    node._metadata_checked = {}
    await node.capture_runtime_metadata("org/model", "http://127.0.0.1:31000")
    assert node.metadata_for("org/model") == payload
    assert json.loads(node.metadata_path.read_text())["org/model"] == payload
    assert node.metadata_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_invalid_runtime_model_metadata_is_not_cached(tmp_path, monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"object": "list", "data": ["invalid"]})
    )
    monkeypatch.setattr(PoolNode, "transport", transport)
    node = PoolNode.__new__(PoolNode)
    node.metadata_path = tmp_path / "runtime-model-metadata.json"
    node.runtime_metadata = {}
    node._metadata_checked = {}
    await node.capture_runtime_metadata("org/model", "http://127.0.0.1:31000")
    assert node.runtime_metadata == {}
    assert not node.metadata_path.exists()


@pytest.mark.asyncio
async def test_capacity_reports_exact_memory_shortfall():
    fit = SimpleNamespace(
        launch_supported=True,
        resident_weights_gib=8.0,
        resident_kv_cache_gib=2.0,
        runtime_overhead_gib=2.0,
        total_required_gib=12.0,
        context_length=32768,
        confidence="architecture",
    )

    async def fit_for_local(request):
        return fit

    node = PoolNode.__new__(PoolNode)
    node.models = SimpleNamespace(fit_for_local=fit_for_local)
    model = {
        "id": "org/model",
        "model_path": "/missing",
        "profile": None,
        "last_settings": ModelLaunchRequest(
            model_id="org/model", served_model_name="model", context_length=32768
        ).model_dump(),
    }
    capacity = await node.model_capacity(model, available_gib=9.5)
    assert capacity == {
        "fits": False,
        "required_gib": 12.0,
        "available_gib": 9.5,
        "missing_gib": 2.5,
        "context_length": 32768,
        "confidence": "architecture",
    }


@pytest.mark.asyncio
async def test_concurrent_pool_inventory_refresh_is_shared(tmp_path):
    pool = FakePool(tmp_path)
    values = await asyncio.gather(*(pool.inventory(force=True) for _ in range(8)))
    assert all(value is values[0] for value in values)
    assert sum(action == "status" for _, action, _ in pool.calls) == 2
    await pool.close()


@pytest.mark.asyncio
async def test_capacity_uses_recorded_metadata_without_network_storage(monkeypatch):
    from pathlib import Path

    fit = SimpleNamespace(
        launch_supported=True,
        resident_weights_gib=2,
        resident_kv_cache_gib=1,
        runtime_overhead_gib=1,
        total_required_gib=4,
        context_length=8192,
        confidence="architecture",
    )
    node = PoolNode.__new__(PoolNode)
    node.models = SimpleNamespace(recorded_fit=lambda _: fit)

    def unexpected_read(*args):
        raise AssertionError("Warm capacity must not touch model storage")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    result = await node.model_capacity({"id": "org/model", "model_path": "/slow-nas/model"}, 8)
    assert result["fits"] and result["required_gib"] == 4


@pytest.mark.asyncio
async def test_transient_cluster_context_targets_that_cluster_without_saving_affinity(tmp_path):
    pool = FakePool(tmp_path)
    request = PoolLoadRequest(model_id="org/model", cluster_id="second")
    job = await pool.enqueue(request)
    await finished(pool)
    assert job["status"] == "ready"
    assert job["cluster_id"] == "second"
    assert pool.preferences.get("org/model", ModelPreferences()).cluster_id is None


@pytest.mark.asyncio
async def test_inventory_keeps_per_cluster_capacity_and_fit_alternatives(tmp_path):
    pool = FakePool(tmp_path)
    pool.nodes["first"]["models"][0]["capacity"] = {
        "fits": False,
        "required_gib": 12.0,
        "available_gib": 8.0,
        "missing_gib": 4.0,
        "context_length": 32768,
        "confidence": "architecture",
    }
    pool.nodes["second"]["models"][0]["capacity"] = {
        "fits": True,
        "required_gib": 12.0,
        "available_gib": 16.0,
        "missing_gib": 0.0,
        "context_length": 32768,
        "confidence": "architecture",
    }
    state = await pool.inventory(force=True)
    model = next(item for item in state["models"] if item["id"] == "org/model")
    assert model["available_clusters"] == ["second"]
    assert model["cluster_capacity"] == [
        {
            "cluster_id": "first",
            "cluster_name": "first",
            "available": True,
            **pool.nodes["first"]["models"][0]["capacity"],
        },
        {
            "cluster_id": "second",
            "cluster_name": "second",
            "available": True,
            **pool.nodes["second"]["models"][0]["capacity"],
        },
    ]


@pytest.mark.asyncio
async def test_load_progress_tracks_global_and_each_machine_then_memory_stats(tmp_path):
    pool = FakePool(tmp_path)
    pool.nodes["first"]["machines"].append(
        {
            "id": "first-node-2",
            "name": "first node 2",
            "address": "10.0.0.3",
            "accelerators": 1,
            "memory_used_gib": 2.0,
            "memory_total_gib": 16.0,
        }
    )
    pool.nodes["second"]["busy"] = True
    pool.gate = asyncio.Event()
    job = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await eventually(lambda: job["status"] == "loading")
    assert job["stage"] == "starting-runtime"
    assert job["progress"] == 20.0
    assert [machine["name"] for machine in job["machines"]] == ["first node 1", "first node 2"]
    await asyncio.sleep(1.1)
    assert job["progress"] > 20
    assert all(machine["status"] == "loading-runtime" for machine in job["machines"])
    pool.gate.set()
    await finished(pool)
    assert job["status"] == "ready"
    assert job["stage"] == "ready"
    assert job["progress"] == 100.0
    assert job["completed_at"] is not None
    assert all(machine["progress"] == 100 for machine in job["machines"])
    assert [machine["memory_after_gib"] for machine in job["machines"]] == [1.0, 2.0]
    assert [machine["memory_total_gib"] for machine in job["machines"]] == [16.0, 16.0]


@pytest.mark.asyncio
async def test_shared_cluster_loads_while_a_different_retained_model_is_busy(tmp_path):
    pool = FakePool(tmp_path)
    pool.nodes["second"]["busy"] = True
    node = pool.nodes["first"]
    node.update(
        multi_residency=True,
        busy=True,
        active_model=None,
        keep_loaded=True,
        resident_models=[
            {
                "model_id": "other/model",
                "active_requests": 1,
                "status": "ready",
                "keep_loaded": True,
            }
        ],
    )
    node["models"].append({"id": "org/second", "loadable": True, "loaded": False})
    pool.gate = asyncio.Event()
    first = await pool.enqueue(PoolLoadRequest(model_id="org/model"))
    await eventually(lambda: first["status"] == "loading")
    second = await pool.enqueue(PoolLoadRequest(model_id="org/second"))
    await eventually(lambda: second["status"] == "loading")
    assert first["cluster_id"] == second["cluster_id"] == "first"
    assert pool.reserved == {("first", "org/model"), ("first", "org/second")}
    pool.gate.set()
    await finished(pool)
    assert first["status"] == second["status"] == "ready"


def test_routing_and_instances_use_per_model_residency_in_a_shared_cluster(tmp_path):
    pool = FakePool(tmp_path)
    first = pool.nodes["first"]
    first.update(
        id="first",
        available=True,
        active_model=None,
        busy=True,
        resident_models=[
            {"model_id": "org/model", "active_requests": 0, "status": "ready"},
            {"model_id": "other/model", "active_requests": 2, "status": "ready"},
        ],
    )
    selected = pool.resident_for("org/model", [first], ModelPreferences())
    assert selected is first
    pool.reserved.add(("first", "new/model"))
    assert pool.resident_for("org/model", [first], ModelPreferences()) is first


@pytest.mark.asyncio
async def test_inventory_exposes_gpu_budgets_for_each_co_resident_model(tmp_path):
    pool = FakePool(tmp_path)
    first = pool.nodes["first"]
    first.update(
        active_model=None,
        resident_models=[
            {
                "model_id": "org/model",
                "gpu_ids": ["gpu-1"],
                "budget_gib": {"gpu-1": 2.5},
                "active_requests": 1,
                "status": "ready",
            }
        ],
    )
    first["models"][0]["loaded"] = True
    state = await pool.inventory(force=True)
    placement = state["models"][0]["placements"][0]
    assert placement["gpu_ids"] == ["gpu-1"]
    assert placement["budget_gib"] == {"gpu-1": 2.5}
    assert placement["active_requests"] == 1


def test_saved_memory_fractions_migrate_to_fixed_but_new_requests_use_auto(tmp_path):
    path = tmp_path / "model-pool.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "org/model": {
                        "last_settings": {
                            "model_id": "org/model",
                            "served_model_name": "model",
                            "gpu_memory_utilization": 0.45,
                        }
                    }
                }
            }
        )
    )
    pool = FakePool(tmp_path)
    assert pool.preferences["org/model"].last_settings.memory_budget_mode == "fixed"
    assert (
        ModelLaunchRequest(model_id="org/new", served_model_name="new").memory_budget_mode == "auto"
    )


def test_auto_budget_covers_engine_concurrency_and_replicated_parallel_kv():
    settings = Settings(active_hardware="nvidia-gpu")
    launch = ModelLaunchRequest(
        model_id="org/model", served_model_name="model", tensor_parallel=2, pipeline_parallel=2
    )
    fit = SimpleNamespace(resident_weights_gib=16, resident_kv_cache_gib=8)
    resources = PoolNode.resources_for_fit(launch, fit, settings)
    assert resources.gpu_count == 4
    assert resources.required_gib_per_gpu == 10  # 4 weights + 4 replicated KV + 2 workspace.
