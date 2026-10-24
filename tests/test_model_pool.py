import asyncio
from types import SimpleNamespace

import pytest

from spark_cluster.config import Settings
from spark_cluster.model_pool import ModelPool, ModelPreferences, PoolLoadRequest
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
