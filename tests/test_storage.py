import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from spark_cluster.app import create_app
from spark_cluster.config import Settings
from spark_cluster.storage import StorageService, StorageShortage, measure
from spark_cluster.store import SnapshotStore


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "state.db",
        setup_state_path=tmp_path / "setup",
        model_cache_path=tmp_path / "models",
        model_admin_token="storage-test",
    )
    store = SnapshotStore(settings.database_path)
    cluster = SimpleNamespace(
        id="local-aaaaaaaaaaaa", name="Node A", kind="local-docker", host_address="node-a"
    )
    models = SimpleNamespace(
        _invalidate_inventory=Mock(),
        _inventory_guard=threading.RLock(),
        _local_fit_cache={},
        runtime_endpoint=AsyncMock(return_value=SimpleNamespace(loaded_model_ids=[])),
        safe_error=str,
    )
    node = SimpleNamespace(
        storage_lock=asyncio.Lock(),
        local_cluster=lambda: cluster,
        models=models,
        demand=SimpleNamespace(residents={}, profiles=[], dynamic=set(), persist=Mock()),
        native={},
        deployments=SimpleNamespace(launches=AsyncMock(return_value=[])),
        machine_inventory=lambda: [{"id": "node-a", "name": "Node A"}],
    )
    registry = SimpleNamespace(list=lambda: [cluster], get=lambda _: cluster)
    library = SimpleNamespace(docker=SimpleNamespace(available=False))
    return StorageService(settings, store, registry, node, library)


def weights(service, repo="owner/model", revision="a" * 40):
    path = service.settings.model_cache_path / repo
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"weights" * 1000)
    (path / ".spark-cluster.json").write_text(json.dumps({"repo_id": repo, "revision": revision}))
    return path


@pytest.mark.asyncio
async def test_inventory_counts_disk_files_without_database_history_and_groups_filesystems(setup):
    path = weights(setup)
    hf = Path(__import__("os").environ["HF_HOME"]) / "hub/models--owner--cached/blobs"
    hf.mkdir(parents=True)
    (hf / "blob").write_bytes(b"x" * 4096)
    state = await setup.local(fresh=True)
    assert {a["model_id"] for a in state["assets"]} == {"owner/model", "owner/cached"}
    assert len(state["locations"]) == 1  # same physical filesystem, never double free-space totals
    assert state["locations"][0]["free_bytes"] > 0
    assert next(a for a in state["assets"] if a["model_id"] == "owner/model")["path"] == str(path)


@pytest.mark.asyncio
async def test_deletion_is_scoped_and_preserves_recipe_definition_and_other_weights(setup):
    path = weights(setup)
    other = weights(setup, "owner/other")
    recipe = {"id": "publisher/recipe", "version": "1.0.0", "model": {"repository": "owner/model"}}
    setup.store.save_runtime_object(
        "recipe-install",
        {
            "id": "install",
            "recipe": recipe,
            "cluster_id": "local-aaaaaaaaaaaa",
            "status": "completed",
        },
    )
    state = await setup.inventory(fresh=True)
    asset = next(a for a in state["assets"] if a["model_id"] == "owner/model")
    assert asset["recipe_refs"] == [{"id": "publisher/recipe", "version": "1.0.0"}]
    result = await setup.delete([asset["id"]])
    assert result["deleted"] == [asset["id"]] and not result["errors"]
    assert not path.exists() and other.exists()
    record = setup.store.runtime_objects("recipe-install")[0]
    assert record["recipe"] == recipe and record["artifacts_present"] is False


@pytest.mark.asyncio
async def test_running_and_changed_models_cannot_be_deleted(setup):
    path = weights(setup)
    entry = (await setup.local(fresh=True))["assets"][0]
    setup.node.demand.residents["model"] = SimpleNamespace(
        profile=SimpleNamespace(model_id="owner/model", model="model")
    )
    with pytest.raises(ValueError, match="In use"):
        await setup.delete_local([entry["id"]])
    assert path.exists()
    setup.node.demand.residents.clear()
    (path / "new.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        await setup.delete_local([entry["id"]])
    assert path.exists()


@pytest.mark.asyncio
async def test_selected_batch_is_validated_before_any_delete_and_symlinks_stay_outside(
    setup, tmp_path
):
    path = weights(setup)
    entry = (await setup.local(fresh=True))["assets"][0]
    with pytest.raises(ValueError):
        await setup.delete_local([entry["id"], "invalid-asset"])
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(b"keep")
    (setup.settings.model_cache_path / "symlink").symlink_to(outside, target_is_directory=True)
    assert all(a["path"] != str(outside) for a in (await setup.local(fresh=True))["assets"])
    assert path.exists() and (outside / "model.safetensors").exists()


def test_reclaimable_estimate_does_not_count_external_hardlinks(setup, tmp_path):
    path = weights(setup)
    before = measure(path)[1]
    import os

    os.link(path / "model.safetensors", tmp_path / "external-link")
    assert measure(path)[1] < before


@pytest.mark.asyncio
async def test_structured_shortfall_excludes_target_model_and_rechecks_after_cleanup(setup):
    weights(setup)
    state = await setup.inventory(fresh=True)
    location = state["locations"][0]
    needed = location["total_bytes"] + 1
    with pytest.raises(StorageShortage) as failure:
        await setup.require({location["id"]: needed}, ["target/model"])
    detail = failure.value.detail()
    assert detail["code"] == "insufficient_storage"
    assert detail["storage"]["requirements"][0]["missing_bytes"] > 0
    assert detail["storage"]["excluded_model_ids"] == ["target/model"]
    await setup.require({location["id"]: 0})


@pytest.mark.asyncio
async def test_remote_node_deletion_does_not_delete_controller_copy(setup, tmp_path):
    local_path = weights(setup)
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    remote_asset = {
        "id": "remote-files",
        "model_id": "owner/model",
        "location_id": "remote-disk",
        "kind": "weights",
        "size_bytes": 500,
        "reclaimable_bytes": 500,
        "blockers": [],
    }
    remote_state = {
        "locations": [
            {
                "id": "remote-disk",
                "name": "Shared weights",
                "free_bytes": 1000,
                "node_ids": ["spark-1", "spark-2"],
                "node_names": ["Spark 1", "Spark 2"],
                "scope": "shared",
            }
        ],
        "assets": [remote_asset],
        "errors": [],
        "model_location_id": "remote-disk",
    }
    remote = SimpleNamespace(
        id="remote-bbbbbbbbbbbb", name="Spark pair", kind="remote-bridge", endpoint="http://remote"
    )
    setup.registry.list = lambda: [setup.node.local_cluster(), remote]
    setup.registry.get = lambda _: remote
    requests = []

    def handler(request):
        assert request.headers.get("x-spark-signature")
        requests.append(request)
        if request.method == "POST":
            assert json.loads(request.content)["asset_ids"] == ["remote-files"]
            return httpx.Response(200, json={"deleted": ["remote-files"]})
        return httpx.Response(200, json=remote_state)

    setup.transport = httpx.MockTransport(handler)
    state = await setup.inventory(fresh=True)
    assert any(location["scope"] == "shared" for location in state["locations"])
    result = await setup.delete([remote.id + ":remote-files"])
    assert result["deleted"] == [remote.id + ":remote-files"]
    assert local_path.exists()


@pytest.mark.asyncio
async def test_docker_recipe_artifacts_are_explicit_and_container_references_are_protected(setup):
    weights(setup)
    image = "docker.io/publisher/runtime@sha256:" + "a" * 64
    recipe = {
        "id": "publisher/recipe",
        "version": "1.0.0",
        "model": {"repository": "owner/model"},
        "runtime": {"image": image},
    }
    setup.store.save_runtime_object(
        "recipe-install", {"id": "recipe", "recipe": recipe, "status": "completed"}
    )

    def response(method, path, **kwargs):
        if path == "/system/df":
            return httpx.Response(
                200,
                json={
                    "Images": [
                        {
                            "Id": "sha256:configuration",
                            "RepoDigests": [image.removeprefix("docker.io/")],
                            "Size": 1000,
                            "SharedSize": 400,
                        }
                    ],
                    "Volumes": [],
                },
            )
        return httpx.Response(
            200,
            json=[
                {
                    "ImageID": "sha256:configuration",
                    "State": "running",
                    "Labels": {"spark-cluster.model-id": "owner/model"},
                }
            ],
        )

    setup.library.docker = SimpleNamespace(available=True, request=AsyncMock(side_effect=response))
    state = await setup.local(fresh=True)
    runtime = next(a for a in state["assets"] if a["kind"] == "runtime-image")
    assert runtime["reclaimable_bytes"] == 600 and runtime["blockers"]
    assert next(a for a in state["assets"] if a["kind"] == "weights")["blockers"]


@pytest.mark.asyncio
async def test_api_requires_token_for_deletion_and_preserves_storage_shortage_payload(setup):
    app = create_app(setup.settings)
    app.state.storage.local = AsyncMock(return_value={"assets": [], "locations": []})
    app.state.storage.download_check = AsyncMock(
        side_effect=StorageShortage(
            [
                {
                    "location_id": "controller:disk",
                    "required_bytes": 100,
                    "available_bytes": 1,
                    "missing_bytes": 99,
                    "name": "Controller",
                }
            ],
            ["owner/model"],
        )
    )
    from spark_cluster.models import ModelFitPlan

    fit = ModelFitPlan(
        repo_id="owner/model",
        revision="main",
        repository_bytes=100,
        weight_bytes=100,
        weights_gib=1,
        kv_cache_gib=0,
        runtime_overhead_gib=1,
        total_required_gib=2,
        cluster_nodes=1,
        usable_per_node_gib=100,
        usable_cluster_gib=100,
        minimum_nodes=1,
        headroom_gib=98,
        memory_fits=True,
        storage_available_gib=0,
        storage_fits=False,
        fits=False,
        confidence="repository",
    )
    app.state.model_manager.fit = AsyncMock(return_value=fit)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (
            await client.post("/api/v1/storage/delete", json={"asset_ids": ["x"]})
        ).status_code == 401
        response = await client.post(
            "/api/v1/models/download",
            json={"repo_id": "owner/model"},
            headers={"Authorization": "Bearer storage-test"},
        )
        assert response.status_code == 507
        assert response.json()["detail"]["storage"]["requirements"][0]["missing_bytes"] == 99
        assert app.state.runtime.store.list_downloads() == []


@pytest.mark.asyncio
async def test_legacy_runtime_without_recipe_still_protects_its_weights(setup):
    weights(setup)

    def response(method, path, **kwargs):
        if path == "/system/df":
            return httpx.Response(200, json={"Images": [], "Volumes": []})
        return httpx.Response(
            200, json=[{"State": "running", "Labels": {"spark-cluster.model-id": "owner/model"}}]
        )

    setup.library.docker = SimpleNamespace(available=True, request=AsyncMock(side_effect=response))
    state = await setup.local(fresh=True)
    assert state["assets"][0]["blockers"]


@pytest.mark.asyncio
async def test_partial_cleanup_reports_deleted_files_and_separate_failures(setup):
    first = weights(setup, "owner/first")
    weights(setup, "owner/second")
    state = await setup.local(fresh=True)
    assets = sorted(state["assets"], key=lambda a: a["model_id"])
    import shutil

    actual = shutil.rmtree

    def remove(path):
        if path.name == "second":
            raise PermissionError("second model could not be removed")
        actual(path)

    from unittest.mock import patch

    with patch("spark_cluster.storage.shutil.rmtree", side_effect=remove):
        result = await setup.delete_local([a["id"] for a in assets])
    assert result["deleted"] == [assets[0]["id"]]
    assert result["errors"] and not first.exists()


@pytest.mark.asyncio
async def test_recipe_space_rejection_does_not_create_installation_job(setup):
    from spark_cluster.recipe_install import RecipeInstaller
    from spark_cluster.recipes import bundled_recipes

    recipe = bundled_recipes()[0]
    catalog = SimpleNamespace(
        get=AsyncMock(return_value={"recipe": recipe.model_dump(mode="json"), "origin": {}}),
        compatible=AsyncMock(return_value=[]),
    )
    download = AsyncMock()
    installer = RecipeInstaller(
        catalog,
        setup.registry,
        setup.store,
        setup.library,
        SimpleNamespace(),
        download,
        AsyncMock(),
        str,
    )
    installer.preflight = AsyncMock(
        side_effect=StorageShortage(
            [
                {
                    "location_id": "node:disk",
                    "required_bytes": 100,
                    "available_bytes": 1,
                    "missing_bytes": 99,
                    "name": "Node",
                }
            ]
        )
    )
    with pytest.raises(StorageShortage):
        await installer.create(recipe.content_hash, "local-aaaaaaaaaaaa")
    assert setup.store.runtime_objects("recipe-install") == []
    assert not installer.tasks
    download.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_disk_scan_returns_pending_without_blocking_inventory(setup):
    import threading

    weights(setup)
    release = threading.Event()
    original = setup._scan

    def slow_scan():
        release.wait(3)
        return original()

    setup._scan = slow_scan
    try:
        pending = await asyncio.wait_for(setup.local(fresh=True), timeout=1)
        assert pending["scanning"] is True
        assert pending["assets"] == []
        assert pending["locations"][0]["free_bytes"] > 0
    finally:
        release.set()
    await asyncio.to_thread(setup.scan_future.result, 2)
    ready = await setup.local(fresh=True)
    assert ready["scanning"] is False
    assert len(ready["assets"]) == 1
