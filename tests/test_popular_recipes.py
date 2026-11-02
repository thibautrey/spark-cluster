import httpx
import pytest

from spark_cluster.cluster_registry import ClusterTarget
from spark_cluster.config import Settings
from spark_cluster.popular_recipes import catalog_page, references, snapshots
from spark_cluster.recipe_probe import probe_recipe
from spark_cluster.recipes import RecipeCatalog, SparkRecipe
from spark_cluster.store import SnapshotStore


def test_catalog_has_over_one_thousand_distinct_pinned_serving_templates():
    items = [entry for snapshot in snapshots() for entry in snapshot.entries]
    assert len(items) >= 1000
    assert len({e.recipe.model.repository for e in items}) == len(items)
    assert len({e.recipe.id for e in items}) == len(items)
    assert {e.origin["category"] for e in items} >= {
        "text",
        "vision",
        "embeddings",
        "classification",
        "speech",
    }
    for entry in items:
        r = entry.recipe
        assert r.validation == "generated-template"
        assert len(r.model.revision) == 40
        assert "@sha256:" in r.runtime.image
        assert entry.origin["library"] in {"transformers", "sentence-transformers", "vllm"}
        assert entry.origin["weight_format"] == "safetensors"
        assert not any(
            marker in r.model.repository.casefold()
            for marker in ["-mlx", "mlx-", "-gguf", "exl2", "exl3"]
        )
        assert entry.origin["architectures"]
        assert entry.origin["query_ranks"]
        assert entry.origin["safetensors"]["parameters"]
        assert r.model.revision in entry.origin["model_card"]
        assert r.model.revision in entry.origin["metadata_url"]
        assert "--trust-remote-code" not in r.runtime.arguments
        if entry.origin["category"] in {"classification", "embeddings"}:
            assert "--runner" in r.runtime.arguments
    refs = references()
    assert len(refs) >= 100
    assert {r["task"] for r in refs} >= {"text-to-image", "text-to-video", "text-to-speech"}
    assert all(r["status"] == "reference-only" for r in refs)
    assert not any("runtime" in r or "launch" in r for r in refs)


@pytest.fixture
def catalog(tmp_path):
    settings = Settings(
        database_path=tmp_path / "state.db",
        setup_state_path=tmp_path / "setup",
        model_cache_path=tmp_path / "models",
    )
    return RecipeCatalog(settings, SnapshotStore(settings.database_path))


@pytest.mark.asyncio
async def test_paginated_catalog_search_filters_and_history_survive_restart(catalog):
    first = await catalog_page(catalog, None, limit=7)
    second = await catalog_page(catalog, None, offset=7, limit=7)
    assert first["catalog_total"] >= 1000 and first["generated_total"] >= 1000
    assert len(first["recipes"]) == len(second["recipes"]) == 7
    assert not ({e["id"] for e in first["recipes"]} & {e["id"] for e in second["recipes"]})
    embedding = await catalog_page(catalog, None, category="embeddings", sort="downloads", limit=10)
    assert all(e["origin"]["category"] == "embeddings" for e in embedding["recipes"])
    assert [e["origin"]["downloads"] for e in embedding["recipes"]] == sorted(
        [e["origin"]["downloads"] for e in embedding["recipes"]], reverse=True
    )
    chosen = SparkRecipe.model_validate(embedding["recipes"][0]["recipe"])
    newer = chosen.model_copy(update={"version": "1.10.0"})
    await catalog.remember(newer, {"kind": "imported"})
    found = await catalog_page(catalog, None, q=chosen.model.repository, limit=1)
    assert [e["recipe"]["version"] for e in found["recipes"]] == ["1.10.0", "1.0.0"]
    reopened = RecipeCatalog(catalog.settings, catalog.store)
    assert (await reopened.get(newer.content_hash))["recipe"]["model"][
        "revision"
    ] == chosen.model.revision
    changed = chosen.model_copy(update={"description": "quietly rewritten release"})
    with pytest.raises(ValueError, match="Immutable"):
        await reopened.remember(changed, {"kind": "imported"})


@pytest.mark.asyncio
async def test_references_are_not_installable_and_target_filter_fails_closed(catalog):
    page = await catalog_page(catalog, None, category="text-to-image", limit=5)
    assert page["recipes"] == [] and len(page["references"]) == 5
    with pytest.raises(KeyError):
        await catalog.get(page["references"][0]["id"])
    target = ClusterTarget(id="remote", name="Spark", kind="remote-bridge", hardware="dgx-spark")
    compatible = await catalog_page(catalog, target, installable=True)
    assert compatible["total"] == 0
    unfiltered = await catalog_page(catalog, target)
    assert all(e["blockers"] for e in unfiltered["recipes"])


@pytest.mark.asyncio
async def test_bulk_bootstrap_is_single_atomic_write_and_idempotent(catalog, monkeypatch):
    saved = []
    actual = catalog.store.save_runtime_objects

    def record(kind, values):
        saved.append(len(values))
        actual(kind, values)

    monkeypatch.setattr(catalog.store, "save_runtime_objects", record)
    entries = await catalog.entries()
    assert saved == [len(entries)]
    await catalog.entries()
    assert len(saved) == 1
    reopened = RecipeCatalog(catalog.settings, catalog.store)
    await reopened.entries()
    assert saved[-1] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category, response, endpoint",
    [
        ("embeddings", {"data": [{"embedding": [0.1, 0.2]}]}, "v1/embeddings"),
        ("classification", {"data": [{"probs": [0.2, 0.8]}]}, "classify"),
        ("speech", {"text": ""}, "v1/audio/transcriptions"),
        ("text", {"choices": [{"text": " Paris"}]}, "v1/completions"),
        ("vision", {"choices": [{"message": {"content": "OK"}}]}, "v1/chat/completions"),
    ],
)
async def test_task_specific_probes_use_actual_endpoint_contract(category, response, endpoint):
    entry = next(e for s in snapshots() for e in s.entries if e.origin["category"] == category)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        await probe_recipe(client, entry.recipe, {}, task=entry.origin["task"])
    assert seen[0].url.path.endswith(endpoint)
    if category == "speech":
        assert seen[0].headers["content-type"].startswith("multipart/form-data")
        assert b"RIFF" in seen[0].content


@pytest.mark.asyncio
async def test_probe_rejects_empty_embeddings_instead_of_claiming_success():
    entry = next(e for s in snapshots() for e in s.entries if e.origin["category"] == "embeddings")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(ValueError, match="embedding"):
            await probe_recipe(client, entry.recipe, {})


@pytest.mark.asyncio
async def test_withdrawn_release_remains_addressable_but_cannot_be_installed(catalog, monkeypatch):
    from spark_cluster import popular_recipes

    entries = await catalog.entries()
    entry = entries[0]
    recipe = SparkRecipe.model_validate(entry["recipe"])
    withdrawal = popular_recipes.RecipeWithdrawal(
        hash=entry["id"], model_id=recipe.model.repository, reason="Withdrawn incompatible format"
    )
    monkeypatch.setattr(popular_recipes, "withdrawals", lambda: {entry["id"]: withdrawal})
    assert not any(e["id"] == entry["id"] for e in await catalog.entries())
    assert (await catalog.get(entry["id"]))["recipe"] == entry["recipe"]
    assert "Withdrawn incompatible format" in await catalog.compatible(recipe, None)
