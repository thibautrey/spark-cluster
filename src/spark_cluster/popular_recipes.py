"""Offline, immutable popular-model serving templates and explicitly non-executable references."""

from collections import Counter
from functools import lru_cache
from typing import Literal

from pydantic import Field

from .recipes import SparkRecipe, StrictModel, bundled_recipes, recipe_directory, version_tuple


class CatalogEntry(StrictModel):
    recipe: SparkRecipe
    origin: dict


class CatalogSnapshot(StrictModel):
    schema_version: Literal["spark-cluster.recipe-catalog/v1"]
    captured_at: str
    ranking: str
    queries: list[str]
    entries: list[CatalogEntry] = Field(max_length=10000)


class WorkloadReference(StrictModel):
    id: str = Field(pattern=r"^[a-f0-9]{64}$")
    model_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    task: str
    library: str
    license: str
    downloads: int = Field(ge=0)
    likes: int = Field(ge=0)
    trending_score: float = Field(ge=0)
    rank_score: float = Field(ge=0)
    model_card: str
    status: Literal["reference-only"]
    blocker: str


class ReferenceSnapshot(StrictModel):
    schema_version: Literal["spark-cluster.recipe-references/v1"]
    captured_at: str
    references: list[WorkloadReference] = Field(max_length=10000)
    upstream_libraries: list[dict]


def read_snapshot(path, model):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Bundled catalog exceeds 32 MiB")
    return model.model_validate_json(path.read_bytes())


@lru_cache(maxsize=1)
def snapshots() -> tuple[CatalogSnapshot, ...]:
    return tuple(
        read_snapshot(p, CatalogSnapshot)
        for p in sorted((recipe_directory() / "popular").glob("*.catalog.json"))
    )


@lru_cache(maxsize=1)
def reference_snapshots() -> tuple[ReferenceSnapshot, ...]:
    return tuple(
        read_snapshot(p, ReferenceSnapshot)
        for p in sorted((recipe_directory() / "popular").glob("*.references.json"))
    )


def bundled_catalog():
    entries = [(recipe, {"kind": "bundled"}) for recipe in bundled_recipes()]
    for snapshot in snapshots():
        entries.extend((e.recipe, e.origin) for e in snapshot.entries)
    return entries


def catalog_metadata():
    sources = {}
    for snapshot in reference_snapshots():
        for source in snapshot.upstream_libraries:
            sources[(source["repository"], source["commit"])] = source
    return {
        "snapshots": [
            {
                "captured_at": s.captured_at,
                "ranking": s.ranking,
                "entries": len(s.entries),
                "queries": s.queries,
            }
            for s in snapshots()
        ],
        "reference_count": len(references()),
        "upstream_libraries": list(sources.values()),
    }


def references():
    # Preserve old snapshots on disk; the browser shows the newest record per exact checkpoint.
    values = {}
    for snapshot in sorted(reference_snapshots(), key=lambda s: s.captured_at):
        for item in snapshot.references:
            values[item.id] = item.model_dump(mode="json")
    return list(values.values())


async def catalog_page(
    catalog, target, *, q="", category="all", sort="popular", offset=0, limit=24, installable=False
):
    entries = await catalog.entries()
    groups = {}
    for entry in entries:
        groups.setdefault(entry["recipe"]["id"], []).append(entry)
    for versions in groups.values():
        versions.sort(key=lambda e: version_tuple(e["recipe"]["version"]), reverse=True)
    newest = [versions[0] for versions in groups.values()]
    categories = Counter(e["origin"].get("category", "curated") for e in newest)
    refs = references()
    ref_categories = Counter(e["task"] for e in refs)
    query = q.casefold().strip()
    filtered = []
    for versions in groups.values():
        entry = versions[0]
        recipe = entry["recipe"]
        if category != "all" and entry["origin"].get("category", "curated") != category:
            continue
        if (
            query
            and query
            not in " ".join(
                [
                    recipe["id"],
                    recipe["title"],
                    recipe["model"]["repository"],
                    entry["origin"].get("task", ""),
                ]
            ).casefold()
        ):
            continue
        blockers = await catalog.compatible(SparkRecipe.model_validate(recipe), target)
        if installable and blockers:
            continue
        filtered.append(versions)
    if sort == "name":
        filtered.sort(key=lambda v: v[0]["recipe"]["title"].casefold())
    else:
        field = {"downloads": "downloads", "trending": "trending_score"}.get(sort, "rank_score")
        filtered.sort(key=lambda v: (-float(v[0]["origin"].get(field, 0)), v[0]["recipe"]["id"]))
    page = []
    for versions in filtered[offset : offset + limit]:
        for entry in versions:
            blockers = await catalog.compatible(SparkRecipe.model_validate(entry["recipe"]), target)
            page.append({**entry, "blockers": blockers})
    reference_page = []
    if category in ref_categories:
        matching = [
            r
            for r in refs
            if not installable
            and r["task"] == category
            and (not query or query in r["model_id"].casefold())
        ]
        key = {"downloads": "downloads", "trending": "trending_score"}.get(sort, "rank_score")
        matching.sort(
            key=(lambda r: r["model_id"].casefold())
            if sort == "name"
            else lambda r: (-r[key], r["model_id"])
        )
        reference_page = matching[offset : offset + limit]
        total = len(matching)
    else:
        total = len(filtered)
    return {
        "recipes": page,
        "references": reference_page,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
        "catalog_total": len(groups),
        "release_total": len(entries),
        "generated_total": sum(e["recipe"]["validation"] == "generated-template" for e in newest),
        "categories": dict(sorted(categories.items())),
        "reference_categories": dict(sorted(ref_categories.items())),
        "catalog": catalog_metadata(),
    }
