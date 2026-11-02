"""Spark Cluster Recipe v1: strict files, immutable releases, cached GitHub discovery."""

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .models import ModelLaunchRequest, RecipeRuntime

SEMVER = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
COMMIT = r"^[a-f0-9]{40}$"
MAX_FILE_BYTES = 128 * 1024


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(map(int, value.split(".")))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecipeSource(StrictModel):
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    commit: str = Field(pattern=COMMIT)
    path: str = Field(min_length=1, max_length=300)
    license: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def safe_path(self):
        if self.path.startswith("/") or ".." in PurePosixPath(self.path).parts:
            raise ValueError("Source must use a repository-relative path")
        return self


class RecipeModel(StrictModel):
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=COMMIT)
    license: str = Field(min_length=1, max_length=100)


class RecipeHardware(StrictModel):
    hardware: Literal["dgx-spark"] = "dgx-spark"
    platform: Literal["linux/arm64"] = "linux/arm64"
    nodes: int = Field(ge=1, le=64)
    memory_gib_per_node: float = Field(gt=0, le=1024)
    disk_gib_per_node: float = Field(gt=0, le=16384)
    # v1 only uses the controller's existing Docker / Ray topology.
    topology: Literal["single", "ray"] = "single"


class RecipeLaunch(ModelLaunchRequest):
    model_config = ConfigDict(extra="forbid")


class SparkRecipe(StrictModel):
    schema_version: Literal["spark-cluster.recipe/v1"]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*$")
    version: str = Field(pattern=SEMVER)
    controller_min_version: str = Field(default="0.1.0", pattern=SEMVER)
    title: str = Field(min_length=1, max_length=150)
    description: str = Field(min_length=1, max_length=2000)
    license: str = Field(min_length=1, max_length=100)
    source: RecipeSource
    model: RecipeModel
    requirements: RecipeHardware
    runtime: RecipeRuntime
    launch: RecipeLaunch
    validation: Literal["source-reviewed", "hardware-verified", "generated-template"] = (
        "source-reviewed"
    )
    notes: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def consistent_contract(self):
        if self.launch.model_id != self.model.repository:
            raise ValueError("Launch model must equal the pinned model repository")
        if self.launch.engine_id != "vllm":
            raise ValueError("Recipe v1 supports the vLLM adapter")
        if self.launch.recipe_runtime or self.launch.recipe_revision:
            raise ValueError("Recipe runtime and revision belong at the top level")
        ranks = self.launch.tensor_parallel * self.launch.pipeline_parallel
        if ranks != self.requirements.nodes:
            raise ValueError("Recipe nodes must equal tensor_parallel × pipeline_parallel")
        if (ranks == 1) != (self.requirements.topology == "single"):
            raise ValueError("Multi-node recipes must use the Ray topology")
        if len(self.runtime.arguments) and not self.runtime.arguments[0].startswith("--"):
            raise ValueError("Recipe arguments must start with an option")
        return self

    @property
    def content_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def launch_request(self) -> ModelLaunchRequest:
        return self.launch.model_copy(
            update={
                "recipe_runtime": self.runtime,
                "recipe_revision": self.model.revision,
            }
        )


def read_recipe(content: bytes | str) -> SparkRecipe:
    if len(content.encode() if isinstance(content, str) else content) > MAX_FILE_BYTES:
        raise ValueError("Recipe exceeds 128 KiB")
    return SparkRecipe.model_validate_json(content)


def bundled_recipes() -> list[SparkRecipe]:
    path = Path(__file__).parent / "recipes"
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / "recipes"
    return [read_recipe(p.read_bytes()) for p in sorted(path.glob("*.spark-recipe.json"))]


def recipe_directory() -> Path:
    packaged = Path(__file__).parent / "recipes"
    return packaged if packaged.exists() else Path(__file__).resolve().parents[2] / "recipes"


class RecipeCatalog:
    """Discovery never runs code. SQLite retains releases and HTTP cache across restarts."""

    def __init__(self, settings, store, transport=None):
        self.settings, self.store, self.transport = settings, store, transport
        self.lock = asyncio.Lock()
        self.refresh_task = None
        self.release_lock = asyncio.Lock()
        self.bundled_loaded = False

    def start(self):
        if not self.refresh_task or self.refresh_task.done():
            self.refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self):
        while True:
            try:
                await self.discover()
            except Exception:
                pass  # Discovery is optional; persisted/bundled recipes remain usable.
            await asyncio.sleep(3600)

    async def stop(self):
        if self.refresh_task:
            self.refresh_task.cancel()
            await asyncio.gather(self.refresh_task, return_exceptions=True)

    async def discovery_state(self):
        values = await asyncio.to_thread(self.store.runtime_objects, "recipe-discovery")
        return values[0] if values else {"found": 0, "errors": [], "checked_at": None}

    async def remember(self, recipe: SparkRecipe, origin: dict) -> dict:
        async with self.release_lock:
            return await self._remember(recipe, origin)

    async def _remember(self, recipe: SparkRecipe, origin: dict) -> dict:
        releases = await asyncio.to_thread(self.store.runtime_objects, "recipe")
        for item in releases:
            old = item["recipe"]
            if old["id"] == recipe.id and old["version"] == recipe.version:
                if item["id"] != recipe.content_hash:
                    raise ValueError(
                        f"Immutable recipe {recipe.id}@{recipe.version} changed content"
                    )
                return item
        value = {
            "id": recipe.content_hash,
            "recipe": recipe.model_dump(mode="json"),
            "origin": origin,
            "discovered_at": time.time(),
        }
        await asyncio.to_thread(self.store.save_runtime_object, "recipe", value)
        return value

    async def entries(self, *, include_withdrawn=False) -> list[dict]:
        if not self.bundled_loaded:
            async with self.release_lock:
                if not self.bundled_loaded:
                    from .popular_recipes import bundled_catalog

                    releases = await asyncio.to_thread(self.store.runtime_objects, "recipe")
                    known = {(e["recipe"]["id"], e["recipe"]["version"]): e for e in releases}
                    incoming = await asyncio.to_thread(bundled_catalog)
                    batch = []
                    for recipe, origin in incoming:
                        identity = recipe.id, recipe.version
                        old = known.get(identity)
                        digest = recipe.content_hash
                        if old and old["id"] != digest:
                            raise ValueError(
                                f"Immutable recipe {recipe.id}@{recipe.version} changed content"
                            )
                        value = {
                            "id": digest,
                            "recipe": recipe.model_dump(mode="json"),
                            "origin": origin,
                            "discovered_at": time.time(),
                        }
                        if not old or old.get("origin") != origin:
                            batch.append(value)
                        known[identity] = value
                    await asyncio.to_thread(self.store.save_runtime_objects, "recipe", batch)
                    self.bundled_loaded = True
        entries = await asyncio.to_thread(self.store.runtime_objects, "recipe")
        if not include_withdrawn:
            from .popular_recipes import withdrawals

            excluded = withdrawals()
            entries = [e for e in entries if e["id"] not in excluded]
        return sorted(
            entries,
            key=lambda e: (e["recipe"]["id"], version_tuple(e["recipe"]["version"])),
            reverse=True,
        )

    async def get(self, digest: str) -> dict:
        value = next(
            (item for item in await self.entries(include_withdrawn=True) if item["id"] == digest),
            None,
        )
        if not value:
            raise KeyError("Recipe release not found")
        return value

    async def _cached_get(self, client, path: str, *, force=False):
        key = hashlib.sha256(path.encode()).hexdigest()
        cached = next(
            (
                v
                for v in await asyncio.to_thread(self.store.runtime_objects, "recipe-http")
                if v["id"] == key
            ),
            None,
        )
        if cached and not force and time.time() - cached["checked_at"] < 3600:
            return cached["body"]
        headers = {"If-None-Match": cached["etag"]} if cached and cached.get("etag") else {}
        response = await client.get("https://api.github.com" + path, headers=headers)
        if response.status_code == 304 and cached:
            value = {**cached, "checked_at": time.time()}
        else:
            response.raise_for_status()
            if len(response.content) > 4 * 1024 * 1024:
                raise ValueError("GitHub response exceeds discovery limit")
            value = {
                "id": key,
                "body": response.json(),
                "etag": response.headers.get("etag"),
                "checked_at": time.time(),
            }
        await asyncio.to_thread(self.store.save_runtime_object, "recipe-http", value)
        return value["body"]

    async def discover(self, *, force=False) -> dict:
        async with self.lock:
            prior = await asyncio.to_thread(self.store.runtime_objects, "recipe-discovery")
            if prior and not force and time.time() - prior[0]["checked_at"] < 3600:
                return prior[0]
            result = {
                "id": "github",
                "checked_at": time.time(),
                "errors": [],
                "found": 0,
                "topic": "spark-cluster-recipes",
                "stale": False,
            }
            headers = {"Accept": "application/vnd.github+json", "User-Agent": "spark-cluster"}
            if self.settings.recipes_github_token:
                headers["Authorization"] = f"Bearer {self.settings.recipes_github_token}"
            async with httpx.AsyncClient(
                transport=self.transport, timeout=20, headers=headers, follow_redirects=False
            ) as client:
                try:
                    # Bounded pagination. Publishers need no central registry registration.
                    repos = []
                    for page in range(1, 4):
                        body = await self._cached_get(
                            client,
                            f"/search/repositories?q=topic:spark-cluster-recipes&per_page=30&page={page}",
                            force=force,
                        )
                        repos.extend(body.get("items", []))
                        if len(body.get("items", [])) < 30:
                            break
                    for repo in repos:
                        name = repo.get("full_name", "")
                        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
                            continue
                        try:
                            ref = await self._cached_get(
                                client, f"/repos/{name}/commits/HEAD", force=force
                            )
                            sha = ref["sha"]
                            if not re.fullmatch(COMMIT, sha):
                                raise ValueError("Invalid repository commit")
                            # Read a small explicit index rather than executing/cloning a repo.
                            index = await self._content(
                                client, name, sha, "spark-cluster-recipes.json"
                            )
                            if index.get("schema_version") != "spark-cluster.recipe-index/v1":
                                raise ValueError("Unsupported recipe index version")
                            paths = index.get("recipes", [])
                            if not isinstance(paths, list) or len(paths) > 40:
                                raise ValueError("Recipe index must contain at most 40 paths")
                            for path in paths:
                                recipe = SparkRecipe.model_validate(
                                    await self._content(client, name, sha, path)
                                )
                                if recipe.id.split("/")[0] != name.split("/")[0].lower():
                                    raise ValueError("Recipe namespace must match its GitHub owner")
                                await self.remember(
                                    recipe,
                                    {
                                        "kind": "github",
                                        "repository": name,
                                        "commit": sha,
                                        "path": path,
                                    },
                                )
                                result["found"] += 1
                        except (httpx.HTTPError, ValueError, KeyError, TypeError):
                            result["errors"].append(f"Could not validate recipes in {name}")
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    result["errors"].append(
                        "GitHub unavailable or rate limited; cached recipes retained"
                    )
                    result["stale"] = True
            await asyncio.to_thread(self.store.save_runtime_object, "recipe-discovery", result)
            return result

    async def _content(self, client, repository: str, sha: str, path: str):
        import base64
        from urllib.parse import quote

        if (
            not isinstance(path, str)
            or not path.endswith(".json")
            or len(path) > 300
            or path.startswith("/")
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError("Invalid recipe path")
        body = await self._cached_get(
            client, f"/repos/{repository}/contents/{quote(path, safe='/')}?ref={sha}"
        )
        if body.get("type") != "file" or body.get("size", 0) > MAX_FILE_BYTES:
            raise ValueError("Invalid recipe file")
        content = base64.b64decode(body["content"])
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("Recipe exceeds 128 KiB")
        return json.loads(content)

    async def compatible(self, recipe: SparkRecipe, target) -> list[str]:
        blockers = []
        from .popular_recipes import withdrawal_reason

        withdrawn = withdrawal_reason(recipe)
        if withdrawn:
            blockers.append(withdrawn)
        if version_tuple(__version__) < version_tuple(recipe.controller_min_version):
            blockers.append(f"Requires Spark Cluster {recipe.controller_min_version} or newer")
        if not target:
            return [*blockers, "Select a compute pool"]
        req = recipe.requirements
        if target.hardware != req.hardware:
            blockers.append("Requires NVIDIA DGX Spark hardware")
        if target.gpu_count < req.nodes:
            blockers.append(f"Requires {req.nodes} Sparks")
        if target.gpu_memory_gib and target.gpu_memory_gib < req.memory_gib_per_node:
            blockers.append(f"Requires {req.memory_gib_per_node:g} GiB per Spark")
        if target.kind != "local-docker":
            # v1 recipes use custom images/flags: require matching bridge support.
            # Distributed bootstrap must preserve worker argv/environment, not just head flags.
            blockers.append("Recipe v1 currently executes on a directly connected Docker Spark")
        if req.topology != "single":
            blockers.append("Distributed recipe bootstrap is not supported by recipe v1")
        return blockers
