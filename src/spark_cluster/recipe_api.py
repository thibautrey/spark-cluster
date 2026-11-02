"""Recipe catalog, version history, discovery, file import, and install jobs."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .recipes import SparkRecipe
from .storage import StorageShortage


class RecipeInstallRequest(BaseModel):
    recipe_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cluster_id: str = Field(min_length=1, max_length=80)
    launch: bool = True


class NodeRecipeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe: SparkRecipe
    recipe_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    launch: bool = False
    request_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    task: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def verify_release(self):
        if self.recipe.content_hash != self.recipe_hash:
            raise ValueError("Recipe body differs from its immutable release hash")
        if len(self.recipe.model_dump_json().encode()) > 128 * 1024:
            raise ValueError("Recipe exceeds 128 KiB")
        return self


def register_recipes(app, catalog, installer, registry, require_token, settings):
    router = APIRouter(prefix="/api/v1/recipes", dependencies=[Depends(require_token)])

    def mutation():
        if not settings.model_admin_token:
            raise HTTPException(503, "Configure MODEL_ADMIN_TOKEN before changing recipes")

    @router.get("/schema")
    async def schema():
        return SparkRecipe.model_json_schema()

    @router.get("")
    async def state(
        cluster_id: str | None = None,
        q: str = Query(default="", max_length=200),
        category: str = Query(default="all", max_length=80),
        sort: Literal["popular", "downloads", "trending", "name"] = "popular",
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=24, ge=1, le=100),
        installable: bool = False,
    ):
        from .popular_recipes import catalog_page

        catalog.start()
        discovery = await catalog.discovery_state()
        target = registry.get(cluster_id) if cluster_id else registry.active()
        if target and target.kind == "remote-bridge" and installer.remote:
            try:
                target = await installer.remote.refresh(target)
            except ValueError:
                target = target.model_copy(update={"status": "offline"})
        page = await catalog_page(
            catalog,
            target,
            q=q,
            category=category,
            sort=sort,
            offset=offset,
            limit=limit,
            installable=installable,
        )
        return {**page, "discovery": discovery, "jobs": await installer.jobs()}

    @router.post("/discover", dependencies=[Depends(mutation)])
    async def discover():
        return await catalog.discover(force=True)

    @router.post("/import", dependencies=[Depends(mutation)])
    async def import_recipe(recipe: SparkRecipe):
        try:
            return await catalog.remember(recipe, {"kind": "imported"})
        except StorageShortage:
            raise
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/install", status_code=202, dependencies=[Depends(mutation)])
    async def install(request: RecipeInstallRequest):
        try:
            return await installer.create(
                request.recipe_hash, request.cluster_id, launch=request.launch
            )
        except KeyError as error:
            raise HTTPException(404, "Recipe release or compute pool not found") from error
        except StorageShortage:
            raise
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    def local_target():
        target = next((c for c in registry.list() if c.kind == "local-docker"), None)
        if not target:
            raise HTTPException(409, "This bridge has no measured Docker accelerator")
        return target

    async def node_check(request):
        target = local_target()
        blockers = await catalog.compatible(request.recipe, target)
        if blockers:
            raise ValueError("; ".join(blockers))
        if installer.preflight:
            await installer.preflight(request.recipe, target.id)
        return target

    @router.post("/node/preflight", dependencies=[Depends(mutation)])
    async def node_preflight(request: NodeRecipeRequest):
        try:
            await node_check(request)
            return {"status": "ready", "recipe_hash": request.recipe_hash}
        except StorageShortage:
            raise
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/node/install", status_code=202, dependencies=[Depends(mutation)])
    async def node_install(request: NodeRecipeRequest):
        if not request.request_id:
            raise HTTPException(422, "Remote installation requires an idempotency key")
        try:
            # Resolve replay before preflight: the original node job is authoritative.
            prior = next(
                (j for j in await installer.jobs() if j.get("request_id") == request.request_id),
                None,
            )
            if prior:
                if (
                    prior["recipe_hash"] != request.recipe_hash
                    or prior.get("launch", True) != request.launch
                ):
                    raise ValueError("Idempotency key belongs to another recipe request")
                return prior
            target = await node_check(request)
            await catalog.remember(request.recipe, {"kind": "remote", "task": request.task})
            return await installer.create(
                request.recipe_hash, target.id, launch=request.launch, request_id=request.request_id
            )
        except StorageShortage:
            raise
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.get("/node/jobs/{job_id}")
    async def node_job(job_id: str):
        job = next(
            (j for j in await installer.jobs() if j["id"] == job_id and not j.get("remote")), None
        )
        if not job:
            raise HTTPException(404, "Node recipe installation not found")
        return job

    app.include_router(router)
