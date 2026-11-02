"""Recipe catalog, version history, discovery, file import, and install jobs."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .recipes import SparkRecipe
from .storage import StorageShortage


class RecipeInstallRequest(BaseModel):
    recipe_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cluster_id: str = Field(min_length=1, max_length=80)


def register_recipes(app, catalog, installer, registry, require_token, settings):
    router = APIRouter(prefix="/api/v1/recipes", dependencies=[Depends(require_token)])

    def mutation():
        if not settings.model_admin_token:
            raise HTTPException(503, "Configure MODEL_ADMIN_TOKEN before changing recipes")

    @router.get("/schema")
    async def schema():
        return SparkRecipe.model_json_schema()

    @router.get("")
    async def state(cluster_id: str | None = None):
        catalog.start()
        discovery = await catalog.discovery_state()
        target = registry.get(cluster_id) if cluster_id else registry.active()
        entries = []
        for entry in await catalog.entries():
            blockers = await catalog.compatible(SparkRecipe.model_validate(entry["recipe"]), target)
            entries.append({**entry, "blockers": blockers})
        return {"recipes": entries, "discovery": discovery, "jobs": await installer.jobs()}

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
            return await installer.create(request.recipe_hash, request.cluster_id)
        except KeyError as error:
            raise HTTPException(404, "Recipe release or compute pool not found") from error
        except StorageShortage:
            raise
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    app.include_router(router)
