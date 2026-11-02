"""Controller-wide storage inventory and authenticated physical-node operations."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .models import ModelDownloadRequest


class StorageDeleteRequest(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=100)


def register_storage(app, storage, require_token, settings, preflight_download):
    router = APIRouter(prefix="/api/v1/storage", dependencies=[Depends(require_token)])

    def configured():
        if not settings.model_admin_token:
            raise HTTPException(503, "Configure MODEL_ADMIN_TOKEN before deleting stored files")

    @router.get("")
    async def inventory():
        return await storage.inventory()

    @router.get("/node")
    async def node_inventory():
        return await storage.local(fresh=True)

    @router.post("/preflight")
    async def preflight(request: ModelDownloadRequest):
        await preflight_download(request)
        return {"ready": True}

    @router.post("/delete", dependencies=[Depends(configured)])
    async def delete(request: StorageDeleteRequest):
        try:
            return await storage.delete(list(dict.fromkeys(request.asset_ids)))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.post("/node/delete", dependencies=[Depends(configured)])
    async def node_delete(request: StorageDeleteRequest):
        try:
            return await storage.delete_local(list(dict.fromkeys(request.asset_ids)))
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    app.include_router(router)
