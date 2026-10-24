"""HTTP surface for the controller runtime catalogue, cache and install queue."""

from fastapi import APIRouter, Depends, HTTPException, Query

from .cluster_setup import SetupFailure
from .runtime_catalog import RegistryError
from .runtime_library import RuntimeDownloadRequest, RuntimeHostRequest, RuntimeImportRequest


def register_runtime_library(app, library, require_model_token, settings):
    router = APIRouter(
        prefix="/api/v1/runtime-library", dependencies=[Depends(require_model_token)]
    )

    def require_admin_configured():
        if not settings.model_admin_token:
            raise HTTPException(503, "Configure MODEL_ADMIN_TOKEN before changing runtimes")

    @router.get("")
    async def state():
        return await library.state()

    @router.get("/runtimes/{runtime_id}/versions")
    async def versions(
        runtime_id: str,
        cursor: str = "",
        limit: int = Query(50, ge=1, le=100),
        platform: str = "linux/amd64",
    ):
        try:
            return await library.remote.versions(runtime_id, cursor, limit, platform)
        except KeyError as error:
            raise HTTPException(404, "Runtime not found") from error
        except RegistryError as error:
            raise HTTPException(502, str(error)) from error

    @router.post("/resolve")
    async def resolve(request: RuntimeDownloadRequest):
        try:
            return await library.remote.resolve(**request.model_dump())
        except KeyError as error:
            raise HTTPException(404, "Runtime not found") from error
        except RegistryError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/downloads", status_code=202, dependencies=[Depends(require_admin_configured)])
    async def download(request: RuntimeDownloadRequest):
        try:
            return await library.download(request)
        except KeyError as error:
            raise HTTPException(404, "Runtime not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/installs", status_code=202, dependencies=[Depends(require_admin_configured)])
    async def install(request: RuntimeImportRequest):
        try:
            return await library.install(request)
        except KeyError as error:
            raise HTTPException(404, "Artifact or compute pool not found") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.post("/hosts", status_code=201, dependencies=[Depends(require_admin_configured)])
    async def register_host(request: RuntimeHostRequest):
        try:
            return await library.register_host(request)
        except (ValueError, SetupFailure) as error:
            raise HTTPException(409, str(error)) from error

    @router.delete(
        "/hosts/{host_id}", status_code=204, dependencies=[Depends(require_admin_configured)]
    )
    async def remove_host(host_id: str):
        try:
            await library.remove_host(host_id)
        except KeyError as error:
            raise HTTPException(404, "Native host not found") from error

    @router.get("/jobs/{job_id}")
    async def job(job_id: str):
        try:
            return await library.job(job_id)
        except KeyError as error:
            raise HTTPException(404, "Runtime job not found") from error

    app.include_router(router)
