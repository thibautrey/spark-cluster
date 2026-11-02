"""Authenticated, idempotent node-owned recipe installation; no weights transit the controller."""

import json
import time
from urllib.parse import quote

import httpx

from .remote_cluster import signed_controller_headers
from .storage import StorageShortage

PROTOCOL = "spark-cluster.remote-recipes/v1"
CAPABILITY = "recipes.v1"


def remap_storage(storage, cluster_id):
    return {**storage, "requirements": [
        {**r, "location_id": cluster_id + ":" + r["location_id"].split(":", 1)[-1]}
        for r in storage.get("requirements", [])]}


class RemoteRecipes:
    def __init__(self, settings, registry, transport=None):
        self.settings, self.registry, self.transport = settings, registry, transport
        self.checked = {}

    async def request(self, cluster, method, path, payload=None):
        if not self.settings.model_admin_token:
            raise ValueError("Configure the shared administration token for remote recipes")
        body = json.dumps(payload).encode() if payload is not None else b""
        headers = signed_controller_headers(self.settings.model_admin_token, method, path, body)
        headers["content-type"] = "application/json"
        try:
            async with httpx.AsyncClient(base_url=cluster.endpoint, transport=self.transport,
                                         timeout=120, trust_env=False) as client:
                response = await client.request(method, path, content=body, headers=headers)
        except httpx.HTTPError as error:
            raise ValueError("Recipe bridge is unavailable; reconnect it and retry") from error
        if response.is_error:
            try:
                detail = response.json().get("detail", {})
            except ValueError:
                detail = {}
            if response.status_code == 507 and isinstance(detail, dict) and detail.get("storage"):
                mapped = remap_storage(detail["storage"], cluster.id)
                raise StorageShortage(mapped["requirements"], mapped.get("excluded_model_ids", []))
            message = detail.get("message") if isinstance(detail, dict) else detail
            raise ValueError(str(message or "The node rejected recipe preparation")[:600])
        return response.json()

    async def refresh(self, target, *, force=False):
        if not force and time.monotonic() - self.checked.get(target.id, -1000) < 20:
            return self.registry.get(target.id)
        payload = await self.request(target, "GET", "/api/v1/controller/verify")
        if payload.get("recipe_protocol") != PROTOCOL or not payload.get("node_target"):
            raise ValueError("Update this bridge to support remote recipe installation")
        target = self.registry.update_remote_hardware(target.id, payload["node_target"], [CAPABILITY])
        self.checked[target.id] = time.monotonic()
        return target

    async def prepare(self, target, recipe, *, launch):
        return await self.request(target, "POST", "/api/v1/recipes/node/preflight", {
            "recipe": recipe.model_dump(mode="json"), "recipe_hash": recipe.content_hash,
            "launch": launch})

    async def install(self, target, job):
        return await self.request(target, "POST", "/api/v1/recipes/node/install", {
            "recipe": job["recipe"], "recipe_hash": job["recipe_hash"],
            "request_id": job["id"], "launch": job.get("launch", True),
            "task": job.get("origin", {}).get("task")})

    async def job(self, target, job_id):
        return await self.request(target, "GET", "/api/v1/recipes/node/jobs/" + quote(job_id, safe=""))
