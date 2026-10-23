"""Bounded, controller-local Hugging Face discovery. Credentials never reach the UI."""

import asyncio
import time
from typing import Literal
from urllib.parse import parse_qs, quote, urlparse

import httpx
from pydantic import BaseModel, Field

from .model_manager import safe_repo_parts

CatalogSort = Literal["trendingScore", "downloads", "likes", "lastModified"]


class CatalogModel(BaseModel):
    id: str
    author: str
    downloads: int = 0
    likes: int = 0
    task: str | None = None
    library: str | None = None
    tags: list[str] = Field(default_factory=list)
    gated: bool = False
    updated: str | None = None
    parameters: int | None = None


class CatalogPage(BaseModel):
    models: list[CatalogModel]
    cursor: str | None = None


class CatalogFile(BaseModel):
    name: str
    size: int | None = None


class CatalogDetail(CatalogModel):
    revision: str
    commit: str | None = None
    license: str | None = None
    architecture: str | None = None
    files: list[CatalogFile]
    repository_bytes: int
    size_complete: bool


def summary(data: dict) -> CatalogModel:
    repo_id = data.get("id") or data["modelId"]
    return CatalogModel(
        id=repo_id,
        author=repo_id.split("/")[0] if "/" in repo_id else "Hugging Face",
        downloads=data.get("downloads") or 0,
        likes=data.get("likes") or 0,
        task=data.get("pipeline_tag"),
        library=data.get("library_name"),
        tags=(data.get("tags") or [])[:30],
        gated=bool(data.get("gated")),
        updated=data.get("lastModified"),
        parameters=(data.get("safetensors") or {}).get("total"),
    )


class ModelBrowser:
    def __init__(self, token: str | None = None):
        self.token = token
        self.cache: dict[tuple, tuple[float, dict | list, str | None]] = {}
        self.lock = asyncio.Lock()
        self.transport: httpx.AsyncBaseTransport | None = None

    async def fetch(self, path: str, params: dict) -> tuple[dict | list, str | None]:
        key = (path, *sorted(params.items()))
        async with self.lock:
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < 60:
                return cached[1], cached[2]
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            async with httpx.AsyncClient(
                base_url="https://huggingface.co",
                timeout=15,
                headers=headers,
                transport=self.transport,
            ) as client:
                response = await client.get(path, params=params)
                response.raise_for_status()
            data = response.json()
            next_url = response.links.get("next", {}).get("url", "")
            cursor = parse_qs(urlparse(next_url).query).get("cursor", [None])[0]
            if len(self.cache) >= 64:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (time.monotonic(), data, cursor)
            return data, cursor

    async def search(
        self,
        query: str,
        sort: CatalogSort,
        task: str,
        library: str,
        access: str,
        limit: int,
        cursor: str | None,
    ) -> CatalogPage:
        params = {
            "search": query,
            "sort": sort,
            "direction": "-1",
            "limit": str(limit),
            "full": "true",
        }
        if task:
            params["pipeline_tag"] = task
        if library:
            params["filter"] = library
        if access:
            params["gated"] = "true" if access == "gated" else "false"
        if cursor:
            params["cursor"] = cursor
        data, next_cursor = await self.fetch("/api/models", params)
        return CatalogPage(models=[summary(item) for item in data], cursor=next_cursor)

    async def detail(self, repo_id: str, revision: str) -> CatalogDetail:
        safe_repo_parts(repo_id)
        path = f"/api/models/{quote(repo_id, safe='/')}/revision/{quote(revision, safe='')}"
        data, _ = await self.fetch(path, {"blobs": "true"})
        files = [
            CatalogFile(name=item["rfilename"], size=item.get("size"))
            for item in data.get("siblings", [])
        ]
        config = data.get("config") or {}
        architectures = config.get("architectures") or []
        return CatalogDetail(
            **summary(data).model_dump(),
            revision=revision,
            commit=data.get("sha"),
            license=(data.get("cardData") or {}).get("license"),
            architecture=architectures[0] if architectures else config.get("model_type"),
            files=files,
            repository_bytes=sum(item.size or 0 for item in files),
            size_complete=bool(files) and all(item.size is not None for item in files),
        )
