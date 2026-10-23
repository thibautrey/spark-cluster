"""Bounded, controller-local Hugging Face discovery. Credentials never reach the UI."""

import asyncio
import hashlib
import json
import re
import time
from typing import Literal
from urllib.parse import parse_qs, quote, urlparse

import httpx
from pydantic import BaseModel, Field, field_validator

from .catalog_cache import CatalogCache
from .catalog_recommendation import (
    RECOMMENDATION_VERSION,
    Recommendation,
    generic_ranking_key,
    ranking_key,
    recommend,
)
from .config import Settings
from .model_manager import (
    DEEPSEEK_V41_ENGRAM_BYTES,
    WEIGHT_SUFFIXES,
    RepositoryMetadata,
    build_fit_plan,
    safe_repo_parts,
)
from .models import ModelFitRequest
from .spark_optimization import CLASSIFIER_VERSION, SparkOptimization, classify_spark

CatalogSort = Literal["spark", "trendingScore", "downloads", "likes", "lastModified"]


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
    thumbnail_url: str
    icon_url: str | None = None
    commit: str | None = None
    spark: SparkOptimization = Field(default_factory=SparkOptimization)
    recommendation: Recommendation = Field(default_factory=Recommendation)


class CatalogPage(BaseModel):
    models: list[CatalogModel]
    cursor: str | None = None
    ranking_note: str | None = None


class CatalogFile(BaseModel):
    name: str
    size: int | None = None


class CatalogDetail(CatalogModel):
    revision: str
    license: str | None = None
    readme: str | None = None
    readme_path: str | None = None
    readme_truncated: bool = False
    architecture: str | None = None
    files: list[CatalogFile]
    repository_bytes: int
    size_complete: bool


class CatalogFitCandidate(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    commit: str | None = Field(default=None, max_length=200)

    @field_validator("id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        safe_repo_parts(value)
        return value


class CatalogFitBatchRequest(BaseModel):
    models: list[CatalogFitCandidate] = Field(min_length=1, max_length=48)
    spark_count: int = Field(ge=1, le=16)
    context_length: int = Field(default=32768, ge=1024, le=1_048_576)
    concurrent_sequences: int = Field(default=1, ge=1, le=128)
    kv_cache_dtype: Literal["auto", "bf16", "fp8_e4m3"] = "auto"


class CatalogFitResult(BaseModel):
    repo_id: str
    compatible: bool
    memory_fits: bool = False
    runtime_fits: bool = False
    launch_supported: bool = False
    fit_status: str = "download-only"
    placement_strategy: str = "unsupported"
    minimum_nodes: int = 0
    weights_gib: float = 0
    kv_cache_gib: float = 0
    detected_format: str = "Unknown"
    engine_recommendation: str = "Unknown"
    cached: bool = False
    error: str | None = None


class CatalogFitBatchResponse(BaseModel):
    results: list[CatalogFitResult]
    cache: dict[str, int]


MODEL_ICON_NAMES = tuple(
    f"{name}.{extension}"
    for name in ("model-icon", "icon", "avatar", "thumbnail", "logo")
    for extension in ("png", "jpg", "jpeg", "webp", "svg")
)


def model_icon_url(data: dict, repo_id: str) -> str | None:
    files = {
        item.get("rfilename", "").casefold(): item.get("rfilename", "")
        for item in (data.get("siblings") or [])
    }
    for wanted in MODEL_ICON_NAMES:
        # Prefer explicit root icons, then an exact filename under assets/images.
        path = files.get(wanted)
        if not path:
            path = next(
                (original for lowered, original in files.items() if lowered.endswith("/" + wanted)),
                None,
            )
        if path:
            revision = data.get("sha") or "main"
            encoded_path = "/".join(quote(part, safe="") for part in path.split("/"))
            return (
                f"https://huggingface.co/{quote(repo_id, safe='/')}/resolve/"
                f"{quote(revision, safe='')}/{encoded_path}"
            )
    return None


def summary(data: dict) -> CatalogModel:
    repo_id = data.get("id") or data["modelId"]
    spark = classify_spark(data)
    return CatalogModel(
        id=repo_id,
        spark=spark,
        recommendation=recommend(data, spark),
        author=data.get("author") or (repo_id.split("/")[0] if "/" in repo_id else "Hugging Face"),
        downloads=data.get("downloads") or 0,
        likes=data.get("likes") or 0,
        task=data.get("pipeline_tag"),
        library=data.get("library_name"),
        tags=(data.get("tags") or [])[:30],
        gated=bool(data.get("gated")),
        updated=data.get("lastModified"),
        parameters=(data.get("safetensors") or {}).get("total"),
        commit=data.get("sha"),
        thumbnail_url=(
            "https://cdn-thumbnails.huggingface.co/social-thumbnails/models/"
            f"{quote(repo_id, safe='/')}.png"
        ),
        icon_url=model_icon_url(data, repo_id),
    )


class ModelBrowser:
    planner_version = "catalog-fit-v1"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.token = settings.hf_token
        cache_path = settings.catalog_cache_path or settings.database_path.with_name(
            f"{settings.database_path.stem}-catalog-cache.db"
        )
        self.persistent_cache = CatalogCache(cache_path)
        self.cache: dict[tuple, tuple[float, dict | list, str | None]] = {}
        self.lock = asyncio.Lock()
        self.transport: httpx.AsyncBaseTransport | None = None

    async def fetch(self, path: str, params: dict) -> tuple[dict | list, str | None]:
        key = (path, *sorted(params.items()))
        persistent_key = json.dumps(key, separators=(",", ":"), default=str)
        async with self.lock:
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < 60:
                return cached[1], cached[2]
        persisted = await asyncio.to_thread(self.persistent_cache.get_metadata, persistent_key)
        if persisted:
            async with self.lock:
                self.cache[key] = (time.monotonic(), persisted[0], persisted[1])
            return persisted
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        async with httpx.AsyncClient(
            base_url="https://huggingface.co",
            timeout=20,
            headers=headers,
            transport=self.transport,
        ) as client:
            response = await client.get(path, params=params)
            response.raise_for_status()
        data = response.json()
        next_url = response.links.get("next", {}).get("url", "")
        cursor = parse_qs(urlparse(next_url).query).get("cursor", [None])[0]
        async with self.lock:
            if len(self.cache) >= 64:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (time.monotonic(), data, cursor)
        immutable_revision = "/revision/" in path and "/revision/main" not in path
        ttl = 30 * 86400 if immutable_revision else 600 if path == "/api/models" else 86400
        await asyncio.to_thread(
            self.persistent_cache.set_metadata, persistent_key, data, cursor, ttl
        )
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
        if sort == "spark":
            return await self._spark_search(query, task, library, access, limit, cursor)
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

    async def _spark_search(
        self,
        query: str,
        task: str,
        library: str,
        access: str,
        limit: int,
        cursor: str | None,
    ) -> CatalogPage:
        # Cache the entire ranked shortlist, not individual pages, so pagination is
        # stable and classification is reused across controller restarts.
        identity = json.dumps(
            (
                CLASSIFIER_VERSION,
                RECOMMENDATION_VERSION,
                time.strftime("%Y-%m-%d", time.gmtime()),
                query,
                task,
                library,
                access,
            )
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        key = f"spark-ranking:{digest}"
        offset = 0
        if cursor:
            parts = cursor.split(":")
            if len(parts) != 3 or parts[:2] != ["spark-v1", digest]:
                raise ValueError("Invalid Spark catalog cursor")
            if not parts[2].isdigit():
                raise ValueError("Invalid Spark catalog offset")
            offset = int(parts[2])
            if not 0 < offset < 288:
                raise ValueError("Invalid Spark catalog offset")
        cached = await asyncio.to_thread(self.persistent_cache.get_metadata, key)
        if cursor and not cached:
            raise ValueError("Expired Spark catalog cursor; restart search")
        if cached:
            snapshot = cached[0]
        else:
            # A bounded union discovers relevant low-precision checkpoints even
            # when they do not make the first trending page. Every upstream query
            # carries the same task, format and access constraints.
            base = {
                "search": query,
                "sort": "trendingScore",
                "direction": "-1",
                "limit": "48",
                "full": "true",
            }
            if task:
                base["pipeline_tag"] = task
            if library:
                base["filter"] = library
            if access:
                base["gated"] = "true" if access == "gated" else "false"
            targeted = []
            for tag in ("dgx-spark", "sm121", "exl3"):
                targeted.append({**base, "filter": ",".join(filter(None, (library, tag)))})
            responses = await asyncio.gather(
                self.fetch("/api/models", base),
                self.fetch("/api/models", {**base, "search": "NVFP4", "sort": "downloads"}),
                *(self.fetch("/api/models", params) for params in targeted),
                self.fetch("/api/models", {**base, "sort": "downloads"}),
                return_exceptions=True,
            )
            # A failed supplemental discovery must not take down regular search.
            if isinstance(responses[0], Exception):
                raise responses[0]
            candidates = {}
            for response in responses:
                if isinstance(response, Exception):
                    continue
                for item in response[0]:
                    repo_id = item.get("id") or item.get("modelId", "")
                    if query and query.casefold() not in repo_id.casefold():
                        continue
                    candidates.setdefault(repo_id, summary(item))
            active_capability = self.settings.active_compute_capability
            spark_hardware = active_capability is None or active_capability >= 10
            models = sorted(
                candidates.values(),
                key=ranking_key if spark_hardware else generic_ranking_key,
            )
            note = (
                (
                    "Evidence → Spark signals → publisher → bounded popularity · "
                    if spark_hardware
                    else "Publisher → bounded popularity for active NVIDIA GPUs · "
                )
                + f"shortlist of {len(models)} models (up to 288 candidates); "
                "not a quality or speed benchmark"
            )
            if any(isinstance(response, Exception) for response in responses[1:]):
                note += " · some discovery sources unavailable"
            snapshot = {"models": [model.model_dump() for model in models], "note": note}
            await asyncio.to_thread(self.persistent_cache.set_metadata, key, snapshot, None, 600)
        models = snapshot["models"]
        if cursor and offset >= len(models):
            raise ValueError("Invalid Spark catalog offset")
        end = offset + limit
        return CatalogPage(
            models=[CatalogModel.model_validate(item) for item in models[offset:end]],
            cursor=f"spark-v1:{digest}:{end}" if end < len(models) else None,
            ranking_note=snapshot["note"],
        )

    async def author_avatar_url(self, author: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", author):
            raise ValueError("Invalid Hugging Face profile")
        key = f"profile-avatar:{author.casefold()}"
        cached = await asyncio.to_thread(self.persistent_cache.get_metadata, key)
        if cached:
            return cached[0]["url"]
        endpoints = (
            f"/api/organizations/{quote(author, safe='')}/overview",
            f"/api/users/{quote(author, safe='')}/overview",
        )
        for path in endpoints:
            try:
                data, _ = await self.fetch(path, {})
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404:
                    continue
                raise
            url = data.get("avatarUrl") if isinstance(data, dict) else None
            if isinstance(url, str) and url.startswith("https://cdn-avatars.huggingface.co/"):
                await asyncio.to_thread(
                    self.persistent_cache.set_metadata, key, {"url": url}, None, 86400
                )
                return url
        return f"https://huggingface.co/avatars/{quote(author, safe='')}.svg"

    async def _readme(
        self, repo_id: str, revision: str, data: dict
    ) -> tuple[str | None, str | None, bool]:
        names = [item.get("rfilename", "") for item in data.get("siblings", [])]
        readme_path = next((name for name in names if name.casefold() == "readme.md"), None)
        if not readme_path:
            return None, None, False
        immutable_revision = revision != "main" and revision == data.get("sha")
        cache_key = json.dumps(
            ("catalog-readme-v1", repo_id, revision, readme_path), separators=(",", ":")
        )
        cached = await asyncio.to_thread(self.persistent_cache.get_metadata, cache_key)
        if cached:
            payload = cached[0]
            return payload["text"], readme_path, payload["truncated"]
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        encoded_path = "/".join(quote(part, safe="") for part in readme_path.split("/"))
        path = f"/{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{encoded_path}"
        limit = 512 * 1024
        content = bytearray()
        async with httpx.AsyncClient(
            base_url="https://huggingface.co",
            timeout=20,
            headers=headers,
            transport=self.transport,
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", path) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > limit:
                        break
        truncated = len(content) > limit
        text = bytes(content[:limit]).decode("utf-8", errors="replace")
        payload = {"text": text, "truncated": truncated}
        await asyncio.to_thread(
            self.persistent_cache.set_metadata,
            cache_key,
            payload,
            None,
            30 * 86400 if immutable_revision else 600,
        )
        return text, readme_path, truncated

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
        readme, readme_path, readme_truncated = await self._readme(repo_id, revision, data)
        return CatalogDetail(
            **summary(data).model_dump(),
            revision=revision,
            license=(data.get("cardData") or {}).get("license"),
            readme=readme,
            readme_path=readme_path,
            readme_truncated=readme_truncated,
            architecture=architectures[0] if architectures else config.get("model_type"),
            files=files,
            repository_bytes=sum(item.size or 0 for item in files),
            size_complete=bool(files) and all(item.size is not None for item in files),
        )

    async def fit_batch(self, request: CatalogFitBatchRequest) -> CatalogFitBatchResponse:
        semaphore = asyncio.Semaphore(4)

        async def plan(candidate: CatalogFitCandidate) -> CatalogFitResult:
            safe_repo_parts(candidate.id)
            revision = candidate.commit or "main"
            key = json.dumps(
                (
                    self.planner_version,
                    candidate.id,
                    revision,
                    request.spark_count,
                    request.context_length,
                    request.concurrent_sequences,
                    request.kv_cache_dtype,
                    self.settings.fallback_node_memory_gib,
                    self.settings.model_usable_memory_fraction,
                ),
                separators=(",", ":"),
            )
            cached = await asyncio.to_thread(self.persistent_cache.get_fit, key)
            if cached:
                return CatalogFitResult.model_validate({**cached, "cached": True})
            try:
                async with semaphore:
                    metadata = await self._repository_metadata(candidate.id, revision)
                capacity_settings = self.settings.model_copy(
                    update={
                        "fallback_node_count": request.spark_count,
                        # Capacity browsing must never touch a busy remote model mount.
                        "model_cache_path": self.settings.database_path.parent,
                    }
                )
                fit = await asyncio.to_thread(
                    build_fit_plan,
                    ModelFitRequest(
                        repo_id=candidate.id,
                        revision=revision,
                        context_length=request.context_length,
                        concurrent_sequences=request.concurrent_sequences,
                        kv_cache_dtype=request.kv_cache_dtype,
                    ),
                    metadata,
                    None,
                    capacity_settings,
                )
                result = CatalogFitResult(
                    repo_id=candidate.id,
                    compatible=fit.runtime_fits,
                    memory_fits=fit.memory_fits,
                    runtime_fits=fit.runtime_fits,
                    launch_supported=fit.launch_supported,
                    fit_status=fit.fit_status,
                    placement_strategy=fit.placement_strategy,
                    minimum_nodes=fit.minimum_nodes,
                    weights_gib=fit.weights_gib,
                    kv_cache_gib=fit.kv_cache_gib,
                    detected_format=fit.detected_format,
                    engine_recommendation=fit.engine_recommendation,
                )
                ttl = 30 * 86400 if candidate.commit else 86400
                await asyncio.to_thread(
                    self.persistent_cache.set_fit,
                    key,
                    result.model_dump(exclude={"cached"}),
                    ttl,
                )
                return result
            except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError):
                return CatalogFitResult(
                    repo_id=candidate.id,
                    compatible=False,
                    error="Model capacity metadata is unavailable",
                )

        results = await asyncio.gather(*(plan(candidate) for candidate in request.models))
        stats = await asyncio.to_thread(self.persistent_cache.stats)
        return CatalogFitBatchResponse(results=results, cache=stats)

    async def _repository_metadata(self, repo_id: str, revision: str) -> RepositoryMetadata:
        path = f"/api/models/{quote(repo_id, safe='/')}/revision/{quote(revision, safe='')}"
        data, _ = await self.fetch(path, {"blobs": "true"})
        files = tuple(
            (str(item.get("rfilename") or ""), int(item.get("size") or 0))
            for item in data.get("siblings", [])
        )
        config = dict(data.get("config") or {})
        model = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
        architectures = model.get("architectures") or config.get("architectures") or []
        architecture = architectures[0] if architectures else model.get("model_type")
        parameter_count = (data.get("safetensors") or {}).get("total")
        model_type = str(model.get("model_type") or config.get("model_type") or "")
        is_deepseek_v41 = model_type == "deepseek_v41" or "deepseek-v4.1" in repo_id.lower()
        engram_bytes = sum(
            size
            for name, size in files
            if re.search(r"model-0004[78]-of-00048\.safetensors$", name)
        )
        external_artifacts_bytes = 0
        external_artifacts: tuple[str, ...] = ()
        if is_deepseek_v41 and "flash-1017" in repo_id.lower() and not engram_bytes:
            external_artifacts_bytes = DEEPSEEK_V41_ENGRAM_BYTES
            external_artifacts = (
                "deepseek-ai/DeepSeek-V4.1-Flash-1017/model-00047-of-00048.safetensors",
                "deepseek-ai/DeepSeek-V4.1-Flash-1017/model-00048-of-00048.safetensors",
            )
        return RepositoryMetadata(
            repo_id=repo_id,
            revision=revision,
            repository_bytes=sum(size for _, size in files),
            weight_bytes=sum(
                size for name, size in files if name.lower().endswith(WEIGHT_SUFFIXES)
            ),
            parameter_count=int(parameter_count) if parameter_count else None,
            architecture=str(architecture) if architecture else None,
            config=config,
            files=files,
            tags=tuple(str(tag) for tag in data.get("tags") or []),
            library_name=data.get("library_name"),
            pipeline_tag=data.get("pipeline_tag"),
            engram_bytes=engram_bytes,
            external_artifacts_bytes=external_artifacts_bytes,
            external_artifacts=external_artifacts,
        )

    def close(self) -> None:
        self.persistent_cache.close()
