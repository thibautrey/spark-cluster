"""Curated inference runtimes and live, paginated public OCI registry discovery.

Tags are distribution versions, not claims of kernel or model compatibility.
Only curated repositories and HTTPS authentication endpoints are contacted.
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, urlparse

import httpx

TAG = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"
DIGEST = r"sha256:[a-f0-9]{64}"
ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


@dataclass(frozen=True)
class CatalogRuntime:
    id: str
    name: str
    repository: str
    project_url: str
    description: str
    hardware: str
    launch_adapter: bool = False
    note: str = "Installation prepares an image; no Spark launch adapter is enabled."


CATALOG = {
    runtime.id: runtime
    for runtime in [
        CatalogRuntime(
            "vllm",
            "vLLM",
            "docker.io/vllm/vllm-openai",
            "https://github.com/vllm-project/vllm",
            "High-throughput LLM serving",
            "NVIDIA / AMD, depending on image",
            True,
            "Launch adapter exists. Upstream images still need GPU/kernel validation.",
        ),
        CatalogRuntime(
            "sglang",
            "SGLang",
            "docker.io/lmsysorg/sglang",
            "https://github.com/sgl-project/sglang",
            "LLM and multimodal serving",
            "NVIDIA / AMD / Ascend, depending on image",
            True,
            "Single-node launch adapter exists; distributed bootstrap is not validated.",
        ),
        CatalogRuntime(
            "llama-cpp",
            "llama.cpp",
            "ghcr.io/ggml-org/llama.cpp",
            "https://github.com/ggml-org/llama.cpp",
            "Portable GGUF inference server",
            "CPU / CUDA / Vulkan, depending on image",
        ),
        CatalogRuntime(
            "ollama",
            "Ollama",
            "docker.io/ollama/ollama",
            "https://github.com/ollama/ollama",
            "Local model management and serving",
            "CPU / NVIDIA / AMD, depending on image",
        ),
        CatalogRuntime(
            "tensorrt-llm",
            "TensorRT-LLM",
            "nvcr.io/nvidia/tensorrt-llm/release",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "Optimized NVIDIA LLM inference",
            "NVIDIA; engine builds and GPU generation constraints apply",
        ),
        CatalogRuntime(
            "triton",
            "Triton Inference Server",
            "nvcr.io/nvidia/tritonserver",
            "https://github.com/triton-inference-server/server",
            "Multi-framework inference server",
            "NVIDIA / CPU, depending on backend",
        ),
        CatalogRuntime(
            "tgi",
            "Text Generation Inference",
            "ghcr.io/huggingface/text-generation-inference",
            "https://github.com/huggingface/text-generation-inference",
            "Hugging Face LLM serving",
            "NVIDIA / AMD / Intel, depending on image",
        ),
        CatalogRuntime(
            "lmdeploy",
            "LMDeploy",
            "docker.io/openmmlab/lmdeploy",
            "https://github.com/InternLM/lmdeploy",
            "TurboMind and PyTorch LLM serving",
            "NVIDIA; architecture support varies by release",
        ),
        CatalogRuntime(
            "xinference",
            "Xinference",
            "docker.io/xprobe/xinference",
            "https://github.com/xorbitsai/inference",
            "Multi-engine model serving",
            "CPU / NVIDIA / other accelerators, depending on image",
        ),
        CatalogRuntime(
            "localai",
            "LocalAI",
            "docker.io/localai/localai",
            "https://github.com/mudler/LocalAI",
            "Local OpenAI-compatible model APIs",
            "CPU / NVIDIA / AMD / Vulkan, depending on image",
        ),
    ]
}


class RegistryError(ValueError):
    pass


def runtime_entry(runtime_id: str) -> CatalogRuntime:
    try:
        return CATALOG[runtime_id]
    except KeyError as error:
        raise KeyError("Unknown runtime") from error


def catalog_entries() -> list[dict]:
    return [asdict(item) for item in CATALOG.values()]


class RuntimeRegistry:
    def __init__(self, transport=None):
        self.transport = transport
        self.cache: dict[tuple, tuple[float, dict]] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def _get(self, client, runtime, path, **kwargs):
        host, repository = runtime.repository.split("/", 1)
        host = "registry-1.docker.io" if host == "docker.io" else host
        url = f"https://{host}/v2/{repository}/{path}"
        response = await client.get(url, **kwargs)
        if response.status_code == 401:
            challenge = response.headers.get("www-authenticate", "")
            fields = dict(re.findall(r'(\w+)="([^"]+)"', challenge))
            realm = fields.get("realm", "")
            permitted = {
                "registry-1.docker.io": "https://auth.docker.io/token",
                "ghcr.io": "https://ghcr.io/token",
                "nvcr.io": "https://nvcr.io/proxy_auth",
            }
            if realm != permitted.get(host):
                raise RegistryError("Registry requested an unsupported authentication endpoint")
            auth = await client.get(
                realm,
                params={
                    "service": fields.get("service", host),
                    "scope": f"repository:{repository}:pull",
                },
            )
            auth.raise_for_status()
            token = auth.json().get("token") or auth.json().get("access_token")
            if not token:
                raise RegistryError("Public registry authentication is unavailable")
            headers = {**kwargs.pop("headers", {}), "Authorization": f"Bearer {token}"}
            response = await client.get(url, headers=headers, **kwargs)
        response.raise_for_status()
        return response

    async def versions(self, runtime_id: str, cursor: str = "", limit: int = 50) -> dict:
        runtime = runtime_entry(runtime_id)
        if cursor and not re.fullmatch(TAG, cursor):
            raise RegistryError("Invalid version cursor")
        key = (runtime_id, cursor, limit)
        async with self.locks.setdefault(runtime_id, asyncio.Lock()):
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < 300:
                return {**cached[1], "cached": True}
            try:
                async with httpx.AsyncClient(transport=self.transport, timeout=30) as client:
                    if runtime.repository.startswith("docker.io/"):
                        if cursor and not cursor.isdecimal():
                            raise RegistryError("Invalid Docker Hub page")
                        repository = runtime.repository.split("/", 1)[1]
                        response = await client.get(
                            f"https://hub.docker.com/v2/repositories/{repository}/tags",
                            params={
                                "page_size": limit,
                                "page": cursor or "1",
                                "ordering": "last_updated",
                            },
                        )
                        response.raise_for_status()
                        body = response.json()
                        tags = [
                            {
                                "version": item["name"],
                                "updated_at": item.get("last_updated"),
                                "platforms": sorted(
                                    {
                                        f"{image['os']}/{image['architecture']}"
                                        for image in item.get("images", [])
                                        if image.get("os")
                                        and image.get("architecture")
                                        and image["architecture"] != "unknown"
                                    }
                                ),
                            }
                            for item in body.get("results", [])
                        ]
                        next_cursor = (
                            parse_qs(urlparse(body["next"]).query).get("page", [""])[0]
                            if body.get("next")
                            else None
                        )
                    else:
                        params = {"n": limit}
                        if cursor:
                            params["last"] = cursor
                        response = await self._get(client, runtime, "tags/list", params=params)
                        names = response.json().get("tags") or []
                        tags = [{"version": name, "platforms": []} for name in names]
                        link = response.headers.get("link", "")
                        match = re.search(r"<([^>]+)>;\s*rel=\"?next", link)
                        next_cursor = (
                            parse_qs(urlparse(match[1]).query).get("last", [""])[0]
                            if match
                            else (names[-1] if len(names) == limit else None)
                        )
                result = {
                    "runtime_id": runtime_id,
                    "repository": runtime.repository,
                    "versions": tags,
                    "next_cursor": next_cursor,
                    "cached": False,
                    "stale": False,
                    "ordering": "recently updated"
                    if runtime.repository.startswith("docker.io/")
                    else "registry order",
                }
                self.cache[key] = (time.monotonic(), result)
                # Bound the in-memory page cache, including obsolete pages.
                while len(self.cache) > 100:
                    self.cache.pop(next(iter(self.cache)))
                return result
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                if cached:
                    return {
                        **cached[1],
                        "cached": True,
                        "stale": True,
                        "warning": "Registry unavailable; showing cached versions",
                    }
                raise RegistryError(
                    "Registry unavailable or access/rate limited; retry later"
                ) from error

    async def resolve(self, runtime_id: str, version: str, platform: str) -> dict:
        runtime = runtime_entry(runtime_id)
        if not re.fullmatch(TAG, version):
            raise RegistryError("Invalid image tag")
        if platform not in {"linux/amd64", "linux/arm64"}:
            raise RegistryError("Choose linux/amd64 or linux/arm64")
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=30) as client:

                async def manifest(reference):
                    response = await self._get(
                        client, runtime, f"manifests/{reference}", headers={"Accept": ACCEPT}
                    )
                    digest = "sha256:" + hashlib.sha256(response.content).hexdigest()
                    expected = response.headers.get("docker-content-digest", digest)
                    if digest != expected or (
                        reference.startswith("sha256:") and digest != reference
                    ):
                        raise RegistryError("Registry manifest digest verification failed")
                    return response.json(), digest

                body, digest = await manifest(version)
                if "manifests" in body:
                    selected = next(
                        (
                            item
                            for item in body["manifests"]
                            if f"{item.get('platform', {}).get('os')}/"
                            f"{item.get('platform', {}).get('architecture')}" == platform
                        ),
                        None,
                    )
                    if not selected:
                        raise RegistryError(f"This version has no {platform} image")
                    if not re.fullmatch(DIGEST, selected["digest"]):
                        raise RegistryError("Unsupported manifest digest")
                    body, digest = await manifest(selected["digest"])
                config_digest = body.get("config", {}).get("digest", "")
                if not re.fullmatch(DIGEST, config_digest):
                    raise RegistryError("Unsupported image configuration digest")
                response = await self._get(
                    client, runtime, f"blobs/{config_digest}", follow_redirects=True
                )
                if "sha256:" + hashlib.sha256(response.content).hexdigest() != config_digest:
                    raise RegistryError("Registry configuration digest verification failed")
                config = response.json()
                if f"{config.get('os')}/{config.get('architecture')}" != platform:
                    raise RegistryError(f"This version has no {platform} image")
                return {
                    "runtime_id": runtime_id,
                    "version": version,
                    "platform": platform,
                    "repository": runtime.repository,
                    "digest": digest,
                    "image": f"{runtime.repository}:{version}",
                    "config_digest": config_digest,
                    "size_bytes": sum(item.get("size", 0) for item in body.get("layers", [])),
                }
        except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RegistryError(
                "Image metadata unavailable; check the tag and registry access"
            ) from error
