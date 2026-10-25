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
STABLE_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-cu(\d+))?$")
CALENDAR_VERSION = re.compile(r"^(\d{2})\.(\d{2})(?:-py3)?$")
SUPPORTED_PLATFORMS = (
    "linux/amd64",
    "linux/arm64",
    "darwin/arm64",
    "darwin/amd64",
    "windows/amd64",
    "windows/arm64",
)
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
    platforms: tuple[str, ...] = ("linux/amd64", "linux/arm64")
    native_repository: str | None = None
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
            launch_adapter=True,
            note="Launch adapter exists. Upstream images still need GPU/kernel validation.",
        ),
        CatalogRuntime(
            "sglang",
            "SGLang",
            "docker.io/lmsysorg/sglang",
            "https://github.com/sgl-project/sglang",
            "LLM and multimodal serving",
            "NVIDIA / AMD / Ascend, depending on image",
            launch_adapter=True,
            note="Single-node launch adapter exists; distributed bootstrap is not validated.",
        ),
        CatalogRuntime(
            "llama-cpp",
            "llama.cpp",
            "ghcr.io/ggml-org/llama.cpp",
            "https://github.com/ggml-org/llama.cpp",
            "Portable GGUF inference server",
            "CPU / CUDA / Vulkan, depending on image",
            SUPPORTED_PLATFORMS,
            "ggml-org/llama.cpp",
            launch_adapter=True,
            note=(
                "Managed GGUF launch is available on Linux compute pools; native packages "
                "remain install-only."
            ),
        ),
        CatalogRuntime(
            "ollama",
            "Ollama",
            "docker.io/ollama/ollama",
            "https://github.com/ollama/ollama",
            "Local model management and serving",
            "CPU / NVIDIA / AMD, depending on image",
            SUPPORTED_PLATFORMS,
            "ollama/ollama",
            launch_adapter=True,
            note="Managed launch imports the selected local model into an isolated Ollama runtime.",
        ),
        CatalogRuntime(
            "tensorrt-llm",
            "TensorRT-LLM",
            "nvcr.io/nvidia/tensorrt-llm/release",
            "https://github.com/NVIDIA/TensorRT-LLM",
            "Optimized NVIDIA LLM inference",
            "NVIDIA; engine builds and GPU generation constraints apply",
            launch_adapter=True,
            note="Managed launch uses the upstream OpenAI-compatible trtllm-serve command.",
        ),
        CatalogRuntime(
            "triton",
            "Triton Inference Server",
            "nvcr.io/nvidia/tritonserver",
            "https://github.com/triton-inference-server/server",
            "Multi-framework inference server",
            "NVIDIA / CPU, depending on backend",
            launch_adapter=True,
            note="Managed launch requires a valid Triton model repository and exposes the V2 API.",
        ),
        CatalogRuntime(
            "tgi",
            "Text Generation Inference",
            "ghcr.io/huggingface/text-generation-inference",
            "https://github.com/huggingface/text-generation-inference",
            "Hugging Face LLM serving",
            "NVIDIA / AMD / Intel, depending on image",
            launch_adapter=True,
            note="Managed launch uses the upstream text-generation-launcher contract.",
        ),
        CatalogRuntime(
            "lmdeploy",
            "LMDeploy",
            "docker.io/openmmlab/lmdeploy",
            "https://github.com/InternLM/lmdeploy",
            "TurboMind and PyTorch LLM serving",
            "NVIDIA; architecture support varies by release",
            launch_adapter=True,
            note="Managed launch uses LMDeploy's OpenAI-compatible API server.",
        ),
        CatalogRuntime(
            "xinference",
            "Xinference",
            "docker.io/xprobe/xinference",
            "https://github.com/xorbitsai/inference",
            "Multi-engine model serving",
            "CPU / NVIDIA / other accelerators, depending on image",
            launch_adapter=True,
            note="Managed launch starts Xinference and loads the selected local model path.",
        ),
        CatalogRuntime(
            "localai",
            "LocalAI",
            "docker.io/localai/localai",
            "https://github.com/mudler/LocalAI",
            "Local OpenAI-compatible model APIs",
            "CPU / NVIDIA / AMD / Vulkan, depending on image",
            ("linux/amd64", "linux/arm64", "darwin/arm64"),
            "mudler/LocalAI",
            launch_adapter=True,
            note=(
                "Managed launch serves a selected local GGUF model; native packages remain "
                "install-only."
            ),
        ),
    ]
}


class RegistryError(ValueError):
    pass


def native_asset_name(runtime_id: str, version: str, platform: str) -> str | None:
    """Return the one conservative upstream package selected for a native host."""
    values = {
        ("llama-cpp", "darwin/arm64"): f"llama-{version}-bin-macos-arm64.tar.gz",
        ("llama-cpp", "darwin/amd64"): f"llama-{version}-bin-macos-x64.tar.gz",
        ("llama-cpp", "windows/amd64"): f"llama-{version}-bin-win-cpu-x64.zip",
        ("llama-cpp", "windows/arm64"): f"llama-{version}-bin-win-cpu-arm64.zip",
        ("ollama", "darwin/arm64"): "ollama-darwin.tgz",
        ("ollama", "darwin/amd64"): "ollama-darwin.tgz",
        ("ollama", "windows/amd64"): "ollama-windows-amd64.zip",
        ("ollama", "windows/arm64"): "ollama-windows-arm64.zip",
        ("localai", "darwin/arm64"): f"local-ai-{version}-darwin-arm64",
    }
    return values.get((runtime_id, platform))


def archive_format(name: str) -> str:
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    return "binary"


def runtime_entry(runtime_id: str) -> CatalogRuntime:
    try:
        return CATALOG[runtime_id]
    except KeyError as error:
        raise KeyError("Unknown runtime") from error


def catalog_entries() -> list[dict]:
    return [asdict(item) for item in CATALOG.values()]


def newest_stable_version(
    versions: list[dict], platform: str, runtime_id: str | None = None
) -> str | None:
    """Choose a reviewed stable channel for a runtime and target platform."""
    if runtime_id == "llama-cpp":
        published = {
            str(item.get("version", ""))
            for item in versions
            if not item.get("platforms") or platform in item.get("platforms", [])
        }
        for channel in ("server-cuda", "server-cuda13", "server"):
            if channel in published:
                return channel
    candidates = []
    for item in versions:
        version = str(item.get("version", ""))
        match = STABLE_VERSION.fullmatch(version)
        platforms = item.get("platforms") or []
        if platforms and platform not in platforms:
            continue
        if runtime_id == "triton":
            calendar = CALENDAR_VERSION.fullmatch(version)
            if calendar:
                year, month = calendar.groups()
                candidates.append(((int(year), int(month), 0), True, version))
            continue
        if not match:
            continue
        major, minor, patch, cuda = match.groups()
        candidates.append(((int(major), int(minor), int(patch)), cuda is None, version))
    return max(candidates)[2] if candidates else None


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

    async def versions(
        self, runtime_id: str, cursor: str = "", limit: int = 50, platform: str = "linux/amd64"
    ) -> dict:
        runtime = runtime_entry(runtime_id)
        if platform not in runtime.platforms:
            raise RegistryError(f"{runtime.name} is not distributed for {platform}")
        if cursor and not re.fullmatch(TAG, cursor):
            raise RegistryError("Invalid version cursor")
        if platform.startswith(("darwin/", "windows/")):
            return await self._native_versions(runtime, cursor, limit, platform)
        key = (runtime_id, cursor, limit, platform)
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

    async def latest_stable(self, runtime_id: str, platform: str) -> dict:
        """Resolve the newest final semantic release to immutable OCI metadata."""
        if runtime_id == "llama-cpp" and platform.startswith("linux/"):
            metadata = await self.resolve(runtime_id, "server-cuda", platform)
            metadata["stale"] = False
            return metadata
        cursor = ""
        seen: list[dict] = []
        stale = False
        for _ in range(5):
            page = await self.versions(runtime_id, cursor=cursor, limit=100, platform=platform)
            seen.extend(page["versions"])
            stale = stale or bool(page.get("stale"))
            version = newest_stable_version(seen, platform, runtime_id)
            if version:
                metadata = await self.resolve(runtime_id, version, platform)
                metadata["stale"] = stale
                return metadata
            cursor = page.get("next_cursor") or ""
            if not cursor:
                break
        raise RegistryError("No supported stable release is published for this platform")

    async def _native_versions(
        self, runtime: CatalogRuntime, cursor: str, limit: int, platform: str
    ) -> dict:
        if not runtime.native_repository or not native_asset_name(runtime.id, "VERSION", platform):
            raise RegistryError(f"{runtime.name} has no managed native package for {platform}")
        if cursor and not cursor.isdecimal():
            raise RegistryError("Invalid GitHub release page")
        key = (runtime.id, cursor, limit, platform)
        async with self.locks.setdefault(f"{runtime.id}:{platform}", asyncio.Lock()):
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < 300:
                return {**cached[1], "cached": True}
            try:
                async with httpx.AsyncClient(transport=self.transport, timeout=30) as client:
                    response = await client.get(
                        f"https://api.github.com/repos/{runtime.native_repository}/releases",
                        params={"per_page": limit, "page": cursor or "1"},
                        headers={"Accept": "application/vnd.github+json"},
                    )
                    response.raise_for_status()
                    releases = response.json()
                versions = []
                for release in releases:
                    version = str(release.get("tag_name", ""))
                    expected = native_asset_name(runtime.id, version, platform)
                    if expected and any(
                        asset.get("name") == expected for asset in release.get("assets", [])
                    ):
                        versions.append(
                            {
                                "version": version,
                                "updated_at": release.get("published_at"),
                                "platforms": [platform],
                            }
                        )
                next_cursor = str(int(cursor or "1") + 1) if len(releases) == limit else None
                result = {
                    "runtime_id": runtime.id,
                    "repository": runtime.native_repository,
                    "versions": versions,
                    "next_cursor": next_cursor,
                    "cached": False,
                    "stale": False,
                    "ordering": "recent GitHub releases",
                }
                self.cache[key] = (time.monotonic(), result)
                while len(self.cache) > 100:
                    self.cache.pop(next(iter(self.cache)))
                return result
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                if cached:
                    return {
                        **cached[1],
                        "cached": True,
                        "stale": True,
                        "warning": "GitHub unavailable; showing cached releases",
                    }
                raise RegistryError("GitHub releases are unavailable or rate limited") from error

    async def resolve(self, runtime_id: str, version: str, platform: str) -> dict:
        runtime = runtime_entry(runtime_id)
        if not re.fullmatch(TAG, version):
            raise RegistryError("Invalid runtime version")
        if platform not in runtime.platforms:
            raise RegistryError(f"{runtime.name} is not distributed for {platform}")
        if platform.startswith(("darwin/", "windows/")):
            return await self._resolve_native(runtime, version, platform)
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

    async def _resolve_native(self, runtime: CatalogRuntime, version: str, platform: str) -> dict:
        expected_name = native_asset_name(runtime.id, version, platform)
        if not runtime.native_repository or not expected_name:
            raise RegistryError(f"{runtime.name} has no managed native package for {platform}")
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=30) as client:
                response = await client.get(
                    f"https://api.github.com/repos/{runtime.native_repository}/releases/tags/{version}",
                    headers={"Accept": "application/vnd.github+json"},
                )
                response.raise_for_status()
                release = response.json()
            asset = next(
                item for item in release.get("assets", []) if item.get("name") == expected_name
            )
            digest = str(asset.get("digest") or "")
            url = str(asset.get("browser_download_url") or "")
            parsed = urlparse(url)
            if (
                not re.fullmatch(DIGEST, digest)
                or parsed.scheme != "https"
                or parsed.hostname != "github.com"
            ):
                raise RegistryError("Native package has no trusted SHA256 release identity")
            return {
                "runtime_id": runtime.id,
                "version": version,
                "platform": platform,
                "artifact_kind": "native",
                "repository": runtime.native_repository,
                "asset_name": expected_name,
                "archive_format": archive_format(expected_name),
                "download_url": url,
                "digest": digest,
                "size_bytes": int(asset.get("size") or 0),
            }
        except StopIteration as error:
            raise RegistryError(f"Release {version} has no {platform} package") from error
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, RegistryError):
                raise
            raise RegistryError("Native release metadata is unavailable") from error
