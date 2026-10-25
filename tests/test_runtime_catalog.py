import hashlib
import json
import time

import httpx
import pytest

from spark_cluster.runtime_catalog import (
    CATALOG,
    RegistryError,
    RuntimeRegistry,
    newest_stable_version,
)


def encoded(value):
    content = json.dumps(value).encode()
    return content, "sha256:" + hashlib.sha256(content).hexdigest()


def test_newest_stable_version_filters_channels_and_prefers_generic_release():
    versions = [
        {"version": "nightly-20261025", "platforms": ["linux/amd64"]},
        {"version": "v0.6.0-rc1", "platforms": ["linux/amd64"]},
        {"version": "v0.5.22-cu130", "platforms": ["linux/amd64"]},
        {"version": "v0.5.22", "platforms": ["linux/amd64", "linux/arm64"]},
        {"version": "v0.5.21", "platforms": ["linux/amd64"]},
        {"version": "v9.0.0", "platforms": ["linux/arm64"]},
    ]
    assert newest_stable_version(versions, "linux/amd64") == "v0.5.22"
    assert newest_stable_version(versions, "linux/s390x") is None


def test_catalogue_runtimes_have_managed_launch_channels():
    assert all(runtime.launch_adapter for runtime in CATALOG.values())
    versions = [
        {"version": "server", "platforms": ["linux/amd64"]},
        {"version": "server-cuda", "platforms": ["linux/amd64", "linux/arm64"]},
        {"version": "server-cuda13", "platforms": ["linux/amd64"]},
    ]
    assert newest_stable_version(versions, "linux/arm64", "llama-cpp") == "server-cuda"
    calendar = [
        {"version": "26.08-py3", "platforms": ["linux/amd64"]},
        {"version": "26.09-py3", "platforms": ["linux/amd64"]},
        {"version": "nightly", "platforms": ["linux/amd64"]},
    ]
    assert newest_stable_version(calendar, "linux/amd64", "triton") == "26.09-py3"


@pytest.mark.asyncio
async def test_docker_hub_pagination_cache_and_stale_fallback():
    calls = []
    offline = False

    def handle(request):
        calls.append(request)
        if offline:
            return httpx.Response(429)
        assert request.url.params["page_size"] == "2"
        assert request.url.params["page"] == "1"
        return httpx.Response(
            200,
            json={
                "next": "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page=2",
                "results": [
                    {
                        "name": "v0.9.2",
                        "images": [
                            {"os": "linux", "architecture": "arm64"},
                            {"os": "unknown", "architecture": "unknown"},
                        ],
                    }
                ],
            },
        )

    registry = RuntimeRegistry(httpx.MockTransport(handle))
    first = await registry.versions("vllm", limit=2)
    assert first["next_cursor"] == "2"
    assert first["versions"][0]["platforms"] == ["linux/arm64"]
    assert (await registry.versions("vllm", limit=2))["cached"]
    assert len(calls) == 1
    key = ("vllm", "", 2, "linux/amd64")
    registry.cache[key] = (time.monotonic() - 301, first)
    offline = True
    result = await registry.versions("vllm", limit=2)
    assert result["stale"] and result["warning"]


@pytest.mark.asyncio
async def test_oci_auth_pagination_and_immutable_architecture_resolution():
    config, config_digest = encoded({"architecture": "arm64", "os": "linux"})
    image, image_digest = encoded({"config": {"digest": config_digest}, "layers": [{"size": 42}]})
    index, _ = encoded(
        {
            "manifests": [
                {"digest": image_digest, "platform": {"os": "linux", "architecture": "arm64"}}
            ]
        }
    )
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path == "/token":
            assert request.url.params["scope"] == "repository:ggml-org/llama.cpp:pull"
            return httpx.Response(200, json={"token": "public-test-token"})
        if request.headers.get("authorization") != "Bearer public-test-token":
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io"'
                },
            )
        if request.url.path.endswith("tags/list"):
            assert request.url.params["last"] == "server-b1"
            return httpx.Response(
                200,
                json={"tags": ["server-b2"]},
                headers={
                    "link": '</v2/ggml-org/llama.cpp/tags/list?last=server-b2&n=1>; rel="next"'
                },
            )
        if request.url.path.endswith("/manifests/server-b2"):
            return httpx.Response(200, content=index)
        if request.url.path.endswith("/manifests/" + image_digest):
            return httpx.Response(200, content=image)
        if request.url.path.endswith("/blobs/" + config_digest):
            return httpx.Response(200, content=config)
        raise AssertionError(request.url)

    registry = RuntimeRegistry(httpx.MockTransport(handle))
    versions = await registry.versions("llama-cpp", "server-b1", limit=1)
    assert versions["next_cursor"] == "server-b2"
    result = await registry.resolve("llama-cpp", "server-b2", "linux/arm64")
    assert result["digest"] == image_digest
    assert result["config_digest"] == config_digest
    assert result["size_bytes"] == 42
    with pytest.raises(RegistryError, match="no linux/amd64"):
        await registry.resolve("llama-cpp", "server-b2", "linux/amd64")


@pytest.mark.asyncio
async def test_registry_rejects_untrusted_auth_realm_and_corrupt_manifest():
    def hostile(request):
        return httpx.Response(
            401, headers={"www-authenticate": 'Bearer realm="http://127.0.0.1/secret",service="x"'}
        )

    registry = RuntimeRegistry(httpx.MockTransport(hostile))
    with pytest.raises(RegistryError, match="unsupported authentication"):
        await registry.resolve("llama-cpp", "server", "linux/arm64")
    registry.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"config": {}}, headers={"docker-content-digest": "sha256:" + "a" * 64}
        )
    )
    with pytest.raises(RegistryError, match="digest verification"):
        await registry.resolve("llama-cpp", "server", "linux/arm64")


@pytest.mark.asyncio
async def test_single_architecture_is_checked_from_image_config():
    config, digest = encoded({"os": "linux", "architecture": "amd64"})
    image, _ = encoded({"config": {"digest": digest}, "layers": []})
    registry = RuntimeRegistry(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=config if "/blobs/" in request.url.path else image
            )
        )
    )
    with pytest.raises(RegistryError, match="no linux/arm64"):
        await registry.resolve("ollama", "0.9.2", "linux/arm64")


@pytest.mark.asyncio
async def test_catalog_is_curated_and_never_interprets_tag_or_cursor_as_url():
    registry = RuntimeRegistry(
        httpx.MockTransport(lambda request: pytest.fail("Unexpected network"))
    )
    assert len(CATALOG) == 10
    with pytest.raises(KeyError):
        await registry.versions("https://internal/secret")
    with pytest.raises(RegistryError):
        await registry.resolve("vllm", "$(whoami)", "linux/arm64")
    with pytest.raises(RegistryError):
        await registry.versions("vllm", "../../etc/passwd")
    with pytest.raises(RegistryError):
        await registry.resolve("vllm", "latest", "darwin/arm64")


@pytest.mark.asyncio
async def test_native_release_versions_and_sha256_asset_resolution():
    digest = "sha256:" + "a" * 64

    def handle(request):
        if request.url.path.endswith("/releases"):
            assert request.url.params["page"] == "1"
            return httpx.Response(
                200,
                json=[
                    {
                        "tag_name": "b1234",
                        "published_at": "2026-10-24T12:00:00Z",
                        "assets": [{"name": "llama-b1234-bin-macos-arm64.tar.gz"}],
                    },
                    {"tag_name": "v0.7.0", "assets": [{"name": "nightly-tag.txt"}]},
                ],
            )
        if request.url.path.endswith("/releases/tags/b1234"):
            return httpx.Response(
                200,
                json={
                    "assets": [
                        {
                            "name": "llama-b1234-bin-macos-arm64.tar.gz",
                            "size": 42,
                            "digest": digest,
                            "browser_download_url": (
                                "https://github.com/ggml-org/llama.cpp/releases/download/"
                                "b1234/llama-b1234-bin-macos-arm64.tar.gz"
                            ),
                        }
                    ]
                },
            )
        raise AssertionError(request.url)

    registry = RuntimeRegistry(httpx.MockTransport(handle))
    versions = await registry.versions("llama-cpp", limit=10, platform="darwin/arm64")
    assert [item["version"] for item in versions["versions"]] == ["b1234"]
    assert versions["ordering"] == "recent GitHub releases"
    package = await registry.resolve("llama-cpp", "b1234", "darwin/arm64")
    assert package["artifact_kind"] == "native"
    assert package["archive_format"] == "tar.gz"
    assert package["digest"] == digest


@pytest.mark.asyncio
async def test_runtime_platform_matrix_refuses_unpublished_native_packages():
    registry = RuntimeRegistry(
        httpx.MockTransport(lambda request: pytest.fail("Unexpected network"))
    )
    assert "windows/arm64" in CATALOG["ollama"].platforms
    assert "darwin/arm64" in CATALOG["localai"].platforms
    assert CATALOG["vllm"].platforms == ("linux/amd64", "linux/arm64")
    with pytest.raises(RegistryError, match="not distributed"):
        await registry.versions("vllm", platform="windows/amd64")
    with pytest.raises(RegistryError, match="not distributed"):
        await registry.resolve("localai", "v4.12.1", "windows/amd64")
