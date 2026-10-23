import httpx
import pytest
from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.config import Settings


@pytest.fixture
def catalog(tmp_path):
    settings = Settings(
        mode="live",
        database_path=tmp_path / "catalog.db",
        model_cache_path=tmp_path / "models",
        setup_state_path=tmp_path / "setup",
        hf_token="server-only-secret",
        remote_cluster_url="http://192.168.50.10:30080",
    )
    app = create_app(settings)
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.host == "huggingface.co"
        assert request.headers["authorization"] == "Bearer server-only-secret"
        if request.url.path == "/api/models":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "acme/spark-model",
                        "downloads": 1200,
                        "likes": 42,
                        "pipeline_tag": "text-generation",
                        "library_name": "transformers",
                        "gated": "manual",
                        "tags": ["safetensors"],
                        "lastModified": "2026-10-23T12:00:00Z",
                    }
                ],
                headers={"Link": '<https://huggingface.co/api/models?cursor=page-two>; rel="next"'},
            )
        return httpx.Response(
            200,
            json={
                "id": "acme/spark-model",
                "sha": "commit123",
                "gated": False,
                "cardData": {"license": "apache-2.0"},
                "config": {"architectures": ["LlamaForCausalLM"]},
                "safetensors": {"total": 8000000000},
                "siblings": [
                    {"rfilename": "model.safetensors", "size": 4096},
                    {"rfilename": "config.json", "size": 128},
                ],
            },
        )

    app.state.model_browser.transport = httpx.MockTransport(handle)
    # Deliberately attach a remote cluster: catalog discovery must stay on the controller.
    with TestClient(app) as client:
        yield client, app.state.model_browser, requests


def test_search_filters_pagination_cache_and_controller_credentials(catalog):
    client, _, requests = catalog
    url = (
        "/api/v1/catalog/models?query=spark&sort=downloads"
        "&task=text-generation&library=gguf&access=open"
    )
    result = client.get(url)
    assert result.status_code == 200
    model = result.json()["models"][0]
    assert model["author"] == "acme"
    assert model["gated"] is True
    assert model["downloads"] == 1200
    assert model["thumbnail_url"] == (
        "https://cdn-thumbnails.huggingface.co/social-thumbnails/models/acme/spark-model.png"
    )
    assert result.json()["cursor"] == "page-two"
    assert "server-only-secret" not in result.text
    params = requests[0].url.params
    assert params["pipeline_tag"] == "text-generation"
    assert params["filter"] == "gguf"
    assert params["gated"] == "false"
    assert client.get(url).json() == result.json()
    assert len(requests) == 1
    assert client.get(url + "&cursor=page-two").status_code == 200
    assert requests[-1].url.params["cursor"] == "page-two"


def test_detail_sizes_files_and_revision(catalog):
    client, _, requests = catalog
    result = client.get("/api/v1/catalog/models/acme/spark-model?revision=release/v2")
    assert result.status_code == 200
    model = result.json()
    assert model["repository_bytes"] == 4224
    assert model["size_complete"] is True
    assert model["parameters"] == 8000000000
    assert model["architecture"] == "LlamaForCausalLM"
    assert model["license"] == "apache-2.0"
    assert model["commit"] == "commit123"
    assert model["thumbnail_url"].endswith("/acme/spark-model.png")
    assert len(model["files"]) == 2
    assert requests[-1].url.params["blobs"] == "true"
    assert "release%2Fv2" in str(requests[-1].url)


def test_unknown_file_sizes_are_not_reported_as_complete(catalog):
    client, browser, _ = catalog
    browser.transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "id": "acme/unknown",
                "siblings": [{"rfilename": "model.gguf"}],
            },
        )
    )
    result = client.get("/api/v1/catalog/models/acme/unknown").json()
    assert result["size_complete"] is False
    assert result["files"][0]["size"] is None


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_upstream_errors_are_actionable_and_redacted(catalog, status):
    client, browser, _ = catalog
    browser.transport = httpx.MockTransport(
        lambda _: httpx.Response(
            status,
            text="server-only-secret upstream internal error",
        )
    )
    result = client.get("/api/v1/catalog/models")
    assert result.status_code == (502 if status == 500 else status)
    assert "server-only-secret" not in result.text
    assert "upstream internal error" not in result.text


def test_timeout_and_validation(catalog):
    client, browser, requests = catalog

    def timeout(request):
        raise httpx.ReadTimeout("sensitive upstream request", request=request)

    browser.transport = httpx.MockTransport(timeout)
    result = client.get("/api/v1/catalog/models")
    assert result.status_code == 502
    assert "sensitive" not in result.text
    for query in ("limit=1000", "limit=0", "sort=invalid", "access=private", "query=" + "x" * 201):
        assert client.get("/api/v1/catalog/models?" + query).status_code == 422
    assert client.get("/api/v1/catalog/models/a/b/c").status_code == 422
    assert requests == []


def test_capacity_filter_caches_metadata_and_context_plan(catalog):
    client, browser, requests = catalog
    payload = {
        "models": [{"id": "acme/spark-model", "commit": "commit123"}],
        "spark_count": 1,
        "context_length": 32768,
        "concurrent_sequences": 1,
        "kv_cache_dtype": "auto",
    }
    first = client.post("/api/v1/catalog/fit", json=payload)
    assert first.status_code == 200
    result = first.json()["results"][0]
    assert result["repo_id"] == "acme/spark-model"
    assert result["compatible"] is True
    assert result["minimum_nodes"] == 1
    assert result["cached"] is False
    metadata_requests = len(requests)

    second = client.post("/api/v1/catalog/fit", json=payload)
    assert second.status_code == 200
    assert second.json()["results"][0]["cached"] is True
    assert len(requests) == metadata_requests
    assert second.json()["cache"] == {"metadata_entries": 1, "fit_entries": 1}

    payload["context_length"] = 131072
    third = client.post("/api/v1/catalog/fit", json=payload)
    assert third.status_code == 200
    assert third.json()["results"][0]["cached"] is False
    assert len(requests) == metadata_requests
    assert third.json()["cache"]["fit_entries"] == 2


def test_capacity_filter_validates_batch_bounds(catalog):
    client, _, _ = catalog
    assert (
        client.post(
            "/api/v1/catalog/fit",
            json={"models": [], "spark_count": 1},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/catalog/fit",
            json={"models": [{"id": "acme/model"}], "spark_count": 0},
        ).status_code
        == 422
    )


def test_spark_discovery_ranks_deduplicates_and_paginates_cached_shortlist(catalog):
    client, browser, requests = catalog

    def handle(request):
        requests.append(request)
        params = request.url.params
        assert params["pipeline_tag"] == "text-generation"
        assert params["gated"] == "false"
        assert "safetensors" in params["filter"]
        assert params["sort"] != "spark"
        if params["search"] == "NVFP4":
            models = [{"id": "nvidia/Qwen-NVFP4"}, {"id": "other/Irrelevant-NVFP4"}]
        elif "dgx-spark" in params["filter"]:
            models = [{"id": "acme/Qwen-NVFP4-DGX-Spark", "tags": ["dgx-spark", "nvfp4"]}]
        elif "sm121" in params["filter"]:
            models = [{"id": "acme/Qwen-NVFP4-DGX-Spark", "tags": ["sm121", "nvfp4"]}]
        else:
            models = [{"id": "popular/Qwen", "downloads": 9999999}]
        return httpx.Response(200, json=models)

    browser.transport = httpx.MockTransport(handle)
    url = (
        "/api/v1/catalog/models?query=Qwen&task=text-generation"
        "&library=safetensors&access=open&limit=1"
    )
    first = client.get(url)
    assert first.status_code == 200
    page = first.json()
    assert page["models"][0]["id"] == "acme/Qwen-NVFP4-DGX-Spark"
    assert page["models"][0]["spark"]["tier"] == "spark-targeted"
    assert "shortlist of 3" in page["ranking_note"]
    assert len(requests) == 4
    cursor = page["cursor"]
    second = client.get(url + "&cursor=" + cursor).json()
    assert second["models"][0]["id"] == "nvidia/Qwen-NVFP4"
    third = client.get(url + "&cursor=" + second["cursor"]).json()
    assert third["models"][0]["id"] == "popular/Qwen"
    assert third["cursor"] is None
    assert len(requests) == 4
    # A new browser instance reuses both classification and ordering from SQLite.
    from spark_cluster.catalog import ModelBrowser

    reopened = ModelBrowser(browser.settings)
    assert (
        reopened.persistent_cache.get_metadata("spark-ranking:" + cursor.split(":")[1])[0][
            "models"
        ][0]["spark"]["tier"]
        == "spark-targeted"
    )
    reopened.close()
    assert (
        client.get(url.replace("query=Qwen", "query=Other") + "&cursor=" + cursor).status_code
        == 422
    )
    assert client.get(url + "&cursor=" + cursor.rsplit(":", 1)[0] + ":999").status_code == 422


def test_spark_discovery_supplement_failure_keeps_regular_results(catalog):
    client, browser, _ = catalog

    def handle(request):
        if request.url.params["search"] == "NVFP4" or request.url.params.get("filter"):
            return httpx.Response(503)
        return httpx.Response(200, json=[{"id": "acme/baseline"}])

    browser.transport = httpx.MockTransport(handle)
    response = client.get("/api/v1/catalog/models")
    assert response.status_code == 200
    assert response.json()["models"][0]["id"] == "acme/baseline"
    assert "unavailable" in response.json()["ranking_note"]


def test_explicit_sort_preserves_hub_order_while_adding_badges(catalog):
    client, browser, requests = catalog
    browser.transport = httpx.MockTransport(
        lambda request: (
            requests.append(request)
            or httpx.Response(
                200,
                json=[
                    {"id": "popular/baseline"},
                    {"id": "acme/model-NVFP4"},
                ],
            )
        )
    )
    response = client.get("/api/v1/catalog/models?sort=downloads")
    assert response.status_code == 200
    assert [model["id"] for model in response.json()["models"]] == [
        "popular/baseline",
        "acme/model-NVFP4",
    ]
    assert response.json()["models"][1]["spark"]["score"] > 0
    assert len(requests) == 1
