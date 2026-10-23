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
