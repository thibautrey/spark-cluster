import json

import httpx
import pytest

from spark_cluster.config import Settings
from spark_cluster.models import BenchmarkRequest, CachePrefixRequest
from spark_cluster.optimization_lab import OptimizationLab
from spark_cluster.store import SnapshotStore
from spark_cluster.telemetry import InferenceTelemetryService


@pytest.mark.asyncio
async def test_benchmark_streams_measured_results_and_persists_them(tmp_path):
    settings = Settings(
        database_path=tmp_path / "state.db",
        openai_api_base="http://runtime.test/v1",
    )
    store = SnapshotStore(settings.database_path)
    telemetry = InferenceTelemetryService(settings, store)
    lab = OptimizationLab(settings, store, telemetry)

    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        body = "\n".join(
            [
                'data: {"choices":[{"delta":{"content":"one"}}]}',
                'data: {"choices":[{"delta":{"content":"two"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(200, text=body)

    lab.transport = httpx.MockTransport(handle)
    result = await lab.benchmark(
        BenchmarkRequest(model_id="acme/model", prompt="private prompt", repetitions=2)
    )
    assert result.measured is True
    assert result.successful_requests == 2
    assert result.prompt_tokens == 24
    assert result.completion_tokens == 4
    assert result.ttft_p50_ms is not None
    assert len(store.list_benchmarks()) == 1
    assert "private prompt" not in settings.database_path.read_bytes().decode(errors="ignore")
    store.close()


@pytest.mark.asyncio
async def test_prefix_lifecycle_persists_hashes_without_raw_prompt(tmp_path):
    settings = Settings(
        database_path=tmp_path / "state.db",
        openai_api_base="http://runtime.test/v1",
    )
    store = SnapshotStore(settings.database_path)
    telemetry = InferenceTelemetryService(settings, store)
    telemetry.transport = httpx.MockTransport(
        lambda _: httpx.Response(200, text="lmcache:cache_hits_total 2\n")
    )
    lab = OptimizationLab(settings, store, telemetry)
    lab.transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"usage": {"prompt_tokens": 42}})
    )
    request = CachePrefixRequest(
        model_id="acme/model",
        tokenizer_id="acme/model",
        prompt="highly private reusable context",
    )
    record = await lab.prewarm(request)
    assert record.state == "prewarmed"
    assert record.prompt_tokens == 42
    assert record.prefix_hash != request.prompt
    persisted = store.list_cache_prefixes()[0]
    assert persisted.id == record.id
    assert request.prompt not in settings.database_path.read_bytes().decode(errors="ignore")
    store.close()
