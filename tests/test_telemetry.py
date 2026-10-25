import time

import httpx
import pytest

from spark_cluster.config import Settings
from spark_cluster.store import SnapshotStore
from spark_cluster.telemetry import InferenceTelemetryService, parse_prometheus

METRICS = """
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1"} 2
vllm:time_to_first_token_seconds_bucket{le="0.2"} 8
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10
vllm:inter_token_latency_seconds_bucket{le="0.01"} 7
vllm:inter_token_latency_seconds_bucket{le="0.02"} 10
vllm:inter_token_latency_seconds_bucket{le="+Inf"} 10
vllm:prompt_tokens_total 70
vllm:generation_tokens_total 140
vllm:gpu_cache_usage_perc 0.75
vllm:spec_decode_draft_acceptance_rate 82
vllm:num_requests_running 3
vllm:num_requests_waiting 1
lmcache:cache_hits_total 11
lmcache:cache_misses_total 2
"""


def test_prometheus_parser_and_measured_runtime_telemetry(tmp_path):
    parsed = parse_prometheus(METRICS)
    assert parsed["vllm_prompt_tokens_total"][0][1] == 70
    store = SnapshotStore(tmp_path / "state.db")
    service = InferenceTelemetryService(Settings(database_path=tmp_path / "state.db"), store)
    service._previous = (time.monotonic() - 2, {"prompt": 50, "decode": 100})
    result = service.from_metrics(METRICS)
    assert result.available is True
    assert result.ttft_p50_ms == 200
    assert result.ttft_p95_ms is None
    assert result.inter_token_p50_ms == 10
    assert 9 < result.prefill_tokens_per_second < 11
    assert 19 < result.decode_tokens_per_second < 21
    assert result.speculative_acceptance_rate == 0.82
    assert result.gpu_kv_cache_usage == 0.75
    assert result.cache_hits == 11
    assert result.active_requests == 3
    store.save_telemetry(result)
    assert store.latest_telemetry() == result
    store.close()


@pytest.mark.asyncio
async def test_multiple_runtime_metrics_keep_independent_counter_baselines(tmp_path):
    store = SnapshotStore(tmp_path / "state.db")
    service = InferenceTelemetryService(Settings(database_path=tmp_path / "state.db"), store)
    counts = {"first": 100, "second": 10}
    targets = {name: f"http://{name}/metrics" for name in counts}
    service.target_provider = lambda: targets

    def metrics(request):
        count = counts[request.url.host]
        return httpx.Response(200, text=f"vllm:prompt_tokens_total {count}\nvllm:num_requests_running 1\n")

    service.transport = httpx.MockTransport(metrics)
    first = await service.sample()
    assert first.input_tokens_total == 110
    assert set(first.runtimes) == {"first", "second"}
    assert first.active_requests == 2
    assert first.ttft_p95_ms is None
    for url in targets.values():
        service._target_previous[url] = (time.monotonic() - 2, service._target_previous[url][1])
    counts.update(first=120, second=0)  # Only the second runtime restarted.
    second = await service.sample()
    assert 9 < second.prefill_tokens_per_second < 11
    assert second.runtimes["second"]["prefill_tokens_per_second"] is None
    targets.pop("second")
    final = await service.sample()
    assert set(final.runtimes) == {"first"}
    assert "http://second/metrics" not in service._target_previous
    store.close()
