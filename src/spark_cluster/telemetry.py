"""Collect measured inference and cache telemetry from Prometheus endpoints."""

import asyncio
import math
import re
import time
from collections import defaultdict
from datetime import UTC, datetime

import httpx

from .config import Settings
from .models import InferenceTelemetry
from .store import SnapshotStore

SAMPLE = re.compile(r"^([^\s{]+)(?:\{([^}]*)\})?\s+([-+\w.eE]+)(?:\s+\d+)?$")
LABEL = re.compile(r'(\w+)="((?:\\.|[^"])*)"')


def _name(value: str) -> str:
    return value.replace(":", "_").replace("-", "_").casefold()


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    result: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line.strip())
        if not match:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        labels = {key: value for key, value in LABEL.findall(match.group(2) or "")}
        result[_name(match.group(1))].append((labels, value))
    return result


def _value(metrics, aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        values = metrics.get(_name(alias), [])
        if values:
            return sum(value for _, value in values)
    return None


def _quantile(metrics, aliases: tuple[str, ...], quantile: float) -> float | None:
    for alias in aliases:
        buckets = metrics.get(_name(alias) + "_bucket", [])
        if not buckets:
            continue
        merged: dict[float, float] = defaultdict(float)
        for labels, value in buckets:
            le = labels.get("le")
            if le is None:
                continue
            boundary = math.inf if le == "+Inf" else float(le)
            merged[boundary] += value
        if not merged:
            continue
        total = merged.get(math.inf, max(merged.values()))
        target = total * quantile
        for boundary, cumulative in sorted(merged.items()):
            if cumulative >= target:
                return boundary if math.isfinite(boundary) else None
    return None


class InferenceTelemetryService:
    def __init__(self, settings: Settings, store: SnapshotStore):
        self.settings = settings
        self.store = store
        self.url = settings.runtime_metrics_url or (
            settings.openai_api_base.rstrip("/").removesuffix("/v1") + "/metrics"
        )
        self.latest = store.latest_telemetry() or InferenceTelemetry(
            notes=["Waiting for a runtime metrics endpoint"]
        )
        self.task: asyncio.Task | None = None
        self.transport: httpx.AsyncBaseTransport | None = None
        self._previous: tuple[float, dict[str, float]] | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def sample(self) -> InferenceTelemetry:
        try:
            async with httpx.AsyncClient(timeout=5, transport=self.transport) as client:
                response = await client.get(self.url)
                response.raise_for_status()
            telemetry = self.from_metrics(response.text)
        except (httpx.HTTPError, ValueError) as error:
            telemetry = InferenceTelemetry(
                source=self.url,
                notes=[f"Runtime metrics unavailable: {type(error).__name__}"],
            )
        self.latest = telemetry
        await asyncio.to_thread(self.store.save_telemetry, telemetry)
        return telemetry

    def from_metrics(self, text: str) -> InferenceTelemetry:
        metrics = parse_prometheus(text)
        now = time.monotonic()
        counters = {
            "prompt": _value(metrics, ("vllm:prompt_tokens_total", "vllm_prompt_tokens_total"))
            or 0,
            "decode": _value(
                metrics, ("vllm:generation_tokens_total", "vllm_generation_tokens_total")
            )
            or 0,
        }
        prefill_rate = decode_rate = None
        if self._previous:
            elapsed = now - self._previous[0]
            if elapsed > 0:
                prefill_rate = max(0, counters["prompt"] - self._previous[1]["prompt"]) / elapsed
                decode_rate = max(0, counters["decode"] - self._previous[1]["decode"]) / elapsed
        self._previous = (now, counters)
        ttft = ("vllm:time_to_first_token_seconds", "vllm_time_to_first_token_seconds")
        itl = ("vllm:inter_token_latency_seconds", "vllm_inter_token_latency_seconds")
        return InferenceTelemetry(
            sampled_at=datetime.now(UTC),
            source=self.url,
            available=True,
            ttft_p50_ms=_milliseconds(_quantile(metrics, ttft, 0.5)),
            ttft_p95_ms=_milliseconds(_quantile(metrics, ttft, 0.95)),
            inter_token_p50_ms=_milliseconds(_quantile(metrics, itl, 0.5)),
            inter_token_p95_ms=_milliseconds(_quantile(metrics, itl, 0.95)),
            prefill_tokens_per_second=prefill_rate,
            decode_tokens_per_second=decode_rate,
            speculative_acceptance_rate=_ratio(
                _value(
                    metrics,
                    (
                        "vllm:spec_decode_draft_acceptance_rate",
                        "vllm:spec_decode_efficiency",
                    ),
                )
            ),
            gpu_kv_cache_usage=_ratio(
                _value(metrics, ("vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc"))
            ),
            cache_hits=_integer(
                _value(metrics, ("lmcache:cache_hits_total", "lmcache_cache_hits_total"))
            ),
            cache_misses=_integer(
                _value(metrics, ("lmcache:cache_misses_total", "lmcache_cache_misses_total"))
            ),
            cache_retrieval_p95_ms=_milliseconds(
                _quantile(
                    metrics,
                    ("lmcache:retrieve_latency_seconds", "lmcache_retrieve_latency_seconds"),
                    0.95,
                )
            ),
            nccl_collective_p95_ms=_milliseconds(
                _quantile(metrics, ("nccl:collective_latency_seconds",), 0.95)
            ),
            expert_imbalance=_value(metrics, ("vllm:expert_load_imbalance",)),
            active_requests=_integer(
                _value(metrics, ("vllm:num_requests_running", "vllm_num_requests_running"))
            ),
            queued_requests=_integer(
                _value(metrics, ("vllm:num_requests_waiting", "vllm_num_requests_waiting"))
            ),
            notes=[
                "Values are read from the runtime; absent metrics remain unknown rather than zero."
            ],
        )

    async def _loop(self) -> None:
        while True:
            await self.sample()
            await asyncio.sleep(max(1, self.settings.runtime_metrics_poll_seconds))


def _milliseconds(value: float | None) -> float | None:
    return round(value * 1000, 3) if value is not None else None


def _integer(value: float | None) -> int | None:
    return int(value) if value is not None else None


def _ratio(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value / 100 if value > 1 else value, 6)
