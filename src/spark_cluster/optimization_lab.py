"""Repeatable runtime benchmarks and privacy-preserving prefix-cache lifecycle tracking."""

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from uuid import uuid4

import httpx

from .config import Settings
from .models import BenchmarkRequest, BenchmarkResult, CachePrefixRecord, CachePrefixRequest
from .store import SnapshotStore
from .telemetry import InferenceTelemetryService


class OptimizationLab:
    def __init__(
        self, settings: Settings, store: SnapshotStore, telemetry: InferenceTelemetryService
    ):
        self.settings = settings
        self.store = store
        self.telemetry = telemetry
        self.transport: httpx.AsyncBaseTransport | None = None

    def _headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {self.settings.openai_api_key}"}
            if self.settings.openai_api_key
            else {}
        )

    async def benchmark(self, request: BenchmarkRequest) -> BenchmarkResult:
        launches = self.store.list_launches()
        launch = next((item for item in launches if item.status == "ready"), None)
        engine_id = (
            launch.execution_plan.engine_id if launch and launch.execution_plan else "unknown"
        )
        engine_image = launch.execution_plan.image if launch and launch.execution_plan else None
        topology = "single"
        if launch:
            ranks = launch.request.tensor_parallel * launch.request.pipeline_parallel
            topology = "single" if ranks == 1 else f"{ranks}-node"
        semaphore = asyncio.Semaphore(request.concurrency)

        async def one() -> dict:
            async with semaphore:
                return await self._stream_completion(
                    request.model_id, request.prompt, request.max_tokens
                )

        started = time.perf_counter()
        samples = await asyncio.gather(
            *(one() for _ in range(request.repetitions * request.concurrency)),
            return_exceptions=True,
        )
        elapsed = max(time.perf_counter() - started, 1e-9)
        successful = [item for item in samples if isinstance(item, dict)]
        errors = [item for item in samples if isinstance(item, Exception)]
        ttfts = [item["ttft_ms"] for item in successful]
        itls = [value for item in successful for value in item["inter_token_ms"]]
        completions = sum(item["completion_tokens"] for item in successful)
        prompts = sum(item["prompt_tokens"] for item in successful)
        result = BenchmarkResult(
            id=uuid4().hex,
            model_id=request.model_id,
            model_revision=launch.fit.revision if launch else None,
            engine_id=engine_id,
            engine_image=engine_image,
            topology=topology,
            context_length=launch.request.context_length if launch else 0,
            concurrency=request.concurrency,
            cache_state=request.cache_state,
            speculative_method=launch.request.speculative_method if launch else "off",
            expert_parallel=launch.request.expert_parallel if launch else False,
            repetitions=request.repetitions,
            successful_requests=len(successful),
            ttft_p50_ms=_percentile(ttfts, 0.5),
            ttft_p95_ms=_percentile(ttfts, 0.95),
            inter_token_p50_ms=_percentile(itls, 0.5),
            inter_token_p95_ms=_percentile(itls, 0.95),
            output_tokens_per_second=round(completions / elapsed, 3) if successful else None,
            prompt_tokens=prompts,
            completion_tokens=completions,
            error=(
                f"{len(errors)} request(s) failed: {type(errors[0]).__name__}" if errors else None
            ),
        )
        await asyncio.to_thread(self.store.save_benchmark, result)
        return result

    async def prewarm(self, request: CachePrefixRequest) -> CachePrefixRecord:
        canonical_config = json.dumps(request.engine_config, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(canonical_config.encode()).hexdigest()
        prefix_hash = hashlib.sha256(request.prompt.encode()).hexdigest()
        identity = "|".join(
            (
                request.model_id,
                request.model_revision,
                request.tokenizer_id,
                request.engine_id,
                config_hash,
                prefix_hash,
            )
        )
        record_id = hashlib.sha256(identity.encode()).hexdigest()
        record = await asyncio.to_thread(self.store.get_cache_prefix, record_id)
        previous_generation = record.runtime_generation if record else None
        before_hits = self.telemetry.latest.cache_hits
        started = time.perf_counter()
        error = None
        prompt_tokens = 0
        try:
            response = await self._completion(request.model_id, request.prompt, 1)
            prompt_tokens = int((response.get("usage") or {}).get("prompt_tokens") or 0)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            error = f"{type(exc).__name__}: runtime prewarm failed"
        ttft = round((time.perf_counter() - started) * 1000, 3)
        await self.telemetry.sample()
        after_hits = self.telemetry.latest.cache_hits
        hit_delta = (
            max(0, after_hits - before_hits)
            if before_hits is not None and after_hits is not None
            else None
        )
        generation = self._runtime_generation()
        if error:
            state = "failed"
        elif record and previous_generation != generation and hit_delta is not None:
            state = "reused" if hit_delta > 0 else "rebuilt"
        else:
            state = "prewarmed"
        record = CachePrefixRecord(
            id=record_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            tokenizer_id=request.tokenizer_id,
            engine_id=request.engine_id,
            engine_config_hash=config_hash,
            prefix_hash=prefix_hash,
            prompt_characters=len(request.prompt),
            prompt_tokens=prompt_tokens or (record.prompt_tokens if record else 0),
            priority=request.priority,
            prewarm_count=(record.prewarm_count if record else 0) + 1,
            verification_count=(record.verification_count if record else 0) + (1 if record else 0),
            last_ttft_ms=ttft,
            last_cache_hit_delta=hit_delta,
            state=state,
            runtime_generation=generation,
            created_at=record.created_at if record else datetime.now(UTC),
            updated_at=datetime.now(UTC),
            error=error,
        )
        await asyncio.to_thread(self.store.save_cache_prefix, record)
        return record

    async def _completion(self, model: str, prompt: str, max_tokens: int) -> dict:
        async with httpx.AsyncClient(
            base_url=self.settings.openai_api_base.rstrip("/"),
            headers=self._headers(),
            timeout=300,
            transport=self.transport,
        ) as client:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()

    async def _stream_completion(self, model: str, prompt: str, max_tokens: int) -> dict:
        started = time.perf_counter()
        first = None
        previous = None
        intervals: list[float] = []
        prompt_tokens = completion_tokens = chunks = 0
        async with httpx.AsyncClient(
            base_url=self.settings.openai_api_base.rstrip("/"),
            headers=self._headers(),
            timeout=300,
            transport=self.transport,
        ) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    payload = json.loads(line[6:])
                    usage = payload.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                    delta = ((payload.get("choices") or [{}])[0].get("delta") or {}).get("content")
                    if not delta:
                        continue
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    elif previous is not None:
                        intervals.append((now - previous) * 1000)
                    previous = now
                    chunks += 1
        if first is None:
            raise ValueError("Runtime returned no generated tokens")
        return {
            "ttft_ms": (first - started) * 1000,
            "inter_token_ms": intervals,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens or chunks,
        }

    def _runtime_generation(self) -> str | None:
        launch = next((item for item in self.store.list_launches() if item.status == "ready"), None)
        return launch.id if launch else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math_floor = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - math_floor
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)
