import asyncio
from types import SimpleNamespace

from spark_cluster.token_usage_cache import TokenUsageCache


def test_precalculation_hits_refresh_and_isolation():
    calls = []

    def read(scope, entity, hours):
        calls.append((scope, entity, hours))
        return {"entities": [{"scope": "model", "entity": "model/a"}],
                "summary": {"input_tokens": len(calls)}}

    async def run():
        cache = TokenUsageCache(SimpleNamespace(performance_summary=read))
        await cache.refresh()
        assert len(calls) == 6
        first = await cache.get("model/a", 24)
        assert first["cache"]["hit"]
        first["summary"]["input_tokens"] = -1
        assert (await cache.get("model/a", 24))["summary"]["input_tokens"] != -1
        assert len(calls) == 6
        results = await asyncio.gather(*(cache.get("custom", 48) for _ in range(8)))
        assert len(calls) == 7
        assert sum(not r["cache"]["hit"] for r in results) == 1
        await cache.refresh()
        assert ("model", "custom", 48) in calls[7:]
        assert len(calls) == 14
        cache.max_entries = 8
        for i in range(20):
            await cache.get(str(i), 1)
        assert len(cache.entries) == 8

    asyncio.run(run())


def test_refresh_failure_keeps_last_good_result():
    def read(*args):
        return {"summary": {"input_tokens": 17}}

    async def run():
        cache = TokenUsageCache(SimpleNamespace(performance_summary=read))
        await cache.get(None, 24)

        def fail(*args):
            raise OSError("unavailable")

        cache.journal.performance_summary = fail
        await cache.start()
        await asyncio.sleep(0.05)
        result = await cache.get(None, 24)
        assert result["summary"]["input_tokens"] == 17
        assert result["cache"]["refresh_error"]
        await cache.stop()

    asyncio.run(run())
