"""Bounded token-history cache, prewarmed independently of MCP requests."""

import asyncio
import contextlib
import copy
import logging
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TokenUsageCache:
    interval = 60
    max_entries = 128
    periods = (24, 168, 720)

    def __init__(self, journal):
        self.journal = journal
        self.entries = OrderedDict()
        self.lock = asyncio.Lock()
        self.task = None
        self.refresh_error = False

    async def _compute(self, key):
        model, hours = key
        work = asyncio.create_task(asyncio.to_thread(
            self.journal.performance_summary, "model" if model else "overall",
            model or "all", hours,
        ))
        try:
            result = await asyncio.shield(work)
        except asyncio.CancelledError:
            await work
            raise
        self.entries[key] = (time.time(), result)
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return self.entries[key]

    async def get(self, model, hours):
        key = (model, hours)
        entry = self.entries.get(key)
        hit = entry is not None
        if entry is None:
            async with self.lock:
                entry = self.entries.get(key)
                hit = entry is not None
                if entry is None:
                    entry = await self._compute(key)
        self.entries.move_to_end(key)
        generated, value = entry
        result = copy.deepcopy(value)
        result["cache"] = dict(
            hit=hit, generated_at=generated, age_seconds=max(0, time.time() - generated),
            refresh_interval_seconds=self.interval, refresh_error=self.refresh_error,
        )
        return result

    async def refresh(self):
        async with self.lock:
            # Refresh custom requested windows too, with bounded cache residency.
            keys = list(self.entries)
            for hours in self.periods:
                await self._compute((None, hours))
            overall = self.entries[(None, 720)][1]
            models = [e["entity"] for e in overall.get("entities", []) if e["scope"] == "model"]
            for model in models[:24]:
                for hours in self.periods:
                    key = (model, hours)
                    if key not in keys:
                        keys.append(key)
            for key in keys:
                if key[0] is not None or key[1] not in self.periods:
                    await self._compute(key)
        self.refresh_error = False

    async def _loop(self):
        while True:
            try:
                await self.refresh()
            except Exception:
                self.refresh_error = True
                logger.exception("Token usage precalculation failed; retaining cached results")
            await asyncio.sleep(self.interval)

    async def start(self):
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
