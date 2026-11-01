"""Bounded private app-prefix profiles, restored opportunistically after runtime restarts.

These are requests to the inference engine only: generated tool calls are never executed.
The runtime's durable KV store owns correctness and invalidation. Profiles only bring
frequently used prefixes and kernels back into the hot tier while the runtime is idle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import httpx

MAX_PROFILES = 12
MAX_TEXT = 32_768
MAX_BYTES = 262_144


class PrefixWarmup:
    def __init__(self, root: Path, external):
        self.root = root
        self.external = external
        self.lock = threading.RLock()
        self.task: asyncio.Task | None = None
        self.attempted: set[str] = set()
        self.last_uptime: dict[str, float] = {}
        self.results: dict[str, dict] = {}
        self.transport = None

    def _read(self) -> list[dict]:
        result = []
        if not self.root.exists():
            return result
        for path in self.root.glob("*.json"):
            try:
                if path.stat().st_size <= MAX_BYTES + 4096:
                    value = json.loads(path.read_text())
                    if (
                        isinstance(value, dict)
                        and value.get("id") == path.stem
                        and isinstance(value.get("body"), dict)
                        and isinstance(value.get("model"), str)
                    ):
                        result.append(value)
            except (OSError, ValueError):
                continue
        return result

    def remember(self, payload: dict) -> None:
        model, messages = payload.get("model"), payload.get("messages")
        if not isinstance(model, str) or not isinstance(messages, list):
            return
        prefix = []
        remaining = MAX_TEXT
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                break  # No image URL refetches, tool results or replayed tool invocations.
            if message.get("role") not in {"system", "developer", "user"}:
                break
            text = message["content"][:remaining]
            if message["role"] == "user" and len(text) < 2048:
                text = "Reply with OK."
            prefix.append({"role": message["role"], "content": text})
            remaining -= len(text)
            if remaining <= 0 or message["role"] == "user":
                break
        if not prefix or sum(len(m["content"]) for m in prefix) < 1024:
            return
        if prefix[-1]["role"] != "user":
            prefix.append({"role": "user", "content": "Reply with OK."})
        body = {
            key: payload[key]
            for key in (
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
                "functions",
                "function_call",
            )
            if key in payload
        }
        body.update(
            model=model,
            messages=prefix,
            temperature=0,
            max_tokens=1,
            stream=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        encoded = json.dumps(body, ensure_ascii=False).encode()
        if len(encoded) > MAX_BYTES:
            return
        identity = {
            "model": model,
            "tools": body.get("tools", body.get("functions")),
            "leading": [{"role": m["role"], "content": m["content"][:1024]} for m in prefix],
        }
        profile_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with self.lock:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.root.chmod(0o700)
            profiles = self._read()
            prior = next((p for p in profiles if p["id"] == profile_id), {})
            value = dict(
                id=profile_id,
                model=model,
                body=body,
                bytes=len(encoded),
                seen=int(prior.get("seen", 0)) + 1,
                updated_at=time.time(),
            )
            fd, temp = tempfile.mkstemp(prefix=".prefix-", dir=self.root)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(value, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, self.root / f"{profile_id}.json")
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            profiles = sorted(
                (p for p in profiles if p["id"] != profile_id),
                key=lambda p: p.get("updated_at", 0),
                reverse=True,
            )
            for old in profiles[MAX_PROFILES - 1 :]:
                (self.root / f"{old['id']}.json").unlink(missing_ok=True)

    def status(self) -> dict:
        with self.lock:
            profiles = self._read()
        return {
            "profiles": [
                {key: p.get(key) for key in ("id", "model", "bytes", "seen", "updated_at")}
                for p in profiles
            ],
            "last_results": list(self.results.values()),
            "max_profiles": MAX_PROFILES,
            "max_text_characters": MAX_TEXT,
            "policy": (
                "Private text/tool prefixes only; one-token warmup while idle. No tools execute."
            ),
        }

    async def start(self) -> None:
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def sample(self) -> None:
        cluster = self.external.registry.active()
        if not cluster or not self.external.configured(cluster):
            return
        base = self.external.base_url(cluster).removesuffix("/v1")
        transport = self.transport or self.external.transport
        async with httpx.AsyncClient(timeout=3, transport=transport, trust_env=False) as client:
            try:
                response = await client.get(base + "/health")
                response.raise_for_status()
                health = response.json()
                if health.get("ok") is not True or health.get("fatal"):
                    return
                uptime = health.get("uptime_s")
                if not isinstance(uptime, (int, float)):
                    return
                previous = self.last_uptime.get(cluster.id)
                if previous is not None and uptime < previous - 1:
                    self.attempted.clear()
                self.last_uptime[cluster.id] = uptime
                if health.get("inflight") != 0 or health.get("queued", 0) != 0:
                    return
                if health.get("idle_s") is not None and health["idle_s"] < 5:
                    return
                probe = await self.external.probe(cluster)
                if probe.status != "ready":
                    return
                models = {str(m["id"]) for m in probe.models}
                with self.lock:
                    profiles = sorted(self._read(), key=lambda p: p.get("seen", 0), reverse=True)
                for profile in profiles:
                    key = cluster.id + ":" + profile["id"]
                    if key in self.attempted or profile.get("model") not in models:
                        continue
                    # Mark before dispatch: a timeout is not a reason for an endless warmup loop.
                    self.attempted.add(key)
                    start = time.monotonic()
                    result = {"id": profile["id"], "model": profile["model"], "ok": False}
                    try:
                        reply = await client.post(
                            base + "/v1/chat/completions",
                            json=profile["body"],
                            headers=self.external._headers(cluster),
                            timeout=60,
                        )
                        reply.raise_for_status()
                        value = reply.json()
                        result["ok"] = bool(value.get("choices")) and not value.get("error")
                        result["usage"] = value.get("usage")
                    except (httpx.HTTPError, ValueError, TypeError) as error:
                        result["error"] = type(error).__name__
                    result["elapsed_ms"] = round((time.monotonic() - start) * 1000, 1)
                    result["at"] = time.time()
                    self.results[profile["id"]] = result
                    if len(self.results) > MAX_PROFILES:
                        self.results.pop(next(iter(self.results)))
                    break  # Recheck health/idle before warming the next app profile.
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                return

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            await self.sample()
