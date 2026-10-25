"""Bounded, content-free request accounting. No IO on the inference hot path."""

import asyncio
import contextlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections import deque
from uuid import uuid4

from starlette.requests import ClientDisconnect

from .performance_history import DAY, PerformanceHistory

logger = logging.getLogger(__name__)
SCOPE_KEY = "spark.inference"
BODY_LIMIT = 256 * 1024
FRAME_LIMIT = 64 * 1024


def counters():
    return dict(
        requests=0,
        successful=0,
        failed=0,
        cancelled=0,
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        unknown_usage=0,
        duration_ms=0.0,
    )


class InferenceJournal:
    """A separate WAL database isolates telemetry from model download contention."""

    def __init__(self, path, retention, settings):
        self.lock = threading.RLock()
        self.retention = retention
        self.retention_days = settings.inference_retention_days
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=1)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests "
            "(id TEXT PRIMARY KEY, finished REAL NOT NULL, payload TEXT NOT NULL)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS requests_finished ON requests(finished DESC)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS state "
            "(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
        )
        self.db.commit()
        self.performance = PerformanceHistory(self.db, settings)

    def restore(self):
        row = self.db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        recent = self.db.execute(
            "SELECT payload FROM requests ORDER BY finished DESC LIMIT 512"
        ).fetchall()
        return json.loads(row[0]) if row else None, [json.loads(r[0]) for r in recent]

    def save(self, events, state, samples=()):
        with self.lock:
            now = time.time()
            maintenance_due = now - self.performance.last_maintenance >= 60
            with self.db:
                for event in events:
                    inserted = self.db.execute(
                        "INSERT OR IGNORE INTO requests VALUES (?,?,?)",
                        (
                            event["id"],
                            event["finished_at"],
                            json.dumps(event, separators=(",", ":")),
                        ),
                    ).rowcount
                    if inserted:
                        self.performance.request(event, now)
                for sample in samples:
                    self.performance.runtime(sample, now)
                self.db.execute("INSERT OR REPLACE INTO state VALUES (1,?)", (json.dumps(state),))
                self.db.execute(
                    "DELETE FROM requests WHERE finished < ?", (now - self.retention_days * DAY,)
                )
                count = self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
                self.db.execute(
                    "DELETE FROM requests WHERE id IN (SELECT id FROM requests "
                    "ORDER BY finished ASC LIMIT ?)",
                    (max(0, count - self.retention),),
                )
                self.performance.maintain(now)
            if maintenance_due:
                self.performance.last_maintenance = now

    def performance_summary(self, scope, entity, hours):
        with self.lock:
            return self.performance.read(scope, entity, hours)


class InferenceTracker:
    def __init__(self, settings):
        path = settings.database_path.with_name(settings.database_path.stem + "-inference.db")
        self.journal = InferenceJournal(path, settings.inference_retention_requests, settings)
        saved, recent = self.journal.restore()
        self.state = saved or dict(since=time.time(), totals=counters(), groups={}, buckets={})
        self.recent = deque(recent, maxlen=512)
        self.pending = deque(maxlen=4096)
        self.runtime_pending = deque(maxlen=4096)
        self.active = {}
        self.dropped = 0
        self.persistence_error = False
        self.last_flush_at = None
        self.resources = lambda cluster_id: None
        self.pool_state = lambda: []
        self.local_pool = lambda: None
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self._loop())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        await self.flush()
        self.journal.db.close()

    async def _loop(self):
        while True:
            await asyncio.sleep(1)
            await self.flush()

    async def flush(self):
        if (
            not self.pending
            and not self.runtime_pending
            and time.time() - self.journal.performance.last_maintenance < 60
        ):
            return
        events = list(self.pending)
        samples = list(self.runtime_pending)
        self.pending.clear()
        self.runtime_pending.clear()
        state = json.loads(json.dumps(self.state))
        # Shield the thread so shutdown cannot close SQLite during an in-flight transaction.
        write = asyncio.create_task(asyncio.to_thread(self.journal.save, events, state, samples))
        try:
            await asyncio.shield(write)
            self.persistence_error = False
            self.last_flush_at = time.time()
        except asyncio.CancelledError:
            await write
            raise
        except (sqlite3.Error, OSError):
            self.persistence_error = True
            for event in reversed(events):
                if len(self.pending) == self.pending.maxlen:
                    self.dropped += 1
                else:
                    self.pending.appendleft(event)
            for sample in reversed(samples):
                if len(self.runtime_pending) == self.runtime_pending.maxlen:
                    self.dropped += 1
                else:
                    self.runtime_pending.appendleft(sample)
            logger.warning("Inference journal write failed; retrying bounded batch")

    def observe_runtime(self, scopes, timestamp, metrics):
        if len(self.runtime_pending) == self.runtime_pending.maxlen:
            self.dropped += 1
        self.runtime_pending.append(dict(scopes=scopes, timestamp=timestamp, metrics=metrics))

    async def track(self, handler, model_id, path, scope, receive, send):
        if SCOPE_KEY in scope or scope.get("method") in {"GET", "HEAD", "OPTIONS"}:
            return await handler(scope, receive, send)
        record = RequestRecord(self, model_id, path)
        if arrival := scope.get("spark.arrival"):
            record.started_at, record.start = arrival
        scope = dict(scope, **{SCOPE_KEY: record})
        self.active[record.id] = record

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                record.client_disconnected = True
            return message

        try:
            return await handler(scope, observed_receive, record.wrap_send(send))
        except (asyncio.CancelledError, ClientDisconnect):
            if record.outcome != "failed" and (record.status is None or record.status < 400):
                record.outcome = "cancelled"
            raise
        except BaseException:
            record.outcome = "failed"
            raise
        finally:
            self.active.pop(record.id, None)
            self.finish(record)

    def finish(self, record):
        event = record.finish()
        self.recent.appendleft(event)
        if len(self.pending) == self.pending.maxlen:
            self.dropped += 1
        self.pending.append(event)
        group_key = json.dumps([event["model_id"], event["cluster_id"]])
        groups = self.state["groups"]
        if group_key not in groups and len(groups) >= 256:
            group_key = "other"
        group = groups.setdefault(
            group_key,
            dict(
                model_id=event["model_id"] if group_key != "other" else "Other models",
                cluster_id=event["cluster_id"] if group_key != "other" else None,
                **counters(),
            ),
        )
        minute = str(int(event["finished_at"] // 60) * 60)
        if minute not in self.state["buckets"]:
            cutoff = int(time.time() // 60) * 60 - 24 * 3600
            self.state["buckets"] = {
                k: v for k, v in self.state["buckets"].items() if int(k) >= cutoff
            }
        bucket = self.state["buckets"].setdefault(minute, counters())
        for target in (self.state["totals"], group, bucket):
            target["requests"] += 1
            target[event["outcome"]] += 1
            target["input_tokens"] += event["input_tokens"] or 0
            target["output_tokens"] += event["output_tokens"] or 0
            target["cached_tokens"] += event["cached_tokens"] or 0
            target["unknown_usage"] += event["input_tokens"] is None
            target["duration_ms"] += event["duration_ms"]

    def prometheus(self):
        lines = []
        values = {
            "active_requests": len(self.active),
            "queued_requests": sum(r.dispatched_at is None for r in self.active.values()),
            "journal_pending": len(self.pending),
            "journal_dropped_total": self.dropped,
            "journal_healthy": int(not self.persistence_error),
        }
        for key, value in self.state["totals"].items():
            values[key + "_total"] = value
        for key, value in values.items():
            name = "spark_cluster_inference_" + key
            kind = "counter" if key.endswith("_total") else "gauge"
            lines.extend([f"# TYPE {name} {kind}", f"{name} {value}"])
        for group in self.state["groups"].values():
            model_label = json.dumps(group["model_id"], ensure_ascii=False)
            pool_label = json.dumps(group["cluster_id"] or "unassigned", ensure_ascii=False)
            labels = f"model={model_label},pool={pool_label}"
            for key in (
                "requests",
                "successful",
                "failed",
                "input_tokens",
                "output_tokens",
                "cached_tokens",
            ):
                lines.append(f"spark_cluster_inference_pool_{key}_total{{{labels}}} {group[key]}")
        return "\n".join(lines) + "\n"

    def summary(self, limit=20):
        def percentile(key, q):
            values = sorted(
                e[key] for e in self.recent if e[key] is not None and e["outcome"] == "successful"
            )
            return round(values[max(0, math.ceil(len(values) * q) - 1)], 3) if values else None

        now = time.time()
        return dict(
            sampled_at=now,
            since=self.state["since"],
            totals=dict(self.state["totals"]),
            active_requests=len(self.active),
            queued_requests=sum(r.dispatched_at is None for r in self.active.values()),
            active=[r.public() for r in list(self.active.values())[:64]],
            latency=dict(
                p50_ms=percentile("duration_ms", 0.5),
                p95_ms=percentile("duration_ms", 0.95),
                ttft_p50_ms=percentile("ttft_ms", 0.5),
                ttft_p95_ms=percentile("ttft_ms", 0.95),
                sample_size=len(self.recent),
            ),
            groups=list(self.state["groups"].values()),
            history=[
                dict(timestamp=int(k), **v)
                for k, v in sorted(self.state["buckets"].items())
                if int(k) >= now - 3600
            ],
            recent=list(self.recent)[:limit],
            pools=self.pool_state(),
            persistence=dict(
                pending=len(self.pending) + len(self.runtime_pending),
                dropped=self.dropped,
                healthy=not self.persistence_error,
                last_flush_at=self.last_flush_at,
            ),
            coverage="Routed pool, node and legacy listeners. "
            "Direct backend traffic is aggregate-only "
            "when runtime metrics are available. "
            "Tokens are provider-reported; missing is unknown.",
            mode="measured",
        )


class RequestRecord:
    def __init__(self, tracker, model_id, path):
        self.tracker = tracker
        self.id = uuid4().hex
        self.model_id = model_id
        self.path = path.split("?", 1)[0][:160]
        self.started_at = time.time()
        self.start = time.perf_counter()
        self.dispatched_at = None
        self.first_output = None
        self.last_body = None
        self.headers_at = None
        self.first_byte_at = None
        self.tail = bytearray()
        self.cluster_id = None
        self.gpu_ids = []
        self.status = None
        self.outcome = None
        self.input_tokens = self.output_tokens = self.cached_tokens = None
        self.bytes_sent = 0
        self.content_type = ""
        self.buffer = bytearray()
        self.overflow = False
        self.complete = False
        self.stream_ended = False
        self.client_disconnected = False
        self.memory_start = None

    def dispatch(self, cluster_id=None):
        if cluster_id:
            self.cluster_id = cluster_id
        if not self.cluster_id:
            local_pool = getattr(self.tracker, "local_pool", None)
            self.cluster_id = local_pool() if local_pool else None
        if self.dispatched_at is None:
            self.dispatched_at = time.perf_counter()
            self.memory_start = self.tracker.resources(self.cluster_id)

    def public(self):
        return dict(
            id=self.id,
            model_id=self.model_id,
            path=self.path,
            cluster_id=self.cluster_id,
            started_at=self.started_at,
            elapsed_ms=round((time.perf_counter() - self.start) * 1000, 3),
            state="running" if self.dispatched_at else "queued",
        )

    def parse(self, raw):
        if raw == b"[DONE]":
            self.stream_ended = True
            return
        if self.first_output is not None and not any(
            key in raw
            for key in (
                b'"usage"',
                b'"error"',
                b'"finish_reason"',
                b"response.failed",
                b".completed",
                b".done",
            )
        ):
            return
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return
        if not isinstance(value, dict):
            return
        if value.get("error") or value.get("type") in {"error", "response.failed"}:
            self.outcome = "failed"
        payload = value.get("response") if isinstance(value.get("response"), dict) else value
        usage = payload.get("usage")
        if isinstance(usage, dict):
            for attr, names in (
                ("input_tokens", ("prompt_tokens", "input_tokens")),
                ("output_tokens", ("completion_tokens", "output_tokens")),
            ):
                number = next((usage[n] for n in names if n in usage), None)
                if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
                    setattr(self, attr, number)
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            if isinstance(details, dict):
                number = details.get("cached_tokens")
                if isinstance(number, int) and number >= 0:
                    self.cached_tokens = number
        choices = value.get("choices")
        if isinstance(choices, list) and any(
            isinstance(c, dict) and c.get("finish_reason") == "error" for c in choices
        ):
            self.outcome = "failed"
        output = (
            any(
                isinstance(c, dict)
                and (
                    c.get("text")
                    or (
                        isinstance(c.get("delta"), dict)
                        and any(
                            c["delta"].get(k)
                            for k in ("content", "tool_calls", "reasoning_content", "audio")
                        )
                    )
                )
                for c in choices
            )
            if isinstance(choices, list)
            else False
        )
        if value.get("type") in {
            "response.output_text.delta",
            "response.function_call_arguments.delta",
        }:
            output = bool(value.get("delta"))
        if value.get("type") in {"response.completed", "transcript.text.done"} or (
            isinstance(choices, list)
            and any(isinstance(c, dict) and c.get("finish_reason") for c in choices)
        ):
            self.stream_ended = True
        if output and "text/event-stream" in self.content_type and self.first_output is None:
            self.first_output = time.perf_counter()

    def observe(self, chunk):
        # Only transient response metadata is parsed. Nothing from a prompt or output is retained.
        if "text/event-stream" in self.content_type:
            start = 0
            while start < len(chunk):
                boundary = chunk.find(b"\n", start)
                end = len(chunk) if boundary < 0 else boundary
                if not self.overflow:
                    if len(self.buffer) + end - start <= FRAME_LIMIT:
                        self.buffer.extend(chunk[start:end])
                    else:
                        self.buffer.clear()
                        self.overflow = True
                if boundary < 0:
                    break
                if not self.overflow and self.buffer.startswith(b"data:"):
                    self.parse(self.buffer[5:].strip())
                self.buffer.clear()
                self.overflow = False
                start = end + 1
        elif "json" in self.content_type:
            self.tail.extend(chunk[-FRAME_LIMIT:])
            if len(self.tail) > FRAME_LIMIT:
                del self.tail[:-FRAME_LIMIT]
            if self.overflow:
                return
            if len(self.buffer) + len(chunk) <= BODY_LIMIT:
                self.buffer.extend(chunk)
            else:
                self.buffer.clear()
                self.overflow = True

    def wrap_send(self, send):
        async def observed(message):
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.headers_at = time.perf_counter()
                self.content_type = (
                    dict(message.get("headers", []))
                    .get(b"content-type", b"")
                    .decode(errors="replace")
                    .lower()
                )
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                self.bytes_sent += len(chunk)
                if chunk and self.first_byte_at is None:
                    self.first_byte_at = time.perf_counter()
                self.observe(chunk)
                if not message.get("more_body", False):
                    if "json" in self.content_type and not self.overflow:
                        self.parse(self.buffer)
                    elif "json" in self.content_type:
                        # Large embedding/image responses: parse only the trailing usage object.
                        matches = list(re.finditer(rb'(?<!\\)"usage"\s*:\s*', self.tail))
                        if matches:
                            try:
                                usage, _ = json.JSONDecoder().raw_decode(
                                    self.tail[matches[-1].end() :].decode()
                                )
                                self.parse(json.dumps({"usage": usage}).encode())
                            except (ValueError, UnicodeDecodeError, RecursionError):
                                pass
                    self.complete = True
                self.last_body = time.perf_counter()
            await send(message)

        return observed

    def finish(self):
        end = self.last_body or time.perf_counter()
        duration = (end - self.start) * 1000
        streaming = "text/event-stream" in self.content_type
        delivered = self.stream_ended if streaming else self.complete
        outcome = self.outcome or (
            "successful"
            if delivered and self.status is not None and self.status < 400
            else "cancelled"
            if self.client_disconnected and (self.status is None or self.status < 400)
            else "failed"
        )
        decode = None
        if (
            self.first_output
            and self.output_tokens
            and self.output_tokens > 1
            and end > self.first_output
        ):
            decode = (self.output_tokens - 1) / (end - self.first_output)
        return dict(
            id=self.id,
            model_id=self.model_id,
            path=self.path,
            cluster_id=self.cluster_id,
            gpu_ids=self.gpu_ids,
            started_at=self.started_at,
            finished_at=time.time(),
            status=self.status,
            outcome=outcome,
            duration_ms=round(duration, 3),
            queue_ms=round(((self.dispatched_at or end) - self.start) * 1000, 3),
            service_ms=round((end - self.dispatched_at) * 1000, 3) if self.dispatched_at else None,
            time_to_headers_ms=round((self.headers_at - self.start) * 1000, 3)
            if self.headers_at
            else None,
            first_byte_ms=round((self.first_byte_at - self.start) * 1000, 3)
            if self.first_byte_at
            else None,
            ttft_ms=round((self.first_output - self.start) * 1000, 3)
            if self.first_output
            else None,
            prefill_ms=round((self.first_output - self.dispatched_at) * 1000, 3)
            if self.first_output and self.dispatched_at
            else None,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            decode_tokens_per_second=decode,
            bytes_sent=self.bytes_sent,
            memory_start=self.memory_start,
            memory_end=self.tracker.resources(self.cluster_id),
        )


def mark_dispatch(scope, cluster_id=None, gpu_ids=None):
    if record := scope.get(SCOPE_KEY):
        record.dispatch(cluster_id)
        if gpu_ids is not None:
            record.gpu_ids = list(gpu_ids)


def request_stream_usage(body, content_type, path):
    """Ask compatible generation routes for usage unless the client opted out."""
    if "application/json" not in content_type or not path.rstrip("/").endswith(
        ("/chat/completions", "/completions")
    ):
        return bytes(body)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return bytes(body)
    if not isinstance(value, dict) or value.get("stream") is not True:
        return bytes(body)
    options = value.get("stream_options", {})
    if not isinstance(options, dict) or "include_usage" in options:
        return bytes(body)
    value["stream_options"] = {**options, "include_usage": True}
    return json.dumps(value, separators=(",", ":")).encode()
