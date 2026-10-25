"""Content-free performance rollups, written only by the inference journal worker."""

import json
import math
import time

DAY = 86400
RUNTIME_FIELDS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "gpu_kv_cache_usage",
    "cache_retrieval_p95_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
)


def merge(target, source):
    for key, value in source.items():
        target[key] = (
            max(target.get(key, value), value)
            if key.endswith("_max")
            else (target.get(key, 0) + value)
        )
    return target


def gpu_key(pool_id, gpu_id):
    # A GPU index or UUID is scoped to its pool, never implicitly global.
    return json.dumps([pool_id, str(gpu_id)], separators=(",", ":"))


def scopes(model_id, pool_id, gpu_ids=()):
    result = [("overall", "all")]
    if model_id:
        result.append(("model", model_id))
    if pool_id:
        result.append(("pool", pool_id))
        result.extend(("gpu", gpu_key(pool_id, gpu)) for gpu in set(gpu_ids))
    return result


class PerformanceHistory:
    def __init__(self, db, settings):
        self.db = db
        self.minute_days = settings.performance_minute_days
        self.hour_days = settings.performance_hour_days
        self.day_days = settings.performance_day_days
        self.max_rows = settings.performance_max_rows
        self.last_maintenance = 0
        db.execute(
            "CREATE TABLE IF NOT EXISTS performance_buckets ("
            "scope TEXT, entity TEXT, resolution INTEGER, timestamp INTEGER, payload TEXT, "
            "PRIMARY KEY(scope,entity,resolution,timestamp))"
        )
        db.execute("CREATE INDEX IF NOT EXISTS performance_time ON performance_buckets(timestamp)")
        db.commit()

    def resolution(self, timestamp, now):
        age = now - timestamp
        return 60 if age < self.minute_days * DAY else (3600 if age < self.hour_days * DAY else DAY)

    def add(self, entities, timestamp, values, now):
        if timestamp < now - self.day_days * DAY:
            return
        resolution = self.resolution(timestamp, now)
        bucket = int(timestamp // resolution) * resolution
        for scope, entity in entities:
            key = (scope, entity, resolution, bucket)
            row = self.db.execute(
                "SELECT payload FROM performance_buckets "
                "WHERE scope=? AND entity=? AND resolution=? AND timestamp=?",
                key,
            ).fetchone()
            payload = merge(json.loads(row[0]) if row else {}, values)
            self.db.execute(
                "INSERT OR REPLACE INTO performance_buckets VALUES (?,?,?,?,?)",
                (*key, json.dumps(payload, separators=(",", ":"))),
            )

    def request(self, event, now):
        values = dict(requests=1, **{event["outcome"]: 1})
        for key in ("input_tokens", "output_tokens", "cached_tokens"):
            value = event.get(key)
            if value is not None:
                values[key] = value
                values[key + "_known"] = 1
        if event.get("cached_tokens") is not None and event.get("input_tokens") is not None:
            values["cache_input_tokens"] = event["input_tokens"]
            values["cache_reported_tokens"] = event["cached_tokens"]
        if event["outcome"] == "successful":
            for key in ("duration_ms", "queue_ms", "ttft_ms", "prefill_ms"):
                value = event.get(key)
                if value is not None:
                    values[key + "_sum"] = value
                    values[key + "_count"] = 1
                    values[key + "_max"] = value
            rate = event.get("decode_tokens_per_second")
            if rate and math.isfinite(rate) and event.get("output_tokens", 0) > 1:
                tokens = event["output_tokens"] - 1
                values.update(decode_tokens=tokens, decode_seconds=tokens / rate)
            if event.get("prefill_ms") and event.get("input_tokens") is not None:
                values.update(
                    prefill_tokens=event["input_tokens"],
                    prefill_seconds=event["prefill_ms"] / 1000,
                )
        self.add(
            scopes(event["model_id"], event["cluster_id"], event.get("gpu_ids", [])),
            event["finished_at"],
            values,
            now,
        )

    def runtime(self, sample, now):
        values = {"runtime_samples": 1}
        for key in RUNTIME_FIELDS:
            value = sample["metrics"].get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[key + "_sum"] = value
                values[key + "_count"] = 1
                values[key + "_max"] = value
        self.add(sample["scopes"], sample["timestamp"], values, now)

    def maintain(self, now):
        if now - self.last_maintenance < 60:
            return
        self.db.execute(
            "DELETE FROM performance_buckets WHERE timestamp < ?", (now - self.day_days * DAY,)
        )
        # Move old fine buckets into the next tier exactly once, atomically with the journal.
        for resolution, cutoff in (
            (60, now - self.minute_days * DAY),
            (3600, now - self.hour_days * DAY),
        ):
            rows = self.db.execute(
                "SELECT scope,entity,resolution,timestamp,payload FROM performance_buckets "
                "WHERE resolution=? AND timestamp < ?",
                (resolution, cutoff),
            ).fetchall()
            for scope, entity, old_resolution, timestamp, payload in rows:
                self.add([(scope, entity)], timestamp, json.loads(payload), now)
                self.db.execute(
                    "DELETE FROM performance_buckets WHERE "
                    "scope=? AND entity=? AND resolution=? AND timestamp=?",
                    (scope, entity, old_resolution, timestamp),
                )
        count = self.db.execute("SELECT COUNT(*) FROM performance_buckets").fetchone()[0]
        if count > self.max_rows:
            self.db.execute(
                "DELETE FROM performance_buckets WHERE rowid IN ("
                "SELECT rowid FROM performance_buckets ORDER BY timestamp LIMIT ?)",
                (count - self.max_rows,),
            )
        # Set only after the enclosing transaction commits (the caller does this).

    @staticmethod
    def metrics(values):
        def ratio(a, b):
            return values.get(a, 0) / values[b] if values.get(b) else None

        result = {
            key: values.get(key, 0)
            for key in (
                "requests",
                "successful",
                "failed",
                "cancelled",
                "runtime_samples",
            )
        }
        for key in ("input_tokens", "output_tokens", "cached_tokens"):
            result[key] = values.get(key) if values.get(key + "_known") else None
            result[key + "_known"] = values.get(key + "_known", 0)
        for key in ("duration_ms", "queue_ms", "ttft_ms", "prefill_ms"):
            result[key + "_mean"] = ratio(key + "_sum", key + "_count")
            result[key + "_max"] = values.get(key + "_max")
        result["decode_tokens_per_second"] = ratio("decode_tokens", "decode_seconds")
        result["effective_prefill_tokens_per_second"] = ratio("prefill_tokens", "prefill_seconds")
        result["cached_input_ratio"] = ratio("cache_reported_tokens", "cache_input_tokens")
        for key in RUNTIME_FIELDS:
            result["runtime_" + key] = ratio(key + "_sum", key + "_count")
            result["runtime_" + key + "_max"] = values.get(key + "_max")
        return result

    def read(self, scope, entity, hours, now=None):
        now = now or time.time()
        start = now - hours * 3600
        # Coarse boundary buckets are included whole and their actual bounds are exposed.
        rows = self.db.execute(
            "SELECT resolution,timestamp,payload FROM performance_buckets "
            "WHERE scope=? AND entity=? AND timestamp+resolution>? AND timestamp<=? "
            "ORDER BY timestamp",
            (scope, entity, start, now),
        ).fetchall()
        resolution = 60 if hours <= 24 else 3600 if hours <= 24 * 30 else DAY
        points = {}
        for stored_resolution, timestamp, payload in rows:
            step = max(resolution, stored_resolution)
            key = (int(timestamp // step) * step, step)
            merge(points.setdefault(key, {}), json.loads(payload))
        total = {}
        history = []
        for (timestamp, step), values in sorted(points.items()):
            merge(total, values)
            history.append(
                dict(timestamp=timestamp, resolution_seconds=step, **self.metrics(values))
            )
        entities = [
            dict(scope=kind, entity=identifier)
            for kind, identifier in self.db.execute(
                "SELECT scope,entity FROM performance_buckets WHERE timestamp+resolution>? "
                "GROUP BY scope,entity ORDER BY MAX(timestamp) DESC LIMIT 256",
                (start,),
            ).fetchall()
        ]
        return dict(
            scope=scope,
            entity=entity,
            start=start,
            end=now,
            summary=self.metrics(total),
            entities=entities,
            history=history,
            retention=dict(
                minute_days=self.minute_days,
                hour_days=self.hour_days,
                day_days=self.day_days,
                max_rows=self.max_rows,
            ),
            coverage="Routed requests and locally scraped runtimes. GPU requests describe the "
            "whole workload placed on that GPU; multi-GPU requests appear on each assigned GPU. "
            "Runtime rates are sample means. Effective prefill includes backend scheduling and "
            "first-token delivery. Missing metrics remain unknown. Boundary buckets are included "
            "whole; history begins when this feature is enabled.",
        )
