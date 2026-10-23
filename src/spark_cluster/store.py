import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .models import (
    BenchmarkResult,
    CachePrefixRecord,
    ClusterSnapshot,
    DownloadJob,
    InferenceTelemetry,
    ModelLaunchJob,
    RuntimeInstallJob,
    RuntimeLog,
)
from .setup_models import SetupJob


class SnapshotStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS snapshots "
            "(timestamp TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS model_downloads "
            "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_logs "
            "(id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, pod TEXT NOT NULL, "
            "container TEXT NOT NULL, level TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS runtime_logs_timestamp ON runtime_logs(timestamp DESC)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS model_launches (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_install_jobs "
            "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS inference_telemetry "
            "(timestamp TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS optimization_benchmarks "
            "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS cache_prefixes (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.commit()

        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS cluster_setups (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._connection.commit()

    def save_setup_job(self, job: SetupJob) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO cluster_setups(id, payload) VALUES (?, ?)",
                (job.id, job.model_dump_json()),
            )
            self._connection.commit()

    def get_setup_job(self, job_id: str) -> SetupJob | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM cluster_setups WHERE id = ?", (job_id,)
            ).fetchone()
        return SetupJob.model_validate_json(row[0]) if row else None

    def list_setup_jobs(self) -> list[SetupJob]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM cluster_setups "
                "ORDER BY json_extract(payload, '$.created_at') DESC LIMIT 50"
            ).fetchall()
        return [SetupJob.model_validate_json(row[0]) for row in rows]

    def append(self, snapshot: ClusterSnapshot) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO snapshots(timestamp, payload) VALUES (?, ?)",
                (snapshot.timestamp.isoformat(), snapshot.model_dump_json()),
            )
            self._connection.commit()

    def latest(self) -> ClusterSnapshot | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return ClusterSnapshot.model_validate_json(row[0]) if row else None

    def history(self, minutes: int, limit: int = 600) -> list[dict]:
        since = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM snapshots WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT ?",
                (since, limit),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def prune(self, retention_hours: int) -> None:
        before = (datetime.now(UTC) - timedelta(hours=retention_hours)).isoformat()
        with self._lock:
            self._connection.execute("DELETE FROM snapshots WHERE timestamp < ?", (before,))
            self._connection.commit()

    def save_download(self, job: DownloadJob) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO model_downloads(id, payload) VALUES (?, ?)",
                (job.id, job.model_dump_json()),
            )
            self._connection.commit()

    def get_download(self, job_id: str) -> DownloadJob | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM model_downloads WHERE id = ?", (job_id,)
            ).fetchone()
        return DownloadJob.model_validate_json(row[0]) if row else None

    def list_downloads(self, limit: int = 100) -> list[DownloadJob]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM model_downloads "
                "ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [DownloadJob.model_validate_json(row[0]) for row in rows]

    def recover_downloads(self) -> list[DownloadJob]:
        recovered: list[DownloadJob] = []
        for job in self.list_downloads():
            if job.status in {
                "queued",
                "resolving",
                "downloading",
                "preparing-transfer",
                "distributing",
            }:
                job.status = "queued"
                job.error = None
                job.message = "Recovered after controller restart"
                job.updated_at = datetime.now(UTC)
                self.save_download(job)
                recovered.append(job)
        return recovered

    def append_runtime_log(self, entry: RuntimeLog) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO runtime_logs"
                "(id, timestamp, pod, container, level, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.timestamp.isoformat(),
                    entry.pod,
                    entry.container,
                    entry.level,
                    entry.model_dump_json(),
                ),
            )
            self._connection.commit()

    def list_runtime_logs(
        self, limit: int = 500, pod: str | None = None, level: str | None = None
    ) -> list[RuntimeLog]:
        clauses: list[str] = []
        parameters: list[str | int] = []
        if pod:
            clauses.append("pod = ?")
            parameters.append(pod)
        if level:
            clauses.append("level = ?")
            parameters.append(level)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT payload FROM runtime_logs {where} ORDER BY timestamp DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [RuntimeLog.model_validate_json(row[0]) for row in reversed(rows)]

    def prune_runtime_logs(self, keep: int) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM runtime_logs WHERE id NOT IN "
                "(SELECT id FROM runtime_logs ORDER BY timestamp DESC LIMIT ?)",
                (keep,),
            )
            self._connection.commit()

    def save_launch(self, job: ModelLaunchJob) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO model_launches(id, payload) VALUES (?, ?)",
                (job.id, job.model_dump_json()),
            )
            self._connection.commit()

    def get_launch(self, job_id: str) -> ModelLaunchJob | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM model_launches WHERE id = ?", (job_id,)
            ).fetchone()
        return ModelLaunchJob.model_validate_json(row[0]) if row else None

    def list_launches(self, limit: int = 50) -> list[ModelLaunchJob]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM model_launches "
                "ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ModelLaunchJob.model_validate_json(row[0]) for row in rows]

    def recover_launches(self) -> list[ModelLaunchJob]:
        recovered: list[ModelLaunchJob] = []
        for job in self.list_launches():
            if job.status not in {"ready", "failed"}:
                job.status = "queued"
                job.message = "Recovered after control-plane restart"
                job.error = None
                job.updated_at = datetime.now(UTC)
                self.save_launch(job)
                recovered.append(job)
        return recovered

    def save_runtime_install(self, job: RuntimeInstallJob) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO runtime_install_jobs(id, payload) VALUES (?, ?)",
                (job.id, job.model_dump_json()),
            )
            self._connection.commit()

    def get_runtime_install(self, job_id: str) -> RuntimeInstallJob | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_install_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return RuntimeInstallJob.model_validate_json(row[0]) if row else None

    def list_runtime_installs(self, limit: int = 20) -> list[RuntimeInstallJob]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM runtime_install_jobs "
                "ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [RuntimeInstallJob.model_validate_json(row[0]) for row in rows]

    def save_telemetry(self, telemetry: InferenceTelemetry) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO inference_telemetry(timestamp, payload) VALUES (?, ?)",
                (telemetry.sampled_at.isoformat(), telemetry.model_dump_json()),
            )
            self._connection.execute(
                "DELETE FROM inference_telemetry WHERE timestamp NOT IN "
                "(SELECT timestamp FROM inference_telemetry ORDER BY timestamp DESC LIMIT 10000)"
            )
            self._connection.commit()

    def latest_telemetry(self) -> InferenceTelemetry | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM inference_telemetry ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return InferenceTelemetry.model_validate_json(row[0]) if row else None

    def telemetry_history(self, limit: int = 600) -> list[InferenceTelemetry]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM inference_telemetry ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [InferenceTelemetry.model_validate_json(row[0]) for row in reversed(rows)]

    def save_benchmark(self, result: BenchmarkResult) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO optimization_benchmarks(id, payload) VALUES (?, ?)",
                (result.id, result.model_dump_json()),
            )
            self._connection.commit()

    def list_benchmarks(self, limit: int = 100) -> list[BenchmarkResult]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM optimization_benchmarks "
                "ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [BenchmarkResult.model_validate_json(row[0]) for row in rows]

    def save_cache_prefix(self, record: CachePrefixRecord) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO cache_prefixes(id, payload) VALUES (?, ?)",
                (record.id, record.model_dump_json()),
            )
            self._connection.commit()

    def get_cache_prefix(self, record_id: str) -> CachePrefixRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM cache_prefixes WHERE id = ?", (record_id,)
            ).fetchone()
        return CachePrefixRecord.model_validate_json(row[0]) if row else None

    def list_cache_prefixes(self, limit: int = 100) -> list[CachePrefixRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM cache_prefixes "
                "ORDER BY json_extract(payload, '$.priority') DESC, "
                "json_extract(payload, '$.updated_at') DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [CachePrefixRecord.model_validate_json(row[0]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
