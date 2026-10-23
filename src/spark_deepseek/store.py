import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .models import ClusterSnapshot, DownloadJob


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
        self._connection.commit()

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
            if job.status in {"queued", "resolving", "downloading"}:
                job.status = "queued"
                job.error = None
                job.updated_at = datetime.now(UTC)
                self.save_download(job)
                recovered.append(job)
        return recovered

    def close(self) -> None:
        with self._lock:
            self._connection.close()
