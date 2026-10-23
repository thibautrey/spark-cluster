"""Small persistent cache for Hugging Face metadata and capacity plans."""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class CatalogCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS catalog_metadata_cache ("
            "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, cursor TEXT, "
            "expires_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS catalog_fit_cache ("
            "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
            "expires_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS catalog_metadata_expiry "
            "ON catalog_metadata_cache(expires_at)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS catalog_fit_expiry ON catalog_fit_cache(expires_at)"
        )
        self._connection.commit()

    def get_metadata(self, key: str) -> tuple[dict | list, str | None] | None:
        now = time.time()
        with self._lock:
            row = self._connection.execute(
                "SELECT payload, cursor FROM catalog_metadata_cache "
                "WHERE cache_key = ? AND expires_at > ?",
                (key, now),
            ).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def set_metadata(
        self,
        key: str,
        payload: dict | list,
        cursor: str | None,
        ttl_seconds: int,
    ) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO catalog_metadata_cache"
                "(cache_key, payload, cursor, expires_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (key, json.dumps(payload), cursor, now + ttl_seconds, now),
            )
            self._prune_locked("catalog_metadata_cache", now, 1000)
            self._connection.commit()

    def get_fit(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM catalog_fit_cache WHERE cache_key = ? AND expires_at > ?",
                (key, now),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def set_fit(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO catalog_fit_cache"
                "(cache_key, payload, expires_at, updated_at) VALUES (?, ?, ?, ?)",
                (key, json.dumps(payload), now + ttl_seconds, now),
            )
            self._prune_locked("catalog_fit_cache", now, 5000)
            self._connection.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            metadata = self._connection.execute(
                "SELECT COUNT(*) FROM catalog_metadata_cache"
            ).fetchone()[0]
            fits = self._connection.execute("SELECT COUNT(*) FROM catalog_fit_cache").fetchone()[0]
        return {"metadata_entries": metadata, "fit_entries": fits}

    def _prune_locked(self, table: str, now: float, keep: int) -> None:
        self._connection.execute(f"DELETE FROM {table} WHERE expires_at <= ?", (now,))
        self._connection.execute(
            f"DELETE FROM {table} WHERE cache_key NOT IN "
            f"(SELECT cache_key FROM {table} ORDER BY updated_at DESC LIMIT ?)",
            (keep,),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
