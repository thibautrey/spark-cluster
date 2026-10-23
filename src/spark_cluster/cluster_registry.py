"""Persistent registry for multiple independently selectable inference clusters."""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .remote_cluster import load_active_connection, validated_cluster_url
from .setup_models import private_address


class GPUDevice(BaseModel):
    index: int = Field(ge=0)
    name: str
    uuid: str
    memory_gib: float = Field(gt=0)
    compute_capability: str | None = None


class ClusterTarget(BaseModel):
    id: str
    name: str
    kind: Literal["remote-bridge", "local-docker"]
    hardware: Literal["dgx-spark", "nvidia-gpu", "other"] = "other"
    status: Literal["ready", "offline", "unknown"] = "unknown"
    endpoint: str | None = None
    host_address: str | None = None
    runtime_port: int = Field(default=30000, ge=1024, le=65535)
    gpu_count: int = Field(default=0, ge=0)
    gpu_memory_gib: float = Field(default=0, ge=0)
    gpu_model: str | None = None
    gpus: list[GPUDevice] = Field(default_factory=list)
    model_storage_path: str | None = None
    active: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    capabilities: list[str] = Field(default_factory=list)


class RemoteClusterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    address: str
    port: int = Field(default=30080, ge=1024, le=65535)
    hardware: Literal["dgx-spark", "nvidia-gpu", "other"] = "other"

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Cluster name is required")
        return value


class LocalClusterRequest(BaseModel):
    name: str = Field(default="Unraid GPUs", min_length=1, max_length=80)


class ClusterRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.path = settings.setup_state_path / "clusters.json"
        self._lock = threading.RLock()
        self._clusters: dict[str, ClusterTarget] = {}
        self._active_id: str | None = None
        self._load()
        self._migrate_legacy()
        self._apply_active()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
            clusters = [ClusterTarget.model_validate(item) for item in payload["clusters"]]
            self._clusters = {item.id: item for item in clusters}
            active = payload.get("active_cluster_id")
            self._active_id = active if active in self._clusters else None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._clusters = {}
            self._active_id = None

    def _migrate_legacy(self) -> None:
        if self._clusters:
            return
        connection = load_active_connection(self.settings)
        if not connection and self.settings.remote_cluster_url:
            endpoint = validated_cluster_url(self.settings.remote_cluster_url)
            connection = {
                "url": endpoint,
                "cluster_name": self.settings.cluster_name,
                "head_address": endpoint.split("://", 1)[1].split(":", 1)[0],
            }
        if connection:
            cluster = ClusterTarget(
                id=self._new_id("remote"),
                name=connection["cluster_name"],
                kind="remote-bridge",
                hardware="dgx-spark",
                status="ready",
                endpoint=connection["url"],
                host_address=connection["head_address"],
                active=True,
                capabilities=["models", "runtime", "telemetry", "logs"],
            )
            self._clusters[cluster.id] = cluster
            self._active_id = cluster.id
            self._save()

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex[:12]}"

    def _save(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        payload = {
            "version": 1,
            "active_cluster_id": self._active_id,
            "clusters": [item.model_dump(mode="json") for item in self._clusters.values()],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.path)

    def _apply_active(self) -> None:
        cluster = self.active()
        if cluster is None:
            self.settings.remote_cluster_url = None
            return
        self.settings.active_hardware = cluster.hardware
        self.settings.cluster_name = cluster.name
        if cluster.kind == "remote-bridge":
            self.settings.remote_cluster_url = cluster.endpoint
            self.settings.active_compute_capability = (
                12.1 if cluster.hardware == "dgx-spark" else None
            )
            return
        self.settings.remote_cluster_url = None
        self.settings.fallback_node_count = max(1, cluster.gpu_count)
        self.settings.fallback_node_memory_gib = max(1, cluster.gpu_memory_gib)
        capabilities = [
            float(item.compute_capability) for item in cluster.gpus if item.compute_capability
        ]
        self.settings.active_compute_capability = min(capabilities) if capabilities else None
        host = cluster.host_address or self.settings.docker_runtime_host
        self.settings.openai_api_base = f"http://{host}:{cluster.runtime_port}/v1"

    def list(self) -> list[ClusterTarget]:
        with self._lock:
            return [
                item.model_copy(update={"active": item.id == self._active_id})
                for item in self._clusters.values()
            ]

    def active(self) -> ClusterTarget | None:
        with self._lock:
            item = self._clusters.get(self._active_id or "")
            return item.model_copy(update={"active": True}) if item else None

    def get(self, cluster_id: str) -> ClusterTarget:
        if not re.fullmatch(r"[a-z]+-[a-f0-9]{12}", cluster_id):
            raise KeyError(cluster_id)
        with self._lock:
            item = self._clusters.get(cluster_id)
            if not item:
                raise KeyError(cluster_id)
            return item.model_copy(update={"active": item.id == self._active_id})

    def select(self, cluster_id: str) -> ClusterTarget:
        with self._lock:
            cluster = self.get(cluster_id)
            self._active_id = cluster.id
            self._save()
            self._apply_active()
            return cluster.model_copy(update={"active": True})

    def add_remote(
        self,
        *,
        name: str,
        endpoint: str,
        head_address: str,
        hardware: str = "other",
        activate: bool = True,
    ) -> ClusterTarget:
        endpoint = validated_cluster_url(endpoint)
        head_address = private_address(head_address)
        with self._lock:
            existing = next(
                (item for item in self._clusters.values() if item.endpoint == endpoint), None
            )
            if existing:
                cluster = existing.model_copy(
                    update={"name": name, "hardware": hardware, "status": "ready"}
                )
            else:
                cluster = ClusterTarget(
                    id=self._new_id("remote"),
                    name=name,
                    kind="remote-bridge",
                    hardware=hardware,
                    status="ready",
                    endpoint=endpoint,
                    host_address=head_address,
                    capabilities=["models", "runtime", "telemetry", "logs"],
                )
            self._clusters[cluster.id] = cluster
            if activate or not self._active_id:
                self._active_id = cluster.id
            self._save()
            self._apply_active()
            return cluster.model_copy(update={"active": cluster.id == self._active_id})

    def add_local(
        self,
        *,
        name: str,
        host_address: str,
        runtime_port: int,
        gpus: list[GPUDevice],
        model_storage_path: str,
        activate: bool = True,
    ) -> ClusterTarget:
        if not gpus:
            raise ValueError("No NVIDIA GPU is visible to the controller")
        with self._lock:
            existing = next(
                (item for item in self._clusters.values() if item.kind == "local-docker"), None
            )
            memory = min(item.memory_gib for item in gpus)
            models = sorted({item.name for item in gpus})
            values = {
                "name": name,
                "hardware": "nvidia-gpu",
                "status": "ready",
                "host_address": host_address,
                "runtime_port": runtime_port,
                "gpu_count": len(gpus),
                "gpu_memory_gib": memory,
                "gpu_model": ", ".join(models),
                "gpus": gpus,
                "model_storage_path": model_storage_path,
                "capabilities": ["models", "docker-runtime", "telemetry", "logs"],
            }
            cluster = (
                existing.model_copy(update=values)
                if existing
                else ClusterTarget(id=self._new_id("local"), kind="local-docker", **values)
            )
            self._clusters[cluster.id] = cluster
            if activate or not self._active_id:
                self._active_id = cluster.id
            self._save()
            self._apply_active()
            return cluster.model_copy(update={"active": cluster.id == self._active_id})

    def delete(self, cluster_id: str) -> ClusterTarget:
        with self._lock:
            cluster = self.get(cluster_id)
            del self._clusters[cluster_id]
            if self._active_id == cluster_id:
                self._active_id = next(iter(self._clusters), None)
            self._save()
            self._apply_active()
            return cluster
