from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

NodeStatus = Literal["healthy", "degraded", "offline"]
NodeRole = Literal["head", "worker", "cache"]


class MemoryStats(BaseModel):
    total_gib: float = Field(ge=0)
    used_gib: float = Field(ge=0)
    kv_cache_gib: float = Field(default=0, ge=0)
    model_gib: float = Field(default=0, ge=0)

    @property
    def utilization(self) -> float:
        return 0 if self.total_gib == 0 else min(100, self.used_gib / self.total_gib * 100)


class NodeSnapshot(BaseModel):
    id: str
    hostname: str
    role: NodeRole
    status: NodeStatus = "healthy"
    address: str
    memory: MemoryStats
    gpu_utilization: float = Field(ge=0, le=100)
    temperature_c: float = Field(ge=0)
    input_tokens_per_second: float = Field(default=0, ge=0)
    output_tokens_per_second: float = Field(default=0, ge=0)
    active_requests: int = Field(default=0, ge=0)
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CacheTier(BaseModel):
    name: str
    kind: Literal["gpu", "cpu", "nvme", "remote"]
    used_gib: float = Field(ge=0)
    capacity_gib: float = Field(ge=0)
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    read_latency_ms: float = Field(default=0, ge=0)
    persistent: bool = False

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return 0 if total == 0 else self.hits / total * 100


class ModelRuntime(BaseModel):
    model: str
    quantization: str
    context_length: int
    tensor_parallel: int
    pipeline_parallel: int
    uptime_seconds: int = Field(default=0, ge=0)
    queued_requests: int = Field(default=0, ge=0)


class ClusterSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    cluster_name: str
    status: NodeStatus = "healthy"
    nodes: list[NodeSnapshot]
    cache_tiers: list[CacheTier]
    runtime: ModelRuntime
    total_input_tokens_per_second: float = 0
    total_output_tokens_per_second: float = 0
    cache_hit_rate: float = 0
    prefix_reuse_tokens: int = 0


class CapacityRequest(BaseModel):
    parameter_count_billions: float = Field(gt=0)
    bits_per_weight: float = Field(default=4, gt=0)
    context_length: int = Field(default=131072, gt=0)
    concurrent_sequences: int = Field(default=1, gt=0)
    kv_bytes_per_token_billions: float = Field(default=0.00025, gt=0)
    node_memory_gib: float = Field(default=128, gt=0)
    usable_fraction: float = Field(default=0.82, gt=0, le=1)


class CapacityPlan(BaseModel):
    weights_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    total_required_gib: float
    usable_per_node_gib: float
    minimum_nodes: int
