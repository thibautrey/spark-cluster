from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

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


class ModelFitRequest(BaseModel):
    repo_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(default="main", min_length=1, max_length=200)
    bits_per_weight: float = Field(default=4, gt=0, le=32)
    context_length: int = Field(default=131072, gt=0, le=1_048_576)
    concurrent_sequences: int = Field(default=1, gt=0, le=1024)
    kv_cache_dtype: Literal["auto", "bf16", "fp8_e4m3"] = "auto"


class ModelFitPlan(BaseModel):
    repo_id: str
    revision: str
    architecture: str | None = None
    parameter_count: int | None = None
    repository_bytes: int
    weight_bytes: int
    weights_gib: float
    kv_cache_gib: float
    runtime_overhead_gib: float
    total_required_gib: float
    cluster_nodes: int
    usable_per_node_gib: float
    usable_cluster_gib: float
    minimum_nodes: int
    headroom_gib: float
    memory_fits: bool
    storage_available_gib: float
    storage_fits: bool
    fits: bool
    confidence: Literal["architecture", "parameters", "repository"]
    notes: list[str] = Field(default_factory=list)
    bits_per_weight: float = 4
    context_length: int = 131072
    concurrent_sequences: int = 1
    kv_cache_dtype: str = "bf16"
    runtime_fits: bool = False
    download_fits: bool = False
    launch_supported: bool = True
    runtime_storage_fits: bool = True
    fit_status: Literal[
        "full-memory",
        "hybrid-ready",
        "engine-required",
        "download-only",
        "insufficient-storage",
    ] = "download-only"
    placement_strategy: Literal[
        "full-memory",
        "engram-nvme",
        "kv-offload",
        "expert-offload",
        "mmap",
        "unsupported",
    ] = "full-memory"
    engine_recommendation: str = "vLLM"
    detected_format: str = "Unknown"
    detected_bits_per_weight: float | None = None
    library_name: str | None = None
    pipeline_tag: str | None = None
    resident_weights_gib: float = 0
    offloaded_weights_gib: float = 0
    resident_kv_cache_gib: float = 0
    offloaded_kv_cache_gib: float = 0
    ssd_required_gib_per_node: float = 0
    additional_artifacts_bytes: int = 0
    runtime_storage_required_gib: float = 0
    performance_impact: Literal["none", "low", "moderate", "high", "unknown"] = "none"
    requirements: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)


class ModelDownloadRequest(ModelFitRequest):
    allow_oversized: bool = False


DownloadStatus = Literal[
    "queued",
    "resolving",
    "downloading",
    "preparing-transfer",
    "distributing",
    "completed",
    "failed",
]


class DownloadJob(BaseModel):
    id: str
    repo_id: str
    revision: str
    status: DownloadStatus = "queued"
    total_bytes: int = 0
    downloaded_bytes: int = 0
    progress: float = Field(default=0, ge=0, le=100)
    transfer_progress: float = Field(default=0, ge=0, le=100)
    transferred_bytes: int = 0
    target_path: str
    staging_path: str
    source: Literal["huggingface", "controller"] = "huggingface"
    destination: Literal["local", "cluster"] = "local"
    controller_url: str | None = None
    controller_job_id: str | None = None
    message: str = "Queued"
    error: str | None = None
    attempts: int = 0
    fit: ModelFitPlan
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ControllerModelImportRequest(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    repo_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(default="main", min_length=1, max_length=200)
    controller_url: str
    total_bytes: int = Field(ge=0)
    fit: ModelFitPlan


class LocalModel(BaseModel):
    id: str
    revision: str | None = None
    source: Literal["managed", "huggingface-cache", "runtime"]
    path: str | None = None
    size_bytes: int = 0
    downloaded_at: datetime | None = None
    loaded: bool = False


class RuntimeEndpoint(BaseModel):
    base_url: str
    models_url: str
    chat_completions_url: str
    status: Literal["ready", "unreachable"]
    loaded_model_ids: list[str] = Field(default_factory=list)


LogLevel = Literal["debug", "info", "warning", "error"]


class RuntimeLog(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    pod: str
    container: str
    level: LogLevel = "info"
    message: str


LaunchStatus = Literal[
    "queued",
    "validating",
    "scaling-workers",
    "starting-runtime",
    "waiting-ready",
    "ready",
    "failed",
]


class ModelLaunchRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    served_model_name: str = Field(min_length=1, max_length=100)
    bits_per_weight: float = Field(default=4, gt=0, le=32)
    context_length: int = Field(default=131072, gt=0, le=1_048_576)
    concurrent_sequences: int = Field(default=1, gt=0, le=1024)
    tensor_parallel: int = Field(default=1, gt=0, le=64)
    pipeline_parallel: int = Field(default=1, gt=0, le=64)
    gpu_memory_utilization: float = Field(default=0.82, gt=0, lt=1)
    optimization_profile: Literal["balanced", "latency", "throughput", "long-context"] = "balanced"
    kv_cache_dtype: Literal["auto", "fp8_e4m3"] = "auto"
    max_num_seqs: int = Field(default=4, gt=0, le=1024)
    max_num_batched_tokens: int = Field(default=8192, ge=512, le=1_048_576)

    @model_validator(mode="after")
    def scheduler_covers_requested_concurrency(self):
        if self.max_num_seqs < self.concurrent_sequences:
            self.max_num_seqs = self.concurrent_sequences
        return self


class ModelLaunchJob(BaseModel):
    id: str
    request: ModelLaunchRequest
    model_path: str
    status: LaunchStatus = "queued"
    progress: float = Field(default=0, ge=0, le=100)
    message: str = "Queued"
    fit: ModelFitPlan
    error: str | None = None
    attempts: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
