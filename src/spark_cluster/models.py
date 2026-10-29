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
    accelerator_used_gib: float = Field(default=0, ge=0)
    kv_cache_source: Literal["measured", "estimated", "unavailable"] = "unavailable"

    @property
    def utilization(self) -> float:
        return 0 if self.total_gib == 0 else min(100, self.used_gib / self.total_gib * 100)


class CpuCoreLoad(BaseModel):
    id: int = Field(ge=0)
    utilization: float | None = Field(default=None, ge=0, le=100)


class HostResources(BaseModel):
    id: str
    hostname: str
    ram_total_gib: float | None = Field(default=None, ge=0)
    ram_used_gib: float | None = Field(default=None, ge=0)
    ram_available_gib: float | None = Field(default=None, ge=0)
    cpu_physical_cores: int | None = Field(default=None, ge=1)
    cpu_logical_cores: int | None = Field(default=None, ge=1)
    cpu_utilization: float | None = Field(default=None, ge=0, le=100)
    cpu_cores: list[CpuCoreLoad] = Field(default_factory=list)
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NodeSnapshot(BaseModel):
    id: str
    hostname: str
    role: NodeRole
    status: NodeStatus = "healthy"
    address: str
    memory: MemoryStats
    host: HostResources | None = None
    gpu_utilization: float = Field(ge=0, le=100)
    temperature_c: float = Field(ge=0)
    input_tokens_per_second: float = Field(default=0, ge=0)
    output_tokens_per_second: float = Field(default=0, ge=0)
    active_requests: int = Field(default=0, ge=0)
    loaded_model_ids: list[str] = Field(default_factory=list)
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
    offload_ranks: int = Field(default=1, ge=1, le=64)
    cpu_offload_gib: float = Field(default=0, ge=0, le=512)
    gpu_layers: int | None = Field(default=None, ge=0, le=10000)


class ModelFitPlan(BaseModel):
    repo_id: str
    revision: str
    checkpoint_commit: str | None = None
    recipe_id: str | None = None
    recipe_source_url: str | None = None
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
        "cpu-offload",
        "mmap",
        "unsupported",
    ] = "full-memory"
    engine_recommendation: str = "vLLM"
    detected_format: str = "Unknown"
    detected_bits_per_weight: float | None = None
    library_name: str | None = None
    pipeline_tag: str | None = None
    model_layers: int | None = None
    host_memory_gib: float = 0
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
    smallest_gguf_transfer_only: bool = False


DownloadStatus = Literal[
    "queued",
    "resolving",
    "downloading",
    "paused",
    "preparing-transfer",
    "distributing",
    "completed",
    "cancelled",
    "deleted",
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
    target_cluster_id: str | None = None
    target_cluster_name: str | None = None
    target_cluster_url: str | None = None
    message: str = "Queued"
    error: str | None = None
    attempts: int = 0
    smallest_gguf_transfer_only: bool = False
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
    launch_supported: bool | None = None


class ModelMutationResult(BaseModel):
    repo_id: str
    status: Literal["deleted", "not-found"]
    controller_deleted: bool = False
    cluster_deleted: bool = False
    freed_bytes: int = 0
    message: str


class RuntimeEndpoint(BaseModel):
    base_url: str
    models_url: str
    chat_completions_url: str
    status: Literal["ready", "unreachable"]
    loaded_model_ids: list[str] = Field(default_factory=list)
    routing_mode: Literal["direct-runtime"] = "direct-runtime"
    controller_required: bool = False


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
    memory_budget_mode: Literal["auto", "fixed"] = "auto"
    cpu_offload_gib: float = Field(default=0, ge=0, le=512)
    gpu_layers: int | None = Field(default=None, ge=0, le=10000)
    optimization_profile: Literal["balanced", "latency", "throughput", "long-context"] = "balanced"
    kv_cache_dtype: Literal["auto", "fp8_e4m3"] = "auto"
    max_num_seqs: int = Field(default=4, gt=0, le=1024)
    max_num_batched_tokens: int = Field(default=8192, ge=512, le=1_048_576)
    engine_id: Literal[
        "auto",
        "vllm",
        "sglang",
        "llama-cpp",
        "ollama",
        "tensorrt-llm",
        "triton",
        "tgi",
        "lmdeploy",
        "xinference",
        "localai",
        "tensorfold",
        "deepinfer",
    ] = "auto"
    speculative_method: Literal["off", "auto", "mtp", "dspark", "eagle3", "dflash"] = "auto"
    speculative_tokens: int = Field(default=0, ge=0, le=5)
    draft_model: str | None = Field(default=None, max_length=200)
    expert_parallel: bool = False
    all2all_backend: Literal[
        "auto",
        "allgather_reducescatter",
        "deepep_low_latency",
        "deepep_high_throughput",
        "naive",
    ] = "auto"
    expert_load_balancing: bool = False
    redundant_experts: int = Field(default=0, ge=0, le=32)
    cuda_graphs: bool = True
    chunked_prefill: bool = True
    cache_cpu_gib: int = Field(default=32, ge=0, le=512)
    cache_disk_gib: int = Field(default=1536, ge=0, le=16384)
    rank_order: list[str] = Field(default_factory=list, max_length=64)
    enforce_rank_order: bool = False

    @model_validator(mode="after")
    def scheduler_covers_requested_concurrency(self):
        if self.cpu_offload_gib and self.gpu_layers is not None:
            raise ValueError("Choose RAM weight offload or CPU/GPU layers, not both")
        if self.max_num_seqs < self.concurrent_sequences:
            self.max_num_seqs = self.concurrent_sequences
        if self.speculative_method == "off":
            self.speculative_tokens = 0
            self.draft_model = None
        elif self.speculative_method in {"mtp", "dspark", "eagle3", "dflash"}:
            if self.speculative_tokens == 0:
                self.speculative_tokens = 3
            if self.speculative_method == "dspark" and self.speculative_tokens < 2:
                raise ValueError("DSpark requires between 2 and 5 speculative tokens")
            if self.speculative_method == "mtp" and self.speculative_tokens > 4:
                raise ValueError("MTP supports between 1 and 4 speculative tokens")
        if self.expert_load_balancing and not self.expert_parallel:
            raise ValueError("Expert load balancing requires expert parallelism")
        if self.redundant_experts and not self.expert_load_balancing:
            raise ValueError("Redundant experts require expert load balancing")
        expected_ranks = self.tensor_parallel * self.pipeline_parallel
        if self.rank_order and len(self.rank_order) != expected_ranks:
            raise ValueError("Rank order must contain exactly one node for every TP x PP rank")
        if len(set(self.rank_order)) != len(self.rank_order):
            raise ValueError("Rank order cannot contain duplicate nodes")
        if self.enforce_rank_order and not self.rank_order:
            raise ValueError("A rank order is required when rank placement is enforced")
        return self


class RuntimeExecutionPlan(BaseModel):
    engine_id: str
    engine_name: str
    image: str | None = None
    executable: bool = False
    command: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    readiness_path: str = "/v1/models"
    inference_path: str | None = "/v1/completions"
    inference_protocol: Literal["completions", "chat", "triton"] = "completions"
    capabilities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class ModelLaunchJob(BaseModel):
    id: str
    request: ModelLaunchRequest
    target_cluster_id: str | None = None
    target_cluster_name: str | None = None
    model_path: str
    status: LaunchStatus = "queued"
    progress: float = Field(default=0, ge=0, le=100)
    message: str = "Queued"
    fit: ModelFitPlan
    execution_plan: RuntimeExecutionPlan | None = None
    error: str | None = None
    attempts: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InferenceTelemetry(BaseModel):
    runtimes: dict[str, dict] = Field(default_factory=dict)
    sampled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "unavailable"
    available: bool = False
    input_tokens_total: int | None = None
    output_tokens_total: int | None = None
    requests_total: int | None = None
    request_p95_ms: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    inter_token_p50_ms: float | None = None
    inter_token_p95_ms: float | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    speculative_acceptance_rate: float | None = None
    gpu_kv_cache_usage: float | None = None
    cache_hits: int | None = None
    cache_misses: int | None = None
    cache_retrieval_p95_ms: float | None = None
    nccl_collective_p95_ms: float | None = None
    expert_imbalance: float | None = None
    active_requests: int | None = None
    queued_requests: int | None = None
    notes: list[str] = Field(default_factory=list)


class BenchmarkRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=2_000_000)
    max_tokens: int = Field(default=128, ge=1, le=8192)
    repetitions: int = Field(default=3, ge=1, le=20)
    concurrency: int = Field(default=1, ge=1, le=64)
    cache_state: Literal["cold", "warm"] = "cold"
    objective: Literal["latency", "throughput", "balanced"] = "balanced"

    @model_validator(mode="after")
    def bounded_sweep(self):
        if self.repetitions * self.concurrency > 64:
            raise ValueError("A benchmark is limited to 64 total requests")
        if len(self.prompt) * self.concurrency > 8_000_000:
            raise ValueError("Prompt size multiplied by concurrency exceeds the 8 MB safety bound")
        return self


class BenchmarkResult(BaseModel):
    id: str
    model_id: str
    model_revision: str | None = None
    engine_id: str
    engine_image: str | None = None
    topology: str
    context_length: int
    concurrency: int
    cache_state: str
    speculative_method: str = "off"
    expert_parallel: bool = False
    repetitions: int
    successful_requests: int
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    inter_token_p50_ms: float | None = None
    inter_token_p95_ms: float | None = None
    output_tokens_per_second: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    measured: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


class CachePrefixRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    model_revision: str = Field(default="main", min_length=1, max_length=200)
    tokenizer_id: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=2_000_000)
    engine_id: str = Field(default="vllm", min_length=1, max_length=50)
    engine_config: dict[str, str | int | float | bool] = Field(default_factory=dict)
    priority: int = Field(default=50, ge=0, le=100)


class CachePrefixRecord(BaseModel):
    id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    engine_id: str
    engine_config_hash: str
    prefix_hash: str
    prompt_characters: int
    prompt_tokens: int = 0
    priority: int = 50
    prewarm_count: int = 0
    verification_count: int = 0
    last_ttft_ms: float | None = None
    last_cache_hit_delta: int | None = None
    state: Literal["registered", "prewarmed", "reused", "rebuilt", "failed"] = "registered"
    runtime_generation: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None


RuntimeEngineId = Literal[
    "vllm",
    "sglang",
    "llama-cpp",
    "ollama",
    "tensorrt-llm",
    "triton",
    "tgi",
    "lmdeploy",
    "xinference",
    "localai",
    "tensorfold",
    "deepinfer",
]
RuntimeNodeStatus = Literal["ready", "missing", "outdated", "unreachable", "installing", "failed"]
RuntimeReadinessState = Literal["ready", "needs_preparation", "partial", "unreachable", "blocked"]


class RuntimeNodeState(BaseModel):
    address: str
    role: Literal["head", "worker"]
    status: RuntimeNodeStatus
    installed_version: str | None = None
    required_version: str
    message: str


class RuntimeReadiness(BaseModel):
    model_id: str
    runtime_id: RuntimeEngineId
    runtime_name: str
    required_version: str
    image: str | None = None
    ready: bool = False
    install_supported: bool = False
    nodes: list[RuntimeNodeState] = Field(default_factory=list)
    message: str


class RuntimeReadinessRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    engine_id: Literal[
        "auto",
        "vllm",
        "sglang",
        "llama-cpp",
        "ollama",
        "tensorrt-llm",
        "triton",
        "tgi",
        "lmdeploy",
        "xinference",
        "localai",
        "tensorfold",
        "deepinfer",
    ] = "auto"
    cluster_id: str | None = Field(default=None, min_length=1, max_length=80)


class RuntimePrepareRequest(BaseModel):
    cluster_id: str = Field(min_length=1, max_length=80)
    engine_id: RuntimeEngineId


class RuntimeInstallLog(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node: str
    level: Literal["info", "success", "error"] = "info"
    message: str


class RuntimeInstallJob(BaseModel):
    id: str
    model_id: str
    runtime_id: RuntimeEngineId
    cluster_id: str | None = None
    cluster_name: str | None = None
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    progress: float = Field(default=0, ge=0, le=100)
    nodes: list[RuntimeNodeState] = Field(default_factory=list)
    logs: list[RuntimeInstallLog] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RuntimeEngineStatus(BaseModel):
    id: RuntimeEngineId
    name: str
    capability: Literal["managed_launch", "package_only", "no_launch_adapter"]
    image: str | None = None
    required_version: str
    state: RuntimeReadinessState
    ready: bool = False
    prepare_supported: bool = False
    nodes_ready: int = 0
    nodes_total: int = 0
    nodes: list[RuntimeNodeState] = Field(default_factory=list)
    message: str
    blockers: list[str] = Field(default_factory=list)
    active_job_id: str | None = None
    active_job_status: Literal["queued", "running"] | None = None
    active_job_progress: float | None = None


class RuntimeEngineCollection(BaseModel):
    cluster_id: str
    cluster_name: str
    cluster_kind: Literal["remote-bridge", "local-docker"]
    platform: str | None = None
    engines: list[RuntimeEngineStatus]


class RuntimeGuideStep(BaseModel):
    id: Literal["resolve", "cache", "install", "activate", "launch", "verify"]
    title: str
    description: str


class RuntimeGuide(BaseModel):
    cluster_id: str
    cluster_name: str
    engine_id: RuntimeEngineId
    engine_name: str
    model_id: str | None = None
    compatible: bool
    prepared: bool = False
    update_available: bool = False
    platform: str
    version: str | None = None
    image: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    summary: str
    blockers: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    steps: list[RuntimeGuideStep] = Field(default_factory=list)


class RuntimeActivationRequest(BaseModel):
    engine_id: str = Field(min_length=1, max_length=40)
    version: str = Field(min_length=1, max_length=200)
    image: str = Field(min_length=1, max_length=1000, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9./:@_-]+$")
    digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    platform: Literal["linux/arm64", "linux/amd64"]
    artifact_id: str | None = None


class RuntimeWorkflowRequest(BaseModel):
    cluster_id: str = Field(min_length=1, max_length=80)
    engine_id: Literal[
        "auto",
        "vllm",
        "sglang",
        "llama-cpp",
        "ollama",
        "tensorrt-llm",
        "triton",
        "tgi",
        "lmdeploy",
        "xinference",
        "localai",
        "tensorfold",
        "deepinfer",
    ] = "auto"
    model_id: str | None = Field(default=None, min_length=1, max_length=200)
    launch: ModelLaunchRequest | None = None

    @model_validator(mode="after")
    def launch_matches_model(self):
        if self.launch and self.model_id and self.launch.model_id != self.model_id:
            raise ValueError("Launch settings must belong to the guided model")
        if self.launch and not self.model_id:
            self.model_id = self.launch.model_id
        return self


class RuntimeWorkflowStep(BaseModel):
    id: Literal["resolve", "cache", "install", "activate", "launch", "verify"]
    title: str
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    message: str = "Waiting"


class RuntimeWorkflowJob(BaseModel):
    id: str
    request: RuntimeWorkflowRequest
    engine_id: RuntimeEngineId
    engine_name: str
    version: str
    image: str
    digest: str
    platform: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    stage: Literal["resolve", "cache", "install", "activate", "launch", "verify"] = "resolve"
    progress: float = Field(default=0, ge=0, le=100)
    steps: list[RuntimeWorkflowStep] = Field(default_factory=list)
    artifact_id: str | None = None
    library_job_id: str | None = None
    install_job_id: str | None = None
    launch_job_id: str | None = None
    response_preview: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RuntimeTerminalRequest(BaseModel):
    address: str


class RuntimeTerminalTicket(BaseModel):
    ticket: str
    address: str
    expires_in: int = 60
