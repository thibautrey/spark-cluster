"""Evidence-backed runtime defaults for DGX Spark model families."""

from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings
from .runtime_engines import capabilities, recommended_engine


class PerformancePlanRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    node_count: int | None = Field(default=None, ge=1, le=8)
    topology: Literal["auto", "single", "direct", "ring", "switch"] = "auto"
    workload: Literal["balanced", "latency", "throughput", "long-context"] = "balanced"
    context_length: int = Field(default=262_144, gt=0, le=1_048_576)
    concurrent_sequences: int = Field(default=1, gt=0, le=1024)


class PerformanceEvidence(BaseModel):
    title: str
    finding: str
    source_url: str


class PerformancePlan(BaseModel):
    model_family: str
    profile: str
    engine: str
    engine_id: str = "vllm"
    engine_executable: bool = True
    recommended_nodes: int
    tensor_parallel: int
    pipeline_parallel: int = 1
    context_length: int
    gpu_memory_utilization: float
    kv_cache_dtype: Literal["auto", "fp8_e4m3"]
    max_num_seqs: int
    max_num_batched_tokens: int
    collective_backend: str
    topology_guidance: str
    engine_guidance: str
    automatic_settings: list[str]
    warnings: list[str]
    evidence: list[PerformanceEvidence]
    speculative_method: str = "off"
    speculative_tokens: int = 0
    expert_parallel: bool = False
    all2all_backend: str = "auto"
    expert_load_balancing: bool = False
    cache_cpu_gib: int = 32
    cache_disk_gib: int = 1536
    context_options: list[int] = Field(default_factory=list)


NVIDIA_SCALE = "https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/"
DEEPSEEK_RECIPE = "https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark"
GLM_RECIPE = "https://github.com/chishiki37/glm-5.3-flash-nvfp4-dgx-spark"


def _topology(nodes: int, requested: str) -> tuple[str, str]:
    topology = requested
    if topology == "auto":
        topology = (
            "single"
            if nodes == 1
            else "direct"
            if nodes == 2
            else "ring"
            if nodes == 4
            else "switch"
        )
    if topology == "switch":
        return (
            "RoCEnante with transparent NCCL fallback",
            "Use the switched fabric for direct any-to-any exchange and independent model pools.",
        )
    if topology == "ring":
        return (
            "NCCL ring over RoCEv2",
            "Use both ConnectX-7 ports per Spark and keep tensor-parallel ranks in "
            "physical ring order.",
        )
    if topology == "direct":
        return (
            "NCCL over direct RoCEv2",
            "Keep model traffic on the 200G fabric and management traffic on Ethernet.",
        )
    return "Local collectives", "No inter-node collective tuning is needed for one Spark."


def build_performance_plan(
    request: PerformancePlanRequest, available_nodes: int, settings: Settings | None = None
) -> PerformancePlan:
    settings = settings or Settings()
    model = request.model_id.lower()
    nodes = max(1, request.node_count or available_nodes or 1)
    collective, topology_guidance = _topology(nodes, request.topology)
    evidence = [
        PerformanceEvidence(
            title="Official DGX Spark scaling",
            finding=(
                "On four Sparks, NVIDIA measured 32.82 tok/s on a physical ring and 36.34 "
                "tok/s on a switch with RoCEnante for Qwen3.8-27B NVFP4 at concurrency 1."
            ),
            source_url=NVIDIA_SCALE,
        )
    ]
    warnings: list[str] = []
    engine_id, engine_notes = recommended_engine(request.model_id, settings)
    warnings.extend(engine_notes)
    engine_capability = capabilities(settings)[engine_id]
    speculative_method = "off"
    speculative_tokens = 0
    expert_parallel = False
    all2all_backend = "auto"
    expert_load_balancing = False
    automatic_settings = [
        "Force NCCL onto the RoCE/IB transport without hard-coding a reboot-volatile GID index.",
        "Bound scheduler concurrency and batched tokens instead of using broad runtime defaults.",
        "Keep CUDA graphs enabled; select a compatibility profile only when the model image "
        "requires eager mode.",
    ]

    if "glm-5.3" in model or "glm5.3" in model:
        profile = "GLM 5.3 Flash · NVFP4/FP8 KV"
        engine = "vLLM SM121 optimized image"
        engine_id = "vllm"
        recommended_nodes = 4 if nodes >= 4 else 2 if nodes >= 2 else 1
        tensor_parallel = recommended_nodes
        context = max(request.context_length, 262_144)
        gpu = 0.85
        kv_dtype = "fp8_e4m3"
        max_seqs = max(request.concurrent_sequences, 6 if recommended_nodes >= 4 else 4)
        batch_tokens = 8192
        engine_guidance = (
            "Use clean compressed-tensors NVFP4 weights and an SM121 vLLM image. Native MTP3 "
            "is the validated balanced speculative path; DFlash2 depends on the workload "
            "and license."
        )
        speculative_method = "mtp"
        speculative_tokens = 3
        expert_parallel = True
        expert_load_balancing = request.workload == "throughput"
        all2all_backend = (
            "allgather_reducescatter" if request.topology in {"auto", "ring"} else "auto"
        )
        warnings.extend(
            [
                "Do not hard-code NCCL_IB_GID_INDEX; the RoCE GID table can change after "
                "reboot or recabling.",
                "Some DFlash2 weights are restricted to non-commercial use. Verify the exact "
                "model license before enabling that path.",
                "Pin and stress-test the KV allocation on real nodes before increasing it "
                "beyond the recommended runtime value.",
            ]
        )
        evidence.append(
            PerformanceEvidence(
                title="GLM 5.3 Flash sweep",
                finding=(
                    "Measured results found FP8 KV doubled capacity, MTP3 beat MTP4, and removing "
                    "eager mode improved single-stream performance by 22–38% on clean weights."
                ),
                source_url=GLM_RECIPE,
            )
        )
    elif "deepseek-v4.1" in model or "deepseek_v4.1" in model:
        profile = "DeepSeek V4.1 Flash · long-context"
        engine = "vLLM executable baseline; TensorFold available after adapter validation"
        recommended_nodes = 4 if nodes >= 4 else 2 if nodes >= 2 else 1
        tensor_parallel = recommended_nodes
        context = request.context_length
        gpu = 0.83
        kv_dtype = "auto"
        max_seqs = max(request.concurrent_sequences, 16 if recommended_nodes >= 4 else 8)
        batch_tokens = 8192
        engine_guidance = (
            "Launch the runnable vLLM compatibility lane. TensorFold with EXL3 remains a "
            "measured specialized option, but is not selected until its image and command "
            "contract are validated on every GB10 node."
        )
        available_speculation = capabilities(settings)["vllm"].speculative_methods
        speculative_method = "dspark" if "dspark" in available_speculation else "mtp"
        speculative_tokens = 3
        if speculative_method != "dspark":
            warnings.append(
                "DSpark is not declared by the pinned vLLM capability profile; use MTP as "
                "the executable baseline and enable DSpark only with a validated image."
            )
        expert_parallel = request.workload == "throughput"
        expert_load_balancing = expert_parallel
        all2all_backend = (
            "allgather_reducescatter" if request.topology in {"auto", "ring"} else "auto"
        )
        warnings.extend(
            [
                "The specialized TensorFold lane needs a pinned engine build and "
                "model-specific weight layout.",
                "Distribute sparse Engram rows to node-local NVMe; reading them over "
                "management Ethernet removes the long-context advantage.",
            ]
        )
        evidence.append(
            PerformanceEvidence(
                title="DeepSeek V4.1 Flash lane",
                finding=(
                    "Four-Spark TensorFold reached 93.6 tok/s at one stream and 270.8 tok/s "
                    "at 16 streams; using both RoCE rails improved measured cells by 4–15%."
                ),
                source_url=DEEPSEEK_RECIPE,
            )
        )
    else:
        profile = "Portable vLLM · balanced"
        engine = "vLLM V1"
        recommended_nodes = nodes
        tensor_parallel = nodes
        context = request.context_length
        gpu = 0.82 if request.workload != "throughput" else 0.85
        kv_dtype = "fp8_e4m3" if request.workload == "long-context" else "auto"
        max_seqs = max(request.concurrent_sequences, 8 if request.workload == "throughput" else 4)
        batch_tokens = 16_384 if request.workload == "throughput" else 8192
        engine_guidance = (
            "Use the model vendor's GB10-compatible vLLM image and validate one warm request "
            "before load testing."
        )
        if any(marker in model for marker in ("moe", "a3b", "a2b", "mixtral")):
            expert_parallel = request.workload == "throughput" and nodes > 1
            expert_load_balancing = expert_parallel
            all2all_backend = (
                "allgather_reducescatter" if request.topology in {"auto", "ring"} else "auto"
            )

    if request.workload == "latency":
        max_seqs = max(1, request.concurrent_sequences)
    elif request.workload == "long-context":
        max_seqs = max(1, request.concurrent_sequences)
        if "deepseek-v4.1" not in model:
            kv_dtype = "fp8_e4m3"
    elif request.workload == "throughput":
        max_seqs = max(max_seqs, request.concurrent_sequences, 8)
        batch_tokens = max(batch_tokens, 16_384)

    cache_cpu_gib = 48 if request.workload == "long-context" else 32
    cache_disk_gib = (
        3072
        if request.workload == "long-context"
        else 2048
        if request.workload == "throughput"
        else 1536
    )
    engine_capability = capabilities(settings)[engine_id]

    return PerformancePlan(
        model_family=request.model_id,
        profile=profile,
        engine=engine,
        engine_id=engine_id,
        engine_executable=engine_capability.executable and bool(engine_capability.image),
        recommended_nodes=recommended_nodes,
        tensor_parallel=tensor_parallel,
        context_length=context,
        gpu_memory_utilization=gpu,
        kv_cache_dtype=kv_dtype,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=batch_tokens,
        collective_backend=collective,
        topology_guidance=topology_guidance,
        engine_guidance=engine_guidance,
        automatic_settings=automatic_settings,
        warnings=warnings,
        evidence=evidence,
        speculative_method=speculative_method,
        speculative_tokens=speculative_tokens,
        expert_parallel=expert_parallel,
        all2all_backend=all2all_backend,
        expert_load_balancing=expert_load_balancing,
        cache_cpu_gib=cache_cpu_gib,
        cache_disk_gib=cache_disk_gib,
        context_options=[16_384, 65_536, 262_144, 524_288, 1_048_576],
    )
