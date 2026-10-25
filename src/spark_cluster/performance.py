"""Evidence-backed runtime defaults for DGX Spark model families."""

from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings
from .deepseek_mia import MODEL_ID, MiaRecipePlan, build_mia_recipe
from .deepseek_tonyd2wild import (
    RECIPE_SOURCE as TONY_RECIPE_SOURCE,
)
from .deepseek_tonyd2wild import (
    TonyRecipePlan,
    build_tony_recipe,
    profile_for_model,
)
from .runtime_engines import capabilities, recommended_engine


class PerformancePlanRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    revision: str | None = Field(default=None, min_length=1, max_length=200)
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
    recipe: MiaRecipePlan | TonyRecipePlan | None = None
    blockers: list[str] = Field(default_factory=list)


NVIDIA_SCALE = "https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/"
DEEPSEEK_RECIPE = TONY_RECIPE_SOURCE
GLM_RECIPE = "https://github.com/chishiki37/glm-5.3-flash-nvfp4-dgx-spark"


def _topology(nodes: int, requested: str, hardware: str) -> tuple[str, str]:
    if hardware != "dgx-spark":
        if nodes == 1:
            return "Local GPU", "No distributed collective tuning is needed."
        return (
            "NCCL over local PCIe",
            "Keep tensor-parallel ranks on the GPUs of this host and verify CUDA peer access.",
        )
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
    is_spark = settings.active_hardware == "dgx-spark"
    if is_spark and request.model_id == MODEL_ID:
        return _mia_performance_plan(request, nodes)
    if is_spark and profile_for_model(request.model_id):
        return _tony_performance_plan(request, nodes)
    collective, topology_guidance = _topology(nodes, request.topology, settings.active_hardware)
    evidence = (
        [
            PerformanceEvidence(
                title="Official DGX Spark scaling",
                finding=(
                    "On four Sparks, NVIDIA measured 32.82 tok/s on a physical ring and 36.34 "
                    "tok/s on a switch with RoCEnante for Qwen3.8-27B NVFP4 at concurrency 1."
                ),
                source_url=NVIDIA_SCALE,
            )
        ]
        if is_spark
        else []
    )
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
        (
            "Force NCCL onto the RoCE/IB transport without hard-coding a reboot-volatile GID index."
            if is_spark
            else "Use local NCCL and keep tensor-parallel ranks on this GPU host."
        ),
        "Bound scheduler concurrency and batched tokens instead of using broad runtime defaults.",
        "Keep CUDA graphs enabled; select a compatibility profile only when the model image "
        "requires eager mode.",
    ]

    if is_spark and ("glm-5.3" in model or "glm5.3" in model):
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
    elif is_spark and ("deepseek-v4.1" in model or "deepseek_v4.1" in model):
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
            if is_spark
            else "Use the pinned vLLM image and validate one warm request before load testing."
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


def _mia_performance_plan(request: PerformancePlanRequest, available_nodes: int) -> PerformancePlan:
    nodes = 4 if available_nodes >= 4 else 3
    context = min(request.context_length, 1_048_576 if nodes == 4 else 262_144)
    if nodes == 3 and request.workload != "long-context":
        context = min(context, 32_768)
    recipe = build_mia_recipe(
        nodes,
        context,
        request.concurrent_sequences,
        request.topology,
        request.revision,
    )
    recipe.environment["CONTEXT_LENGTH"] = str(context)
    warnings = [
        "Published MIA results are external evidence for a patched SGLang recipe, "
        "not throughput measured by this controller.",
        "Never use RAM Engram offload or expandable_segments:True on this unified-memory stack.",
        "Named loop errors, malformed-history repair and the output cap require the recipe's "
        "adapters; environment variables alone do not add them to stock SGLang.",
    ]
    if context != request.context_length:
        warnings.append(f"The profile reduces the requested context to {context:,} tokens.")
    if available_nodes < 3:
        recipe.blockers.append(
            "The native checkpoint needs at least three DGX Sparks; two do not fit."
        )
    if nodes == 3:
        topology = "ConnectX-7 RoCE triangle"
        topology_guidance = "Use both ports on each of three Sparks to form the complete triangle."
        warnings.append(
            "TP3 uses 32k context for unattended serving. The 256k configured ceiling needs "
            "head-memory and long-prefill validation; choose long-context to explore it."
        )
    elif request.topology == "ring":
        topology = "Patched NCCL switchless RoCE ring"
        topology_guidance = (
            "Use four cables, physical rank order and the upstream NCCL patch on every rank."
        )
    else:
        topology = "Switched RoCE fabric"
        topology_guidance = "Use a RoCE switch; the switchless ring is a separate opt-in profile."
    return PerformancePlan(
        model_family=request.model_id,
        profile=f"DeepSeek V4.1 Flash · MIA SGLang TP{nodes}",
        engine="MIA patched SGLang · native MXFP4 experts / FP8 dense weights",
        engine_id="sglang",
        engine_executable=False,
        recommended_nodes=nodes,
        tensor_parallel=nodes,
        context_length=context,
        gpu_memory_utilization=float(recipe.environment["MEM_FRACTION_STATIC"]),
        kv_cache_dtype="auto",
        max_num_seqs=int(recipe.environment["MAX_RUNNING_REQUESTS"]),
        max_num_batched_tokens=int(recipe.environment["CHUNKED_PREFILL_SIZE"]),
        collective_backend=topology,
        topology_guidance=topology_guidance,
        engine_guidance=(
            "Prepare the pinned upstream recipe using the profile below. Managed launch "
            "requires a dedicated multi-node adapter with packed Engram mounts and patched "
            "kernels; configuring an SGLang image alone is insufficient."
        ),
        automatic_settings=recipe.prerequisites,
        warnings=warnings,
        evidence=[
            PerformanceEvidence(
                title=f"MIA TP{nodes} measurements", finding=finding, source_url=recipe.recipe_url
            )
            for finding in recipe.measurements
        ],
        speculative_method="dspark",
        speculative_tokens=5,
        expert_parallel=False,
        cache_cpu_gib=0,
        cache_disk_gib=0,
        context_options=[16_384, 32_768, 65_536, 131_072, 262_144]
        + ([524_288, 1_048_576] if nodes == 4 else []),
        recipe=recipe,
        blockers=recipe.blockers,
    )


def _tony_performance_plan(
    request: PerformancePlanRequest, available_nodes: int
) -> PerformancePlan:
    profile = profile_for_model(request.model_id)
    if not profile:  # Kept explicit for type narrowing and direct unit calls.
        raise ValueError("No reviewed Tech2wild profile exists for this model")
    context = min(request.context_length, 1_048_576)
    recipe = build_tony_recipe(
        request.model_id,
        context,
        request.concurrent_sequences,
        request.topology,
        request.revision,
    )
    if available_nodes < profile.nodes:
        recipe.blockers.append(
            f"The reviewed TP{profile.nodes} lane needs {profile.nodes} DGX Sparks; "
            f"only {available_nodes} are selected."
        )
    warnings = [
        "Tech2wild's results are external measurements for one pinned TensorFold, weight, "
        "kernel-kit and fabric combination; they are not controller benchmarks.",
        "The TensorFold position pool is shared by active streams. The 1M context window "
        "does not mean every configured stream can simultaneously hold 1M tokens.",
        "TensorFold owns DSpark draft length and its native KV layout; generic vLLM KV and "
        "speculative-token controls do not translate to this lane.",
        "The dashboard scheduler token field is not a TensorFold setting and is ignored by "
        "the external launcher.",
        "Verify clocks and decode-shaped memory bandwidth before comparing throughput; the "
        "external fleet observed a hidden GB10 slow state not visible in nvidia-smi.",
        "The external server documents reference mismatches for double-encoded tool argument "
        "strings and text-only content arrays; include them in client compatibility tests.",
    ]
    if context != request.context_length:
        warnings.append(f"The profile reduces the requested context to {context:,} tokens.")
    if profile.nodes == 2:
        topology = "Dual-rail RoCE TP2"
        topology_guidance = (
            "Use both ConnectX-7 rails after probing routable addresses, HCA names and the "
            "RoCE v2 GID on both ranks."
        )
    else:
        topology = "Dual-rail RoCE four-rank fabric"
        topology_guidance = (
            "Keep the recorded rank order, verify every rail and confirm balanced traffic. "
            "Do not copy the author's interface names or GID index."
        )
    return PerformancePlan(
        model_family=request.model_id,
        profile=f"DeepSeek V4.1 Flash · Tech2wild TensorFold TP{profile.nodes}",
        engine=f"TensorFold native Zig · EXL3 {profile.bits_per_weight:g} bpw",
        engine_id="tensorfold",
        engine_executable=False,
        recommended_nodes=profile.nodes,
        tensor_parallel=profile.nodes,
        context_length=context,
        gpu_memory_utilization=round(profile.runtime_required_gib_per_rank / 121.7, 3),
        kv_cache_dtype="auto",
        max_num_seqs=profile.streams,
        max_num_batched_tokens=8192,
        collective_backend=topology,
        topology_guidance=topology_guidance,
        engine_guidance=(
            "Prepare the pinned native TensorFold lane and its generated per-rank inputs. "
            "The managed controller cannot launch it from a generic image or model folder."
        ),
        automatic_settings=recipe.prerequisites,
        warnings=warnings,
        evidence=[
            PerformanceEvidence(
                title=f"Tech2wild TP{profile.nodes} measurements",
                finding=finding,
                source_url=recipe.recipe_url,
            )
            for finding in recipe.measurements
        ],
        speculative_method="dspark",
        speculative_tokens=0,
        expert_parallel=False,
        cache_cpu_gib=0,
        cache_disk_gib=0,
        context_options=[16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576],
        recipe=recipe,
        blockers=recipe.blockers,
    )
