"""Executable inference-engine adapters and their explicit compatibility boundaries."""

import json
from dataclasses import dataclass

from .config import Settings
from .deepseek_mia import MANAGED_LAUNCH_BLOCKER, MODEL_ID
from .deepseek_tonyd2wild import (
    MANAGED_LAUNCH_BLOCKER as TONY_MANAGED_LAUNCH_BLOCKER,
)
from .deepseek_tonyd2wild import profile_for_model
from .models import ModelLaunchRequest, RuntimeExecutionPlan


@dataclass(frozen=True)
class EngineCapability:
    id: str
    name: str
    image: str | None
    executable: bool
    supports_multi_node: bool
    speculative_methods: tuple[str, ...]
    supports_expert_parallel: bool
    supports_lmcache: bool
    blockers: tuple[str, ...] = ()


def capabilities(settings: Settings) -> dict[str, EngineCapability]:
    vllm_speculation = tuple(
        method.strip()
        for method in settings.runtime_vllm_speculative_methods.split(",")
        if method.strip() in {"off", "auto", "mtp", "dspark", "eagle3", "dflash"}
    )
    return {
        "vllm": EngineCapability(
            id="vllm",
            name="vLLM V1 for NVIDIA GPUs",
            image=settings.runtime_vllm_image,
            executable=bool(settings.runtime_vllm_image),
            supports_multi_node=True,
            speculative_methods=vllm_speculation or ("off",),
            supports_expert_parallel=True,
            supports_lmcache=True,
        ),
        "sglang": EngineCapability(
            id="sglang",
            name="SGLang",
            image=settings.runtime_sglang_image,
            executable=bool(settings.runtime_sglang_image),
            supports_multi_node=False,
            speculative_methods=("off", "auto", "mtp", "eagle3"),
            supports_expert_parallel=True,
            supports_lmcache=False,
            blockers=(
                "This control plane currently supports SGLang on one Spark; distributed rank "
                "bootstrap needs an explicitly validated image and launcher.",
            ),
        ),
        "tensorfold": EngineCapability(
            id="tensorfold",
            name="TensorFold specialized lane",
            image=settings.runtime_tensorfold_image,
            executable=False,
            supports_multi_node=True,
            speculative_methods=("off", "auto", "dspark"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            blockers=(
                "A pinned image alone is insufficient: configure a tested TensorFold command "
                "contract and model-specific weight layout before enabling launches.",
            ),
        ),
        "deepinfer": EngineCapability(
            id="deepinfer",
            name="DeepInfer experimental",
            image=settings.runtime_deepinfer_image,
            executable=False,
            supports_multi_node=True,
            speculative_methods=("dspark",),
            supports_expert_parallel=True,
            supports_lmcache=False,
            blockers=(
                "DeepInfer targets SM100 kernels. DGX Spark uses GB10 / SM121; compiling for "
                "sm_121 does not validate its FlashAttention, MoE, NCCL, or MLA kernels.",
            ),
        ),
    }


def recommended_engine(model_id: str, settings: Settings) -> tuple[str, list[str]]:
    model = model_id.casefold()
    notes: list[str] = []
    if model_id == MODEL_ID and settings.active_hardware == "dgx-spark":
        return "sglang", [MANAGED_LAUNCH_BLOCKER]
    if profile_for_model(model_id) and settings.active_hardware == "dgx-spark":
        return "tensorfold", [TONY_MANAGED_LAUNCH_BLOCKER]
    if "deepseek-v4.1" in model or "deepseek_v4.1" in model:
        notes.append(
            "TensorFold is a specialized measured option, but this installation has no "
            "validated executable adapter; vLLM remains the runnable baseline."
        )
    if "gguf" in model:
        notes.append(
            "GGUF needs a llama.cpp-family runtime; the Kubernetes adapter is not enabled."
        )
    return "vllm", notes


def build_execution_plan(
    request: ModelLaunchRequest, model_path: str, settings: Settings
) -> RuntimeExecutionPlan:
    matrix = capabilities(settings)
    engine_id = request.engine_id
    warnings: list[str] = []
    if engine_id == "auto":
        engine_id, recommendations = recommended_engine(request.model_id, settings)
        warnings.extend(recommendations)
    engine = matrix[engine_id]
    blockers = list(engine.blockers)
    if request.model_id == MODEL_ID:
        blockers.append(MANAGED_LAUNCH_BLOCKER)
    if profile_for_model(request.model_id):
        blockers.append(TONY_MANAGED_LAUNCH_BLOCKER)
        if engine_id != "tensorfold":
            blockers.append(
                "The reviewed EXL3 pack requires the pinned native TensorFold lane; the "
                f"managed {engine.name} adapter is not an equivalent fallback."
            )
    ranks = request.tensor_parallel * request.pipeline_parallel
    if not engine.image:
        blockers.append(f"No pinned {engine.name} image is configured.")
    if ranks > 1 and not engine.supports_multi_node:
        blockers.append(f"{engine.name} is enabled only for a single Spark in this release.")
    if request.speculative_method not in engine.speculative_methods:
        blockers.append(
            f"{engine.name} does not expose the {request.speculative_method} speculative method."
        )
    if request.expert_parallel and not engine.supports_expert_parallel:
        blockers.append(f"{engine.name} has no executable expert-parallel adapter.")
    if request.cache_disk_gib and not engine.supports_lmcache:
        warnings.append(f"{engine.name} does not use the configured LMCache disk tier.")
    if "gguf" in request.model_id.casefold() and engine_id == "vllm":
        blockers.append(
            "This GGUF repository needs a validated llama.cpp-family adapter; the vLLM "
            "multi-node launcher is not used as an implicit fallback."
        )

    environment = {
        "ENGINE_ID": engine_id,
        "MODEL_ID": model_path,
        "SERVED_MODEL_NAME": request.served_model_name,
        "MAX_MODEL_LEN": str(request.context_length),
        "TENSOR_PARALLEL_SIZE": str(request.tensor_parallel),
        "PIPELINE_PARALLEL_SIZE": str(request.pipeline_parallel),
        "GPU_MEMORY_UTILIZATION": str(request.gpu_memory_utilization),
        "KV_CACHE_DTYPE": request.kv_cache_dtype,
        "MAX_NUM_SEQS": str(request.max_num_seqs),
        "MAX_NUM_BATCHED_TOKENS": str(request.max_num_batched_tokens),
        "CUDA_GRAPHS": "1" if request.cuda_graphs else "0",
        "CHUNKED_PREFILL": "1" if request.chunked_prefill else "0",
        "EXPERT_PARALLEL": "1" if request.expert_parallel else "0",
        "ALL2ALL_BACKEND": request.all2all_backend,
        "EXPERT_LOAD_BALANCING": "1" if request.expert_load_balancing else "0",
        "REDUNDANT_EXPERTS": str(request.redundant_experts),
        "CACHE_CPU_GIB": str(request.cache_cpu_gib),
        "CACHE_DISK_GIB": str(request.cache_disk_gib),
        "RANK_ORDER": ",".join(request.rank_order),
        "ENFORCE_RANK_ORDER": "1" if request.enforce_rank_order else "0",
    }
    if request.speculative_method not in {"off", "auto"}:
        speculative = {
            "method": request.speculative_method,
            "num_speculative_tokens": request.speculative_tokens,
        }
        if request.draft_model:
            speculative["model"] = request.draft_model
        environment["SPECULATIVE_CONFIG"] = json.dumps(speculative, separators=(",", ":"))
    else:
        environment["SPECULATIVE_CONFIG"] = ""

    command = _command_preview(engine_id, request)
    exposed = ["OpenAI-compatible API"]
    if engine.supports_multi_node:
        exposed.append("multi-node")
    if engine.supports_lmcache:
        exposed.append("persistent LMCache")
    exposed.extend(
        method.upper() for method in engine.speculative_methods if method not in {"off", "auto"}
    )
    if engine.supports_expert_parallel:
        exposed.append("expert parallelism")
    return RuntimeExecutionPlan(
        engine_id=engine_id,
        engine_name=engine.name,
        image=engine.image,
        executable=engine.executable and bool(engine.image) and not blockers,
        command=command,
        environment=environment,
        capabilities=exposed,
        warnings=warnings,
        blockers=blockers,
    )


def _command_preview(engine_id: str, request: ModelLaunchRequest) -> list[str]:
    if engine_id == "sglang":
        return [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "$MODEL_ID",
            "--served-model-name",
            "$SERVED_MODEL_NAME",
            "--tp",
            "$TENSOR_PARALLEL_SIZE",
            "--context-length",
            "$MAX_MODEL_LEN",
        ]
    if engine_id != "vllm":
        return []
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "$MODEL_ID",
        "--served-model-name",
        "$SERVED_MODEL_NAME",
        "--tensor-parallel-size",
        "$TENSOR_PARALLEL_SIZE",
        "--pipeline-parallel-size",
        "$PIPELINE_PARALLEL_SIZE",
    ]
    if request.speculative_method not in {"off", "auto"}:
        command.extend(["--speculative-config", "$SPECULATIVE_CONFIG"])
    if request.expert_parallel:
        command.append("--enable-expert-parallel")
    return command
