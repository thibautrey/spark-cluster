"""Executable inference-engine adapters and their explicit compatibility boundaries."""

import json
from dataclasses import dataclass
from pathlib import Path

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
    supports_tensor_parallel: bool = False
    supports_pipeline_parallel: bool = False
    readiness_path: str = "/v1/models"
    inference_path: str | None = "/v1/completions"
    inference_protocol: str = "completions"
    blockers: tuple[str, ...] = ()


MANAGED_ENGINE_IDS = (
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
)


def capabilities(
    settings: Settings, image_overrides: dict[str, str] | None = None
) -> dict[str, EngineCapability]:
    images = image_overrides or {}
    vllm_speculation = tuple(
        method.strip()
        for method in settings.runtime_vllm_speculative_methods.split(",")
        if method.strip() in {"off", "auto", "mtp", "dspark", "eagle3", "dflash"}
    )

    def selected(runtime_id: str, configured: str | None) -> str | None:
        return images.get(runtime_id, configured)

    return {
        "vllm": EngineCapability(
            id="vllm",
            name="vLLM V1 for NVIDIA GPUs",
            image=images.get("vllm", settings.runtime_vllm_image),
            executable=bool(images.get("vllm", settings.runtime_vllm_image)),
            supports_multi_node=True,
            speculative_methods=vllm_speculation or ("off",),
            supports_expert_parallel=True,
            supports_lmcache=True,
            supports_tensor_parallel=True,
            supports_pipeline_parallel=True,
        ),
        "sglang": EngineCapability(
            id="sglang",
            name="SGLang",
            image=images.get("sglang", settings.runtime_sglang_image),
            executable=bool(images.get("sglang", settings.runtime_sglang_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            supports_tensor_parallel=True,
        ),
        "llama-cpp": EngineCapability(
            id="llama-cpp",
            name="llama.cpp CUDA server",
            image=selected("llama-cpp", settings.runtime_llama_cpp_image),
            executable=bool(selected("llama-cpp", settings.runtime_llama_cpp_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            readiness_path="/health",
        ),
        "ollama": EngineCapability(
            id="ollama",
            name="Ollama",
            image=selected("ollama", settings.runtime_ollama_image),
            executable=bool(selected("ollama", settings.runtime_ollama_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            readiness_path="/api/version",
            inference_path="/v1/chat/completions",
            inference_protocol="chat",
        ),
        "tensorrt-llm": EngineCapability(
            id="tensorrt-llm",
            name="TensorRT-LLM",
            image=selected("tensorrt-llm", settings.runtime_tensorrt_llm_image),
            executable=bool(selected("tensorrt-llm", settings.runtime_tensorrt_llm_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=True,
            supports_lmcache=False,
            supports_tensor_parallel=True,
            supports_pipeline_parallel=True,
        ),
        "triton": EngineCapability(
            id="triton",
            name="Triton Inference Server",
            image=selected("triton", settings.runtime_triton_image),
            executable=bool(selected("triton", settings.runtime_triton_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            readiness_path="/v2/health/ready",
            inference_path=None,
            inference_protocol="triton",
        ),
        "tgi": EngineCapability(
            id="tgi",
            name="Text Generation Inference",
            image=selected("tgi", settings.runtime_tgi_image),
            executable=bool(selected("tgi", settings.runtime_tgi_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            supports_tensor_parallel=True,
            readiness_path="/health",
            inference_path="/v1/chat/completions",
            inference_protocol="chat",
        ),
        "lmdeploy": EngineCapability(
            id="lmdeploy",
            name="LMDeploy",
            image=selected("lmdeploy", settings.runtime_lmdeploy_image),
            executable=bool(selected("lmdeploy", settings.runtime_lmdeploy_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            supports_tensor_parallel=True,
        ),
        "xinference": EngineCapability(
            id="xinference",
            name="Xinference",
            image=selected("xinference", settings.runtime_xinference_image),
            executable=bool(selected("xinference", settings.runtime_xinference_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
        ),
        "localai": EngineCapability(
            id="localai",
            name="LocalAI",
            image=selected("localai", settings.runtime_localai_image),
            executable=bool(selected("localai", settings.runtime_localai_image)),
            supports_multi_node=False,
            speculative_methods=("off", "auto"),
            supports_expert_parallel=False,
            supports_lmcache=False,
            readiness_path="/readyz",
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
        return "llama-cpp", ["GGUF selected the managed llama.cpp adapter."]
    return "vllm", notes


def build_execution_plan(
    request: ModelLaunchRequest,
    model_path: str,
    settings: Settings,
    image_overrides: dict[str, str] | None = None,
    cluster_kind: str | None = None,
) -> RuntimeExecutionPlan:
    matrix = capabilities(settings, image_overrides)
    engine_id = request.engine_id
    warnings: list[str] = []
    managed_path = Path(model_path)
    has_gguf = managed_path.is_file() and managed_path.suffix.casefold() == ".gguf"
    if managed_path.is_dir():
        has_gguf = any(managed_path.rglob("*.gguf"))
    if engine_id == "auto":
        engine_id, recommendations = recommended_engine(request.model_id, settings)
        warnings.extend(recommendations)
        if has_gguf:
            engine_id = "llama-cpp"
            warnings.append("Managed GGUF files selected the llama.cpp adapter.")
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
    if request.tensor_parallel > 1 and not engine.supports_tensor_parallel:
        blockers.append(f"{engine.name} does not expose tensor parallelism through this adapter.")
    if request.pipeline_parallel > 1 and not engine.supports_pipeline_parallel:
        blockers.append(f"{engine.name} does not expose pipeline parallelism through this adapter.")
    if ranks > 1 and not engine.supports_multi_node and cluster_kind != "local-docker":
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
    if managed_path.exists() and engine_id in {"llama-cpp", "localai"}:
        if not has_gguf:
            blockers.append(f"{engine.name} requires a GGUF file in the managed model directory.")
    if managed_path.exists() and engine_id == "triton":
        if not any(managed_path.rglob("config.pbtxt")):
            blockers.append("Triton requires a model repository containing config.pbtxt.")

    if request.cpu_offload_gib:
        if engine_id not in {"vllm", "sglang"}:
            blockers.append(
                f"{engine.name} does not support CPU weight offload through this adapter."
            )
        warnings.append("Degraded mode transfers weights from RAM; inference may be much slower.")
    if request.gpu_layers is not None:
        if engine_id != "llama-cpp":
            blockers.append("CPU/GPU layer splitting is available only with llama.cpp.")
        warnings.append("Degraded mode runs RAM-resident layers and KV operations on CPU.")
    if (
        request.cpu_offload_gib or request.gpu_layers is not None
    ) and cluster_kind != "local-docker":
        blockers.append("Degraded loading requires a local Docker pool with host RAM admission.")

    environment = {
        "ENGINE_ID": engine_id,
        "CPU_OFFLOAD_GIB": str(request.cpu_offload_gib),
        "GPU_LAYERS": str(request.gpu_layers if request.gpu_layers is not None else 999),
        "CPU_KV": "1" if request.gpu_layers is not None else "0",
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
        "RUNTIME_PORT": "8000",
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
    if request.cpu_offload_gib and engine_id in {"vllm", "sglang"}:
        command.extend(["--cpu-offload-gb", str(request.cpu_offload_gib)])
    if request.gpu_layers is not None and engine_id == "llama-cpp":
        command.extend(["--n-gpu-layers", str(request.gpu_layers), "--no-kv-offload"])
    exposed = (
        ["Triton V2 API"] if engine.inference_protocol == "triton" else ["OpenAI-compatible API"]
    )
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
        readiness_path=engine.readiness_path,
        inference_path=engine.inference_path,
        inference_protocol=engine.inference_protocol,
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
    if engine_id == "vllm":
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
    commands = {
        "llama-cpp": ["/app/llama-server", "--model", "$MODEL_ID", "--port", "$RUNTIME_PORT"],
        "ollama": ["ollama", "serve", "+", "ollama", "create", "$SERVED_MODEL_NAME"],
        "tensorrt-llm": [
            "trtllm-serve",
            "$MODEL_ID",
            "--tp_size",
            "$TENSOR_PARALLEL_SIZE",
            "--pp_size",
            "$PIPELINE_PARALLEL_SIZE",
            "--host",
            "0.0.0.0",
            "--port",
            "$RUNTIME_PORT",
        ],
        "triton": [
            "tritonserver",
            "--model-repository=$MODEL_ID",
            "--http-port=$RUNTIME_PORT",
        ],
        "tgi": [
            "text-generation-launcher",
            "--model-id",
            "$MODEL_ID",
            "--port",
            "$RUNTIME_PORT",
            "--num-shard",
            "$TENSOR_PARALLEL_SIZE",
        ],
        "lmdeploy": [
            "lmdeploy",
            "serve",
            "api_server",
            "$MODEL_ID",
            "--server-port",
            "$RUNTIME_PORT",
            "--tp",
            "$TENSOR_PARALLEL_SIZE",
        ],
        "xinference": [
            "xinference-local",
            "-p",
            "$RUNTIME_PORT",
            "+",
            "xinference",
            "launch",
            "--model-path",
            "$MODEL_ID",
        ],
        "localai": ["local-ai", "run", "$MODEL_ID"],
    }
    return commands.get(engine_id, [])


def container_launch_script(engine_id: str) -> str:
    """Return a static shell contract; launch values enter only through quoted env vars."""

    def script(*lines: str) -> str:
        return "\n".join(("set -eu", *lines))

    scripts = {
        "vllm": script(
            "cat > /tmp/lmcache.yaml <<EOF",
            "chunk_size: 256",
            "local_cpu: true",
            "max_local_cpu_size: $CACHE_CPU_GIB",
            "local_disk: file:///var/lib/lmcache",
            "max_local_disk_size: $CACHE_DISK_GIB",
            "save_decode_cache: true",
            "enable_blending: true",
            "EOF",
            'ray start --head --node-ip-address="$NODE_IP" --port=6379 --dashboard-host=0.0.0.0',
            'set -- --model "$MODEL_ID" --served-model-name "$SERVED_MODEL_NAME" '
            "--distributed-executor-backend ray "
            '--tensor-parallel-size "$TENSOR_PARALLEL_SIZE" '
            '--pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" '
            '--max-model-len "$MAX_MODEL_LEN" '
            '--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" '
            '--kv-cache-dtype "$KV_CACHE_DTYPE" --max-num-seqs "$MAX_NUM_SEQS" '
            '--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" '
            "--enable-prefix-caching "
            "--kv-transfer-config "
            '\'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}\' '
            '--host 0.0.0.0 --port "$RUNTIME_PORT"',
            'if test "${CPU_OFFLOAD_GIB:-0}" != 0; then set -- "$@" '
            '--cpu-offload-gb "$CPU_OFFLOAD_GIB"; fi',
            'if test -n "$SPECULATIVE_CONFIG"; then set -- "$@" '
            '--speculative-config "$SPECULATIVE_CONFIG"; fi',
            'if test "$CUDA_GRAPHS" != 1; then set -- "$@" --enforce-eager; fi',
            'if test "$CHUNKED_PREFILL" = 1; then set -- "$@" --enable-chunked-prefill; fi',
            'if test "$EXPERT_PARALLEL" = 1; then',
            '  set -- "$@" --enable-expert-parallel',
            '  if test "$ALL2ALL_BACKEND" != auto; then set -- "$@" '
            '--all2all-backend "$ALL2ALL_BACKEND"; fi',
            "fi",
            'if test "$EXPERT_LOAD_BALANCING" = 1; then set -- "$@" '
            "--enable-eplb --eplb-config "
            '"{\\"num_redundant_experts\\":$REDUNDANT_EXPERTS}"; fi',
            'exec python -m vllm.entrypoints.openai.api_server "$@"',
        ),
        "sglang": script(
            'exec python -m sglang.launch_server --model-path "$MODEL_ID" '
            '--served-model-name "$SERVED_MODEL_NAME" --tp "$TENSOR_PARALLEL_SIZE" '
            '--context-length "$MAX_MODEL_LEN" '
            '--mem-fraction-static "$GPU_MEMORY_UTILIZATION" '
            '--host 0.0.0.0 --port "$RUNTIME_PORT"'
        ),
        "llama-cpp": script(
            'if test -f "$MODEL_ID"; then model_file="$MODEL_ID"; else '
            "model_file=$(find \"$MODEL_ID\" -type f -name '*.gguf' "
            "-printf '%s\t%p\n' | sort -n | head -n 1 | cut -f2-); fi",
            "test -n \"$model_file\" || { echo 'No GGUF file found' >&2; exit 64; }",
            'set -- --model "$model_file" --alias "$SERVED_MODEL_NAME" '
            '--host 0.0.0.0 --port "$RUNTIME_PORT" --ctx-size "$MAX_MODEL_LEN" '
            '--parallel "$MAX_NUM_SEQS" --n-gpu-layers "${GPU_LAYERS:-999}"',
            'if test "${CPU_KV:-0}" = 1; then set -- "$@" --no-kv-offload; fi',
            'exec /app/llama-server "$@"',
        ),
        "ollama": script(
            'export OLLAMA_HOST="0.0.0.0:$RUNTIME_PORT"',
            "ollama serve &",
            "server_pid=$!",
            "trap 'kill \"$server_pid\" 2>/dev/null || true' EXIT INT TERM",
            "until ollama list >/dev/null 2>&1; do sleep 1; done",
            "printf 'FROM \"%s\"\\nPARAMETER num_ctx %s\\n' "
            '"$MODEL_ID" "$MAX_MODEL_LEN" > /tmp/Modelfile',
            'ollama create "$SERVED_MODEL_NAME" --file /tmp/Modelfile',
            'wait "$server_pid"',
        ),
        "tensorrt-llm": script(
            'exec trtllm-serve "$MODEL_ID" --tp_size "$TENSOR_PARALLEL_SIZE" '
            '--pp_size "$PIPELINE_PARALLEL_SIZE" --host 0.0.0.0 '
            '--port "$RUNTIME_PORT"'
        ),
        "triton": script(
            'exec tritonserver --model-repository="$MODEL_ID" --http-address=0.0.0.0 '
            '--http-port="$RUNTIME_PORT"'
        ),
        "tgi": script(
            'exec text-generation-launcher --model-id "$MODEL_ID" --hostname 0.0.0.0 '
            '--port "$RUNTIME_PORT" --num-shard "$TENSOR_PARALLEL_SIZE" '
            '--max-total-tokens "$MAX_MODEL_LEN" '
            '--max-concurrent-requests "$MAX_NUM_SEQS"'
        ),
        "lmdeploy": script(
            'exec lmdeploy serve api_server "$MODEL_ID" --server-name 0.0.0.0 '
            '--server-port "$RUNTIME_PORT" --tp "$TENSOR_PARALLEL_SIZE" '
            '--session-len "$MAX_MODEL_LEN"'
        ),
        "xinference": script(
            'xinference-local -H 0.0.0.0 -p "$RUNTIME_PORT" &',
            "server_pid=$!",
            "trap 'kill \"$server_pid\" 2>/dev/null || true' EXIT INT TERM",
            'until xinference list --endpoint "http://127.0.0.1:$RUNTIME_PORT" '
            ">/dev/null 2>&1; do sleep 1; done",
            "model_engine=transformers",
            'if test -f "$MODEL_ID" && test "${MODEL_ID##*.}" = gguf; then '
            "model_engine=llama.cpp; fi",
            'if test -d "$MODEL_ID" && find "$MODEL_ID" -type f -name \'*.gguf\' '
            "-print -quit | grep -q .; then model_engine=llama.cpp; fi",
            'xinference launch --endpoint "http://127.0.0.1:$RUNTIME_PORT" '
            '--model-path "$MODEL_ID" --model-engine "$model_engine" '
            '--model-uid "$SERVED_MODEL_NAME" -n "$SERVED_MODEL_NAME"',
            'wait "$server_pid"',
        ),
        "localai": script(
            'if test -f "$MODEL_ID"; then model_file="$MODEL_ID"; else '
            "model_file=$(find \"$MODEL_ID\" -type f -name '*.gguf' "
            "-printf '%s\t%p\n' | sort -n | head -n 1 | cut -f2-); fi",
            "test -n \"$model_file\" || { echo 'No GGUF file found' >&2; exit 64; }",
            'export LOCALAI_ADDRESS="0.0.0.0:$RUNTIME_PORT"',
            'exec local-ai run "$model_file"',
        ),
    }
    try:
        return scripts[engine_id]
    except KeyError as error:
        raise ValueError(f"No container launch command is defined for {engine_id}") from error
