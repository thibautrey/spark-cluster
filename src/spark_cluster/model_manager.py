import asyncio
import contextlib
import json
import math
import os
import re
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .config import Settings
from .deepseek_mia import (
    KV_BYTES_PER_TOKEN_PER_RANK,
    MANAGED_LAUNCH_BLOCKER,
    RESIDENT_WEIGHTS_GIB,
    build_mia_recipe,
    is_mia_checkpoint,
)
from .deepseek_mia import (
    MODEL_ID as MIA_MODEL_ID,
)
from .model_relay import ModelRelay
from .models import (
    ClusterSnapshot,
    ControllerModelImportRequest,
    DownloadJob,
    LocalModel,
    ModelDownloadRequest,
    ModelFitPlan,
    ModelFitRequest,
    ModelMutationResult,
    RuntimeEndpoint,
)
from .store import SnapshotStore

GIB = 2**30
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
REPO_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEEPSEEK_V41_ENGRAM_BYTES = 203_073_077_576


@dataclass(frozen=True)
class RepositoryMetadata:
    repo_id: str
    revision: str
    repository_bytes: int
    weight_bytes: int
    parameter_count: int | None
    architecture: str | None
    config: dict[str, Any]
    files: tuple[tuple[str, int], ...] = ()
    tags: tuple[str, ...] = ()
    library_name: str | None = None
    pipeline_tag: str | None = None
    engram_bytes: int = 0
    external_artifacts_bytes: int = 0
    external_artifacts: tuple[str, ...] = ()
    kv_bytes_per_token_bf16: float | None = None
    commit: str | None = None


def safe_repo_parts(repo_id: str) -> tuple[str, ...]:
    parts = tuple(part for part in repo_id.strip().strip("/").split("/") if part)
    if not 1 <= len(parts) <= 2 or any(not REPO_PART.fullmatch(part) for part in parts):
        raise ValueError("Use a Hugging Face model id such as organization/model-name")
    return parts


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class HuggingFaceCatalog:
    def __init__(self, token: str | None = None, download_workers: int = 4):
        self.token = token
        self.download_workers = download_workers
        self.api = HfApi(token=token)

    def inspect(self, repo_id: str, revision: str) -> RepositoryMetadata:
        safe_repo_parts(repo_id)
        info = self.api.model_info(
            repo_id, revision=revision, files_metadata=True, token=self.token
        )
        siblings = info.siblings or []
        files = tuple(
            (
                str(getattr(item, "rfilename", "")),
                int(getattr(item, "size", 0) or 0),
            )
            for item in siblings
        )
        repository_bytes = sum(int(getattr(item, "size", 0) or 0) for item in siblings)
        weight_bytes = sum(
            int(getattr(item, "size", 0) or 0)
            for item in siblings
            if str(getattr(item, "rfilename", "")).lower().endswith(WEIGHT_SUFFIXES)
        )
        config = dict(info.config or {})
        try:
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                revision=revision,
                token=self.token,
            )
            config.update(json.loads(Path(config_path).read_text()))
        except Exception:
            # Some repositories omit config.json; size-based planning remains available.
            pass
        architecture = None
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            architecture = str(architectures[0])
        elif config.get("model_type"):
            architecture = str(config["model_type"])

        parameter_count = None
        safetensors = getattr(info, "safetensors", None)
        total = getattr(safetensors, "total", None)
        if total:
            parameter_count = int(total)
        elif isinstance(safetensors, dict) and safetensors.get("total"):
            parameter_count = int(safetensors["total"])

        tags = tuple(str(tag) for tag in (getattr(info, "tags", None) or []))
        model_type = str(config.get("model_type") or "")
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            model_type = str(text_config.get("model_type") or model_type)
        is_deepseek_v41 = model_type == "deepseek_v41" or "deepseek-v4.1" in repo_id.lower()
        engram_files = tuple(
            (name, size)
            for name, size in files
            if re.search(r"model-0004[78]-of-00048\.safetensors$", name)
        )
        engram_bytes = sum(size for _, size in engram_files)
        external_artifacts_bytes = 0
        external_artifacts: tuple[str, ...] = ()
        if is_deepseek_v41 and "flash-1017" in repo_id.lower() and not engram_bytes:
            base_model = next(
                (
                    tag.removeprefix("base_model:")
                    for tag in tags
                    if tag.startswith("base_model:") and not tag.startswith("base_model:quantized:")
                ),
                "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            )
            try:
                base_info = self.api.model_info(
                    base_model, revision="main", files_metadata=True, token=self.token
                )
                base_files = tuple(
                    (
                        str(getattr(item, "rfilename", "")),
                        int(getattr(item, "size", 0) or 0),
                    )
                    for item in (base_info.siblings or [])
                    if re.search(
                        r"model-0004[78]-of-00048\.safetensors$",
                        str(getattr(item, "rfilename", "")),
                    )
                )
                external_artifacts_bytes = sum(size for _, size in base_files)
                if not external_artifacts_bytes:
                    raise ValueError("Base Engram shard metadata unavailable")
                external_artifacts = tuple(f"{base_model}/{name}" for name, _ in base_files)
            except Exception:
                external_artifacts_bytes = DEEPSEEK_V41_ENGRAM_BYTES
                external_artifacts = (
                    f"{base_model}/model-00047-of-00048.safetensors",
                    f"{base_model}/model-00048-of-00048.safetensors",
                )

        return RepositoryMetadata(
            repo_id=repo_id,
            revision=revision,
            commit=getattr(info, "sha", None),
            repository_bytes=repository_bytes,
            weight_bytes=weight_bytes,
            parameter_count=parameter_count,
            architecture=architecture,
            config=config,
            files=files,
            tags=tags,
            library_name=getattr(info, "library_name", None),
            pipeline_tag=getattr(info, "pipeline_tag", None),
            engram_bytes=engram_bytes,
            external_artifacts_bytes=external_artifacts_bytes,
            external_artifacts=external_artifacts,
        )

    def download(self, repo_id: str, revision: str, target: Path) -> None:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=target,
            max_workers=self.download_workers,
            token=self.token,
        )


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _kv_bytes_per_token(config: dict[str, Any], dtype_bytes: int = 2) -> tuple[float, bool]:
    model = _model_config(config)
    layers = int(model.get("num_hidden_layers") or model.get("n_layer") or 0)
    if not layers:
        return 0, False
    kv_lora_rank = int(model.get("kv_lora_rank") or 0)
    rope_dim = int(model.get("qk_rope_head_dim") or 0)
    if kv_lora_rank:
        return layers * (kv_lora_rank + rope_dim) * dtype_bytes, True

    attention_heads = int(model.get("num_attention_heads") or model.get("n_head") or 0)
    kv_heads = int(model.get("num_key_value_heads") or attention_heads)
    hidden_size = int(model.get("hidden_size") or model.get("n_embd") or 0)
    head_dim = int(
        model.get("head_dim") or (hidden_size // attention_heads if attention_heads else 0)
    )
    if not kv_heads or not head_dim:
        return 0, False
    return layers * 2 * kv_heads * head_dim * dtype_bytes, True


def _quantization(metadata: RepositoryMetadata) -> tuple[str, float | None]:
    repo = metadata.repo_id.lower()
    names = " ".join(name.lower() for name, _ in metadata.files)
    tags = " ".join(tag.lower() for tag in metadata.tags)
    combined = f"{repo} {names} {tags}"
    quant = metadata.config.get("quantization_config")
    if not isinstance(quant, dict):
        quant = _model_config(metadata.config).get("quantization_config")
    quant = quant if isinstance(quant, dict) else {}
    method = str(quant.get("quant_method") or quant.get("format") or "").lower()

    gguf_files = [name for name, _ in metadata.files if name.lower().endswith(".gguf")]
    if gguf_files and not any(
        name.lower().endswith((".safetensors", ".bin")) for name, _ in metadata.files
    ):
        matches = {
            int(value)
            for name in gguf_files
            for value in re.findall(r"[-_.]q([2-8])", name.lower())
        }
        if len(matches) == 1:
            bits = float(next(iter(matches)))
            return f"GGUF Q{int(bits)}", bits
        return (f"GGUF repository ({len(gguf_files)} variants)", None)
    if method == "exl3" or "exl3" in combined:
        bits = float(quant.get("avg_bits") or 0) or None
        if bits is None:
            match = re.search(r"(\d+(?:\.\d+)?)\s*bpw", combined)
            bits = float(match.group(1)) if match else None
        return (f"EXL3 {bits:g} bpw" if bits else "EXL3", bits)
    if "nvfp4" in combined or "nvfp4" in method:
        return "NVFP4", 4
    if any(token in combined for token in ("gptq", "awq")):
        bits = float(quant.get("bits") or 4)
        return ("GPTQ" if "gptq" in combined else "AWQ"), bits
    if "fp8" in combined or method == "fp8":
        return "FP8", 8
    if any(token in combined for token in ("int4", "4-bit", "4bit", "fp4")):
        return "4-bit quantized", 4
    if any(token in combined for token in ("int8", "8-bit", "8bit")):
        return "8-bit quantized", 8

    dtype = str(
        _model_config(metadata.config).get("dtype")
        or _model_config(metadata.config).get("torch_dtype")
        or metadata.config.get("dtype")
        or metadata.config.get("torch_dtype")
        or ""
    ).lower()
    if "float32" in dtype:
        return "FP32", 32
    if "float16" in dtype or "bfloat16" in dtype:
        return "BF16 / FP16", 16
    if metadata.parameter_count and metadata.weight_bytes:
        effective = metadata.weight_bytes * 8 / metadata.parameter_count
        if 1.5 <= effective <= 32:
            return f"Approximately {effective:.1f} bpw", round(effective, 2)
    return "Repository weights", None


def _runtime_weight_bytes(metadata: RepositoryMetadata) -> tuple[int, list[str]]:
    notes: list[str] = []
    groups: dict[str, int] = {}
    for name, size in metadata.files:
        suffix = Path(name).suffix.lower()
        if suffix in WEIGHT_SUFFIXES:
            groups[suffix] = groups.get(suffix, 0) + size
    if not groups:
        return metadata.weight_bytes, notes
    if len(groups) > 1:
        selected_suffix = next(
            suffix
            for suffix in (".safetensors", ".bin", ".gguf", ".pt", ".pth")
            if suffix in groups
        )
        selected_bytes = groups[selected_suffix]
        notes.append(
            "The repository contains multiple weight formats. Runtime memory prioritizes "
            f"the primary format ({selected_suffix}); the download still includes every file."
        )
        return selected_bytes, notes
    suffix, total = next(iter(groups.items()))
    if suffix == ".gguf":
        files = [(name, size) for name, size in metadata.files if name.lower().endswith(".gguf")]
        unsharded = [item for item in files if not re.search(r"-\d{5}-of-\d{5}\.gguf$", item[0])]
        if len(unsharded) > 1:
            selected = min(unsharded, key=lambda item: item[1])
            notes.append(
                "This GGUF repository contains several quantization variants. Runtime memory uses "
                f"the smallest file ({selected[0]}); downloading currently retrieves all variants."
            )
            return selected[1], notes
    return total, notes


def _node_requirement(
    capacities: list[float],
    nodes: int,
    resident_weights: float,
    resident_kv: float,
    kv_heads: int = 0,
    minimum_overhead: float = 8.0,
) -> tuple[bool, float, float, float]:
    weight_per_node = resident_weights / nodes
    kv_per_node = (
        resident_kv * math.ceil(kv_heads / nodes) / kv_heads if kv_heads else resident_kv / nodes
    )
    overhead_per_node = max(minimum_overhead, weight_per_node * 0.08)
    required_per_node = weight_per_node + kv_per_node + overhead_per_node
    selected = capacities[:nodes]
    fits = len(selected) == nodes and all(capacity >= required_per_node for capacity in selected)
    total = resident_weights + kv_per_node * nodes + overhead_per_node * nodes
    headroom = sum(selected) - total
    return fits, overhead_per_node * nodes, total, headroom


def _minimum_nodes(
    capacities: list[float],
    resident_weights: float,
    resident_kv: float,
    kv_heads: int = 0,
    minimum_overhead: float = 8.0,
) -> tuple[int, float, float, float]:
    for nodes in range(1, len(capacities) + 1):
        fits, overhead, total, headroom = _node_requirement(
            capacities, nodes, resident_weights, resident_kv, kv_heads, minimum_overhead
        )
        if fits:
            return nodes, overhead, total, headroom
    _, overhead, total, headroom = _node_requirement(
        capacities, len(capacities), resident_weights, resident_kv, kv_heads, minimum_overhead
    )
    return len(capacities) + 1, overhead, total, headroom


def build_fit_plan(
    request: ModelFitRequest,
    metadata: RepositoryMetadata,
    snapshot: ClusterSnapshot | None,
    settings: Settings,
    cached_repository_bytes: int = 0,
) -> ModelFitPlan:
    available_nodes = (
        [node for node in snapshot.nodes if node.status != "offline"] if snapshot else []
    )
    if available_nodes:
        node_capacities = sorted(
            (
                node.memory.total_gib * settings.model_usable_memory_fraction
                for node in available_nodes
            ),
            reverse=True,
        )
    else:
        node_capacities = [
            settings.fallback_node_memory_gib * settings.model_usable_memory_fraction
        ] * settings.fallback_node_count

    runtime_weight_bytes, notes = _runtime_weight_bytes(metadata)
    if not available_nodes:
        notes.append("Capacity uses configured fallback nodes, not measured available hardware.")
    notes.append(
        "Memory is a capacity estimate with a runtime reserve, not a speed guarantee. "
        "On unified-memory systems CPU offload shares the same physical RAM; "
        "it is not extra capacity."
    )
    detected_format, detected_bits = _quantization(metadata)
    if runtime_weight_bytes:
        repository_weights_gib = runtime_weight_bytes / GIB
        weight_confidence = "repository"
    elif metadata.parameter_count:
        repository_weights_gib = (
            metadata.parameter_count * (detected_bits or request.bits_per_weight) / 8 / GIB
        )
        weight_confidence = "parameters"
    else:
        repository_weights_gib = metadata.weight_bytes / GIB
        weight_confidence = "repository"

    model_config = _model_config(metadata.config)
    model_type = str(model_config.get("model_type") or metadata.config.get("model_type") or "")
    repo_lower = metadata.repo_id.lower()
    tags = {tag.lower() for tag in metadata.tags}
    is_deepseek_v41 = model_type == "deepseek_v41" or "deepseek-v4.1" in repo_lower
    is_glm53 = model_type == "glm5_next" or "glm-5.3" in repo_lower
    is_moe = bool(
        model_config.get("n_routed_experts") or model_config.get("num_experts") or "moe" in tags
    )
    library = str(metadata.library_name or "").lower()
    pipeline = str(metadata.pipeline_tag or "").lower()

    kv_dtype = request.kv_cache_dtype
    if kv_dtype == "auto":
        kv_dtype = "fp8_e4m3" if is_glm53 else "bf16"
    kv_dtype_bytes = 1 if kv_dtype == "fp8_e4m3" else 2
    if metadata.kv_bytes_per_token_bf16 is not None:
        bytes_per_token = metadata.kv_bytes_per_token_bf16 * kv_dtype_bytes / 2
        architecture_known = True
    else:
        bytes_per_token, architecture_known = _kv_bytes_per_token(metadata.config, kv_dtype_bytes)
    if architecture_known:
        kv_cache_gib = bytes_per_token * request.context_length * request.concurrent_sequences / GIB
        confidence = "architecture"
    else:
        kv_cache_gib = 250_000 * request.context_length * request.concurrent_sequences / GIB
        confidence = weight_confidence
        notes.append(
            "KV memory uses a conservative fallback because architecture metadata is incomplete."
        )

    kv_heads = (
        1
        if model_config.get("kv_lora_rank")
        else int(
            model_config.get("num_key_value_heads") or model_config.get("num_attention_heads") or 0
        )
    )
    notes.append(
        "KV uses full-context attention as an upper bound, with GQA/MQA head replication "
        "and replicated MLA latent cache under tensor parallelism. Sliding-window, recurrent "
        "or compressed attention may use less with a verified engine. Pipeline parallelism "
        "can change the placement."
    )
    engram_bytes = metadata.engram_bytes or metadata.external_artifacts_bytes
    engram_gib = engram_bytes / GIB
    complete_weights_gib = repository_weights_gib + metadata.external_artifacts_bytes / GIB
    # Discrete GPUs need CUDA/engine workspace, without the Spark host-memory reserve.
    minimum_overhead = 2.0 if settings.active_hardware == "nvidia-gpu" else 8.0
    full_minimum, full_overhead, full_total, full_headroom = _minimum_nodes(
        node_capacities, complete_weights_gib, kv_cache_gib, kv_heads, minimum_overhead
    )
    memory_fits = full_minimum <= len(node_capacities)

    placement = "full-memory"
    runtime_fits = memory_fits
    launch_supported = True
    engine = "vLLM V1"
    resident_weights = complete_weights_gib
    offloaded_weights = 0.0
    resident_kv = kv_cache_gib
    offloaded_kv = 0.0
    ssd_per_node = 0.0
    minimum_nodes = full_minimum
    runtime_overhead = full_overhead
    total_required = full_total
    headroom = full_headroom
    performance_impact = "none"
    mia_recipe = None
    requirements: list[str] = []
    tradeoffs: list[str] = []

    unsupported_library = library in {"mlx", "transformers.js", "litert-lm", "diffusers"}
    unsupported_pipeline = bool(pipeline) and pipeline not in {
        "text-generation",
        "image-text-to-text",
        "conversational",
    }
    if detected_format.startswith("GGUF"):
        launch_supported = False
        engine = "llama.cpp with GGUF mmap"
        if not memory_fits:
            placement = "mmap"
            runtime_fits = False
            performance_impact = "high"
            tradeoffs.append(
                "SSD-backed mmap can start a model larger than memory, but sustained page "
                "faults can severely reduce decode speed. Treat this as compatibility mode, "
                "not a fast path."
            )
        requirements.append("Select one GGUF quantization file and use a llama.cpp runtime.")
    elif unsupported_library or unsupported_pipeline:
        launch_supported = False
        engine = f"A runtime compatible with {metadata.library_name or metadata.pipeline_tag}"
        placement = "unsupported" if not memory_fits else "full-memory"
        requirements.append(
            "The current vLLM launcher does not support this repository's library or task."
        )

    if is_deepseek_v41 and "flash-1017" in repo_lower and engram_bytes:
        resident_weights = max(0.0, complete_weights_gib - engram_gib)
        hybrid_minimum, hybrid_overhead, hybrid_total, hybrid_headroom = _minimum_nodes(
            node_capacities, resident_weights, kv_cache_gib, kv_heads, minimum_overhead
        )
        placement = "engram-nvme"
        runtime_fits = hybrid_minimum <= len(node_capacities)
        launch_supported = False
        engine = "TensorFold DeepSeek V4.1 lane with disk-backed Engram"
        offloaded_weights = engram_gib
        resident_kv = kv_cache_gib
        minimum_nodes = hybrid_minimum
        runtime_overhead = hybrid_overhead
        total_required = hybrid_total
        headroom = hybrid_headroom
        selected_nodes = min(hybrid_minimum, len(node_capacities))
        ssd_per_node = engram_gib / max(1, selected_nodes)
        performance_impact = "low"
        requirements.extend(
            [
                "Place each rank's sparse Engram rows on node-local NVMe, not on "
                "management Ethernet.",
                "Use the pinned TensorFold/EXL3 or model-specific vLLM lane and verify "
                "the Engram row index.",
            ]
        )
        if metadata.external_artifacts:
            requirements.append(
                "Fetch the two Engram shards from the base DeepSeek V4.1 checkpoint in addition "
                "to this repository."
            )
        tradeoffs.append(
            "Sparse Engram lookup can avoid full table residency with limited overhead in the "
            "referenced recipe. Throughput depends on local NVMe latency, layout and engine. "
            "Preparation and cold reads add latency. Node SSD capacity is not measured here."
        )
        notes.append(
            f"Engram placement removes about {engram_gib:.1f} GiB from resident model memory."
        )

    if metadata.repo_id == MIA_MODEL_ID:
        launch_supported = False
        requirements.append(MANAGED_LAUNCH_BLOCKER)
        engine = "MIA patched SGLang with DSpark and NVMe Engram"
        checkpoint_commit = metadata.commit or metadata.revision
        if (
            is_mia_checkpoint(metadata.repo_id, checkpoint_commit)
            and engram_bytes
            and settings.active_hardware == "dgx-spark"
        ):
            physical = (
                sorted((node.memory.total_gib for node in available_nodes), reverse=True)
                if available_nodes
                else [settings.fallback_node_memory_gib] * len(node_capacities)
            )
            # The static pool is a recipe-specific budget, not the generic vLLM reserve.
            # KV is compressed and replicated per TP rank. The pin is allocated even
            # for a short request, so estimate the complete pool, not only active tokens.
            candidates = {}
            for candidate in (3, 4):
                recipe = build_mia_recipe(
                    candidate,
                    request.context_length,
                    request.concurrent_sequences,
                    revision=checkpoint_commit,
                )
                pool = int(recipe.environment["MAX_TOTAL_TOKENS"])
                supported = (
                    request.context_length <= recipe.max_context_length
                    and request.concurrent_sequences
                    <= int(recipe.environment["MAX_RUNNING_REQUESTS"])
                    and request.context_length * request.concurrent_sequences <= pool
                )
                capacities = [
                    value * float(recipe.environment["MEM_FRACTION_STATIC"]) for value in physical
                ]
                per_rank_kv = pool * KV_BYTES_PER_TOKEN_PER_RANK / GIB
                candidate_fits = (
                    supported
                    and _node_requirement(
                        capacities,
                        candidate,
                        max(RESIDENT_WEIGHTS_GIB, resident_weights),
                        per_rank_kv * candidate,
                    )[0]
                )
                candidates[candidate] = (supported, candidate_fits)
            minimum_nodes = next(
                (candidate for candidate in (3, 4) if candidates[candidate][1]),
                3 if candidates[3][0] else 4,
            )
            selected = next(
                (candidate for candidate in (4, 3) if candidates[candidate][1]),
                4 if len(physical) >= 4 or not candidates[3][0] else 3,
            )
            mia_recipe = build_mia_recipe(
                selected,
                request.context_length,
                request.concurrent_sequences,
                revision=checkpoint_commit,
            )
            pool = int(mia_recipe.environment["MAX_TOTAL_TOKENS"])
            node_capacities = [
                value * float(mia_recipe.environment["MEM_FRACTION_STATIC"]) for value in physical
            ]
            resident_weights = max(RESIDENT_WEIGHTS_GIB, resident_weights)
            kv_cache_gib = pool * KV_BYTES_PER_TOKEN_PER_RANK / GIB * selected
            resident_kv = kv_cache_gib
            runtime_fits, runtime_overhead, total_required, headroom = _node_requirement(
                node_capacities, selected, resident_weights, resident_kv
            )
            runtime_fits = runtime_fits and (
                request.context_length <= mia_recipe.max_context_length
                and request.concurrent_sequences
                <= int(mia_recipe.environment["MAX_RUNNING_REQUESTS"])
                and request.context_length * request.concurrent_sequences <= pool
            )
            minimum_nodes = max(minimum_nodes, 3)
            ssd_per_node = max(engram_gib / selected, 48 if selected == 4 else 63)
            placement = "engram-nvme"
            offloaded_weights = engram_gib
            detected_format = "MXFP4 experts / FP8 dense (MIA recipe)"
            detected_bits = None  # Mixed native precision cannot be summarized as uniform FP8.
            confidence = "architecture"
            requirements = [item for item in requirements if "TensorFold" not in item]
            requirements.extend(mia_recipe.prerequisites)
            requirements.extend(mia_recipe.blockers[1:])
            notes.extend(
                [
                    f"MIA TP{selected} capacity uses the reviewed static fraction "
                    f"{mia_recipe.environment['MEM_FRACTION_STATIC']}, a ~305 GiB expanded "
                    "resident footprint (including padding/draft costs), and the complete "
                    f"{pool:,}-token compressed KV pool at 1670.75 bytes/token/rank. "
                    "These are recipe estimates, not local runtime measurements.",
                    "Checkpoint disk size minus Engram understates runtime residency. "
                    "The packed rank rows also occupy NVMe alongside the original checkpoint.",
                ]
            )
            if request.kv_cache_dtype != "auto":
                notes.append(
                    "The reviewed recipe uses its native compressed KV layout; generic KV "
                    "precision does not rescale it."
                )
            if selected == 3 and request.context_length > 32_768:
                tradeoffs.append(
                    "TP3 long prompts need head-memory stress testing; 32768 is the "
                    "unattended default."
                )
            kv_dtype = "native-compressed"
        else:
            requirements.append(
                "Resolve the reviewed MIA checkpoint revision and Engram shards on DGX Spark "
                "before applying its compressed-KV and resident-memory estimates."
            )

    if not runtime_fits and launch_supported and placement == "full-memory" and kv_cache_gib:
        weights_fit = _minimum_nodes(node_capacities, complete_weights_gib, 0)[0]
        if weights_fit <= len(node_capacities):
            placement = "kv-offload"
            engine = "vLLM with a smaller active context; LMCache for prefix reuse"
            performance_impact = "unknown"
            requirements.append(
                "Reduce active context or concurrency, or validate a runtime with active KV "
                "offload. LMCache prefix storage alone does not shrink active decode KV."
            )
            tradeoffs.append(
                "Cold prefixes can be stored on CPU/NVMe and restored for reuse. All KV needed "
                "by active full-attention sequences must still be accessible during decoding."
            )

    if is_moe and not runtime_fits and placement in {"full-memory", "kv-offload"}:
        placement = "expert-offload"
        engine = "Model-specific expert-offload runtime"
        experts = int(model_config.get("n_routed_experts") or model_config.get("num_experts") or 0)
        active = int(model_config.get("num_experts_per_tok") or 0)
        ratio = active / experts if experts and active else 0
        detail = f" Only about {ratio:.1%} of experts are active per token." if ratio else ""
        notes.append(
            "This is a mixture-of-experts model."
            + detail
            + " Inactive experts still consume memory unless the selected engine explicitly "
            "offloads them."
        )
        requirements.append(
            "Choose and benchmark an engine with expert offload or expert parallelism for "
            "this exact architecture."
        )
        tradeoffs.append(
            "CPU or SSD expert fetches can reduce resident memory but may become the "
            "decode bottleneck."
        )
        performance_impact = "unknown"

    settings.model_cache_path.mkdir(parents=True, exist_ok=True)
    storage_available_gib = (
        shutil.disk_usage(settings.model_cache_path).free
        + min(cached_repository_bytes, metadata.repository_bytes)
    ) / GIB
    repository_storage_gib = metadata.repository_bytes / GIB
    runtime_storage_required_gib = (
        metadata.repository_bytes + metadata.external_artifacts_bytes
    ) / GIB + (ssd_per_node if mia_recipe else 0)
    storage_fits = repository_storage_gib <= storage_available_gib
    runtime_storage_fits = runtime_storage_required_gib <= storage_available_gib
    download_fits = storage_fits

    if not storage_fits:
        notes.append("The selected repository is larger than free controller model storage.")
    elif not runtime_storage_fits:
        notes.append(
            "The repository fits, but the complete runnable artifact set needs more storage."
        )
    if not memory_fits and placement == "full-memory":
        notes.append("The complete model and KV cache exceed the in-memory cluster envelope.")
    if detected_bits and abs(detected_bits - request.bits_per_weight) >= 1:
        notes.append(
            f"The repository appears to be {detected_format}; selecting "
            f"{request.bits_per_weight:g}-bit does not convert the downloaded weights."
        )

    if not available_nodes and snapshot and snapshot.nodes:
        runtime_fits = False
        requirements.append(
            "Bring cluster nodes online before launching; all reported nodes are offline."
        )
    if not memory_fits and kv_cache_gib and complete_weights_gib:
        low, high = 0, request.context_length
        while low < high:
            middle = (low + high + 1) // 2
            required = _minimum_nodes(
                node_capacities,
                complete_weights_gib,
                kv_cache_gib * middle / request.context_length,
                kv_heads,
            )[0]
            if required <= len(node_capacities):
                low = middle
            else:
                high = middle - 1
        if low:
            notes.append(
                f"Alternative full-memory plan: at most about {low:,} tokens per sequence "
                f"at concurrency {request.concurrent_sequences}, at this KV precision."
            )
    if not runtime_weight_bytes and not metadata.parameter_count:
        runtime_fits = False
        requirements.append("Weight sizes and parameter count are unknown; inspect the checkpoint.")
    if (
        detected_format.startswith("NVFP4")
        and settings.active_compute_capability is not None
        and settings.active_compute_capability < 10
    ):
        runtime_fits = False
        launch_supported = False
        requirements.append(
            "NVFP4 requires NVIDIA Blackwell-class kernels; the active cluster reports "
            f"compute capability {settings.active_compute_capability:g}. Choose an AWQ, GPTQ, "
            "FP16/BF16 or other format supported by this GPU generation."
        )
        notes.append("The checkpoint may be downloaded, but it cannot launch on the active GPUs.")
    if detected_format.startswith("EXL3"):
        launch_supported = False
        requirements.append(
            "EXL3 requires its compatible engine; the standard vLLM launcher cannot load it."
        )
    if not download_fits:
        fit_status = "insufficient-storage"
    elif runtime_fits and launch_supported and runtime_storage_fits:
        fit_status = "full-memory" if placement == "full-memory" else "hybrid-ready"
    elif runtime_fits and not launch_supported:
        fit_status = "engine-required"
    else:
        fit_status = "download-only"
    fits = runtime_fits and launch_supported and runtime_storage_fits

    return ModelFitPlan(
        repo_id=request.repo_id,
        revision=request.revision,
        checkpoint_commit=metadata.commit
        or (metadata.revision if re.fullmatch(r"[0-9a-f]{40}", metadata.revision) else None),
        recipe_id=mia_recipe.id if mia_recipe else None,
        recipe_source_url=mia_recipe.recipe_url if mia_recipe else None,
        architecture=metadata.architecture,
        parameter_count=metadata.parameter_count,
        repository_bytes=metadata.repository_bytes,
        weight_bytes=metadata.weight_bytes,
        weights_gib=round(complete_weights_gib, 2),
        kv_cache_gib=round(kv_cache_gib, 2),
        runtime_overhead_gib=round(runtime_overhead, 2),
        total_required_gib=round(total_required, 2),
        cluster_nodes=len(node_capacities),
        usable_per_node_gib=round(sum(node_capacities) / len(node_capacities), 2),
        usable_cluster_gib=round(sum(node_capacities), 2),
        minimum_nodes=minimum_nodes,
        headroom_gib=round(headroom, 2),
        memory_fits=memory_fits,
        storage_available_gib=round(storage_available_gib, 2),
        storage_fits=storage_fits,
        fits=fits,
        confidence=confidence,
        notes=notes,
        bits_per_weight=request.bits_per_weight,
        context_length=request.context_length,
        concurrent_sequences=request.concurrent_sequences,
        kv_cache_dtype=kv_dtype,
        runtime_fits=runtime_fits,
        download_fits=download_fits,
        launch_supported=launch_supported,
        runtime_storage_fits=runtime_storage_fits,
        fit_status=fit_status,
        placement_strategy=placement,
        engine_recommendation=engine,
        detected_format=detected_format,
        detected_bits_per_weight=detected_bits,
        library_name=metadata.library_name,
        pipeline_tag=metadata.pipeline_tag,
        resident_weights_gib=round(resident_weights, 2),
        offloaded_weights_gib=round(offloaded_weights, 2),
        resident_kv_cache_gib=round(resident_kv, 2),
        offloaded_kv_cache_gib=round(offloaded_kv, 2),
        ssd_required_gib_per_node=round(ssd_per_node, 2),
        additional_artifacts_bytes=metadata.external_artifacts_bytes,
        runtime_storage_required_gib=round(runtime_storage_required_gib, 2),
        performance_impact=performance_impact,
        requirements=requirements,
        tradeoffs=tradeoffs,
    )


class ModelManager:
    def __init__(
        self,
        settings: Settings,
        store: SnapshotStore,
        runtime: Any,
        relay: ModelRelay | None = None,
    ):
        self.settings = settings
        self.store = store
        self.runtime = runtime
        self.catalog = HuggingFaceCatalog(settings.hf_token, settings.hf_download_workers)
        self.relay = relay
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker: asyncio.Task | None = None
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._inventory_cache: list[LocalModel] | None = None
        self._local_fit_cache: dict[str, tuple[datetime, ModelFitPlan, int]] = {}
        self._inventory_cached_at = 0.0
        self._inventory_guard = threading.Lock()
        self._inventory_generation = 0
        self._inventory_refresh_thread: threading.Thread | None = None
        self._runtime_cache: RuntimeEndpoint | None = None
        self._runtime_cached_at = 0.0
        self._runtime_lock = asyncio.Lock()
        self.settings.model_cache_path.mkdir(parents=True, exist_ok=True)
        (self.settings.model_cache_path / ".partial").mkdir(exist_ok=True)

    async def start(self) -> None:
        for job in await asyncio.to_thread(self.store.recover_downloads):
            await self.queue.put(job.id)
        self.worker = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self.worker:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass

    async def fit(self, request: ModelFitRequest) -> ModelFitPlan:
        metadata = await asyncio.to_thread(self.catalog.inspect, request.repo_id, request.revision)
        return await asyncio.to_thread(
            build_fit_plan, request, metadata, self.runtime.latest, self.settings
        )

    async def fit_for_local(self, request: ModelFitRequest) -> ModelFitPlan:
        parts = safe_repo_parts(request.repo_id)
        model_path = self.settings.model_cache_path.joinpath(*parts)
        with self._inventory_guard:
            cached = self._local_fit_cache.get(request.repo_id)
        if cached:
            _, previous, cached_bytes = cached
        else:
            downloads = await asyncio.to_thread(self.store.list_downloads)
            self._remember_download_fits(downloads)
            with self._inventory_guard:
                cached = self._local_fit_cache.get(request.repo_id)
            if cached:
                _, previous, cached_bytes = cached
            else:
                manifest_path = model_path / ".spark-cluster.json"
                try:
                    manifest = await asyncio.to_thread(
                        lambda: json.loads(manifest_path.read_text())
                    )
                    previous = ModelFitPlan.model_validate(manifest["fit"])
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    return await self.fit(request)
                cached_bytes = previous.repository_bytes

        if request.repo_id == MIA_MODEL_ID and request.revision not in {
            "main",
            previous.revision,
            previous.checkpoint_commit,
        }:
            return await self.fit(request)
        previous_sequences = max(1, previous.concurrent_sequences)
        previous_context = max(1, previous.context_length)
        kv_bytes_per_token = previous.kv_cache_gib * GIB / (previous_context * previous_sequences)
        if previous.kv_cache_dtype == "fp8_e4m3":
            kv_bytes_per_token *= 2
        metadata = RepositoryMetadata(
            repo_id=request.repo_id,
            revision=request.revision,
            commit=previous.checkpoint_commit,
            repository_bytes=previous.repository_bytes,
            weight_bytes=previous.weight_bytes,
            parameter_count=previous.parameter_count,
            architecture=previous.architecture,
            config={},
            library_name=previous.library_name,
            pipeline_tag=previous.pipeline_tag,
            engram_bytes=(
                int(previous.offloaded_weights_gib * GIB)
                if not previous.additional_artifacts_bytes
                else 0
            ),
            external_artifacts_bytes=previous.additional_artifacts_bytes,
            external_artifacts=("base Engram shards (see original repository metadata)",)
            if previous.additional_artifacts_bytes
            else (),
            kv_bytes_per_token_bf16=kv_bytes_per_token,
        )
        return await asyncio.to_thread(
            build_fit_plan,
            request,
            metadata,
            self.runtime.latest,
            self.settings,
            cached_bytes,
        )

    async def managed_model(self, model_id: str) -> LocalModel | None:
        for model in await self.local_models():
            if model.id == model_id and model.source == "managed":
                return model
        model_path = self.settings.model_cache_path.joinpath(*safe_repo_parts(model_id))
        manifest_path = model_path / ".spark-cluster.json"
        try:
            manifest = await asyncio.to_thread(lambda: json.loads(manifest_path.read_text()))
            fit = ModelFitPlan.model_validate(manifest["fit"])
            return LocalModel(
                id=model_id,
                revision=manifest.get("revision"),
                source="managed",
                path=str(model_path),
                size_bytes=fit.repository_bytes,
                downloaded_at=datetime.fromisoformat(manifest["downloaded_at"]),
                launch_supported=fit.launch_supported,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    async def enqueue(
        self,
        request: ModelDownloadRequest,
        *,
        destination: str = "local",
        controller_url: str | None = None,
        fit_plan: ModelFitPlan | None = None,
        target_cluster_id: str | None = None,
        target_cluster_name: str | None = None,
        target_cluster_url: str | None = None,
    ) -> DownloadJob:
        fit = fit_plan or await self.fit(request)
        if not fit.download_fits:
            raise ModelDoesNotFit(fit)
        parts = safe_repo_parts(request.repo_id)
        target = self.settings.model_cache_path.joinpath(*parts)
        local_exists = (target / ".spark-cluster.json").exists()
        if local_exists and destination == "local":
            raise ModelAlreadyInstalled(request.repo_id)
        active = [
            job
            for job in await asyncio.to_thread(self.store.list_downloads)
            if job.repo_id == request.repo_id
            and job.revision == request.revision
            and job.target_cluster_id == target_cluster_id
            and job.status
            in {
                "queued",
                "resolving",
                "downloading",
                "paused",
                "preparing-transfer",
                "distributing",
            }
        ]
        if active:
            return active[0]

        job_id = uuid4().hex
        job = DownloadJob(
            id=job_id,
            repo_id=request.repo_id,
            revision=request.revision,
            total_bytes=fit.repository_bytes,
            target_path=str(target),
            staging_path=str(self.settings.model_cache_path / ".partial" / job_id),
            destination=destination,
            controller_url=controller_url,
            target_cluster_id=target_cluster_id,
            target_cluster_name=target_cluster_name,
            target_cluster_url=target_cluster_url,
            message=(
                "Queued for controller download and LAN delivery"
                if destination == "cluster"
                else "Queued for download"
            ),
            fit=fit,
        )
        await asyncio.to_thread(self.store.save_download, job)
        await self.queue.put(job.id)
        return job

    async def enqueue_from_controller(self, request: ControllerModelImportRequest) -> DownloadJob:
        existing = await asyncio.to_thread(self.store.get_download, request.job_id)
        if existing:
            return existing
        parts = safe_repo_parts(request.repo_id)
        target = self.settings.model_cache_path.joinpath(*parts)
        available = shutil.disk_usage(self.settings.model_cache_path).free
        if request.total_bytes > available:
            raise ValueError(
                f"Shared cluster model storage needs {request.total_bytes} bytes but only "
                f"{available} bytes are available"
            )
        job = DownloadJob(
            id=request.job_id,
            repo_id=request.repo_id,
            revision=request.revision,
            total_bytes=request.total_bytes,
            target_path=str(target),
            staging_path=str(self.settings.model_cache_path / ".partial" / request.job_id),
            source="controller",
            controller_url=request.controller_url,
            controller_job_id=request.job_id,
            message="Queued for LAN copy from the controller",
            fit=request.fit,
        )
        if (target / ".spark-cluster.json").exists():
            job.status = "completed"
            job.downloaded_bytes = directory_size(target)
            job.progress = 100
            job.message = "Model already exists in shared cluster storage"
        await asyncio.to_thread(self.store.save_download, job)
        if job.status != "completed":
            await self.queue.put(job.id)
        return job

    async def retry(self, job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(self.store.get_download, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "failed":
            return job
        job.status = "queued"
        job.error = None
        job.message = "Retry queued"
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)
        await self.queue.put(job.id)
        return job

    async def pause(self, job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(self.store.get_download, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == "paused":
            return job
        if job.status not in {
            "queued",
            "resolving",
            "downloading",
            "preparing-transfer",
            "distributing",
        }:
            raise ValueError(f"A {job.status} download cannot be paused")
        previous_status = job.status
        job.status = "paused"
        job.message = "Paused; downloaded files are preserved"
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)

        process = self._active_processes.get(job_id)
        if process and process.returncode is None:
            process.send_signal(signal.SIGSTOP)
        elif previous_status in {"preparing-transfer", "distributing"} and self.relay:
            await self.relay.pause_remote(job)
        else:
            task = self._active_tasks.get(job_id)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        return job

    async def resume(self, job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(self.store.get_download, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "paused":
            raise ValueError(f"A {job.status} download cannot be resumed")

        process = self._active_processes.get(job_id)
        if process and process.returncode is None:
            process.send_signal(signal.SIGCONT)
            job.status = "downloading"
            job.message = "Resuming download from preserved partial files"
        elif job.destination == "cluster" and Path(job.target_path).exists() and self.relay:
            remote_resumed = await self.relay.resume_remote(job)
            if remote_resumed:
                job.status = "distributing"
                job.message = "Resuming LAN delivery to the cluster"
            else:
                job.status = "queued"
                job.message = "Resume queued"
                await self.queue.put(job.id)
        else:
            job.status = "queued"
            job.message = "Resume queued from preserved partial files"
            await self.queue.put(job.id)
        job.error = None
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)
        return job

    async def cancel(self, job_id: str) -> DownloadJob:
        job = await asyncio.to_thread(self.store.get_download, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == "cancelled":
            return job
        if job.status in {"completed", "deleted"}:
            raise ValueError("Completed models must be deleted from the model library")

        previous_status = job.status
        job.status = "cancelled"
        job.message = "Cancelled; partial files were removed from controller and cluster storage"
        job.error = None
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)

        if previous_status in {"preparing-transfer", "distributing", "paused"} and self.relay:
            await self.relay.cancel_remote(job)
        process = self._active_processes.get(job_id)
        if process and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGCONT)
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
        task = self._active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await asyncio.to_thread(self._remove_job_files, job)
        if self.relay:
            await asyncio.to_thread(self.relay.remove_manifest, job.id)
        self._invalidate_inventory()
        return job

    async def delete_model(self, repo_id: str) -> ModelMutationResult:
        safe_repo_parts(repo_id)
        endpoint = await self.runtime_endpoint()
        if repo_id in endpoint.loaded_model_ids:
            raise ModelDeletionBlocked("Stop the running model before deleting its files")

        cluster_result = None
        if self.settings.remote_cluster_url and self.relay:
            cluster_result = await self.relay.delete_remote_model(repo_id)

        target = self.settings.model_cache_path.joinpath(*safe_repo_parts(repo_id))
        controller_deleted = (target / ".spark-cluster.json").is_file()
        freed_bytes = await asyncio.to_thread(directory_size, target) if controller_deleted else 0
        if controller_deleted:
            await asyncio.to_thread(shutil.rmtree, target)
            if target.parent != self.settings.model_cache_path:
                with contextlib.suppress(OSError):
                    target.parent.rmdir()
        cluster_deleted = bool(cluster_result and cluster_result.cluster_deleted)
        freed_bytes += cluster_result.freed_bytes if cluster_result else 0
        if not controller_deleted and not cluster_deleted:
            raise KeyError(repo_id)

        for job in await asyncio.to_thread(self.store.list_downloads):
            if job.repo_id == repo_id and job.status == "completed":
                job.status = "deleted"
                job.message = "Model files deleted from controller and cluster storage"
                job.updated_at = datetime.now(UTC)
                await asyncio.to_thread(self.store.save_download, job)
                if self.relay:
                    await asyncio.to_thread(self.relay.remove_manifest, job.id)
        with self._inventory_guard:
            self._local_fit_cache.pop(repo_id, None)
        self._invalidate_inventory()
        return ModelMutationResult(
            repo_id=repo_id,
            status="deleted",
            controller_deleted=controller_deleted,
            cluster_deleted=cluster_deleted or (not self.settings.remote_cluster_url),
            freed_bytes=freed_bytes,
            message="Model removed from the controller and all attached cluster storage",
        )

    async def refresh_huggingface_token(self, token: str) -> list[str]:
        """Replace credentials and restart HF workers without removing partial files."""
        self.settings.hf_token = token
        self.catalog.token = token
        if isinstance(self.catalog, HuggingFaceCatalog):
            self.catalog.api.token = token
        restarted = []
        for job in await self.downloads():
            if job.source != "huggingface" or job.status not in {
                "resolving",
                "downloading",
                "paused",
            }:
                continue
            was_paused = job.status == "paused"
            if not was_paused:
                await self.pause(job.id)
            # SIGCONT alone would keep the old process environment and old token.
            task = self._active_tasks.get(job.id)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if not was_paused:
                await self.resume(job.id)
                restarted.append(job.id)
        return restarted

    async def downloads(self) -> list[DownloadJob]:
        return await asyncio.to_thread(self.store.list_downloads)

    @staticmethod
    def _remove_job_files(job: DownloadJob) -> None:
        for path in (Path(job.staging_path), Path(job.target_path)):
            if path.exists():
                shutil.rmtree(path)

    async def runtime_endpoint(self) -> RuntimeEndpoint:
        if self._runtime_cache and time.monotonic() - self._runtime_cached_at < 1:
            return self._runtime_cache
        async with self._runtime_lock:
            if self._runtime_cache and time.monotonic() - self._runtime_cached_at < 1:
                return self._runtime_cache
            base = self.settings.direct_runtime_base_url
            models_url = f"{base}/models"
            loaded: list[str] = []
            status = "unreachable"
            try:
                headers = (
                    {"Authorization": f"Bearer {self.settings.openai_api_key}"}
                    if self.settings.openai_api_key
                    else {}
                )
                async with httpx.AsyncClient(timeout=2, headers=headers) as client:
                    response = await client.get(models_url)
                    response.raise_for_status()
                    loaded = [str(item["id"]) for item in response.json().get("data", [])]
                    status = "ready"
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                pass
            endpoint = RuntimeEndpoint(
                base_url=base,
                models_url=models_url,
                chat_completions_url=f"{base}/chat/completions",
                status=status,
                loaded_model_ids=loaded,
            )
            self._runtime_cache = endpoint
            self._runtime_cached_at = time.monotonic()
            return endpoint

    async def local_models(self) -> list[LocalModel]:
        endpoint = await self.runtime_endpoint()
        loaded = set(endpoint.loaded_model_ids)
        seeded = False
        with self._inventory_guard:
            needs_seed = self._inventory_cache is None
        if needs_seed:
            downloads = await asyncio.to_thread(self.store.list_downloads)
            self._remember_download_fits(downloads)
            seed = self._seed_inventory(downloads)
            with self._inventory_guard:
                if self._inventory_cache is None:
                    self._inventory_cache = seed
                    self._inventory_cached_at = time.monotonic()
                    seeded = True
        self._start_inventory_refresh(force=seeded)
        with self._inventory_guard:
            inventory = list(self._inventory_cache or ())
        models = [model.model_copy(update={"loaded": model.id in loaded}) for model in inventory]
        known = {model.id for model in models}
        models.extend(
            LocalModel(id=model_id, source="runtime", loaded=True)
            for model_id in sorted(loaded - known)
        )
        return sorted(models, key=lambda model: (not model.loaded, model.id.lower()))

    @staticmethod
    def _seed_inventory(downloads: list[DownloadJob]) -> list[LocalModel]:
        models: dict[str, LocalModel] = {}
        for job in downloads:
            if job.status != "completed":
                continue
            models[job.repo_id] = LocalModel(
                id=job.repo_id,
                revision=job.revision,
                source="managed",
                path=job.target_path,
                size_bytes=job.total_bytes or job.downloaded_bytes,
                downloaded_at=job.updated_at,
                launch_supported=job.fit.launch_supported,
            )
        return list(models.values())

    def _remember_download_fits(self, downloads: list[DownloadJob]) -> None:
        latest: dict[str, DownloadJob] = {}
        for job in downloads:
            if job.status != "completed":
                continue
            current = latest.get(job.repo_id)
            if current is None or job.updated_at > current.updated_at:
                latest[job.repo_id] = job
        with self._inventory_guard:
            for repo_id, job in latest.items():
                cached = self._local_fit_cache.get(repo_id)
                if cached is None or job.updated_at >= cached[0]:
                    cached_bytes = (
                        job.total_bytes or job.downloaded_bytes or job.fit.repository_bytes
                    )
                    self._local_fit_cache[repo_id] = (job.updated_at, job.fit, cached_bytes)

    def _invalidate_inventory(self) -> None:
        with self._inventory_guard:
            self._inventory_cache = None
            self._inventory_cached_at = 0.0
            self._inventory_generation += 1
            # A blocked network-filesystem scan is daemonized and may finish later. Detach it so
            # its stale result cannot replace the inventory after a model mutation.
            self._inventory_refresh_thread = None

    def _start_inventory_refresh(self, force: bool = False) -> None:
        with self._inventory_guard:
            stale = time.monotonic() - self._inventory_cached_at > 10
            if not force and not stale:
                return
            if self._inventory_refresh_thread and self._inventory_refresh_thread.is_alive():
                return
            generation = self._inventory_generation
            thread = threading.Thread(
                target=self._refresh_inventory,
                args=(generation,),
                daemon=True,
                name="spark-model-inventory",
            )
            self._inventory_refresh_thread = thread
        thread.start()

    def _refresh_inventory(self, generation: int) -> None:
        try:
            models = self._scan_managed_models(set())
            known = {model.id for model in models}
            models.extend(self._scan_huggingface_cache(set(), known))
        except (OSError, RuntimeError, ValueError):
            models = None
        with self._inventory_guard:
            if generation != self._inventory_generation:
                return
            if models is not None:
                merged = {model.id: model for model in self._inventory_cache or ()}
                merged.update({model.id: model for model in models})
                self._inventory_cache = list(merged.values())
                self._inventory_cached_at = time.monotonic()
            if self._inventory_refresh_thread is threading.current_thread():
                self._inventory_refresh_thread = None

    def _scan_managed_models(self, loaded: set[str]) -> list[LocalModel]:
        models: list[LocalModel] = []
        root = self.settings.model_cache_path
        if not root.exists():
            return models
        manifest_paths: list[Path] = []
        for owner in root.iterdir():
            if not owner.is_dir() or owner.name.startswith("."):
                continue
            direct_manifest = owner / ".spark-cluster.json"
            if direct_manifest.is_file():
                manifest_paths.append(direct_manifest)
                continue
            try:
                manifest_paths.extend(
                    model / ".spark-cluster.json"
                    for model in owner.iterdir()
                    if model.is_dir() and (model / ".spark-cluster.json").is_file()
                )
            except OSError:
                continue
        for manifest_path in manifest_paths:
            try:
                manifest = json.loads(manifest_path.read_text())
                downloaded_at = datetime.fromisoformat(manifest["downloaded_at"])
                repo_id = str(manifest["repo_id"])
                models.append(
                    LocalModel(
                        id=repo_id,
                        revision=manifest.get("revision"),
                        source="managed",
                        path=str(manifest_path.parent),
                        size_bytes=directory_size(manifest_path.parent),
                        downloaded_at=downloaded_at,
                        loaded=repo_id in loaded,
                        launch_supported=(
                            manifest["fit"].get("launch_supported")
                            if isinstance(manifest.get("fit"), dict)
                            and isinstance(manifest["fit"].get("launch_supported"), bool)
                            else None
                        ),
                    )
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return models

    def _scan_huggingface_cache(self, loaded: set[str], known: set[str]) -> list[LocalModel]:
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = hf_home / "hub"
        models: list[LocalModel] = []
        if not hub.exists():
            return models
        for path in hub.glob("models--*"):
            repo_id = path.name.removeprefix("models--").replace("--", "/", 1)
            if repo_id in known:
                continue
            snapshots = path / "snapshots"
            weight_files = [
                file
                for file in snapshots.glob("**/*")
                if file.is_file() and file.name.lower().endswith(WEIGHT_SUFFIXES)
            ]
            if not weight_files:
                continue
            unique_weight_files = {file.resolve() for file in weight_files}
            models.append(
                LocalModel(
                    id=repo_id,
                    source="huggingface-cache",
                    path=str(path),
                    size_bytes=sum(file.stat().st_size for file in unique_weight_files),
                    loaded=repo_id in loaded,
                )
            )
        return models

    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            task = asyncio.create_task(self._run_job(job_id))
            self._active_tasks[job_id] = task
            try:
                await task
            except asyncio.CancelledError:
                job = await asyncio.to_thread(self.store.get_download, job_id)
                if not job or job.status not in {"paused", "cancelled"}:
                    raise
            except Exception as error:
                job = await asyncio.to_thread(self.store.get_download, job_id)
                if job and job.status not in {"paused", "cancelled"}:
                    job.status = "failed"
                    job.error = self.safe_error(error)
                    job.message = "Model transfer failed and can be retried"
                    job.updated_at = datetime.now(UTC)
                    await asyncio.to_thread(self.store.save_download, job)
            finally:
                self._active_tasks.pop(job_id, None)
                self.queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = await asyncio.to_thread(self.store.get_download, job_id)
        if job is None:
            return
        if job.status in {"paused", "cancelled", "completed", "deleted"}:
            return
        existing_target = Path(job.target_path)
        if (existing_target / ".spark-cluster.json").exists():
            await self._finish_or_distribute(job, existing_target)
            return
        job.status = "downloading"
        job.attempts += 1
        job.error = None
        job.message = (
            "Copying model files from the controller over the LAN"
            if job.source == "controller"
            else "Downloading model from Hugging Face to this controller"
        )
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)
        staging = Path(job.staging_path)
        staging.mkdir(parents=True, exist_ok=True)
        if job.source == "controller":
            if not self.relay:
                raise RuntimeError("Model relay is unavailable")
            await self.relay.pull(job)
            await self._install_staging(job, staging)
            return

        if isinstance(self.catalog, HuggingFaceCatalog):
            completed = await self._run_download_process(job, staging)
            if not completed:
                return
        else:
            completed = await self._run_inprocess_download(job, staging)
            if not completed:
                return

        try:
            manifest = {
                "repo_id": job.repo_id,
                "revision": job.revision,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "fit": job.fit.model_dump(mode="json"),
            }
            (staging / ".spark-cluster.json").write_text(json.dumps(manifest, indent=2))
            await self._install_staging(job, staging)
            return
        except (OSError, ValueError) as error:
            job.status = "failed"
            job.error = self.safe_error(error)
            job.message = "Controller download failed"
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)

    async def _run_download_process(self, job: DownloadJob, staging: Path) -> bool:
        environment = os.environ.copy()
        if self.catalog.token:
            environment["SPARK_CLUSTER_HF_TOKEN"] = self.catalog.token
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "spark_cluster.download_worker",
            job.repo_id,
            job.revision,
            str(staging),
            str(self.catalog.download_workers),
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        self._active_processes[job.id] = process
        try:
            while process.returncode is None:
                await asyncio.sleep(0.5)
                current = await asyncio.to_thread(self.store.get_download, job.id)
                if current is None or current.status == "cancelled":
                    return False
                if current.status == "paused":
                    continue
                job = current
                job.downloaded_bytes = min(
                    job.total_bytes,
                    await asyncio.to_thread(self._staging_progress, staging),
                )
                job.progress = (
                    round(job.downloaded_bytes / job.total_bytes * 100, 2) if job.total_bytes else 0
                )
                job.updated_at = datetime.now(UTC)
                await asyncio.to_thread(self.store.save_download, job)
            return_code = await process.wait()
            current = await asyncio.to_thread(self.store.get_download, job.id)
            if current and current.status in {"paused", "cancelled"}:
                return False
            if return_code:
                raise RuntimeError(f"Hugging Face download worker exited with code {return_code}")
            return True
        except asyncio.CancelledError:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.send_signal(signal.SIGCONT)
                    process.terminate()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=10)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            raise
        finally:
            self._active_processes.pop(job.id, None)

    async def _run_inprocess_download(self, job: DownloadJob, staging: Path) -> bool:
        outcome: dict[str, BaseException | None] = {"error": None}

        def download() -> None:
            try:
                self.catalog.download(job.repo_id, job.revision, staging)
            except BaseException as error:
                outcome["error"] = error

        thread = threading.Thread(target=download, daemon=True, name=f"hf-{job.id[:8]}")
        thread.start()
        while thread.is_alive():
            current = await asyncio.to_thread(self.store.get_download, job.id)
            if current and current.status in {"paused", "cancelled"}:
                return False
            job.downloaded_bytes = min(
                job.total_bytes,
                await asyncio.to_thread(self._staging_progress, staging),
            )
            job.progress = (
                round(job.downloaded_bytes / job.total_bytes * 100, 2) if job.total_bytes else 0
            )
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_download, job)
            await asyncio.sleep(0.5)
        thread.join()

        if outcome["error"] is not None:
            job.status = "failed"
            job.error = self.safe_error(outcome["error"])
            job.message = "Controller download failed"
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_download, job)
            return False
        return True

    async def _install_staging(self, job: DownloadJob, staging: Path) -> None:
        target = Path(job.target_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Target model directory already exists: {target}")
            staging.replace(target)
            self._invalidate_inventory()
            await self._finish_or_distribute(job, target)
        except (OSError, ValueError) as error:
            job.status = "failed"
            job.error = self.safe_error(error)
            job.message = "Model installation failed"
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_download, job)

    async def _finish_or_distribute(self, job: DownloadJob, target: Path) -> None:
        job.downloaded_bytes = job.total_bytes or directory_size(target)
        job.progress = 100
        job.error = None
        if job.destination == "cluster":
            if not self.relay:
                raise RuntimeError("Model relay is unavailable")
            job.status = "preparing-transfer"
            job.message = "Verifying files before LAN delivery to the cluster"
            job.updated_at = datetime.now(UTC)
            await asyncio.to_thread(self.store.save_download, job)
            try:
                completed = await self.relay.distribute(job)
                if not completed:
                    return
            except (httpx.HTTPError, OSError, RuntimeError, ValueError) as error:
                job.status = "failed"
                job.error = self.safe_error(error)
                job.message = "Controller download is safe; cluster delivery can be retried"
                job.updated_at = datetime.now(UTC)
                await asyncio.to_thread(self.store.save_download, job)
                return
            job.transfer_progress = 100
            job.transferred_bytes = job.total_bytes
            job.message = "Available on the controller and in shared cluster storage"
        else:
            job.message = (
                "Copied from the controller into shared cluster storage"
                if job.source == "controller"
                else "Available in local model storage"
            )
        job.status = "completed"
        job.updated_at = datetime.now(UTC)
        await asyncio.to_thread(self.store.save_download, job)
        self._remember_download_fits([job])
        self._invalidate_inventory()

    @staticmethod
    def _staging_progress(staging: Path) -> int:
        completed = sum(
            path.stat().st_size
            for path in staging.rglob("*")
            if path.is_file() and ".cache" not in path.parts
        )
        partial = sum(
            path.stat().st_size
            for path in (staging / ".cache").rglob("*.incomplete")
            if path.is_file()
        )
        return completed + partial

    def safe_error(self, error: BaseException) -> str:
        message = str(error).strip() or error.__class__.__name__
        if self.settings.hf_token:
            message = message.replace(self.settings.hf_token, "[redacted]")
        message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", message)
        return message[:1000]


class ModelDoesNotFit(Exception):
    def __init__(self, fit: ModelFitPlan):
        super().__init__("Model does not fit the current cluster or model-cache storage")
        self.fit = fit


class ModelAlreadyInstalled(Exception):
    pass


class ModelDeletionBlocked(Exception):
    pass
