"""Explainable discovery hints; these are not runtime or performance certifications."""

import re

from pydantic import BaseModel, Field

CLASSIFIER_VERSION = "spark-signals-v1"


class SparkOptimization(BaseModel):
    tier: str = "general"
    score: int = 0
    label: str = "General checkpoint"
    badges: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    version: str = CLASSIFIER_VERSION


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def classify_spark(data: dict) -> SparkOptimization:
    repo_id = data.get("id") or data.get("modelId", "")
    # Do not use author, base_model tags or prose to attribute this checkpoint.
    namespace, _, name = repo_id.partition("/")
    tags = {_normalized(tag) for tag in (data.get("tags") or []) if ":" not in tag}
    name = f"-{_normalized(name)}-"
    config = data.get("config") or {}
    card = data.get("cardData") or {}
    quant = config.get("quantization_config") or data.get("quantization_config") or {}
    quant = quant if isinstance(quant, dict) else {}
    quant_signals = {
        _normalized(value)
        for value in (
            quant.get("quant_method"),
            quant.get("quant_algo"),
            quant.get("format"),
            card.get("quantization"),
        )
        if isinstance(value, str)
    }
    nvfp4_values = {"nvfp4", "nvfp4-awq", "nvfp4-pack-quantized"}
    config_nvfp4 = bool(quant_signals & nvfp4_values)
    contradicts_nvfp4 = bool(quant_signals & {"awq", "gptq", "mxfp4", "bitsandbytes"})
    nvfp4_tag = bool(tags & nvfp4_values)
    nvfp4_name = "-nvfp4-" in name and not any(
        negative in name for negative in ("-not-nvfp4-", "-no-nvfp4-", "-without-nvfp4-")
    )
    targeted_tags = tags & {"dgx-spark", "nvidia-dgx-spark", "gb10", "sm121", "sm-121"}
    targeted_name = any(
        f"-{signal}-" in name for signal in ("dgx-spark", "gb10", "sm121", "sm-121")
    )
    datacenter_only = bool(tags & {"sm100-only", "sm-100-only", "b200-only", "gb200-only"})
    other_target = bool(tags & {"sm100", "sm-100", "b200", "gb200"}) or (
        (data.get("library_name") or "").casefold() == "deepinfer"
    )
    gguf = "gguf" in tags or "-gguf-" in name
    result = SparkOptimization()
    if targeted_tags or targeted_name:
        result.tier, result.score, result.label = "spark-targeted", 100, "DGX Spark targeted"
        result.badges.append("GB10 / SM121")
        result.reasons.append(
            "Publisher explicitly labels this checkpoint for DGX Spark / GB10 / SM121."
        )
    if config_nvfp4 or ((nvfp4_tag or nvfp4_name) and not contradicts_nvfp4):
        result.score += 60 if config_nvfp4 or nvfp4_tag else 40
        if result.tier == "general":
            result.tier, result.label = "nvfp4", "NVFP4 · Spark candidate"
        result.badges.append("NVFP4")
        evidence = (
            "quantization metadata" if config_nvfp4 else "tags" if nvfp4_tag else "repository name"
        )
        result.reasons.append(
            f"NVFP4 declared in {evidence}; a Blackwell-oriented low-precision format."
        )
        if namespace.casefold() == "nvidia":
            result.score += 10
            result.badges.append("NVIDIA checkpoint")
            result.reasons.append("Checkpoint published in NVIDIA's Hugging Face namespace.")
    if contradicts_nvfp4 and not config_nvfp4 and (nvfp4_name or nvfp4_tag):
        result.caveats.append(
            "Quantization metadata conflicts with the NVFP4 label; inspect the model card."
        )
    if datacenter_only:
        result.score = 0
        result.tier, result.label = "other-hardware", "Other GPU target"
        result.caveats.append(
            "Publisher declares a datacenter-only GPU target; GB10 support is not established."
        )
    if other_target and not (targeted_tags or targeted_name):
        result.caveats.append(
            "SM100 / datacenter or DeepInfer targeting does not establish SM121 support."
        )
    if result.score:
        result.caveats.append(
            "Publisher signals, not a measured speedup or verified GB10 runtime compatibility."
        )
        result.caveats.append(
            "Check memory, context, exact checkpoint revision and installed engine before launch."
        )
        if gguf:
            result.caveats.append(
                "GGUF requires a compatible llama.cpp-family runtime and NVFP4 kernels; "
                "format alone is insufficient."
            )
        else:
            result.caveats.append(
                "Acceleration needs an ARM64 / SM121-compatible engine and kernels."
            )
    # Blackwell, FP4, FP8, INT4, MTP or a vendor name alone never earn a boost.
    return result
