"""Versioned, deterministic discovery evidence, separate from capacity and launch gates."""

import math
import re
from datetime import UTC, date, datetime

from pydantic import BaseModel, Field

from .deepseek_mia import (
    MODEL_ID,
    MODEL_REVISION,
    RECIPE_SOURCE,
)
from .deepseek_mia import (
    REVIEWED as MIA_REVIEWED,
)
from .spark_optimization import SparkOptimization

RECOMMENDATION_VERSION = "evidence-v2-mia"
# Exact model revisions inspected on the review date. These are community reports,
# not certifications. A changed revision never inherits an old recipe's priority.
REPORTS = {
    "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw": {
        "commit": "0be63c3337efbb2a0061d5b8bb4006fa186d433f",
        "nodes": [2],
        "runtime": "TensorFold DeepSeek V4.1 TP2 fork",
        "source": "https://forums.developer.nvidia.com/t/386125",
        "note": "TP2 community measurements: 56.8 tok/s prose, 92.2 code in one report; "
        "not comparable to unrelated prompt sets. Requires official 1017 Engram shards. "
        "OOM reports exist; do not assume 1M context is safe for every workload.",
    },
    "HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark": {
        "commit": "f53132cb17e18a5a9ddd7f213965912a575f0dd5",
        "nodes": [2],
        "runtime": "Patched vLLM + EXL3 kernels",
        "source": "https://forums.developer.nvidia.com/t/386295",
        "note": "320K context reported; requires ~203 GB of official 1017 Engram shards "
        "in addition to ~199 GB of quantized weights on each node. Custom image required.",
    },
    "Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw": {
        "commit": "45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b",
        "nodes": [2],
        "runtime": "TensorFold DeepSeek V4.1 fork",
        "source": "https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw",
        "note": "Publisher documents compatibility with the TensorFold TP2 recipe; "
        "not a separate independent benchmark. Requires official 1017 Engram shards.",
    },
    "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw": {
        "commit": "04a492fc3b4d25ddbab8f2d2f07f235ec77c9010",
        "nodes": [4],
        "runtime": "TensorFold DeepSeek V4.1 TP4 fork",
        "source": "https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark",
        "note": "Four-Spark recipe reported; does not fit two Sparks. Requires official "
        "1017 Engram shards. Published throughput depends on prompt, concurrency and RoCE rails.",
    },
    MODEL_ID: {
        "commit": MODEL_REVISION,
        "nodes": [3, 4],
        "runtime": "MIA patched SGLang with DSpark and NVMe Engram",
        "source": RECIPE_SOURCE,
        "reviewed": MIA_REVIEWED,
        "note": "Native MXFP4 experts / FP8 dense weights need at least three Sparks, "
        "~305 GiB resident plus local NVMe Engram. TP3 unattended context 32k; TP4 1M "
        "context and 16 scheduler slots share an 8M-token KV pool. Published TP4 prose "
        "87.7 tok/s c1 and 342.7 aggregate c16 require the production overlay and fabric. "
        "Stock SGLang and the managed vLLM launcher do not implement this recipe.",
    },
}
REVIEWED = date(2026, 10, 23)
EXPERIMENTAL = {
    "Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF": (
        "Publisher warns of experimental ~1.6-bit expert weights, "
        "reasoning loops and degraded output."
    )
}
# Namespace is an identity hint only when it matches the model family. It is never
# evidence of Spark support or model quality.
PUBLISHERS = {
    "deepseek-ai": "deepseek",
    "qwen": "qwen",
    "zai-org": "glm",
    "google": "gemma",
    "stepfun-ai": "step",
    "minimaxai": "minimax",
    "moonshotai": "kimi",
    "nvidia": "nemotron",
}


class Recommendation(BaseModel):
    version: str = RECOMMENDATION_VERSION
    label: str = "Unverified candidate"
    priority: int = 0
    evidence: int = 0
    publisher: str = "community"
    variant: str = "standard"
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    reported_nodes: list[int] = Field(default_factory=list)
    runtime: str | None = None
    reviewed: str | None = None


def recommend(data: dict, spark: SparkOptimization, today: date | None = None) -> Recommendation:
    repo_id = data.get("id") or data.get("modelId", "")
    namespace, _, name = repo_id.casefold().partition("/")
    card = data.get("cardData") or {}
    tags = data.get("tags") or []
    text = " ".join([name, *tags]).casefold()
    result = Recommendation()
    if re.search(r"abliterat|uncensored|(?:^|[\s_-])merge(?:[\s_-]|$)", text):
        result.variant = "modified"
        result.caveats.append(
            "Behavior-modified derivative; evaluate against the original for your task."
        )
    if repo_id in EXPERIMENTAL:
        result.variant = "experimental"
        result.caveats.append(EXPERIMENTAL[repo_id])
        result.sources.append(f"https://huggingface.co/{repo_id}")
    family = PUBLISHERS.get(namespace)
    if family and name.startswith(family):
        result.publisher = "original"
        result.reasons.append("Model-family publisher namespace; not proof of Spark support.")
    elif namespace == "nvidia":
        result.publisher = "vendor-conversion"
        result.reasons.append(
            "NVIDIA-published conversion; check GB10 support for this architecture."
        )
    if not card.get("license") and not any(tag.startswith("license:") for tag in tags):
        result.caveats.append("License not reported in catalog metadata; inspect the model card.")
    if data.get("gated"):
        result.caveats.append("Hugging Face access approval or authentication may be required.")
    library = (data.get("library_name") or "").casefold()
    if library == "mlx" or re.search(r"(?:^|[\s_-])mlx(?:[\s_-]|$)", name):
        result.variant = "other-hardware"
        result.caveats.append("MLX targets Apple hardware; not a GB10 CUDA launch candidate.")
    if spark.tier in {"metadata-only", "other-hardware"}:
        result.variant = spark.tier
    if spark.score:
        structured = (data.get("config") or {}).get("quantization_config") or {}
        declared = any(
            tag.casefold() in {"nvfp4", "dgx-spark", "gb10", "sm121", "sm-121", "nvidia-dgx-spark"}
            for tag in tags
        )
        result.priority = int(bool(declared or structured))
        result.label = "Spark candidate · unverified"
        result.evidence = int(declared) + int(bool(structured))
        result.reasons.extend(spark.reasons)
    report = REPORTS.get(repo_id)
    if report:
        reviewed = report.get("reviewed", REVIEWED)
        result.sources.append(report["source"])
        result.reviewed = reviewed.isoformat()
        result.reported_nodes = report["nodes"]
        result.runtime = report["runtime"]
        result.caveats.append(report["note"])
        age = ((today or datetime.now(UTC).date()) - reviewed).days
        if data.get("sha") == report["commit"] and 0 <= age <= 30:
            result.priority, result.evidence = 2, 2
            result.label = "Documented Spark recipe"
            result.reasons.append(
                "Exact repository revision reviewed with a linked community recipe."
            )
        else:
            result.caveats.append(
                "Recipe review expired or checkpoint revision changed; revalidation required."
            )
        result.caveats.append(
            "Community evidence only; runtime installation and launch are separate checks."
        )
    if result.variant != "standard":
        result.priority = -1
        result.label = result.variant.replace("-", " ").capitalize()
    return result


def ranking_key(model):
    """Lexicographic gates prevent likes/downloads overriding stronger evidence."""
    recommendation = model.recommendation
    popularity = round(
        min(6, math.log10(max(0, model.downloads) + 1))
        + min(3, math.log10(max(0, model.likes) + 1)),
        4,
    )
    return (
        -recommendation.priority,
        -recommendation.evidence,
        -int(recommendation.publisher != "community") if recommendation.priority == 0 else 0,
        -model.spark.score,
        -int(recommendation.publisher != "community"),
        -popularity,
        model.id.casefold(),
        model.id,
    )


def generic_ranking_key(model):
    """Rank general NVIDIA candidates without inheriting DGX Spark/NVFP4 evidence."""
    recommendation = model.recommendation
    risky = recommendation.variant != "standard"
    popularity = round(
        min(6, math.log10(max(0, model.downloads) + 1))
        + min(3, math.log10(max(0, model.likes) + 1)),
        4,
    )
    return (
        risky,
        -int(recommendation.publisher != "community"),
        -popularity,
        model.id.casefold(),
        model.id,
    )
