"""Revision-bound TensorFold TP2/TP4 recipes reported by Tech2wild.

The external repository is treated as evidence and a preparation contract. Its
scripts, patches, binaries and site-specific addresses are neither vendored nor
executed by Spark Cluster.
"""

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

RECIPE_REPOSITORY = "https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks"
RECIPE_REVISION = "ab2d381e7a6cd266581229768efe9123e1e5134b"
RECIPE_SOURCE = f"{RECIPE_REPOSITORY}/tree/{RECIPE_REVISION}"
REVIEWED = date(2026, 10, 25)

BASE_MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE_MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
TP2_MODEL_ID = "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
TP2_MODEL_REVISION = "0be63c3337efbb2a0061d5b8bb4006fa186d433f"
TP4_MODEL_ID = "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw"
TP4_MODEL_REVISION = "04a492fc3b4d25ddbab8f2d2f07f235ec77c9010"

CONTAINER_IMAGE = "nvcr.io/nvidia/pytorch:26.07-py3"
TP2_ENGINE_REVISION = "3ff3b2ac1d71a789cb930068c46d9abed025eefa"
TP4_ENGINE_REVISION = "59402086381a42c74d53053443224eec2790b4c5"
TP2_RANK_CACHE_REVISION = TP4_ENGINE_REVISION
TP2_KIT_TAG = "v0.6.1"
TP2_KIT_REVISION = "1b5ebbee3cc8b63b8b875e46a314135eb6d669db"
TP4_KIT_TAG = "v0.1.1"
TP4_KIT_REVISION = "6033ea48a27004b2715db82190e2a61f33411d92"

MANAGED_LAUNCH_BLOCKER = (
    "The managed TensorFold adapter has no native Zig command contract, kernel-kit and "
    "rank-cache lifecycle, per-rank Engram mounts, or validated RoCE bootstrap. Prepare "
    "the pinned Tech2wild lane on every rank before serving."
)


@dataclass(frozen=True)
class TonyProfile:
    model_id: str
    model_revision: str
    nodes: int
    bits_per_weight: float
    engine_branch: str
    engine_revision: str
    kit_tag: str
    kit_revision: str
    streams: int
    pool_positions: int
    model_memory_gib_per_rank: float
    runtime_required_gib_per_rank: float
    rank_cache_gib_per_rank: float
    local_nvme_gib_per_rank: float
    load_seconds: str


PROFILES = {
    TP2_MODEL_ID: TonyProfile(
        model_id=TP2_MODEL_ID,
        model_revision=TP2_MODEL_REVISION,
        nodes=2,
        bits_per_weight=2.9,
        engine_branch="deepseek-v41-zig",
        engine_revision=TP2_ENGINE_REVISION,
        kit_tag=TP2_KIT_TAG,
        kit_revision=TP2_KIT_REVISION,
        streams=4,
        pool_positions=1_048_576,
        model_memory_gib_per_rank=105.04,
        # The measured nodes left 6.7 and 8.2 GiB of 121.7 GiB available.
        runtime_required_gib_per_rank=115.0,
        rank_cache_gib_per_rank=98.9,
        local_nvme_gib_per_rank=290.0,
        load_seconds="27.1",
    ),
    TP4_MODEL_ID: TonyProfile(
        model_id=TP4_MODEL_ID,
        model_revision=TP4_MODEL_REVISION,
        nodes=4,
        bits_per_weight=3.5,
        engine_branch="deepseek-v41-zig-tp4",
        engine_revision=TP4_ENGINE_REVISION,
        kit_tag=TP4_KIT_TAG,
        kit_revision=TP4_KIT_REVISION,
        streams=16,
        pool_positions=2_097_152,
        model_memory_gib_per_rank=71.87,
        # The head was the limiting rank with about 28-29 GiB left.
        runtime_required_gib_per_rank=93.7,
        rank_cache_gib_per_rank=64.2,
        local_nvme_gib_per_rank=255.0,
        load_seconds="107-120",
    ),
}


def profile_for_model(model_id: str) -> TonyProfile | None:
    return PROFILES.get(model_id)


def is_reviewed_profile(model_id: str, revision: str | None) -> bool:
    profile = profile_for_model(model_id)
    return bool(profile and revision == profile.model_revision)


class TonyRecipePlan(BaseModel):
    id: str
    source_url: str = RECIPE_SOURCE
    recipe_url: str = RECIPE_SOURCE
    recipe_revision: str = RECIPE_REVISION
    checkpoint_revision: str
    base_checkpoint_revision: str = BASE_MODEL_REVISION
    reviewed: str = REVIEWED.isoformat()
    license: str = "MIT recipe; Apache-2.0 engine and kit; MIT weights"
    node_count: int
    launcher: str
    env_file: str
    max_context_length: int = 1_048_576
    environment: dict[str, str]
    prerequisites: list[str]
    blockers: list[str] = Field(default_factory=list)
    measurements: list[str]


def build_tony_recipe(
    model_id: str,
    context_length: int,
    concurrent_sequences: int,
    topology: str = "auto",
    revision: str | None = None,
) -> TonyRecipePlan:
    profile = profile_for_model(model_id)
    if not profile:
        raise ValueError("No reviewed Tech2wild TensorFold profile exists for this model")
    tp2 = profile.nodes == 2
    context = min(context_length, 1_048_576)
    environment = {
        "IMAGE": CONTAINER_IMAGE,
        "MODEL_REVISION": profile.model_revision,
        "BASE_CHECKPOINT_REVISION": BASE_MODEL_REVISION,
        "ENGINE_BRANCH": profile.engine_branch,
        "ENGINE_REVISION": profile.engine_revision,
        "KIT_TAG": profile.kit_tag,
        "KIT_REVISION": profile.kit_revision,
        "CONTEXT": str(context),
        "PARALLEL": str(profile.streams),
        "STREAMS": str(profile.streams),
        "TF_DS_STREAMS": str(profile.streams),
        "TF_DS_WINDOW": str(profile.pool_positions),
        "TF_SEED_RANDOM": "1",
        "TF_TOP_K_MAX": "56",
        "TEMPERATURE": "1.0",
        "TOP_P": "1.0",
        "TOP_K": "20",
        "MAX_TOKENS": "32768",
    }
    if tp2:
        environment.update(
            {
                "RANK_CACHE_REVISION": TP2_RANK_CACHE_REVISION,
                "TF_DS_REPLAY_FLOOR": "1",
            }
        )
    else:
        environment.update(
            {
                "TF_DS_2D_GU": "parity",
                "TF_DS_ROUND_ROWS": "48",
                "TF_DS_SERVED_K": "solo",
            }
        )
    prerequisites = [
        f"Use {profile.model_id} at {profile.model_revision} and the base checkpoint at "
        f"{BASE_MODEL_REVISION} for Engram shards 47 and 48.",
        f"Use {profile.nodes} dedicated ARM64 GB10 / SM121 nodes with about 121.7 GiB "
        "unified memory each.",
        f"Build TensorFold {profile.engine_branch} at {profile.engine_revision}, then apply "
        "only the patches recorded by the pinned run folder.",
        f"Use kernel kit {profile.kit_tag} at {profile.kit_revision} and verify its manifest, "
        "trellis-width coverage, 1017 image-routing bias and RoPE tables before launch.",
        f"Reserve about {profile.local_nvme_gib_per_rank:.0f} GiB local NVMe per rank for "
        "the two Engram shards and generated rank cache; verify row coverage and token maps.",
        "Probe the RoCE v2 GID index and HCA names on every node; do not copy the author's "
        "site-specific interface names, addresses or GID index.",
        "Use both ConnectX-7 rails only after routing and equal traffic are verified; preserve "
        "physical rank order and keep management traffic separate.",
        "Probe decode-shaped memory bandwidth and clocks before comparing performance. The "
        "author observed a hidden GB10 slow state that nvidia-smi did not expose; treat any "
        "physical power-cycle remedy as a separate operator action.",
        "Install the byte-checked DeepSeek 1017 chat template and validate chat, SSE, tools, "
        "tool results, thinking controls, images, concurrent equality and sampling.",
        "Bind the API to a trusted network and configure authentication; the author's public "
        "bind without an API key is evidence of their test endpoint, not a safe default.",
    ]
    if tp2:
        prerequisites.insert(
            3,
            f"Cherry-pick the no-GPU rank-cache writer at {TP2_RANK_CACHE_REVISION} before "
            "applying the TP2 run folder's patches and resolving only the documented conflicts.",
        )
    blockers = [MANAGED_LAUNCH_BLOCKER]
    if revision != profile.model_revision:
        blockers.append("Resolve and select the reviewed weight revision before deployment.")
    if context_length > 1_048_576:
        blockers.append("The reviewed TensorFold lanes expose at most 1,048,576 tokens.")
    if concurrent_sequences > profile.streams:
        blockers.append(
            f"The validated TP{profile.nodes} lane serves at most {profile.streams} streams."
        )
    if context_length * concurrent_sequences > profile.pool_positions:
        blockers.append(
            f"Requested full contexts exceed the {profile.pool_positions:,}-position shared "
            "pool; the context window and stream count are independent ceilings."
        )
    if topology == "single" or (profile.nodes == 4 and topology == "direct"):
        blockers.append(f"The TP{profile.nodes} lane requires a complete RoCE fabric.")
    if tp2:
        measurements = [
            "Tech2wild reports 73.1 aggregate tok/s at C1 and 135.8 at C4, greedy, "
            "thinking off, both RoCE rails; mean TTFT 0.24 s at C1.",
            "Cold unique-prefix prefill was 2,044-2,200 tok/s at 11.6K-186.8K tokens. "
            "A 1M-token needle passed in 641.5 s; this does not prove four concurrent 1M prompts.",
            "The external 69-scenario suite scored 97.1 (66 pass, 2 partial, 1 fail). "
            "Treat it as one author's workload, not an independent quality benchmark.",
            "The external server documents two chat-template gaps: double-encoded tool "
            "argument strings and text-only content arrays do not exactly match the reference.",
        ]
    else:
        measurements = [
            "Tech2wild reports 93.6 aggregate tok/s at C1 and 270.8 at C16 on both rails; "
            "a later live-endpoint sweep measured 98.6 at C1 and may include traffic variance.",
            "Cold unique-prefix prefill was 2,140-2,215 tok/s at 11.6K-93.3K tokens. "
            "Both rails improved reported C1/C6 decode by 4-5% and long prefill by 14-15%.",
            "Needles passed through 500K; 1M is configured but was not rerun on the fixed "
            "stack. The external 69-scenario suite scored 97.1.",
            "Long-prompt concurrent-equality and retained-prefix gates were not run on the live "
            "endpoint; the reported TF_DS_PROMPT_SP optimization was not public or enabled.",
            "The external server documents two chat-template gaps: double-encoded tool "
            "argument strings and text-only content arrays do not exactly match the reference.",
        ]
    return TonyRecipePlan(
        id=f"tonyd2wild-deepseek-v41-tp{profile.nodes}",
        checkpoint_revision=profile.model_revision,
        node_count=profile.nodes,
        launcher=(
            "runs/2026-10-18-1017-tensorfold-tp2/launch/start_tp2.sh"
            if tp2
            else "runs/2026-10-17-1017-tensorfold/launch/start_fleet_rail2_p4.sh"
        ),
        env_file=(
            "runs/2026-10-18-1017-tensorfold-tp2/launch/config.tp2.env"
            if tp2
            else "runs/2026-10-17-1017-tensorfold/launch/config.fleet-8000-rail2-p4.env"
        ),
        environment=environment,
        prerequisites=prerequisites,
        blockers=blockers,
        measurements=measurements,
    )
