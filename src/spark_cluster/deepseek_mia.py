"""Reviewed MIA recipe contracts, independent of executable engine adapters.

Only public configuration facts are represented here. The upstream AGPL adapters
remain in their separately licensed repository and are not bundled or executed.
"""

from datetime import date

from pydantic import BaseModel, Field

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
MODEL_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
RECIPE_REVISION = "64b02dea8681fb1d8e3a659ea432a8e9e9ba3995"
RECIPE_REPOSITORY = "https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks"
RECIPE_SOURCE = f"{RECIPE_REPOSITORY}/tree/{RECIPE_REVISION}"
ARTICLE_URL = "https://mia-ai.net/models/DeepSeek-v4.1-Flash-DGX-Sparks"
REVIEWED = date(2026, 10, 25)
BASE_IMAGE = (
    "lmsysorg/sglang:dev-dsv41@sha256:"
    "3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349"
)
# The recipe reports expanded runtime residency, not simply checkpoint minus Engram.
RESIDENT_WEIGHTS_GIB = 305.0
KV_BYTES_PER_TOKEN_PER_RANK = 1670.75
MANAGED_LAUNCH_BLOCKER = (
    "The managed launcher has no MIA multi-node bootstrap, packed Engram mounts or "
    "patched-kernel contract. Prepare the pinned upstream recipe on every rank first; "
    "a stock SGLang image does not enable this lane."
)


def is_mia_checkpoint(model_id: str, revision: str | None) -> bool:
    return model_id == MODEL_ID and revision == MODEL_REVISION


class MiaRecipePlan(BaseModel):
    id: str
    source_url: str = ARTICLE_URL
    recipe_url: str = RECIPE_SOURCE
    recipe_revision: str = RECIPE_REVISION
    checkpoint_revision: str = MODEL_REVISION
    reviewed: str = REVIEWED.isoformat()
    license: str = "AGPL-3.0-or-later (recipe); checkpoint license is separate"
    node_count: int
    launcher: str
    env_file: str
    max_context_length: int
    environment: dict[str, str]
    prerequisites: list[str]
    blockers: list[str] = Field(default_factory=list)
    measurements: list[str]


def build_mia_recipe(
    nodes: int,
    context_length: int,
    concurrent_sequences: int,
    topology: str = "auto",
    revision: str | None = None,
) -> MiaRecipePlan:
    if nodes not in {3, 4}:
        raise ValueError("The MIA native checkpoint recipe supports exactly three or four ranks")
    tp4 = nodes == 4
    max_context = 1_048_576 if tp4 else 262_144
    max_requests = 16 if tp4 else 4
    pool = 8_000_000 if tp4 else 750_000
    environment = {
        "TP_SIZE": str(nodes),
        "EP_SIZE": "1",
        "NNODES": str(nodes),
        "CONTEXT_LENGTH": str(min(context_length, max_context)),
        "MAX_TOTAL_TOKENS": str(pool),
        "MAX_RUNNING_REQUESTS": str(max_requests),
        "MEM_FRACTION_STATIC": "0.80" if tp4 else "0.95",
        "CHUNKED_PREFILL_SIZE": "4096" if tp4 else "768",
        "OFFLOAD_MODE": "nvme",
        "SPEC_ALGO": "DSPARK",
        "DSPARK_BLOCK_SIZE": "5",
        "DSV41_VERIFY_CAP": "conf:0.1",
        "DSV41_BLOCK_VERIFY": "1",
        "DSV41_ENGRAM_PREFETCH": "1",
        "DSV41_IO_THREADS": "96",
        "DSV41_CACHE_GIB": "4" if tp4 else "0",
        "DSV41_TP_PAD": "0" if tp4 else "1",
        "DSV41_MXFP8_BACKEND": "b12x",
        "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "DSV41_MAX_NEW_TOKENS": "32768",
        "DSV41_LOOP_ABORT": "1",
        "DSV41_LOOP_NAMED_EXIT": "1",
        "DSV41_HISTORY_REPAIR": "1",
        "NCCL_BUFFSIZE": "1048576",
        "NCCL_LL128_BUFFSIZE": "262144",
        "NCCL_PROTO": "^LL128",
        "NCCL_MAX_NCHANNELS": "8",
    }
    if tp4:
        # This is an addition to the production profile's EXTRA_CONTAINER_ENV,
        # not a replacement for its other gated adapters or fabric settings.
        environment["DSV41_INDEXER_CHUNKED"] = "1"
    prerequisites = [
        f"Use {MODEL_ID} at {MODEL_REVISION}; build the recipe at {RECIPE_REVISION}.",
        f"Use {nodes} dedicated ARM64 GB10 / SM121 nodes with about 121.7 GiB each.",
        "Keep Engram rows on each rank's local NVMe; host RAM is shared with the GPU.",
        f"Reserve about {48 if tp4 else 63} GiB local NVMe per rank, plus the "
        "~476 GiB checkpoint on the head, shared to workers over the fabric.",
        "Run the upstream doctor, download, build, share and pack steps before serve.",
        "Preserve the recipe's FlashInfer FP8 GEMM backend when adding SGLang arguments.",
    ]
    if tp4:
        prerequisites.extend(
            [
                "Use the TP4 production Dockerfile.canary-roce and its full gated adapter "
                "profile with EP_SIZE=1; the base .env.tp4.example uses EP_SIZE=2.",
                "4096-token prefill requires the chunked indexer patch and an 8M-token "
                "KV pin at memory fraction 0.80.",
                "Use a RoCE switch, or the opt-in four-cable ring with patched NCCL on "
                "every rank. Configure local NIC/HCA names from fabric probes.",
            ]
        )
    else:
        prerequisites.extend(
            [
                f"Build the TP3 overlay from the reviewed ARM64 base {BASE_IMAGE}.",
                "Use a ConnectX-7 cable triangle; TP3 needs padding of heads 64 to 96, "
                "groups 8 to 12 and draft experts 128 to 129, including repaired scales.",
                "Use 32768 context for unattended TP3 serving; longer prefills need "
                "workload validation on the head.",
            ]
        )
    blockers = [MANAGED_LAUNCH_BLOCKER]
    if revision != MODEL_REVISION:
        blockers.append("Resolve and select the reviewed checkpoint revision before deployment.")
    if context_length > max_context:
        blockers.append(f"The TP{nodes} recipe context ceiling is {max_context} tokens.")
    if concurrent_sequences > max_requests:
        blockers.append(f"The TP{nodes} recipe supports at most {max_requests} active requests.")
    if context_length * concurrent_sequences > pool:
        blockers.append(
            f"Requested full contexts exceed the {pool}-token KV pool; the context ceiling "
            "and scheduler slots cannot all be filled simultaneously."
        )
    if topology in {"single", "direct"}:
        blockers.append(f"A {nodes}-rank recipe needs a complete RoCE fabric.")
    if tp4 and topology == "ring":
        blockers.append("Verify the upstream switchless-ring NCCL patch on all four ranks.")
    return MiaRecipePlan(
        id=f"mia-deepseek-v41-tp{nodes}",
        node_count=nodes,
        launcher="./start-tp4.sh" if tp4 else "./start.sh",
        env_file=".env.tp4" if tp4 else ".env",
        max_context_length=max_context,
        environment=environment,
        prerequisites=prerequisites,
        blockers=blockers,
        measurements=[
            (
                "MIA reports TP4 prose 87.7 tok/s at concurrency 1 and 342.7 tok/s "
                "aggregate at concurrency 16 (22.2 per request), greedy, thinking off, "
                "256 output tokens, switched fabric; not local measurements."
                if tp4
                else "MIA reports TP3 prose 51.0 tok/s at concurrency 1, TTFT 223 ms, "
                "85.4 tok/s aggregate at concurrency 4; greedy, thinking off, triangle "
                "fabric; not local measurements."
            ),
            (
                "TP4 real-text cold prefill: 4.9k-5.0k tok/s at 16k-128k; repeated-token "
                "synthetic filler benefits from a hot Engram row and is not comparable. "
                "A 1,011,084-token needle passed with the chunked indexer."
                if tp4
                else "TP3 prefill about 2000 tok/s; 256k is configured, not a safe "
                "unattended prompt guarantee. Head memory limits long prefills."
            ),
        ],
    )
