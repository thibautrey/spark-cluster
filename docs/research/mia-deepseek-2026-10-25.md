# MIA DeepSeek V4.1 Flash integration

Reviewed 25 October 2026 against [MIA's article](https://mia-ai.net/models/DeepSeek-v4.1-Flash-DGX-Sparks)
and [recipe revision 64b02dea8681](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/tree/64b02dea8681fb1d8e3a659ea432a8e9e9ba3995).
The checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. These findings do not transfer to EXL3,
GGUF, other checkpoints or arbitrary NVIDIA GPUs.

## Profiles

| Setting | TP3 | TP4 production |
|---|---|---|
| Hardware | 3 dedicated GB10 / SM121 Sparks, ~121.7 GiB each | 4 dedicated Sparks |
| Fabric | Two-port ConnectX-7 triangle | RoCE switch; patched switchless ring is opt-in |
| Weights | Native MXFP4 experts, FP8 dense; ~305 GiB expanded resident total | Same weights |
| Engram | Local NVMe, ~63 GiB per rank | Local NVMe, ~48 GiB per rank |
| Head checkpoint | ~476 GiB, shared to workers over the fabric | Same |
| Context | 32,768 unattended default; 262,144 configured ceiling | Up to 1,048,576 |
| Pinned KV pool | 750,000 tokens | 8,000,000 tokens |
| Scheduler slots | 4 | 16 |
| Static memory fraction | 0.95 | 0.80 |
| Prefill chunks | 768 | 4096, requires the chunked indexer patch |
| Speculation | DSpark, 5 drafts, confidence cap and block verification | Same |
| Expert parallel | EP=1 | EP=1 production; the base TP4 example uses EP=2 |

The TP4 context ceiling and scheduler slots are independent limits: sixteen full 1M-token
contexts do not fit an 8M-token pool. Fit checks reject such a requested concurrency.
TP3 long prompts require head-memory stress testing even below the configured context ceiling.

The capacity estimate uses MIA's expanded ~305 GiB resident footprint rather than simply
subtracting Engram from checkpoint size, includes the whole pinned KV pool at 1670.75 bytes
per token **per rank**, and applies the reviewed profile's static fraction with a runtime
reserve. It is an estimate for that recipe, not a measured allocation or a generic KV-dtype
conversion. Packed rows require additional local disk alongside the original shards. The
controller cannot establish that each node has sufficient local NVMe from controller storage.

## What the controller integrates

- Catalog evidence for the exact checkpoint revision, reviewed date, TP3/TP4 node counts and
  pinned source. The priority expires after 30 days or a checkpoint revision change.
- Specialized capacity estimates for resolved checkpoint metadata on DGX Spark. Unresolved
  revisions and other hardware retain conservative estimates. Old persisted fit estimates
  use a different cache namespace.
- `/api/v1/performance/plan` returns `recipe`, `blockers` and `engine_executable=false` for
  the official checkpoint. `recipe` contains the source pin, launcher, env-file name,
  reviewed settings, prerequisites and measurement conditions. Supply `revision` to bind
  the plan to the reviewed checkpoint.
- The read-only MCP tool `get_performance_plan` returns the same plan for AI clients.
- The dashboard's Plan details shows these requirements and settings. Merge settings into
  the **complete** pinned upstream profile, preserving its gated production adapters and
  site-specific fabric configuration.
- The official checkpoint cannot fall through to a stock vLLM or SGLang managed launch,
  even if a generic SGLang image is configured. Derivative recipes keep their separate gates.

The current managed SGLang adapter does not implement MIA's distributed bootstrap, packed
Engram mounts or patched kernels. A capacity-compatible checkpoint is therefore
**engine-required**, not launch-ready. Preparing the upstream recipe is a distinct deployment
step; this integration neither installs nor runs its scripts.

## Runtime requirements

Use the pinned recipe's `doctor`, `download`, `build`, `share` and `pack` steps before
`serve`; use its `smoke` and workload checks after readiness. The TP3 overlay uses the ARM64
SGLang base digest `sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349`.
The published TP4 production figures require `Dockerfile.canary-roce`, EP=1 and the full
production adapter profile. The stock `.env.tp4.example` is not that full production line.
The recipe pin retains the exact build sources; a base-image digest alone is insufficient.

Keep Engram offload on NVMe: Spark CPU and GPU share RAM. Use FlashInfer's b12x route for
32x32 FP8 projections, the TP3 padding/scale repairs, and the TP4 chunked indexer for large
prefills. Keep fused MoE finalize disabled and `expandable_segments:False`; upstream reports
nondeterminism and NaN logits with the respective alternatives. Preserve the NCCL buffer
limits, named loop-abort errors, malformed tool-history repair and the 32,768 output cap for
requests that omit `max_tokens`. These features require code in the upstream overlay;
environment variables alone do not implement them.

## External measurements and limits

MIA reports TP3 prose 51.0 tok/s at concurrency 1 (TTFT 223 ms), and 85.4 aggregate at
concurrency 4. The TP4 production line reports 87.7 tok/s prose at concurrency 1 and
342.7 aggregate at concurrency 16 (22.2 per request), greedy decoding, thinking disabled,
256 output tokens and switched fabric. Prompt types and concurrency change draft acceptance.

TP4 real-text cold prefill is reported at 4.9k–5.0k tok/s for 16k–128k prompts. The faster
repeated-token synthetic prefill benefits from a hot Engram row. A 1,011,084-token needle test
passed with the chunked indexer and pinned pool; this is not a general quality guarantee.
None of these figures are benchmarks performed by this controller. Compare exact checkpoint,
engine, fabric, prompts, concurrency, cache state and decoding settings before claiming gains.

The upstream recipe is AGPL-3.0-or-later, with its own notices; checkpoint licensing is
separate. Spark Cluster represents public configuration facts and links the external recipe.
It does not vendor the recipe's adapters or change this project's license.
