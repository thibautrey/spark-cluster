# Selecting useful DGX Spark models

Research snapshot: **23 October 2026**. This is an evidence review, not a universal model leaderboard. Model cards and community reports can change. Measurements below are their authors' results, not controller benchmarks.

## What the DeepSeek 1017 example teaches us

Searching `DeepSeek-V4.1-Flash-1017`, sorting by downloads and selecting only `text-generation` is insufficient:

- The official checkpoint and several important conversions have the task `image-text-to-text`. The catalog now defaults to all tasks so these are discoverable.
- The downloads-first Hub snapshot led with `nazeshinjite/...-ds4-GGUF` (100,738 downloads), the official model (57,715), and `peasantsmith/...-Maya-GGUF` (53,767). These counts establish distribution, not GB10 compatibility, fidelity or throughput.
- The most useful two-Spark evidence was about **EXL3 + a specific engine fork**, not an NVFP4 label. Searching only NVFP4 misses it.
- The Elevatormusic Hugging Face repository is a recipe. Downloading a recipe is different from downloading its referenced weights.
- Some conversions omit **~203 GB of official Engram tables**. A 199–211 GB repository is not the complete disk requirement. The 1017 tables must match 1017 weights. Engram SSD placement requires an engine implementation; ordinary weight offload is not interchangeable with it.
- Apple MLX conversions, datacenter-only kernels, altered-behavior derivatives and experimental extreme quantizations should not silently become the default simply because they are popular or small.

### Concrete shortlist

| Situation | Weight repository | Evidence and limitations |
|---|---|---|
| Two Sparks; vLLM API, images and priority scheduling | [HeNryous EXL3 2.75bpw](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark) | [Forum recipe](https://forums.developer.nvidia.com/t/386295), custom vLLM image and kernels, TP2 over RoCE, 320K context. Author reports KLD 0.0201 on a specified panel. ~199 GB pack + ~203 GB Engram **per node**. Not stock vLLM. |
| Two Sparks; interactive text/code and long context | [LibertAI EXL3 2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | [TensorFold report](https://forums.developer.nvidia.com/t/386125): 56.8 tok/s prose, 92.2 code, TP2, with long-context needle tests. Custom fork; OOM and shutdown reports in replies. A successful needle does not prove arbitrary 1M-context concurrency is safe. |
| Two Sparks; alternative fidelity-oriented conversion | [Mia-AiLab EAQ v1 2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw) | Card describes plain EXL3, source revision, conversion process, evaluation and compatibility with the TensorFold recipe. This is publisher compatibility evidence, **not an independent performance comparison** against LibertAI. Engram is separate. |
| Four Sparks; higher-bit EXL3 | [LibertAI EXL3 3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) | [TP4 recipe and run records](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark). Author reports 93.6 aggregate tok/s with one user and dual rails; workload and clocks matter. Card explicitly says **does not fit two Sparks**: ~121.5 GiB weights per rank exceeds practical available memory. |
| One Spark; exploratory extreme quantization | [Baekpica VQ GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) | Card explicitly warns of ~1.6-bit experts, looping reasoning, broken output and degraded quality. Requires its specialized engine. Available under experimental variants, not a standard recommendation. |
| Fidelity reference / larger native deployment | [Official DeepSeek 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | Canonical model identity and reference behavior. Its presence does not prove a practical two-node native launch. Use the exact native recipe and full memory/storage plan. |

[Reddit's TensorFold post](https://www.reddit.com/r/LocalLLaMA/comments/1x8q6t6/deepseek_v41_flash_1017_on_2x_dgx_spark_57_toks/) repeats the same author's forum experiment. These are **one evidence origin**, not two independent confirmations. The later vLLM report explicitly cautions that “2.1–2.3× vLLM” refers to an older kit, not every vLLM image. Its follow-up changes cold prefill results and compares clocks and prompts. Do not combine unrelated headline numbers into a speed ranking.

## Other prominent families

Discovery included live Hub trending and downloads snapshots plus [daily coding reports](https://www.reddit.com/r/LocalLLaMA/comments/1wmz86p/coding_models_and_quants_on_dgx_spark/). Reddit contains useful concrete configurations, but “best”, “85% of cloud” and subjective coding impressions are not standardized evaluations. The following are candidates to inspect, not certified winners.

| Family | Useful starting points | Deterministic treatment |
|---|---|---|
| Qwen 3.8 | [27B FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), [Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 27B is a practical single-node baseline. Users report 27B Q8 at 128K and Flash Next FP8 on two Sparks. Prefer the original instruction checkpoint, then a format supported by the chosen engine. Flash Next support depends on architecture/version, not just a “Qwen” label. MLX derivatives are a different hardware path. |
| GLM 5.3 | [Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), [full model](https://huggingface.co/zai-org/GLM-5.3) | Flash has dual-Spark user reports, including MiaAI EXL3, but posters describe patched runtimes. Full GLM and Flash are distinct capacity problems. Do not inherit the Flash report across the family or assume active-parameter count determines weight memory. |
| Gemma 4 | [31B instruction](https://huggingface.co/google/gemma-4-31B-it), [26B A4B instruction](https://huggingface.co/google/gemma-4-26B-A4B-it) | Strong distribution signals and explicit multimodal model cards. Useful smaller baselines. Filter vision explicitly when needed; do not exclude them with text-only task filtering. No Spark benchmark was established in this review, so no recipe-level priority. |
| Step 3.7 Flash | [Official NVFP4](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) | Official format plus documented serving instructions is a useful candidate. Card points to a Step-specific runtime image. Check ARM64/SM121 kernels, the architecture and full resident memory before recommending a node count. |
| NVIDIA Nemotron | [Nano 30B A3B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) | Official quantization and fidelity documentation; useful efficiency candidate. Hybrid Mamba/attention architecture needs matching engine support. Nemotron Nano, Super and Ultra are separate capacity/quality choices. NVIDIA authorship does not certify all these models on GB10. |
| MiniMax M3 | [NVIDIA NVFP4 conversion](https://huggingface.co/nvidia/MiniMax-M3-NVFP4) | NVIDIA model card describes a 428B source. A vendor NVFP4 conversion is stronger format provenance than a name-only claim, but still not a Spark benchmark. Inspect its license and hardware compatibility. |
| Kimi K3 | [Official checkpoint](https://huggingface.co/moonshotai/Kimi-K3) | Highly visible and large; not automatically a useful small-cluster result. MXFP4 is **not NVFP4**. Establish exact weight size, runtime and sharding support first. Distilled Qwen derivatives are different models. |
| New trending entries | [Live trending text models](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) | The snapshot included Flint, Spark-X2.6, Sage-A1.5, Hy-Memory, Dory and Beam. These deserve discovery, but this review did not establish comparable Spark measurements. Trending alone must not confer recommendation evidence. |

The HF safetensors `total` field is not a reliable original parameter count for every packed quantization. Do not derive memory from it alone. Use actual shards, quantization configuration, non-weight dependencies, KV cache, activations, duplicated tensors, engine overhead and per-rank limits. Aggregate cluster RAM is not a single shared GPU address space. Three or five machines are not automatically valid tensor-parallel degrees.

## Implemented selection policy

1. **Bounded discovery:** union trending, downloads, NVFP4 search and `dgx-spark`, `sm121`, `exl3` tag searches, at most 288 raw candidates. Preserve task, library, access and query constraints. A supplemental failure leaves regular search available. This is a shortlist, not exhaustive Hub coverage.
2. **Separate evidence from feasibility:** capacity and launch support remain in the existing context-sensitive fit planner. A documented external recipe does not mark an unsupported engine as installed or launchable.
3. **Lexicographic ranking:** fresh exact-revision community recipe first; then declared technical metadata candidates; then publisher identity and bounded log popularity. A name-only hardware claim cannot acquire recipe priority. Final ties use normalized repository ID, independent of API insertion order. This ranks discovery evidence, **not intelligence or measured speed**.
4. **Explainable selection:** default standard checkpoints; optional documented recipes, family publisher/NVIDIA or all variants. Modified behavior is a preference, not proof of worse quality. Experimental and other-hardware entries remain discoverable. These filters apply to loaded results; load more to expand the shortlist.
5. **Context:** selected capacity uses the existing full fit calculation. In recommended order, launch-supported fits come first; a report matching the selected node count breaks ties. Reported topology is not generalized to another node count. Explicit Hub sorts retain Hub order.
6. **Persistent, versioned cache:** raw metadata and ranking snapshots use the existing SQLite cache. Ranking identity includes rule versions, filters and UTC date; ten-minute snapshots support stable pagination. Recipe priority expires after 30 days and never transfers to a changed checkpoint SHA. Expired or changed evidence is shown as requiring revalidation.

## What would justify a stronger “best for my task” recommendation

A future benchmark registry should pin **model SHA + engine/image digest + patches + GPU architecture + node count + interconnect + context + concurrency + clocks + prompt set + sampling + draft model**, and retain raw artifacts and independent origin. Compare quality on the same evaluation harness, then TTFT/prefill, decode latency, aggregate throughput and completion cost on matched workloads. Show a Pareto frontier rather than one weighted “quality score”. Quantization perplexity, KLD and tool success rates from different datasets cannot be averaged fairly.

This implementation deliberately does not infer tool support from a family name, infer stock-runtime compatibility from a custom recipe, scrape social sentiment into a quality score, or automatically execute third-party instructions. The external recipe registry is intentionally small and reviewable; expanding coverage requires new dated evidence, not a publisher allowlist masquerading as benchmarking.
