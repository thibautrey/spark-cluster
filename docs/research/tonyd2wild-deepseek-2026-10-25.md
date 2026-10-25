# Tech2wild DeepSeek V4.1 Flash TensorFold integration

Reviewed 25 October 2026 against
[`tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks` at `ab2d381e7a6c`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks/tree/ab2d381e7a6cd266581229768efe9123e1e5134b).
The repository is MIT. TensorFold and the kernel-kit recipes are Apache-2.0; the reviewed
weights are MIT. Spark Cluster stores public configuration and evidence only. It does not
vendor or execute the external patches, binaries or scripts.

## Exact lanes

| Setting | TP2 | TP4 |
|---|---|---|
| Weight repository | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw` |
| Weight revision | `0be63c3337ef` | `04a492fc3b4d` |
| Nodes | 2 dedicated GB10 / SM121 Sparks | 4 dedicated GB10 / SM121 Sparks |
| TensorFold | `deepseek-v41-zig` at `3ff3b2ac1d71`, rank-cache commit `59402086381a`, then recorded patches | `deepseek-v41-zig-tp4` at `59402086381a` plus recorded patches |
| Kernel kit | v0.6.1, commit `1b5ebbee3cc8` | v0.1.1, commit `6033ea48a270` |
| Engine-reported model memory | 105.04 GiB per rank | 71.87 GiB per rank |
| Conservative serving envelope | 115.0 GiB per rank | 93.7 GiB per rank, based on the limiting head |
| Context ceiling | 1,048,576 | 1,048,576 |
| Shared position pool | 1,048,576 | 2,097,152 |
| Validated streams | 4 | 16 |
| Local NVMe planning | 290 GiB per rank | 255 GiB per rank |

Both lanes additionally require Engram shards 47 and 48 from
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The local NVMe values include both
Engram shards plus the generated rank cache, rounded upward. Shared model storage and
per-rank local storage are separate gates.

The context window, stream count and position pool are independent limits. TP2 cannot serve
four simultaneous full 1M contexts from a 1,048,576-position pool. TP4 cannot serve sixteen
full 1M contexts from a 2,097,152-position pool. Capacity checks enforce the product of
requested context and concurrency against the shared pool.

## Integrated behavior

- Catalog evidence is bound to the exact weight revision and pinned recipe source. It expires
  after 30 days or when a weight revision changes.
- Capacity uses the run's measured serving envelope on physical 121.7 GiB nodes instead of
  the generic vLLM reserve. `kv_cache_gib` stays zero because the run reports shared positions
  and a combined serving footprint, not an independently measured KV GiB allocation.
- `ssd_required_gib_per_node` includes local Engram plus generated rank cache. Runnable storage
  also accounts for the original weight pack and base Engram artifacts.
- `/api/v1/performance/plan`, the dashboard and MCP `get_performance_plan` expose engine,
  model, base-checkpoint and kit pins; streams and position pool; prerequisites; measurement
  conditions; and blockers.
- Automatic engine selection chooses TensorFold for these two exact repositories. Managed
  launch remains disabled because Spark Cluster does not implement the native Zig command,
  kernel-kit, rank-cache, per-rank Engram and RoCE lifecycle. vLLM is not used as an implicit
  fallback for these EXL3 packs.

## Preparation and operating boundaries

Use the pinned run folder as the source of truth. Build for ARM64 SM121 with Zig 0.17.0 inside
`nvcr.io/nvidia/pytorch:26.07-py3`. Verify kernel coverage, the expected 1017 image-routing
bias difference, RoPE tables, rank-cache hashes, token map and Engram row coverage before
serving. Install the byte-checked 1017 chat template and rerun chat, streaming, tool call,
tool-result, thinking, image, concurrency and sampling gates.

Probe HCA names and the RoCE v2 GID index on every machine. Values such as GID 3 and the
author's interface names and IP addresses describe that fleet and are not portable defaults.
Use the second rail only after addresses, reachability and balanced traffic are verified.
Expose the OpenAI-compatible API only on a trusted network with authentication; the external
run's `0.0.0.0` bind without an API key is not copied.

Before comparing throughput, probe decode-shaped memory bandwidth and clocks. The author
observed a fast and slow GB10 state (about 230 versus 70 GB/s in their probe) that
`nvidia-smi` did not reveal. A reboot did not clear their affected machines. Spark Cluster
surfaces this as a benchmark gate and does not automate the reported physical power-cycle
remedy.

The reviewed sampler defaults are temperature 1.0, top-p 1.0 and top-k 20. The patched exact
sampler accepts top-k from 1 to 56, maps non-positive values to the server default and clamps
larger values. Requests without a seed receive a random seed; explicit seeds stay reproducible,
and temperature 0 remains deterministic. These behaviors require the recorded engine patch.

The external server documents two compatibility gaps. A double-encoded tool argument string
does not exactly follow the reference encoder's second decode, and text-only content arrays
are concatenated without the reference's blank-line separator. Clients should include both
cases in compatibility tests.

## External measurements

TP2 reports 73.1 aggregate tok/s at C1 and 135.8 at C4, with mean TTFT 0.24 s at C1.
Cold unique-prefix prefill was 2,044-2,200 tok/s across 11.6K-186.8K tokens. A 1M-token
needle passed in 641.5 seconds. The author's 69-scenario tool suite scored 97.1.

TP4 reports 93.6 aggregate tok/s at C1 and 270.8 at C16 on both rails. A later sweep on the
live endpoint measured 98.6 at C1 and may include traffic variance. Cold unique-prefix prefill
was 2,140-2,215 tok/s across 11.6K-93.3K tokens. Adding the second rail improved reported
C1/C6 decode by 4-5% and long prefill by 14-15%. Needles passed through 500K; 1M remained
configured but was not rerun on the fixed stack. The same external tool suite scored 97.1.
The recipe's long-prompt concurrent-equality and retained-prefix gates were not run against
the live TP4 endpoint. The reported `TF_DS_PROMPT_SP=rows` prefill optimization was not public
and was not enabled, so Spark Cluster does not recommend it.

These numbers are not controller benchmarks. Compare exact weights, engine, patches, kit,
clocks, rank inputs, prompt set, decoding settings, stream count, cache state and fabric.
