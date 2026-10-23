# Runtime optimization lab

Spark Cluster now treats optimization as a measured control-plane workflow. It does not embed or
fork an inference engine. A launch resolves to an executable engine adapter, a pinned image, a
validated environment, and an exact command preview. A recommendation that lacks an adapter stays
visible with blockers and cannot be launched.

## Engine adapters

- **vLLM V1** is the executable multi-node baseline. The controller passes TP/PP, KV precision,
  scheduler bounds, CUDA graph, chunked-prefill, speculative-decoding, expert-parallel and EPLB
  options into the Kubernetes runtime.
- **SGLang** is executable only when a pinned image is configured and only on one Spark. Distributed
  SGLang bootstrap remains blocked until an image-specific multi-node launcher is validated.
- **TensorFold** remains a documented specialized DeepSeek V4.1 option. A container image by itself
  does not define its command or model layout, so the adapter intentionally refuses launches.
- **DeepInfer** remains experimental and blocked. Its published SM100 support does not prove that
  its FlashAttention, MLA, MoE and NCCL kernels work on GB10/SM121.

The automatic DeepSeek V4.1 plan now preserves the requested context length. One million tokens is
shown as an option rather than silently replacing a 256K request. The automatic runnable baseline is
vLLM until a specialized adapter is validated.

## Speculation and MoE

The launch form exposes baseline, MTP, DSpark, EAGLE3 and DFlash modes. It validates token bounds and
engine capabilities before queuing work. Expert parallelism supports bounded all-to-all choices,
optional EPLB and redundant experts. EPLB cannot be enabled without expert parallelism. A four-node
ring defaults to `allgather_reducescatter`; DeepEP remains opt-in because it must be measured on the
actual fabric.

Strict rank placement accepts one unique node per TP x PP rank. After Kubernetes schedules the pods,
the controller compares the real node order with the validated fabric order and fails with a useful
message if they differ.

## Measured telemetry

The controller polls the runtime Prometheus endpoint and stores a bounded SQLite history. It parses
TTFT and inter-token p50/p95, prefill and decode throughput, speculative acceptance, measured GPU KV
occupancy, LMCache hits/misses and retrieval latency, NCCL latency, request queues and expert
imbalance when the runtime exposes them. Missing metrics remain unknown instead of becoming zero.

The node agent no longer reports total NVML allocation as KV-cache usage. It reports accelerator
allocation separately and marks KV allocation unavailable until a runtime metric provides it.

## Benchmarks

The optimization lab sends deterministic OpenAI-compatible streaming requests. It records measured
TTFT, inter-token latency, aggregate output throughput, token counts, model revision, engine image,
topology, cache state, speculative method and expert-parallel state. Runs are bounded to 64 requests.
Results persist in SQLite and never store the benchmark prompt.

Use the same prompt and output length for baseline and optimized launches. Compare one variable at a
time, then retain separate winners for latency, throughput and long-context workloads.

## Prefix-cache lifecycle

Prewarming keys a prefix by model revision, tokenizer, engine configuration and a SHA-256 prompt
hash. Raw prompts are sent to the local runtime and are never stored by the controller. Records track
priority, token count, runtime generation, TTFT and LMCache hit deltas. Repeating a prefix after a new
runtime generation classifies measured reuse as `reused` or reconstruction as `rebuilt` when LMCache
metrics are available.

Cache CPU and NVMe budgets now follow the workload profile and are written into LMCache configuration
on both head and worker pods for each launch.

## Remaining hardware gates

CUDA kernel porting, real DSpark acceptance, SWA bounded replay, DeepEP benefits, rank-to-ring mapping,
restart-surviving prefix reuse and capacity calibration require physical GB10 measurements. The UI and
APIs expose the controls and evidence needed to run those checks without claiming an unmeasured gain.
