# Runtime scheduling and bounded admission

vLLM, SGLang and llama.cpp own request queueing, continuous batching and their
KV/token budgets. The gateway forwards concurrent requests immediately, including
requests beyond the engine execution-slot count. The executing machine still caps
all running and waiting requests across its models at 64, returning HTTP 503 with
Retry-After when full. This bounds connections and model-start waits.

For these engines, saved adaptive limits, response latency, errors and KV occupancy
do not introduce a second scheduling loop. Status reports `mode: runtime-managed`,
`limit: 64` (outstanding gateway requests), and `runtime_ceiling` (engine execution
capacity, when known). `running` counts forwarded requests, including engine-side
waiting; `waiting` counts only gateway admission waiting. Engine queue metrics remain
available independently. Existing runtime launch settings are preserved; llama.cpp
slots and context allocation are never increased by this policy. Models remain
pinned until every forwarded request completes or is cancelled.

Unknown Docker backends retain the adaptive fallback described below. Managed native
Kubernetes runtimes use runtime scheduling too. The engine's scheduler is authoritative;
this change does not claim a measured throughput improvement for every workload.

References: [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
and [vLLM scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py).

## Adaptive fallback for unknown backends

Unrecognized Docker services retain the previous automatic policy. They begin with
four admissions, bounded by their known engine ceiling, and learn from actual
responses without synthetic warm-up prompts. The score uses reported output tokens
per busy wall-clock second, input tokens for embedding services, or explicitly
labelled requests/s when usage is unavailable. Concurrent durations are not summed.

Throughput gains above 3% permit higher admission. Overload, failed streams and
sustained active KV pressure can reduce it. Two comparable latency regressions
without a throughput gain are required for latency backoff. Client validation errors
and cancellations do not count as capacity failures. Learned fallback decisions are
saved privately beside setup state and keyed by model, revision, image, settings and
GPU allocation. Changing these revalidates the fallback limit.

New validated vLLM/SGLang launches keep their existing maximum-64 scheduler defaults
and startup capacity retry behavior. Existing running engines are no longer restarted
just to enlarge their scheduler ceiling: they can queue above their execution ceiling.
Other launch settings, memory fractions, context lengths and GPU reservations stay
under their existing runtime configuration and placement rules.

The UI's concurrency count reports outstanding gateway requests, rather than the
number of executing sequences inside the engine. Routing still prefers less-loaded
instances. Startup readiness, recovery supervision, cancellation and request tracking
remain active independently of this scheduling policy.
