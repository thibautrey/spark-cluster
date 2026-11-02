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


The pool automatically learns a request-admission limit for each running model
instance. There is no tuning form or required configuration. It learns from actual
responses; it does not generate prompts or send background benchmark requests.

A new runtime begins with four admitted requests, bounded by its engine ceiling.
Successful measurement
windows compare aggregate reported output tokens per busy wall-clock second. For
embedding services they use reported input tokens; without reported token usage they
use requests per second, explicitly labelled. Concurrent request durations are never
summed to calculate aggregate throughput. Queueing, model startup and idle periods
are excluded from the service measurement. Workload or measurement-unit changes
start a fresh comparison.

Under sustained demand, the controller probes higher limits and retains increases
that improve throughput by more than 3%. It prefers a lower limit within 3% of the
same throughput, and periodically probes again after stable windows. Aggregate
throughput takes priority over individual response speed: four requests generating
12 output tokens/s each contribute 48 output tokens/s, not 12. A slower first token
or service time does not undo a measured throughput gain. Latency only triggers a
reduction after two consecutive comparable windows regress by more than twice the
latency at the best throughput, without a greater-than-3% aggregate throughput gain.
Workload/unit changes and healthy windows reset this confirmation.
Overload responses, failed streams and timeouts still reduce admissions immediately;
existing requests finish normally. After a reduction, eight successful observations
allow a new parallelism probe instead of waiting for 32 observations.
Client validation errors and cancellations do not count as capacity failures.

Runtime KV-cache occupancy can block upward probes. Sustained high occupancy during
active work reduces admissions; idle prefix caches and total allocated GPU memory
are not treated as proof of overload. Missing telemetry remains unknown and the
controller continues using request observations. Without simultaneous demand, the
status explicitly reports that more traffic is needed to explore a higher limit.

## Engine and lifecycle behavior

The gateway's existing 64-request bound is also the upper exploration bound.
The validated vLLM and SGLang Docker adapters set their continuous-batching scheduler
ceiling accordingly. Memory fractions, KV-cache precision and requested context are
preserved. Raising the scheduler ceiling does not reserve 64 full contexts: the
engine manages its bounded KV pool and the adaptive gate determines actual admission.
An already running older Docker recipe is upgraded only when active streams have
drained, including a retained runtime; streams are never interrupted for tuning.
If the engine reports a startup memory failure, batching capacity is halved and
retried within the same reservation. The successful ceiling is retained, so a smaller
GPU does not repeatedly retry an oversized CUDA-graph setup. Unrelated errors and
exit code 137 without OOM evidence do not trigger this retry policy.

Other engines retain their validated scheduling settings. In particular, llama.cpp
admission respects the configured parallel-slot count because increasing it would
split its context allocation. The controller cannot infer extra safe slots from
unused VRAM alone. A serial backend can converge to one useful admission even when
its HTTP server accepts several outstanding requests.

Kubernetes keeps one resident model per bridge while allowing concurrent requests
to that model. Model replacement and unload wait for all pinned requests, including
requests in its adaptive queue. New managed launches expose the batching ceiling;
existing runtimes remain bounded by their known execution configuration.

Limits, running requests, queued requests, learning state, observed throughput unit
and decision reason are available under `resident_models[].concurrency` in the pool
and inference status APIs, and under each model placement. The UI shows a compact,
read-only automatic-concurrency summary. Routing prefers the instance with lower
queue pressure relative to its learned limit.

Learned limits are saved privately beside setup state in `adaptive-concurrency.json`.
Identities include the model revision, engine image, launch settings and GPU memory
allocation; changing these resets learning. The saved cache is bounded and expires
old results. Request bodies, generated text and credentials are never persisted by
the learning controller. Without live GPU validation, local simulated-runtime tests
prove admission behavior, not a hardware throughput gain or a universal optimum.

Saved decisions carry a learning-policy version. Legacy decisions below four are
revalidated from four (or the engine ceiling), so an old latency-first limit of one
does not survive indefinitely. Newly measured serial limits and overload reductions
remain valid across restarts. The status API exposes `concurrency.learning_version`.
