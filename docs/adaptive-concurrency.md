# Automatic request concurrency

The pool automatically learns a request-admission limit for each running model
instance. There is no tuning form or required configuration. It learns from actual
responses; it does not generate prompts or send background benchmark requests.

A new runtime begins with at most two admitted requests. Successful measurement
windows compare aggregate reported output tokens per busy wall-clock second. For
embedding services they use reported input tokens; without reported token usage they
use requests per second, explicitly labelled. Concurrent request durations are never
summed to calculate aggregate throughput. Queueing, model startup and idle periods
are excluded from the service measurement. Workload or measurement-unit changes
start a fresh comparison.

Under sustained demand, the controller probes higher limits and retains increases
that improve throughput by more than 3%. It prefers a lower limit within 3% of the
same throughput, backs off for significant first-token/service-latency regressions,
and periodically probes again after stable windows. Overload responses, failed
streams and timeouts reduce admissions immediately; existing requests finish normally.
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
