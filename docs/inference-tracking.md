# Inference activity

The overview reads `/api/v1/inference/activity`. It accounts for requests through the
global `/v1`, model pool, authenticated node, and legacy on-demand listeners. Nested
local routing counts a request once. A remote controller and its worker each keep
their own journal; do not sum their counters as distinct client requests.

Records include model, route without query parameters, assigned pool, HTTP status,
success/failure/cancellation, total latency, waiting/loading time, service time,
headers/first byte latency, streaming time to first actual output, reported input,
output and cached tokens, response bytes, and fresh shared pool memory/GPU samples.
Streaming decode throughput excludes waiting and prefill. A complete JSON response
cannot expose token generation timing; its TTFT stays unknown. Pool memory is a
shared sample, not an estimate of allocation by an individual request.

The dashboard retains totals, a one-hour minute chart, recent requests, per-model
and per-pool totals, pool occupancy and resource usage. Percentiles describe the
latest 512 retained requests (successful observations with the relevant measurement).
Streaming chat/completion routes request a final usage chunk by default; an explicit
client `include_usage: false` remains respected. Missing usage stays unknown and is
counted explicitly. Nothing is inferred from
response length, SSE chunk count, or GPU activity. Earlier unrecorded requests
cannot be reconstructed. Historical records survive runtime unloads and restarts.

An observed client disconnect before completion is a cancellation, including when
the ASGI streaming response handles the disconnect without raising an exception.
A terminal SSE finish event counts as completed generation even if the client
closes before `[DONE]`. Provider errors and upstream streams that end without a
terminal event remain failures. Historical outcomes are retained as recorded;
missing response content cannot safely establish why an older stream stopped.

## Cost and bounds

There is no database, filesystem, network, GPU query, or additional I/O wait in the
request observer. After the first output it skips parsing token-only SSE frames.
The bounded parser reads at most 64 KiB per SSE line, or 256 KiB for JSON, plus a
64 KiB JSON tail for usage in large embedding/image responses. Binary/audio output
is counted without parsing. Streaming bytes and backpressure remain unchanged.

One background task batches completed metadata records every second into a separate
SQLite WAL database next to the configured controller database, with suffix
`-inference.db`. This avoids the model-download store's lock. Normal shutdown drains
the journal. An abrupt process/machine failure can lose the unflushed second and
in-flight requests. Storage failure never awaits a retry on an inference request;
the dashboard and Prometheus expose persistence health, pending and dropped records.
The write backlog is bounded to 4,096 events, in-memory history to 512 requests,
group cardinality to 256 plus an overflow group, and minute buckets to 24 hours.
`SPARK_CLUSTER_INFERENCE_RETENTION_REQUESTS` sets raw journal retention (default
50,000, minimum 100, maximum 1,000,000). Lifetime totals survive raw event pruning.
Prompts, generated content, request bodies, headers, credentials and query values
are never persisted. Model identifiers remain visible like the existing inventory.

The overview polls one small, bounded endpoint every two seconds while visible.
Resource samples reuse the existing snapshot and pool inventory. They expire after
15 seconds; unavailable GPU access is shown as offline, not zero measured usage.

## Runtime metrics and bypass coverage

The Prometheus scraper follows the actual active on-demand engine's internal port,
or the configured native runtime. An explicit `SPARK_CLUSTER_RUNTIME_METRICS_URL`
overrides discovery. It reuses its HTTP connection and resets rate baselines after
endpoint changes, failures, missing counters and counter resets. vLLM, SGLang and
llama.cpp token/queue aliases and modern vLLM KV occupancy are supported where
exported. Unsupported measurements stay null. Histogram percentiles are bucket
upper bounds. Runtime lifetime counters and rates are shown separately from routed
request accounting and are never added to it.

Clients calling a backend directly bypass the request observer. Only aggregate
engine metrics can cover that traffic while the engine remains available; move
clients to a pool or legacy listener for durable request-level history. Direct
backend traffic, stopped engines and older releases cannot supply historical
request details retroactively. A runtime without Prometheus still has complete
routed request outcomes and provider-reported usage.

`/metrics` exports `spark_cluster_inference_*` request/token counters, active/waiting
gauges and journal health. GPU monitoring in the controller container requires
NVIDIA's NVML library and devices (NVIDIA container runtime, `utility` capability).
Without that access the dashboard explicitly shows unavailable resource samples.

## Performance sheets and tiered history

Open **Performance history** from Inference activity, **Speed history** from the
throughput card, or a model/pool/GPU performance button in the library, recent
requests, model details or launch workspace. The same sheet switches between
Overall, Model, Pool and GPU, with ranges from one hour to 180 days. It refreshes
every ten seconds while open and visible; closing it stops reads. The read-only
endpoint is `/api/v1/inference/performance?scope=model&entity=org/model&hours=24`.
GPU entity identifiers are compact JSON arrays `[pool_id,gpu_id]`, so GPU index 0
in two pools never collides.

The sheet separates routed request measurements from runtime telemetry. Streaming
decode is weighted by measured decode duration, not an average of request rates.
Effective prefill is reported input tokens divided by dispatch-to-first-output
time; it includes backend scheduling and delivery, and is not a GPU kernel rate.
JSON/non-streaming responses cannot supply this timing. Cached input percentages
use only requests that report both cached and input tokens. Missing token usage,
cache data and timing stay unknown, and coverage counts are visible. Failed and
cancelled requests do not contribute successful timing/speed measurements.

Known local Docker runtime placements attribute scraper measurements to their
model, pool and exact assigned GPUs. Rates for simultaneous residents of one
scope are summed before temporal averaging; shared cache gauges are averaged.
GPU charts describe whole workloads assigned to the device, including shared
multi-GPU workloads, and must not be added together as independent traffic.
Remote controller requests retain model/pool attribution, but remote GPU placement
and runtime measurements are unavailable unless observed by that worker. Direct
backend traffic contributes only scraped runtime measurements. Hardware resource
utilization does not establish token throughput or cache occupancy.

Performance buckets share the separate `-inference.db` journal, using its background
writer and a lock for concurrent sheet reads. Requests and their rollups commit in
one transaction; retries cannot double-count a request. No database work is added
to the inference response path. Runtime samples use a bounded background queue.
History begins on upgrade; previously retained requests and lifetime counters stay
available in Activity but are not backfilled into performance charts.

Defaults, all configurable with the `SPARK_CLUSTER_` prefix:

| Setting | Default | Behavior |
| --- | --- | --- |
| `INFERENCE_RETENTION_DAYS` | 7 | Raw requests expire by age as well as the existing request-count cap. |
| `PERFORMANCE_MINUTE_DAYS` | 1 | Keep one-minute detail for this many days (1–7). |
| `PERFORMANCE_HOUR_DAYS` | 30 | Reduce older detail to hourly buckets until this age (7–90 days). |
| `PERFORMANCE_DAY_DAYS` | 180 | Reduce older hours to daily buckets, then expire (90–730 days). |
| `PERFORMANCE_MAX_ROWS` | 150000 | Bound total scope/time buckets; oldest buckets expire first under pressure. |

Maintenance runs at least once per minute while the controller is running, even
without new inference requests. Coarsening merges counts, sums, durations, known
measurement counts and peaks; it never averages existing averages. Range boundary
buckets are included whole and expose their actual timestamp and resolution. The
storage cap may shorten history. SQLite reuses pages released by expiry rather
than repeatedly vacuuming an active journal. Lifetime Activity counters survive
raw and bucket expiry. A crash can lose the unflushed second; persistence failures,
pending records and dropped records are shown in the sheet.
