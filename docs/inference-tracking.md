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
Missing usage stays unknown and is counted explicitly. Nothing is inferred from
response length, SSE chunk count, or GPU activity. Earlier unrecorded requests
cannot be reconstructed. Historical records survive runtime unloads and restarts.

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
