# Inference reliability

Spark borrows two operational patterns from GPUStack: check an engine after it
becomes ready, and back off recovery rather than restarting on every request.
The implementation is independent and preserves Spark's GPU reservations,
adaptive admission, retention preferences and controller-independent runtimes.

## Managed Docker runtime recovery

The background monitor probes each idle resident's configured `readiness_path`.
Defaults are a 15-second interval, a 3-second timeout, and three consecutive
failures before marking the runtime `unhealthy`. Profile fields
`health_interval_seconds`, `health_timeout_seconds` and `health_failure_threshold`
adjust these bounds. Only successful HTTP responses establish readiness; an
authentication rejection is not proof of readiness. Use a runtime health endpoint
that the controller can access.

Probes do not hold the global scheduling lock and never retire a runtime with an
active request. A successful probe resets consecutive health failures. Unhealthy
residents stop being advertised as ready and retain their GPU reservation until
their managed container is removed. Idle, unretained unhealthy runtimes can also
be evicted to admit another model.

Failed startups, dead containers and failed readiness checks start a per-profile
cooldown: 5 seconds, then exponential backoff up to 300 seconds. A runtime must
remain ready for 60 seconds and pass a probe before its failure budget resets.
Cancellation does not count as an engine failure. Cooldowns and failure counts
are process-local and visible in `/api/v1/inference/status`; they reset on a
controller restart.

Retained models recover in separate background tasks after the cooldown, so a
slow model load does not stop health monitoring. Unretained models recover on
their next request. Recovery does not change retention preferences, replay
inference requests, force-stop active streams, or mutate unmanaged containers.
An initial failed load that has never established retention remains an explicit
load failure. GPU capacity and existing placement rules still apply.

## Gateway behavior

The global `/v1`, pool and legacy Docker gateways reject request bodies above
64 MiB while reading them. Forwarding uses a 5-second connection/pool timeout,
30-second write timeout, 900-second idle read timeout, and existing 1800-second
request deadlines. These allow long generation while bounding unreachable peers.
Connection failures return an OpenAI-shaped 503; upstream timeouts return 504.
Queue saturation and recovery cooldowns include `Retry-After` where available.
Backend error responses retain their original status and body.

Hop-by-hop headers, including custom names listed in `Connection`, are filtered
on both legs. End-to-end authentication, request IDs and raw streaming bytes are
preserved. A backend failure after response headers closes the incomplete stream;
it cannot manufacture a success terminator or send a second HTTP status. POSTs
are never automatically replayed, including failures before response headers.

A disconnected client waiting for managed Docker startup or adaptive admission
releases its queue position and reservation. The disconnect watcher stops before
streaming starts so it does not compete with Starlette's receive loop. Existing
stream leases last through completion, failure or client cancellation.

These recovery checks cover controller-managed Docker residents. Native and
Kubernetes runtimes retain their existing lifecycle supervision; direct-runtime
traffic bypasses the gateway. A health endpoint alone cannot prove model output
quality or distributed inference correctness. Automated tests exercise faults and
concurrency; an actual model call is a separate integration check.

## Source study

Reviewed GPUStack commit `4aaf598ee12d656115ba59acaf16577e77e17010`:

- [ServeManager health supervision](https://github.com/gpustack/gpustack/blob/4aaf598ee12d656115ba59acaf16577e77e17010/gpustack/worker/serve_manager.py#L1179):
  periodic probes, consecutive-failure thresholds and successful-traffic signals.
- [ServeManager restart backoff](https://github.com/gpustack/gpustack/blob/4aaf598ee12d656115ba59acaf16577e77e17010/gpustack/worker/serve_manager.py#L2806):
  bounded exponential delays and rechecking state before recovery.
- [Server controllers](https://github.com/gpustack/gpustack/blob/4aaf598ee12d656115ba59acaf16577e77e17010/gpustack/server/controllers.py):
  explicit readiness/unreachability states and reconciliation after missed events.

Spark uses cheap readiness probes by default. GPUStack's opt-in inference probes
are not enabled here because they spend model capacity and require workload-specific
requests. Spark's regression and integration checks send actual completion traffic.
