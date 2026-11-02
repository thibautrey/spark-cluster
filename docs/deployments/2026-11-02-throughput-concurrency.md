# Throughput-first request admission — 2026-11-02

Source commit: `b76fd11` (local main, not pushed).

The previous controller tested a latency regression before evaluating aggregate
throughput. A single slow measurement window could therefore halve admissions
even when parallel execution increased total output tokens per second.

The new policy starts from four admissions, bounded by the engine ceiling. It
evaluates aggregate throughput first, accepts gains above 3% despite slower
individual responses, and requires two comparable latency-regression windows
without a throughput gain before latency backoff. Engine overload and sustained
KV pressure still reduce admissions. Recovery probes resume after eight successful
observations. Versioned learning revalidates legacy limits below four while keeping
newly measured serial limits valid across restarts.

## Applied on Spark

- Bridge: `spark-cluster-bridge.service` on `192.168.1.59`.
- Updated module inside the existing bridge virtualenv:
  `/home/thibautrey/spark-cluster-deployments/blazux-faefdb5e6fe9/bridge/venv/lib/python3.12/site-packages/spark_cluster/adaptive_concurrency.py`.
- Previous module retained at the same path with suffix `.before-b76fd11`.
- Deployed module SHA-256:
  `c432213b7a8733a7da7d246ed26143b0fcdf5cdb28a516c3206b5847f150dbaf`.
- The bridge was idle before restart. A runtime-only systemd drop-in at
  `/run/systemd/system/spark-cluster-bridge.service.d/throughput-grace.conf`
  extends shutdown grace to 1,900 seconds for any request racing the idle check.
- Runtime `spark-cluster-demand-blazux-qwen38`, ID `7b3972b72bff`, stayed running.
- Pool status exposes `learning_version: 2`, initially `limit: 4`, then `limit: 5`
  after real requests, with `runtime_ceiling: 8` unchanged.

## Live checks

Streaming requests through `http://sparkcluster.lan/v1/chat/completions`, model
`qwen3.8-flash-next`, thinking disabled, 160 output tokens per request:

| Requests submitted together | Output tokens | Batch wall time | Aggregate output tokens/s |
| --- | --- | --- | --- |
| 4 | 640 | 6.198 s | 103.27 |
| 8 | 1,280 | 11.652 s | 109.86 |

All twelve responses returned HTTP 200, provider-reported token usage and `[DONE]`.
Batch rates include admission waiting and network time. Eight requests submitted
together do not imply eight simultaneous engine executions: the adaptive gate
queues requests beyond its current limit. These short synthetic arithmetic-list
requests demonstrate concurrent service, not long-context performance or a
universal optimum. Generated response text was not retained.

Validation on local main: 501 backend tests passed (one existing FastAPI/HTTPX
deprecation warning), including 28 adaptive-concurrency tests. Targeted Ruff checks
and `git diff --check` passed.

## Unraid controller

Built but **not activated**: `spark-cluster:throughput-b76fd11`, image ID
`sha256:baabe5f424d67807ca26b60b18526c5507846ce47f45f806f2b2dae4b8828b09`.
It derives from the running `spark-cluster:machine-queue-e856bb7` image and replaces
only the admission module. The controller was left running because it was serving
long embedding requests. Its local runtimes therefore still use the previous
learning policy. Qwen admission is executed by the updated Spark bridge and is
already effective through the unchanged routing controller.
