# Runtime-owned inference queues — 2026-11-02

Executable source: `930aead`, integrated into local main without pushing.

The adaptive gateway previously started at four requests and could hold a batching
engine below its configured capacity. Its latency/error/KV feedback duplicated the
runtime scheduler, using observations across heterogeneous request sizes. vLLM and
SGLang already schedule their token and KV budgets; llama.cpp already schedules its
configured slots and continuous batches. An extra learned concurrency gate can
prevent these engines from seeing enough work to batch efficiently.

Known engines now receive requests up to the existing machine-wide 64-outstanding
bound, including requests exceeding their execution slots. The engine owns waiting
and batching. Unknown Docker backends retain the adaptive fallback. Startup waits,
request pinning, cancellation, readiness supervision and recovery remain enforced.
Existing engine launch settings are preserved, including llama.cpp slots/context;
no running engine is replaced just to enlarge its execution ceiling.

## Deployment

- Unraid controller: `192.168.1.149:18771`.
- Image: `spark-cluster:runtime-queue-930aead`.
- Image ID: `sha256:9e666bff3df00c971c7776dcac641fcac90b45f6d50de7e0eecbd507bc57874f`.
- Image derives from the previously running `spark-cluster:throughput-b76fd11`,
  replacing only `adaptive_concurrency.py`, `on_demand.py`, and `model_pool.py`.
- Previous container: `spark-cluster-controller-before-runtime-queue-930aead`.
- Private descriptor and deployment scripts:
  `/mnt/user/appdata/spark-cluster-controller/deploy-runtime-queue-930aead/`.
- Controller shutdown allowed 1,900 seconds for requests to drain before replacement.
- Spark bridge: same three modules replaced in the existing bridge virtualenv;
  previous modules retained with `.before-runtime-queue-930aead` suffix.
- Bridge restarted with its existing 1,900-second systemd stop grace.
- Bridge and local adaptive module SHA-256 matched:
  `c5b3c2b698f201f5678885175a542a93398f7a51d6021cef7a0aef3df38e8df6`.
- Qwen runtime `7b3972b72bff` remained running (12 hours uptime).
- The three local runtime container IDs remained unchanged.

## Verification

On local main: 505 tests passed, one existing FastAPI/HTTPX deprecation warning;
Ruff and `git diff --check` passed. Regression coverage checks forwarding eight
requests above a two-slot ceiling, ignoring old learned limits, overload/KV feedback,
and runtime reconfiguration without reinstating the adaptive gate. Existing tests
continue covering machine admission, cancellations and active-request pinning.

Live controller health and Docker healthcheck passed. Pool status reports
`runtime-managed` for both local vLLM embedding services and remote Qwen. Speech
retains adaptive fallback. Qwen reports gateway limit 64 and engine ceiling 8.

Two twelve-request streaming bursts through the public controller API completed:

| Burst | Output tokens | Wall time | Result |
| --- | --- | --- | --- |
| First | 1,332 | 8.522 s | 12 HTTP 200, usage and final DONE |
| Monitored | 1,332 | 7.336 s | 12 HTTP 200, usage and final DONE |

The monitored burst reached 12 forwarded requests at once with no gateway waiting,
above Qwen's eight execution slots. This demonstrates that runtime queueing is
reachable; it does not mean 12 simultaneous executing sequences. Two requests to
`BAAI/bge-small-en-v1.5` returned HTTP 200 and 384-dimensional embeddings.

These short synthetic prompts validate the deployed scheduling behavior, not a
controlled before/after performance comparison or long-context capacity. No generated
response text was retained. No live llama.cpp instance was exercised in this check.
