# Blazux Qwen3.8 Flash deployment — 2026-11-02

The Spark Cluster controller is deployed on Unraid (`192.168.1.149:18771`).
The exact [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
recipe serves `qwen3.8-flash-next` on the DGX Spark (`192.168.1.59`).

## Immutable inputs

| Component | Pin |
| --- | --- |
| Recipe source | `faefdb5e6fe9fa7ffe3899d8674746b91e6bb20e` |
| Base ARM64 vLLM image | `docker.io/vllm/vllm-openai@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc` |
| Built patched image | `sha256:f620527ca171f006e50b5a23e6279ea9ad749566f61e3c8536a60a5be08c852a` |
| Weights | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Weight revision | `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Controller executable source | `d6d9c76` |
| Controller image | `spark-cluster:recipes-qwen-d6d9c76` |

The source Dockerfile was changed to use the exact base digest above. Upstream
hybrid preparation succeeded: 300 tensors over nine shards, 1,836 FP8 side-layer
tensors, worst maximum relative error 0.0354 (upstream limit 0.10).

Serving uses upstream hybrid mode, YaRN, 500,000 context, MTP=2, eight sequences,
GPU memory fraction 0.80, prefix caching, deterministic top-k, and draft vocabulary.
A configured 500,000 context is not proof of a full-length 500,000-token test.

## Runtime integration

Deployment root on Spark:
`/home/thibautrey/spark-cluster-deployments/blazux-faefdb5e6fe9`.
The root contains source, pinned HF cache, compile cache, `pins.json`,
`deployment.json`, preparation and smoke-test logs, and the bridge configuration.

- Runtime container: `spark-cluster-demand-blazux-qwen38`, restart `unless-stopped`.
- Runtime API is loopback-only, `127.0.0.1:18300` on Spark.
- `spark-cluster-bridge.service` exposes the authenticated node bridge on port 30081.
- Controller pool registration: `remote-3b50a6f21ecd`, **Spark Qwen3.8**.
- Public model API: `http://192.168.1.149:18771/v1`, model `qwen3.8-flash-next`.
- Model-specific API: `/api/v1/pool/serve/c5c5640d676c90eb/v1` on the controller.

This is an imported, prepared runtime profile. Recipe v1 does not execute this
repository's custom image build and hybrid tensor transformation. The catalog's
bundled Qwen starter is a separate recipe; it must not be presented as this tested
deployment. Generic one-click execution of arbitrary GitHub recipes remains unsupported.

The private `bridge/profiles.json` captures Docker Config/HostConfig, immutable
image ID, exact weight revision, the served model alias, and fixed launch settings.
Profile id is `blazux-qwen38`, upstream port 18300, listener port 18301,
startup timeout 1,800 seconds. Persisted on-demand state retains this profile.
`Labels.spark-cluster.preserve-launch-settings=true` in the profile prevents
recovery from expanding the pinned eight-sequence configuration. The running
container has `spark-cluster.adaptive-batching=true` so adoption does not upgrade it.
Two bridge restarts were verified to preserve the same final container ID.

GB10's NVML memory query returns unavailable. The deployment's bridge launch
adapter supplies measured `/proc/meminfo` total/used unified memory to admission,
with empty managed attribution and `accounting_complete=false`. It never claims
complete GPU process accounting; additional sharing is blocked while Qwen is
resident. This adapter is deployment-specific, not generic GB10 telemetry support.
The controller's general telemetry panel is not validated as live GPU telemetry.

Bridge model storage is `/home/thibautrey/models`; HF_HOME is the deployment's
`hf` directory. Storage inventory lists the original DeepSeek pack/Engram and Qwen
cache. Qwen deletion is blocked by runtime references and mounted-container checks.
DeepSeek private sessions/state are outside the recognized weight entries.

## Observed validation

The upstream smoke test passed:

- Coherent answer and `reasoning_effort=high` compatibility.
- Cold prefill: 10,665 tokens in 5.20 s, about 2,051 tokens/s.
- Repeated prompt: 0.90 s; measured prefix-cache hits increased from 0 to 8,000.
- Identical first-token log probabilities across repeated deterministic requests.
- Decode: 400 tokens in 10.85 s, about 36.9 tokens/s including time to first token.

Real controller requests returned `4` for `2 + 2` and emitted a structured
`get_temperature` tool call with `{"city":"Paris"}`. The unified `/v1/chat/completions`
endpoint returned `12` for `7 + 5`, including after the final controller deployment.
These are synthetic validation prompts only.

Backend full run: 470 passed, one warning. Later changes passed model-pool tests
(30), adaptive-concurrency tests (20), and GPU admission tests (21).
Frontend validation from this deployment: 39 tests and production build passed.
UI inspection through a temporary authenticated read-only loopback proxy confirmed
Spark storage entries, deletion protection, and versioned catalog controls. No
existing weights were deleted to test deletion. NAS scans can remain pending;
UI/API report that state without blocking the inventory response.

## Space and rollback

Approximately 17.7 GiB remained free on Spark after preparation and compilation.
No old weights or private DeepSeek KV/session state were deleted. DeepSeek's
`dsv41` container and its watchdog are stopped; they must not run alongside Qwen
in the same unified memory budget. Rollback inspection and previous health records
are private under the deployment root's `rollback` directory.

To roll back serving, drain Qwen requests, remove its retained state, stop its
bridge and Qwen runtime, then start `dsv41` and its watchdog. Verify health before
restoring the former controller registration. The former unreachable port-30080
registration was retired; its descriptor was backed up privately on Unraid and
its runtime credentials were retained. Never print or commit those credentials.

Unraid retains stopped controller containers for rollback. A consistent database
backup and subsequent private registry/container descriptors are under controller
data/deployment backup directories. No Git push was performed.
