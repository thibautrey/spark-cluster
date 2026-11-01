# Durable native DeepSeek serving

The controller proxies inference to the attached native engine. Its model catalog,
request history, metrics, and app-prefix profiles survive controller restarts.
The engine owns token matching and KV compatibility; a successful HTTP response
from `/v1/models` alone does not establish engine health. Check `/health` for
`ok: true` and no `fatal` state. An unavailable attached model returns HTTP 503
with a retry hint rather than starting an unrelated download.

## Cache layers and limits

- Prepared weights remain on the inference host's NVMe. A clean reload should
  report prepared files read, with zero built or written.
- `TF_DSV41_SESSION_DURABLE=1` checkpoints immutable KV pages to NVMe while
  retaining the hot RAM copy. `TF_DSV41_SESSION_DELTA=1` shares validated ancestor
  pages. The configured NVMe budget is 64 GiB, with dependent-aware LRU eviction.
- `TF_DSV41_SESSION_NAMESPACE` identifies the model, quantization, and runtime
  cache ABI. Change it whenever those properties change. Never force reuse of
  incompatible state. The indexed directory also depends on the physical layout.
- Prefix checkpoints and conversation prompt checkpoints have distinct semantic
  tags. The reader rebuilds legacy cross-tag deltas using their full token prefix
  and verifies their original identity. New writers avoid cross-tag delta bases.
- `TF_DSV41_REASONING_MEMORY_DIR` must be inside the same model/runtime's private
  state namespace. The server restores omitted assistant reasoning by tool-call
  identity or content digest. Its atomic, checksummed store is bounded to 1,024
  entries and 16 MiB, with 1 MiB per reasoning entry. Directories are 0700 and
  files 0600. This contains private conversation state; do not publish it.
- The controller stores at most 12 private warmup profiles, 2 MiB each. The tool
  catalog and exact thinking/reasoning parameters are retained; initial text is
  bounded to 32,768 characters. Short user questions, tool results and image URLs
  are not replayed. Warmups generate one token while idle and never execute tools.
  Runtime uptime resets re-arm warmup. `/api/v1/inference/cache-profiles` exposes
  metadata and usage counters without prompt bodies.

Caches are bounded, not permanent archives. Changed prompts, tools, reasoning
settings, cache identity, or evicted entries require computation. Even an exact
restore can require a small tail to be evaluated. A large client tool catalog
has different latency from a short synthetic prompt.

## Applying and validating the engine patch series

Use the reviewed kvfix4 source export, preserving a pristine copy. The pinned
external recipe is commit `3c4340f4532ae526da83f6bccdbf15fc6aa6a620` of
`Loke-60000/deepseek-v4.1-1017-1xGB10-recipe`. Repository claims are reference
measurements, not measurements of this deployment.

```sh
python3 deploy/apply-dsv41-runtime-patches.py /path/to/isolated/source
zig build test-kv -j2 --summary all
zig build test-server -j2 --summary all
```

Build the `serve-dsv41` target with the host's CUDA toolchain and SM121 settings.
Wait for terminal build success before copying the binary. Record its SHA-256,
then verify the same SHA inside the running container. The source patch helper
supports repeated application to the fully patched export.

CPU gates cover bit-identical KV restoration, restart reconstruction of cross-tag
legacy delta chains, checksummed reasoning persistence, and tool/content history
restoration. Live gates must additionally verify real client usage, engine health,
prepared-cache reuse, and restoration after a full host reboot.

## Deployment and recovery

`deploy/replace-inference-container.py` preserves the existing Docker settings and
writes a private inspect backup. It requires a healthy, idle inference observation
before replacing a running container. Controller health is `/healthz`, on the
configured listen address; it need not bind the loopback address. Use
`--idle-url` to check the actual inference engine when replacing the controller.
Use a sufficient stop timeout for request drainage.

`--replace-stopped` is reserved for recovering a known failed, stopped candidate.
It refuses a running container and restores the original stopped state if the new
candidate fails. Never use it to bypass an active request. A startup observation
deadline leaves a still-running process alone; it is not evidence of failure.

The watchdog restarts only its configured inference container after three explicit
fatal health observations. It has a five-minute cooldown and at most three
recoveries per hour. Failed observations, a slow load and an intentional stop do
not trigger recovery. Do not restart the Docker daemon or unrelated runtimes.

Keep the previous image and stopped rollback container. If rolling back to a reader
that cannot reconstruct newer checkpoints, preserve those files and disable session
reuse for the temporary recovery; do not delete or reinterpret the cache. Updating
only the reader and private reasoning persistence does not change the KV page ABI.

## Storage and benchmark checks

Keep enough free NVMe space for the full configured KV budget, initialization and
Kubernetes eviction thresholds. An old prepared namespace, incompatible KV cache,
or original conversion input can be archived only after verifying that the serving
process does not map it and that its active mounts use the retained converted data.
Verify the NAS copy before removing the source. Record exact restore locations
privately. Do not archive active prepared files, converted expert weights or Engram.

Benchmark after transfers and warmups finish. Report cold initialization, cold
prompt prefill, warm TTFT and warm decode separately. For concurrency, report both
per-stream speed and aggregate speed. Compare only equivalent prompt lengths,
output lengths, sampling and reasoning settings. Keep synthetic fixtures separate
from real Codex, OpenCode and LibreChat proof. Public recipe figures are roughly
29–30 tokens/s single-stream and 92–97 tokens/s aggregate for four warm greedy
512-token streams; they do not promise those latencies for large app prompts.
