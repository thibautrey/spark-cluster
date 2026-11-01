# GB10 DeepSeek verification — 2026-11-01

The warm deployment reaches the published single-GB10 reference range. This is a
single Dell GB10 serving DeepSeek-V4.1-Flash-1017 through TensorFold, with the
controller and MultiVibe on Unraid. All measurements below are live measurements,
not capacity estimates or dashboard simulations.

## Comparable reference benchmark

Reference: [the pinned recipe](https://github.com/Loke-60000/deepseek-v4.1-1017-1xGB10-recipe/tree/3c4340f4532ae526da83f6bccdbf15fc6aa6a620),
`recipe/bench/speed.py`, prose prompt about the history of navigation at sea.
Greedy sampling, thinking disabled for this benchmark only, 512 generated tokens
per request, no overlapping inference or file transfer.

| Measurement | Observed | Published reference |
| --- | ---: | ---: |
| First request for this prose shape, decode | 19.5 tokens/s | Warm figures only |
| Warm single stream, decode | 29.39 tokens/s | 29–30 tokens/s |
| Warm single stream, first token | 0.419 s | 0.3–0.4 s |
| Four streams, aggregate including first-token delay | 94.21 tokens/s | 92–97 tokens/s |
| Four streams, decode per stream | 24.07–24.55 tokens/s | 24–25 tokens/s |

Exactly six requests completed, as expected; zero runtime errors. All six outputs
had SHA-256 `03df0fd9d082c794e569f304577e88d326664a2495681fabd78261beaef95aca`.
The engine remained healthy, idle and without a fatal state afterward. GPU
observation during four-stream execution: P0, 57 degrees C, no reported active
clock-event reason at the observation instant.

These are workload-specific warm results. A separate short rain-explanation
prompt measured 20.74 tokens/s single-stream and 57.26 aggregate, with differing
output hashes across executions. It had no reused KV tokens. Do not extrapolate
one prose benchmark to every prompt or claim universal bit-identical model output.
The recipe also documents slow expert-page-cache recovery after a fresh long
prefill. The first request for a shape remains slower than its warm repeat.

## Full host reboot and real clients

A changed kernel boot ID confirmed a full Dell reboot. Docker restarted the
inference container, and the watchdog started automatically. The engine restored
89 NVMe checkpoints and five private reasoning-memory entries. It read all 41
prepared-weight files (111.2 GiB), built zero and wrote zero. Weight loading took
172 seconds on this cold boot; KV persistence does not eliminate weight loading.

| Client after reboot | Input tokens | Restored tokens | Server first token |
| --- | ---: | ---: | ---: |
| Codex CLI through MultiVibe Responses | 234,347 | 234,336 | 3.760 s |
| OpenCode through the controller | 7,516 | 7,504 | 1.173 s |

The earlier fresh Codex catalog had required approximately 581 seconds to its
first token. Complete ephemeral CLI runs also include client and MCP startup:
18.31 seconds for the post-reboot Codex marker test and 10.34 seconds for OpenCode.
Server first-token measurements above exclude that client initialization.

Real Codex shell execution and OpenCode's normal interactive `read` tool both read
a dedicated synthetic fixture and returned its exact content. Headless OpenCode
rejects unresolved permission prompts; the successful interactive test used no
`--auto` flag and changed no global permissions. LibreChat returned `42` for both
17+25 before reboot and 21+21 afterward. Its original new-chat agent was restored.
An initial literal-marker test in LibreChat contained a spelling error, so that
literal test is not counted as passing. This campaign validates transport, cache,
restart behavior and tool protocols rather than general model-answer accuracy.

A separate synthetic tool history deliberately omitted assistant reasoning from
its replay, as clients can do. Before and after the host reboot it returned `42`
with identical output SHA-256. After reboot it restored 3,712 of 3,726 input tokens.
This exercises the durable reasoning memory as well as KV restoration.

## Client context experiment

A one-off Codex run with `features.apps=false` reduced the initial context from
about 234,000 to 30,000 tokens. Both variants performed the same fixture-read task
correctly. The warm experiment took 39.20 seconds versus 79.71 seconds with all
connectors, although generated reasoning/output lengths differed (563 versus 462
tokens), so this is an observed task duration rather than a controlled speed ratio.
The first uncached reduced-context run took 87.10 seconds.

No default model, reasoning level or connector configuration was changed. An
optional code-only profile can trade connector availability for smaller context;
it is separate from the normal full-tool workflow.

## Deployment and limits

- Controller image: `spark-cluster:reliability-9c4e216`.
- Engine image: `dsv41:durable-20261101e`.
- Engine binary SHA-256, verified inside the live container:
  `e2ae06c217ce9d889f416353b26ae0c920f3e9541861829b2d8a49e7419bae38`.
- Four slots, 1,048,576 context ceiling, 2,097,152-token shared KV pool, 64 GiB
  durable KV disk budget and 2 GiB hot snapshot budget. The shared pool is a
  shared capacity bound, not four independent million-token allocations.
- Draft/speculative decoding disabled; existing weight quantization and
  `TF_DSV41_EXPERT_TOPP=0.85` preserved.
- Archived inactive prepared weights, incompatible old KV files and an unused
  original conversion shard on the NAS. Verified transfers completed before
  source removal. Active converted weights and Engram remain local. Free NVMe
  space afterward: 241 GiB; Kubernetes disk pressure cleared and telemetry is live.

Known model identities now remain discoverable during engine unavailability,
with unavailable status and explicit HTTP 503 inference responses. Discovery no
longer waits unboundedly for a secondary pool's inventory: an observed 22-second
catalog delay was reduced to approximately three seconds in the live test.
MultiVibe recovered its catalog automatically; its account settings were not changed.

See [the cache and recovery runbook](durable-native-runtime.md) for namespaces,
privacy, eviction, prewarming and guarded deployment. These caches are bounded;
changed prompts or settings, incompatible model/runtime identities and evicted
entries still require computation.

## Source and validation

The Spark source changes are integrated into main and published. The engine
changes are reproducible patch files under `deploy/runtime-patches/`; their source
export and build inputs are separately preserved. Validation:

- 445 Python tests passed; Ruff check and format checks passed.
- Native KV tests: 44 passed, including cross-tag legacy-delta reconstruction and
  bit-identical restored host pages.
- Native server tests: 71 passed, two skipped; JSON and engine test targets retained
  their previously passing cached results.
- [GitHub CI for 9c4e216](https://github.com/thibautrey/spark-cluster/actions/runs/41309177015)
  passed, including the full Docker build.

Private client transcripts, screenshots, transfer records and numeric benchmark
artifacts remain in the local audit directory; they are not published with source.
