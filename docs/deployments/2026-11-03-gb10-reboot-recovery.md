# GB10 retained model recovery — 2026-11-03

After the host reboot, the bridge was active but Qwen was absent. Its persisted
`blazux-qwen38` retained flag survived. Recovery repeatedly failed GPU admission
before creating a container: the default 82% usable budget minus measured host RAM
usage was smaller than the unchanged 80% runtime reservation. GB10 shares RAM;
NVML does not expose separate device memory on this host.

The deployment's `bridge/launch.py` now sets
`model_usable_memory_fraction=0.85`. Actual measured external use is still
subtracted, and incomplete process attribution still prevents sharing a resident
GPU. Qwen's 80% engine fraction, eight sequences and 500,000 context are unchanged.
This is a host-specific admission setting, not a global default change.

Source fix `e5b4566` also reconciles missing retained models without requiring a
previous in-memory failure record. Admission failures now use the existing
exponential backoff instead of retrying and logging every second. Unretained
models remain demand-loaded; explicit unload removes retention.

The corrected `on_demand.py` was installed in the GB10 bridge virtualenv and the
bridge restarted. Previous module and launch script are retained beside their
active files with `.before-reboot-e5b4566` suffixes. No Docker daemon restart,
host reboot, unrelated container restart, or Git push was performed.

Targeted validation on local main: 54 tests passed; Ruff passed. Regression tests
cover reconstruction from persisted retention without failure history, admission
backoff, and disabling retention before another retry.

Full validation on local main: 510 tests passed, one existing Starlette warning.
Installed module SHA-256 matched local source:
`be496393061ed5e62e2dac0bb5a9d46e969220162c058688345e01ed2430f10a`.
The bridge automatically recreated the missing Qwen container. Following cold
weight loading and engine initialization, `/health` returned HTTP 200. A request
through the public controller `/v1/chat/completions` returned HTTP 200, model
`qwen3.8-flash-next`, answer `4` for `2 + 2`, and usage 26 prompt / 2 output tokens.
An earlier request during initialization returned 503. No second physical host
reboot was performed; reboot reconstruction is covered by the regression test,
and missing-container recovery was observed live after the bridge restart.

## Follow-up: residency independent of retention

A later host boot exposed a separate lifecycle gap. Qwen was absent and the
persisted node state contained `retained: []`; the central pool had no Qwen
retention preference. Docker runtimes created by the on-demand path forcibly used
restart policy `no`, overriding the imported recipe's `unless-stopped` setting.
The earlier recovery fix only covered retained profiles, so it had no desired
model to restore in this state. The observation does not establish which previous
client operation cleared retention.

Source through `6f1527f` now persists successful Docker residency separately from
retention. Host/bridge startup restores recorded models using the same saved
profile and admission/backoff checks. Explicit unload, idle unload and eviction
remove resume intent. Clean controller shutdown preserves it. Retained runtimes
also receive Docker `unless-stopped` supervision when loaded or retained.
An adopted Docker process remains `starting` until readiness/preload succeeds,
with the full configured cold-load deadline; ordinary health probes cannot
mistake its weight loading for a runtime failure and repeatedly recreate it.

Qwen was reloaded through the public controller pool and Keep loaded enabled in
both the central preferences and node state. Its exact model revision, 500,000
context, eight sequences and 80% engine memory fraction were preserved. The
updated node modules were installed in the existing Spark bridge virtualenv;
original modules are retained with `.before-resume-08ee20e` suffixes. No other
runtime or Docker daemon was restarted. The central controller was not replaced.

Live verification after the final bridge restart:

- Qwen `/health`: HTTP 200; node reports `ready` and no error.
- Public controller `/v1/chat/completions`: HTTP 200, model
  `qwen3.8-flash-next`, answer `7` for `3 + 4` (26 prompt, 2 output tokens).
- Two bridge restarts preserved container
  `7f3d28dbdb4618ea848c2d0dc3d0458b115ccfb1a7c03d0f707b020225fef8cf`,
  started `2026-11-03T16:33:42.467932231Z`.
- Docker restart policy is `unless-stopped`; bridge is enabled with
  `Restart=on-failure`; persisted `loaded` and `retained` both contain
  `blazux-qwen38`.
- Installed/local SHA-256: `on_demand.py`
  `66df17d2f81b82179ac83d3af8e3e369b83c756b14b2db67db30184a22458b71`;
  `model_pool.py`
  `804d95e737fee1a9f0678955437d2bbed076023c3d3bdb429bf0c1e761ce023f`.

Regression coverage includes clean/crash restart reconstruction without retention,
explicit unload of an absent model, eviction, idle expiry, admission backoff,
Docker restart policy, and cold Docker adoption without recreation. No additional
physical host reboot was performed; that complete hardware gate remains untested.
Changes are integrated into local main without pushing.

Final validation on local main: 518 tests passed with one existing Starlette
HTTPX deprecation warning; Ruff lint/format and `git diff --check` passed.
