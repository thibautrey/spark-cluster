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
