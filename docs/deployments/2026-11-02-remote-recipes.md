# Remote recipe installation on DGX Spark — 2026-11-02

Implemented controller-to-node recipe delegation with signed requests, immutable release hashes,
node-owned persistent jobs and idempotent replay. Installation and launch are separate actions.
GB10 discovery uses measured unified RAM; unknown process memory attribution remains conservative.
Docker 29 manifest-based image identities are verified alongside classic config-based identities.

## Deployment

- Unraid controller: `spark-cluster:remote-recipes-9f664e1` (final UI over the validated bridge wheel).
- Image ID: `sha256:de74b865972ed121f1cb0fc870886e1242363fc30c7788edd9ab101b718c000f`.
- Spark bridge: updated wheel, `spark-cluster.remote-recipes/v1`, one GB10, 121.63 GiB.
- Wheel SHA256: `c34d65c53bf3b2f2ec90087825ebe044204e8013d82b1f10cd01fc3aad340fd8`.
- Controller rollback container: `spark-cluster-controller-before-20261102T110540Z`.
- Private rollback configurations and build logs remain in the existing deployment data directory.
- Bridge package and launch-script backup: `bridge/backups/remote-recipes-20261102` under its deployment.

## Live evidence

`BAAI/bge-small-en-v1.5` release
`975a2919258fb82187d9aed30c07b69e91a212d85fc8ee4eb69bccaaf30a8c4d`
installed through the Unraid controller onto the Spark:

- Controller job `21b8b6e1084541dbb5b02b41327616f6`.
- Node job `464695c26f284b6d98bda4ee25a4e4e9`.
- Completed stage `installed`, `launch=false`, artifacts present, inference not verified.
- Weights revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 401,223,680 logical bytes.
- Files stored under `/home/thibautrey/models/BAAI/bge-small-en-v1.5` and listed by node storage.
- Exact pinned ARM64 vLLM image prepared; existing Docker layers reused.
- Replaying the same node idempotency key returned the same completed node job.
- The initial live attempt exposed Docker 29's image-ID difference; it remains in job history.
  The subsequent completed attempt validates the correction.

A `Qwen/Qwen3-30B-A3B` install was rejected before transfer with HTTP 507:
61,151,296,255 required bytes versus 18,569,777,152 available bytes. The response identifies
the Spark disk and excludes the requested model from deletion suggestions.

An explicit launch of the installed embedding recipe was rejected by memory admission while
Qwen retained the GPU. Job `481d624fdf554247834abd197568de18` reports that files may remain
installed and memory is reserved. No second runtime was started. Real embedding inference
therefore remains unverified; generated templates retain their untested label.

The existing Qwen container stayed running with the same ID
`7b3972b72bfff6a9ce62e01ae03a22cb4b71e3dbf527293d88e50296d60ec506`.
A request through the controller's unified chat-completions endpoint returned `12` for `7 + 5`.
No existing weights were deleted and no unrelated containers were restarted.

## Validation

- 488 backend tests passed (one existing Starlette/httpx deprecation warning).
- 42 frontend tests passed; TypeScript and production build passed.
- Ruff passed.
- Source integrated into local main; no push.

The final UI also distinguishes memory-admission failures from disk-shortage cleanup actions.
