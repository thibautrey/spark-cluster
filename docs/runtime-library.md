# Runtime library

Open **Runtimes** in the dashboard. Spark maintains a curated catalogue of public
inference runtimes, discovers their image versions, caches a selected image on
the controller, and imports it on explicitly selected machines. No model is
started, stopped, or reconfigured by an installation.

## Catalogue and sources

| Runtime | Main use | Official distribution repository |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput LLM inference | `docker.io/vllm/vllm-openai` |
| [SGLang](https://github.com/sgl-project/sglang) | LLM and multimodal serving | `docker.io/lmsysorg/sglang` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Portable GGUF inference | `ghcr.io/ggml-org/llama.cpp` |
| [Ollama](https://github.com/ollama/ollama) | Local models and serving | `docker.io/ollama/ollama` |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA optimized LLM inference | `nvcr.io/nvidia/tensorrt-llm/release` |
| [Triton](https://github.com/triton-inference-server/server) | Multi-framework inference | `nvcr.io/nvidia/tritonserver` |
| [TGI](https://github.com/huggingface/text-generation-inference) | Hugging Face LLM serving | `ghcr.io/huggingface/text-generation-inference` |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | TurboMind / PyTorch inference | `docker.io/openmmlab/lmdeploy` |
| [Xinference](https://github.com/xorbitsai/inference) | Multi-engine serving | `docker.io/xprobe/xinference` |
| [LocalAI](https://github.com/mudler/LocalAI) | Local OpenAI-compatible APIs | `docker.io/localai/localai` |

These are widely used projects, not a popularity ranking. MLX / MLX-LM is also
an important inference family on Apple Silicon; its native macOS package lifecycle
is outside this Linux container installer. Ray Serve, BentoML and KServe are
serving/orchestration layers; ONNX Runtime, OpenVINO and PyTorch are lower-level
execution frameworks. They are not interchangeable container launch contracts.

Versions are **published distribution tags**, including stable releases, backend
variants, and rolling/nightly tags. Tags are discovered live, never inferred from
a hardcoded version list. Docker Hub pages are ordered by last update; GHCR/NGC
use registry order. Use **Next versions** for further pages or type an exact tag.
Platform information absent from a version page is resolved before downloading.
Version pages are cached for five minutes; an expired page returned during a
registry outage is explicitly marked stale. Registry errors are never an empty
successful version list.

## Controller download

The controller requires `skopeo`, supplied by the Docker image. For a native
installation, install it through the host package manager. Configure
`SPARK_CLUSTER_MODEL_ADMIN_TOKEN` before allowing mutations. All library routes
use the existing administrator authentication and controller signature checks.

By default archives live in `<model_cache_path>/.runtime-images`, on the large
model-storage volume rather than the small state/database volume. Override with
`SPARK_CLUSTER_RUNTIME_CACHE_PATH` if needed. Persist that directory and the
controller database together. The cache directory is ignored by model inventory.

Choose `linux/arm64` for DGX Spark or `linux/amd64` for x86 servers. Spark verifies
the registry manifest, its image configuration, and the selected platform, then
pins the **platform-specific SHA256 manifest digest** before invoking skopeo.
The controller produces a Docker image archive with its original repository/tag,
records its checksum and size, and publishes it only after a successful copy.
It reserves a conservative disk-space allowance; layer expansion can still exceed
this estimate and will fail the job without publishing an incomplete artifact.

Downloads are serialized, duplicate in-flight requests reuse a job, and an
existing artifact is reused only after checksum verification. A moved upstream
tag resolves to a new digest and creates a new artifact. Failed partial downloads
are removed. Jobs and artifact metadata survive restart; interrupted jobs fail
explicitly rather than replaying automatically. This service assumes one
controller process owns a cache/database; it has no distributed writer lock.

## Machine installation

The cached artifact and destination pool are explicit. The API additionally
accepts a subset of the pool's managed SSH addresses. Trust comes from the
existing provisioning fingerprints, SSH identity and known-hosts file. A bridge
registered without locally managed SSH nodes cannot receive imports through this
path; provision/trust its nodes first. Jobs never fall back to another pool.

Every selected SSH machine must report the artifact's Linux architecture before
the first import. The archive is streamed over trusted SSH to a private temporary
file, verified with `sha256sum`, and imported into k3s/containerd's **k8s.io**
namespace or Docker. Containerd's exact image reference must be present after
import. The controller archive checksum and its config blob digest bind the downloaded
configuration; Docker must expose the imported tag with the selected platform. Docker's
containerd image store can give its imported manifest a different local ID, so Spark does
not compare that transformed ID to the remote registry config digest.
Temporary files are removed, including a best-effort cleanup after disconnects.
CRI-only machines without an accessible containerd or Docker import command are
unsupported. Container storage and `/var/tmp` need sufficient disk capacity.

For a local Docker pool, the controller streams the archive through Docker Engine's
image-load API, checks error events even when HTTP status is 200, and verifies
the resulting tag and image platform. It checks the daemon architecture before
loading. The archive never needs to be buffered fully in controller memory.

Imports are serialized. Each SSH node has an independent outcome, so a failure
does not erase successful nodes. A completed job proves image preparation only:
it does not prove CUDA/ROCm drivers, SM121 kernels, model format support,
TensorRT engine compilation, multi-node inference, or endpoint readiness.

The existing vLLM and single-node SGLang launch adapters remain separately
configured using their pinned image settings. Importing a new tag does not switch
running models or implicitly enable a launcher. Configure and validate that
engine's launch image separately. llama.cpp, Ollama and the other listed engines
can be cached/imported but have no universal Spark launch adapter yet.

## API

All paths below are controller-local, including when a remote pool is active.

| Method and path | Purpose |
| --- | --- |
| `GET /api/v1/runtime-library` | Catalogue, artifacts, recent jobs and downloader availability |
| `GET /api/v1/runtime-library/runtimes/{id}/versions?cursor=&limit=50` | Paginated published tags and continuation cursor |
| `POST /api/v1/runtime-library/resolve` | Resolve a runtime/version/platform to immutable image metadata |
| `POST /api/v1/runtime-library/downloads` | Queue controller download; returns HTTP 202 and durable job |
| `POST /api/v1/runtime-library/installs` | Queue cached image import; returns HTTP 202 and durable job |
| `GET /api/v1/runtime-library/jobs/{id}` | Inspect phase, progress, artifact ID and per-node outcomes |

Download/resolve JSON: `{"runtime_id":"vllm","version":"v0.9.2","platform":"linux/arm64"}`.
The version is an example, not a compatibility recommendation. An absent platform
is rejected. Installation JSON:
`{"artifact_id":"<32 hex characters>","cluster_id":"<pool id>","addresses":["<managed node>"]}`.
Omit `addresses` to import on every SSH-managed node in that pool, or for a Docker
pool to import once on its host. Runtimes and source repositories are curated in
`runtime_catalog.py`; arbitrary external image URLs and shell commands are not
accepted by these endpoints.

The separation between runtime definitions, versions and launch contracts is
inspired by [GPUStack's inference backend catalogue](https://github.com/gpustack/gpustack/blob/main/gpustack/schemas/inference_backend.py).
Spark uses its own controller-owned archives, pool registry and pinned SSH trust.
