# Runtime library

Open **Runtimes** in the dashboard. The default view recommends the newest stable
runtime compatible with the selected compute pool. **Set up** resolves the release to
an immutable digest, caches it, installs it, activates it for that pool and verifies the
result in one resumable workflow. A model launch uses the same workflow and continues
through a real inference request before reporting **Ready**.

The active image is stored per compute pool and engine. It takes precedence over the
legacy environment setting without rewriting deployment configuration or requiring a
controller restart. Existing environment images remain available until an administrator
chooses the guided update. Stable selection accepts final semantic releases and excludes
development, nightly, main and release-candidate channels.

The **Advanced runtime packages** disclosure is a separate distribution tool. Spark
maintains a curated catalogue of public runtimes, discovers their image versions,
caches a selected image on the controller, and imports it on explicitly selected
machines. A cached or imported package is not evidence that Spark has an executable
launch adapter for it. No model is started, stopped, or reconfigured by a package
installation.

Runtime readiness has five states:

- **Prepared**: the exact configured image is present on every target node and the engine
  has an executable Spark adapter. **Ready** is reserved for a launched model that returned
  a real inference response.
- **Needs preparation**: every managed node is reachable, but the image is missing or
  outdated.
- **Partial**: nodes disagree, an installation failed, or only part of the pool is
  reachable.
- **Unreachable**: the target cannot be inspected.
- **Blocked**: no pinned image, managed nodes, or executable adapter is available.

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

These are widely used projects, not a popularity ranking. Ray Serve, BentoML and KServe are
serving/orchestration layers; ONNX Runtime, OpenVINO and PyTorch are lower-level
execution frameworks. They are not interchangeable container launch contracts.

## Operating-system support

Spark distinguishes an OCI image from a native runtime package. It never treats a
Linux image tag as proof that the same version is installable on another OS.

| Runtime | Linux AMD64/ARM64 | macOS Apple/Intel | Windows x64/ARM64 |
| --- | --- | --- | --- |
| vLLM, SGLang, TensorRT-LLM, Triton, TGI, LMDeploy, Xinference | OCI, when the selected tag publishes that architecture | — | — |
| llama.cpp | OCI | Native upstream archive | Native upstream CPU archive |
| Ollama | OCI | Native upstream archive | Native upstream archive |
| LocalAI | OCI | Native Apple Silicon binary | — |

The matrix is returned by the catalogue API and drives the platform selector.
Unsupported combinations are rejected before a job is queued. llama.cpp uses the
portable CPU archive on Windows so installation does not assume CUDA, ROCm, Vulkan,
OpenVINO or SYCL. An optimized accelerator build can be added later as an explicit
package variant; Spark never silently substitutes one.

Versions are **published distribution tags**, including stable releases, backend
variants, and rolling/nightly tags. Tags are discovered live, never inferred from
a hardcoded version list. Docker Hub pages are ordered by last update; GHCR/NGC
use registry order. Use **Next versions** for further pages or type an exact tag.
Platform information absent from a version page is resolved before downloading.
Version pages are cached for five minutes; an expired page returned during a
registry outage is explicitly marked stale. Registry errors are never an empty
successful version list.

The guided path chooses the most recent final semantic release automatically. The
advanced package view continues to expose every published channel for deliberate manual
testing.

## Controller download

The controller requires `skopeo`, supplied by the Docker image. For a native
installation, install it through the host package manager. Configure
`SPARK_CLUSTER_MODEL_ADMIN_TOKEN` before allowing mutations. All library routes
use the existing administrator authentication and controller signature checks.

By default archives live in `<model_cache_path>/.runtime-images`, on the large
model-storage volume rather than the small state/database volume. Override with
`SPARK_CLUSTER_RUNTIME_CACHE_PATH` if needed. Persist that directory and the
controller database together. The cache directory is ignored by model inventory.

Choose `linux/arm64` for DGX Spark or `linux/amd64` for x86 servers. Choose
`darwin/arm64`, `darwin/amd64`, `windows/amd64`, or `windows/arm64` only when the
selected runtime publishes a managed native package. Spark verifies
the registry manifest, its image configuration, and the selected platform, then
pins the **platform-specific SHA256 manifest digest** before invoking skopeo.
The controller produces a Docker image archive with its original repository/tag,
records its checksum and size, and publishes it only after a successful copy.
It reserves a conservative disk-space allowance; layer expansion can still exceed
this estimate and will fail the job without publishing an incomplete artifact.

Native versions come from the runtime project's GitHub Releases API. Spark selects
one exact curated asset name, requires GitHub's `sha256:` release digest, restricts
redirects to GitHub release-asset hosts, checks the published byte length while
streaming, and verifies the SHA256 before publishing the controller artifact.
Arbitrary URLs, installer scripts, MSI/EXE execution and package-manager commands
are not accepted.

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

macOS and Windows machines are registered separately under **Standalone macOS & Windows
hosts**. Registration accepts a private LAN address, SSH username and an Ed25519
host-key fingerprint. Spark verifies and pins that fingerprint before its first SSH
command, then detects the actual OS and architecture. The controller must already
have an SSH identity authorized for that user; no password or private key is sent
through the browser.

Native packages install without administrator privileges:

- macOS: `~/Library/Application Support/Spark Cluster/runtimes/<runtime>/<version>`
- Windows: `%LOCALAPPDATA%\SparkCluster\runtimes\<runtime>\<version>`

The package streams over pinned SSH, is verified again on the destination, expands
through `tar`/`ditto` on macOS or `Expand-Archive` on Windows, and is moved into its
version directory only after success. A checksum marker makes retries idempotent.
Spark records the current version inside the runtime directory but does not alter
the user's global `PATH`, install a service, execute an upstream installer, request
elevation, or launch the runtime automatically.

For a local Docker pool, the controller streams the archive through Docker Engine's
image-load API, checks error events even when HTTP status is 200, and verifies
the resulting tag and image platform. It checks the daemon architecture before
loading. The archive never needs to be buffered fully in controller memory.

Imports are serialized. Each SSH node has an independent outcome, so a failure
does not erase successful nodes. A completed job proves image preparation only:
it does not prove CUDA/ROCm drivers, SM121 kernels, model format support,
TensorRT engine compilation, multi-node inference, or endpoint readiness.

Cached archives can be removed from the advanced view. Deletion is refused while an
active job references the archive. Removing the controller copy does not remove images
or native packages already installed on target machines.

The existing vLLM and single-node SGLang launch adapters remain separately
configured using their pinned image settings. Importing a new tag does not switch
running models or implicitly enable a launcher. Configure and validate that
engine's launch image separately. llama.cpp, Ollama and the other listed engines
can be cached/imported but have no universal Spark launch adapter yet.

## API

All paths below are controller-local, including when a remote pool is active.

| Method and path | Purpose |
| --- | --- |
| `GET /api/v1/runtime-management/engines?cluster_id=<id>` | Read configured engine readiness for one explicit pool |
| `POST /api/v1/runtime-management/prepare` | Prepare a configured engine on a frozen pool target; returns HTTP 202 |
| `GET /api/v1/runtime-library` | Catalogue, artifacts, recent jobs and downloader availability |
| `GET /api/v1/runtime-library/runtimes/{id}/versions?cursor=&limit=50` | Paginated published tags and continuation cursor |
| `POST /api/v1/runtime-library/resolve` | Resolve a runtime/version/platform to immutable image metadata |
| `POST /api/v1/runtime-library/downloads` | Queue controller download; returns HTTP 202 and durable job |
| `POST /api/v1/runtime-library/installs` | Queue cached image import; returns HTTP 202 and durable job |
| `POST /api/v1/runtime-library/hosts` | Verify and register a native macOS/Windows SSH host |
| `DELETE /api/v1/runtime-library/hosts/{id}` | Remove native-host registration without changing the machine |
| `DELETE /api/v1/runtime-library/artifacts/{id}` | Remove an unused controller archive without changing target machines |
| `GET /api/v1/runtime-library/jobs/{id}` | Inspect phase, progress, artifact ID and per-node outcomes |

Preparation JSON is `{"cluster_id":"<pool id>","engine_id":"vllm"}`. The engine
must have both a configured image and an executable adapter. The returned job retains
the pool ID and its frozen node list so changing the dashboard context cannot redirect it.

Download/resolve JSON: `{"runtime_id":"vllm","version":"v0.9.2","platform":"linux/arm64"}`.
The version is an example, not a compatibility recommendation. An absent platform
is rejected. Installation JSON:
`{"artifact_id":"<32 hex characters>","cluster_id":"<pool id>","addresses":["<managed node>"]}`.
Omit `addresses` to import on every SSH-managed node in that pool, or for a Docker
pool to import once on its host. Runtimes and source repositories are curated in
`runtime_catalog.py`; arbitrary external image URLs and shell commands are not
accepted by these endpoints.

For a native host, send
`{"artifact_id":"<32 hex characters>","host_id":"host-<12 hex characters>"}`.
Host registration JSON contains `name`, private-LAN `address`, `username`, and the
displayed `SHA256:` Ed25519 fingerprint. Exactly one of `cluster_id` and `host_id`
is accepted for an installation.

The separation between runtime definitions, versions and launch contracts is
inspired by [GPUStack's inference backend catalogue](https://github.com/gpustack/gpustack/blob/main/gpustack/schemas/inference_backend.py).
Spark uses its own controller-owned archives, pool registry and pinned SSH trust.
