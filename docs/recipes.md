# Spark Cluster Recipe v1

**[Authoring and publishing skill](../skills/spark-cluster-recipes/SKILL.md)** ·
[Starter files](../recipes) · [JSON Schema](../recipes/schema-v1.json)

Models → **Recipes** lists bundled, imported, and GitHub-discovered releases.
The [popular-model catalog](popular-recipes.md) adds over 1,000 explicitly generated,
version-pinned serving templates, workload references, search and pagination. Pick a version
and choose **Install & launch**. The controller validates hardware and the immutable ARM64
image, runs model memory/storage admission, downloads the pinned weights, installs the image,
launches through the existing compute-pool scheduler, and checks for a real generated response.
Progress, failures, and retry controls remain visible across page reloads.

## Supported execution

The v1 executor supports **one DGX Spark connected directly through Docker**, using the vLLM
adapter. It supports model-specific literal arguments, runtime environment, persistent compiler
and disk-backed PLE caches, and explicit unconfined container settings when a recipe needs them.
It does not run arbitrary repository shell scripts, install Python packages on the host, or
convert native multi-node launch instructions into Ray commands. Remote Kubernetes bridges,
Ray multi-node recipe bootstrap, other engines, and host setup steps show compatibility blockers.
The JSON contract represents topology explicitly so unsupported configurations cannot silently
run as something else. The rest of Spark Cluster retains its existing launch paths.

The runtime image is executable software supplied by its publisher. Discovery is not a signature
or a safety review. Inspect the source, licenses, runtime flags, and container permissions before
installing a third-party release. Credentials belong in controller configuration, never recipe files.
Mutating recipe endpoints require a configured `SPARK_CLUSTER_MODEL_ADMIN_TOKEN` and authentication.

## File contract

A recipe is a UTF-8 JSON file ending in `.spark-recipe.json`, at most 128 KiB. The authoritative
[JSON Schema](../recipes/schema-v1.json) is also available from `GET /api/v1/recipes/schema`.
Unknown fields are rejected throughout the document, including launch settings.

| Field | Contract |
| --- | --- |
| `schema_version` | Exactly `spark-cluster.recipe/v1`; unknown versions are rejected, never guessed. |
| `id` | Stable `github-owner/recipe-name`; discovery checks the namespace against the actual publisher. |
| `version` | Strict numeric `MAJOR.MINOR.PATCH`, without prerelease/build suffixes in v1. |
| `controller_min_version` | Minimum Spark Cluster application version, checked before installation. |
| `title`, `description`, `license` | Name the actual model variant and recipe license. |
| `source` | Upstream GitHub repository, full Git commit, relative path, and license. |
| `model` | Hugging Face repository, full weights commit, and model license. |
| `requirements` | DGX Spark, Linux ARM64, node count, memory/storage budget per node, `single` or `ray` topology. |
| `runtime` | Image pinned by digest; literal extra vLLM arguments; environment; explicit `unconfined` boolean. |
| `launch` | Existing Spark Cluster typed launch settings: exact model ID, explicit `vllm`, context, parallelism, memory and concurrency. |
| `validation` | `generated-template`, `source-reviewed` or publisher-asserted `hardware-verified`. None is automatic local proof. |
| `notes` | Caveats, prerequisites, original measured setup, and verification evidence. |

The controller owns model/path, host/port, weights and code revisions, served name, context,
concurrency, memory fraction, parallelism, and distributed bootstrap flags. Do not duplicate
those in `runtime.arguments`. Put model-specific options such as quantization, cache dtype,
attention backend, speculative configuration, tool and reasoning parsers in that array.
Arguments are passed as an argv array, never interpolated into shell source. Environment keys
use supported runtime prefixes (`VLLM_`, `B12X_`, `CUTE_`, `SAFETENSORS_`, `NCCL_`, `PYTORCH_`).
Only public Docker Hub, GHCR, and NVIDIA registry image references are accepted in v1.
Runtime metadata resolution verifies manifest/configuration digests and Linux ARM64 architecture.

The memory/storage fields document the recipe's hardware budget. Memory is checked against pool
metadata; actual model storage/memory admission also runs before downloading. Runtime-image and
compiler-cache disk usage can still exhaust target storage; such failures are reported by the
runtime installation. The exact cached weights revision must match before launching. No
`latest` weights or runtime replacement is substituted when a pin is unavailable.

## Author and publish

1. Copy a starter file and follow the [skill](../skills/spark-cluster-recipes/SKILL.md).
2. Validate every release:

   ```sh
   spark-cluster recipe validate recipes/*.spark-recipe.json
   spark-cluster recipe schema > recipes/schema-v1.json
   ```

3. Create a repository-root **`spark-cluster-recipes.json`** index:

   ```json
   {
     "schema_version": "spark-cluster.recipe-index/v1",
     "recipes": ["recipes/example.spark-recipe.json"]
   }
   ```

4. Commit the recipe and index to the public repository's default branch. Tag the release,
   for example `recipe-example-v1.0.0`. Add the GitHub repository topic **`spark-cluster-recipes`**.
   A central registration PR is unnecessary. Publishing/pushing remains an explicit author action.
5. Select **Discover on GitHub** in Models → Recipes, or wait for automatic discovery.
   A local JSON file can also be imported from that section without publishing it.

For separate releases, use separate file paths and retain old paths in the index. This repository's
[index](../spark-cluster-recipes.json) is a working publisher example. Third-party formats such as
sparkrun YAML, vLLM recipe YAML, or README commands are research inputs, not this execution format.

## Versioning, caching, and rollback

There are separate version identities:

- **File format:** `spark-cluster.recipe/v1` and `spark-cluster.recipe-index/v1`. A future incompatible
  format needs a new version and an explicit migration; v1 defaults must remain stable.
- **Recipe release:** immutable `(id, version)`. Reusing it with different normalized content is
  rejected. Semantic numeric ordering makes `1.10.0` newer than `1.9.0`.
- **Discovery source:** full publisher Git commit plus file path. Upstream recipe evidence has its
  own independently pinned `source.commit`.
- **Execution:** pinned Hugging Face commit and platform-specific OCI digest, plus the normalized
  recipe SHA-256 and full launch snapshot persisted in each installation job.

Any change to a recipe release, including pins or metadata, requires a version bump. Suggested
policy: patch for metadata corrections, minor for opt-in tuning changes, major for changed model,
runtime requirements, topology, or incompatible behavior. Updates never alter installed jobs or
resident runtimes automatically. Old releases remain selectable; after unloading running models,
select and install the previous release for rollback. Changing weights also requires explicitly
removing the old downloaded revision first; the controller does not delete it for you. Failed
installations retain partial downloads/runtime artifacts for retry and do not automatically undo
a successfully started runtime if the final response check fails.

Discovery runs in the background on startup and hourly, and can be refreshed manually. It searches
up to three pages of 30 public repositories with the topic, reads up to 40 indexed recipes per repo,
resolves each HEAD to a full commit, and reads files at that commit. GitHub HTTP responses, ETags,
discovery status, release files, and installation snapshots persist in the controller's existing
SQLite database. Valid cached responses are reused for one hour; expired ones are revalidated with
ETags. Bundled releases load without network access. A GitHub outage/rate limit keeps old releases
and exposes a discovery error. Optional `SPARK_CLUSTER_RECIPES_GITHUB_TOKEN` raises API limits;
it is never sent to arbitrary download hosts or stored in recipe metadata.

## Included starters and upstream research

Both bundled recipes are **source-reviewed, not locally hardware-verified**. Published performance
numbers do not prove these exact pins or reduced-context starter configurations work on your pool.

- [Qwen3.8 Flash-Next FP4-Local, one Spark](../recipes/qwen38-flash-next.spark-recipe.json):
  adapted from [NVIDIA's pinned configuration](https://github.com/NVIDIA/local-ai-inference-recipes/blob/7b2ee453234617f58e1e100c195ac3be7f79e233/models/qwen/qwen3.8-flash-next/dgx-spark/linux/vllm/fp4-local/recipe.yaml).
  Uses custom B12x kernels, disk-backed PLE, MTP4, and model-native parsers. Runtime tag
  `nightly-20261015` was resolved to an ARM64 digest on 2026-11-02; the file stores the digest.
  Source attribution/license caveats stay in the file. Model access requires its own license review.
- [DeepSeek R1 Distill Qwen 32B BF16, one Spark](../recipes/deepseek-r1-distill-qwen32b.spark-recipe.json):
  a smaller runnable starting point based on [DeepSeek's pinned serving guidance](https://github.com/deepseek-ai/DeepSeek-R1/blob/0cf78561f1d51c84a21b2190626b21116d5c68bb/README.md).
  Uses the ARM64 manifest of vLLM `v0.11.0`. This is the 32B distillation, **not full DeepSeek R1/V4**.

For full multi-Spark DeepSeek, these sources are concrete starting points for a future adapter:

- [NVIDIA DeepSeek V4 Flash 0731, two Sparks](https://github.com/NVIDIA/local-ai-inference-recipes/blob/7b2ee453234617f58e1e100c195ac3be7f79e233/models/deepseek/deepseek-v4-flash-0731/dgx-spark-2gpu/linux/vllm/fp4-fp8-mixed/recipe.yaml):
  native rank-specific processes, CX7/RoCE settings, B12x, DSpark5, and privileged Docker.
- [vLLM DeepSeek V4 Flash recipes](https://github.com/vllm-project/recipes/blob/2851f5f7d731d18852bcb236f058b472a8858c99/models/deepseek-ai/DeepSeek-V4-Flash.yaml).
- [HiTechAI-VN dual-Spark recipe](https://github.com/HiTechAI-VN/deepseek-v4-flash-dual-spark-recipe/tree/d4acc420f9c44495bcc29702f752dbad7a0863f3).

These are documented as research rather than mislabeled as executable single-Spark recipes.
