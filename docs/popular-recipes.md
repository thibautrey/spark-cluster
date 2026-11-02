# Popular-model recipe catalog

The offline snapshot captured on **2026-11-02** adds **1,073 distinct serving templates**
(one per Hugging Face repository), alongside the two previously reviewed recipes.
It also includes **160 explicitly non-executable workload references**. References
are not counted toward the 1,000-template minimum.

| Workload | Generated serving templates |
| --- | ---: |
| Text generation, code, reasoning and multilingual models | 546 |
| Vision-language models | 200 |
| Embeddings and retrieval | 220 |
| Classification, including ranking checkpoints | 80 |
| Whisper speech recognition | 27 |
| **Total** | **1,073** |

Image generation (60), image editing (30), video generation (30) and speech synthesis
(40) appear under separate workload filters as references. They link to exact model
revisions and their instructions. They cannot be installed through the vLLM adapter.
A diffusion checkpoint is not a chat model, and a reference link is not an executable recipe.

## What the entries mean

The generated recipes are **not 1,073 independently published or hardware-tested Spark
recipes**. They combine actual public checkpoint metadata with architectures listed in
[vLLM v0.32.0's pinned registry](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/models/registry.py).
Eligible weights must use a supported Transformers/Sentence Transformers/vLLM library and include full safetensors weights. MLX, GGUF, EXL, adapter-only and unrelated library formats are excluded even when architecture names match.

Their status is `generated-template`, visibly distinct from `source-reviewed` and
`hardware-verified`. A supported architecture does not prove a checkpoint's tokenizer,
quantization, multimodal processors, kernels or memory behavior work on GB10.

Each serving template has a real model repository, full weights commit, model license,
source commit, explicit launch settings, estimated memory/storage budgets, a normalized
content hash and semantic release version. All use the verified Linux ARM64 manifest
of the official `vllm/vllm-openai:v0.32.0` image:

```
sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
```

Registry metadata verified config digest
`sha256:4ceb6c60eed291c1d179c35674bd8ece7acfd4f9abe8b9be83a6a4dcc3c70ea2`
and about 10.3 GB compressed image layers. The catalog build downloads metadata only.

Budgets derive from published safetensors dtype counts plus overhead and remain
estimates. Multi-node-sized models show the existing topology blocker. Gated model
access and licenses remain the operator's responsibility. No model remote code is
implicitly trusted. Catalog publication neither downloads weights nor changes a
running model, installed release, storage selection or compute-pool selection.

## Popularity and provenance

The builder queried current [Hugging Face Hub model metadata](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.list_models)
using both `trendingScore` and rolling download counts, globally and per task.
It collected **5,278 distinct repositories** before eligibility checks. Ranking
uses reciprocal rank fusion with `k=50` and workload quotas so downloads of older
text models do not crowd out vision, embeddings or speech. This is a reproducible
selection of popular candidates, not a claim that every model or recipe on the
internet has been found or that download counts measure recipe adoption.

Each generated entry retains its source-query ranks, snapshot time, downloads,
likes, trend score, architecture names, safetensors counts, pinned model card and
revision-specific metadata URL. Popularity belongs to the dated snapshot; the UI
does not label a bundled historical snapshot as continuously live rankings.

The source panel also links to pinned trees of:

- [NVIDIA Local AI Inference Recipes](https://github.com/NVIDIA/local-ai-inference-recipes/tree/7b2ee453234617f58e1e100c195ac3be7f79e233): 122 `recipe.yaml` files across hardware targets.
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker/tree/b7f6b4cd4d85c655c5102eeb579f6de6a5bfe3d3): 45 files under its recipe directories.

These are source-library inventories, not extra executable imports or proof of
compatibility. Their original formats, custom builds and hardware requirements
must be preserved when adapting a specific upstream recipe.

## Browsing and installation

Models → Recipes provides server-side search, workload filters, popularity/trending/
download/name sorting, a selected-pool compatibility filter and 24-recipe pages.
Versions of one recipe stay together on a page. Metadata exposes catalog counts,
snapshot provenance and source libraries. The browser never renders a thousand cards
at once or refetches the whole catalog every five seconds.

`GET /api/v1/recipes` accepts `q`, `category`, `sort`, `offset`, `limit` (1–100),
`installable` and optional `cluster_id`. Its `recipes` array contains the requested
page, `total` counts matching distinct recipes, and `catalog_total` counts all
available distinct serving recipes. Existing release history remains addressable
by immutable hash even when it is not on the current page. Reference workload
filters return a separate `references` array; reference hashes are not installable.

Installation uses the existing target, image, memory and storage checks. Remote
bridge recipe bootstrap and multi-node recipes still have their existing explicit
execution blockers. Adding a model to this catalog does not bypass those limits.
The running Blazux Qwen deployment is preserved and remains separate from these templates.

The post-launch probe now follows the workload:

- Text-generation templates: `/v1/completions`, so base models need no chat template.
- Vision-language and existing reviewed chat recipes: `/v1/chat/completions`; this
  proves text generation only, not image-understanding quality.
- Embeddings: `/v1/embeddings`, requiring a nonempty finite vector.
- Classification: `/classify`, requiring classification probabilities.
- Whisper: `/v1/audio/transcriptions` with a synthetic silent WAV, checking response
  shape only; an empty transcript is valid and is not an accuracy benchmark.

Pinned recipe concurrency settings remain fixed through adoption/recovery. Generated
catalog entries are never promoted to hardware-verified by schema validation or an
API listing.

## Explicit snapshot refresh

Use a fresh metadata cache to capture new rankings. No recurring refresh is scheduled.
From the repository root:

```sh
python3 scripts/build_popular_recipes.py \
  --cache /tmp/spark-recipes-new-snapshot \
  --output recipes/popular/YYYY-MM-DD.catalog.json \
  --version 1.1.0
python3 scripts/build_recipe_references.py \
  --cache /tmp/spark-recipes-new-snapshot \
  --output recipes/popular/YYYY-MM-DD.references.json
```

Both scripts use public metadata and fetch no weights. The template builder refuses
to output fewer than 1,000 eligible distinct repositories, refuses overwriting a
snapshot with changed content, and checks all existing bundled snapshots for
immutable `(id, version)` conflicts. Bump the version when model pins, runtime pins,
settings or recipe metadata change. Old snapshots/releases stay available for rollback.
Popularity changes alone live in origin metadata, outside the execution content hash.
Catalog bootstrap validates the batch before saving it atomically to SQLite.

The root publisher index remains the original two independently reviewed JSON files.
The larger bundled snapshot has its own versioned bulk-catalog format; it is not
silently fed through GitHub discovery's 40-file publisher limit. Docker and wheel
packaging include the nested catalog automatically through the existing recipes directory.

## Withdrawn format mismatches

The initial catalog smoke deployment exposed 178 checkpoints whose architecture names
matched but whose library or weight format did not. Their exact release hashes are
withdrawn from browsing and new installation. Persisted release records remain
addressable for audit and existing installation snapshots are untouched. Withdrawal
does not delete files, unload a runtime or rewrite release contents. The corrected
active snapshot contains 1,073 format-checked templates and the two reviewed starters.
