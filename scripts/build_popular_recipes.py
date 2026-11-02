#!/usr/bin/env python3
"""Build immutable, evidence-backed recipes from public model metadata; never download weights.

This is an explicit maintainer operation, not an hourly job that changes installed models.
Each build reads immutable runtime source and records live popularity separately from releases.
"""

import argparse
import ast
import concurrent.futures
import datetime
import hashlib
import json
import math
import pathlib
import re
import time
import urllib.parse
import urllib.request

VLLM_COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
VLLM_IMAGE = (
    "docker.io/vllm/vllm-openai@sha256:"
    "6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc"
)
FIELDS = [
    "sha",
    "pipeline_tag",
    "library_name",
    "downloads",
    "likes",
    "trendingScore",
    "config",
    "safetensors",
    "cardData",
    "tags",
    "gated",
    "siblings",
    "lastModified",
]
TASKS = {
    "text-generation": 1000,
    "image-text-to-text": 300,
    "feature-extraction": 400,
    "sentence-similarity": 300,
    "text-classification": 300,
    "automatic-speech-recognition": 200,
    "text-to-speech": 100,
    "text-to-image": 100,
    "image-to-image": 60,
    "text-to-video": 60,
}
QUOTAS = {"text": 700, "vision": 200, "embeddings": 220, "classification": 80, "speech": 40}
REFERENCE_QUOTAS = {
    "text-to-image": 60,
    "image-to-image": 30,
    "text-to-video": 30,
    "text-to-speech": 40,
}
DTYPE_BITS = {
    "F64": 64,
    "F32": 32,
    "BF16": 16,
    "F16": 16,
    "I64": 64,
    "I32": 32,
    "I16": 16,
    "I8": 8,
    "U8": 8,
    "BOOL": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spark-cluster-catalog/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_queries(cache):
    queries = [
        ("global-trending", {"sort": "trendingScore", "limit": 500}),
        ("global-downloads", {"sort": "downloads", "limit": 500}),
    ]
    for task, limit in TASKS.items():
        for sort in ["trendingScore", "downloads"]:
            queries.append(
                (task + "-" + sort, {"pipeline_tag": task, "sort": sort, "limit": limit})
            )

    def fetch(pair):
        name, params = pair
        path = cache / (name + ".json")
        if path.exists():
            return
        params = {**params, "direction": -1, "expand[]": FIELDS}
        url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(params, doseq=True)
        data = json.loads(get(url))
        for model in data:
            config = model.get("config") or {}
            model["config"] = {
                k: v
                for k, v in config.items()
                if k in ["architectures", "model_type", "quantization_config", "auto_map"]
            }
            model["siblings"] = [s["rfilename"] for s in model.get("siblings", [])]
            card = model.get("cardData") or {}
            model["cardData"] = {
                k: v for k, v in card.items() if k in ["license", "base_model", "language"]
            }
        path.write_text(json.dumps({"url": url, "fetched_at": time.time(), "models": data}))
        print(name, len(data), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, queries))
    return [cache / (name + ".json") for name, _ in queries]


def architecture_groups(cache):
    path = cache / "vllm-registry.py"
    if not path.exists():
        path.write_bytes(
            get(
                f"https://raw.githubusercontent.com/vllm-project/vllm/{VLLM_COMMIT}/"
                "vllm/model_executor/models/registry.py"
            )
        )
    groups = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    groups[target.id] = {
                        key.value
                        for key in node.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
    return groups


def category(model, groups):
    task = model.get("pipeline_tag")
    architectures = set((model.get("config") or {}).get("architectures") or [])
    if task == "text-generation" and architectures & groups["_TEXT_GENERATION_MODELS"]:
        return "text"
    if task == "image-text-to-text" and architectures & groups["_MULTIMODAL_MODELS"]:
        return "vision"
    if task in {"feature-extraction", "sentence-similarity"} and architectures & (
        groups["_EMBEDDING_MODELS"] | groups["_TEXT_GENERATION_MODELS"]
    ):
        return "embeddings"
    if task == "text-classification" and architectures & groups["_SEQUENCE_CLASSIFICATION_MODELS"]:
        return "classification"
    # Whisper has a documented transcription API. Other speech families may require custom setup.
    if (
        task == "automatic-speech-recognition"
        and "WhisperForConditionalGeneration" in architectures
    ):
        return "speech"
    return None


def supported_weight_format(model):
    if model.get("library_name") not in {"transformers", "sentence-transformers", "vllm"}:
        return False
    if any(
        marker in model["id"].casefold() for marker in ("-mlx", "mlx-", "-gguf", "exl2", "exl3")
    ):
        return False
    return any(
        name.endswith(".safetensors") and "adapter_model" not in name
        for name in model.get("siblings", [])
    )


def make_recipe(model, kind, version):
    repo, sha = model["id"], model["sha"]
    stable = re.sub(r"[^a-z0-9-]", "-", repo.casefold()).strip("-")[:85]
    slug = stable + "-" + hashlib.sha256(repo.encode()).hexdigest()[:10]
    parameters = model["safetensors"]["parameters"]
    weight_bytes = sum(n * DTYPE_BITS.get(dtype, 32) / 8 for dtype, n in parameters.items())
    weight_gib = weight_bytes / 2**30
    nodes = max(1, min(64, math.ceil((weight_gib * 1.2 + 10) / 105)))
    memory = round(min(1024, max(4, weight_gib * 1.2 / nodes + 10)), 1)
    disk = round(min(16384, max(30, weight_gib * 1.4 + 30)), 1)
    args = ["--dtype", "auto"]
    if kind == "embeddings":
        args += ["--runner", "pooling", "--convert", "embed"]
    elif kind == "classification":
        args += ["--runner", "pooling", "--convert", "classify"]
    context = 448 if kind == "speech" else 512 if kind in {"embeddings", "classification"} else 4096
    bits = 16
    quant = (model.get("config") or {}).get("quantization_config") or {}
    if quant.get("bits") in {2, 3, 4, 8}:
        bits = quant["bits"]
    elif "fp8" in str(quant.get("quant_method", "")):
        bits = 8
    notes = [
        "Generated serving template, not a published model-specific Spark recipe or hardware test.",
        "Architecture is advertised by the pinned vLLM release; "
        "this does not verify GB10 kernels, quantization, or this checkpoint.",
        "Memory and disk budgets are estimates from published safetensors dtype counts plus "
        "overhead; live admission still applies.",
        "Weights and runtime stay pinned. Catalog updates never download "
        "weights, launch models, or replace installed releases.",
        "Review the pinned model card and license; "
        "gated checkpoints require separately configured access.",
        "No remote model code is trusted or executed by this template.",
    ]
    if nodes > 1:
        notes.append(
            "Multi-node estimate only; recipe v1 cannot bootstrap this distributed configuration."
        )
    if quant:
        notes.append("Checkpoint quantization metadata: " + canonical(quant)[:700])
    return {
        "schema_version": "spark-cluster.recipe/v1",
        "id": "popular-models/" + slug,
        "version": version,
        "controller_min_version": "0.1.0",
        "title": repo[:150],
        "description": f"{kind.capitalize()} serving template for {repo}. "
        "Derived from pinned public metadata; untested on DGX Spark.",
        "license": "Apache-2.0",
        "source": {
            "repository": "vllm-project/vllm",
            "commit": VLLM_COMMIT,
            "path": "vllm/model_executor/models/registry.py",
            "license": "Apache-2.0",
        },
        "model": {
            "repository": repo,
            "revision": sha,
            "license": str(model["cardData"]["license"])[:100],
        },
        "requirements": {
            "hardware": "dgx-spark",
            "platform": "linux/arm64",
            "nodes": nodes,
            "memory_gib_per_node": memory,
            "disk_gib_per_node": disk,
            "topology": "single" if nodes == 1 else "ray",
        },
        "runtime": {"image": VLLM_IMAGE, "arguments": args, "environment": {}, "unconfined": False},
        "launch": {
            "model_id": repo,
            "served_model_name": slug[:100],
            "engine_id": "vllm",
            "bits_per_weight": bits,
            "context_length": context,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 4096,
            "tensor_parallel": nodes,
            "pipeline_parallel": 1,
            "gpu_memory_utilization": 0.8,
            "memory_budget_mode": "fixed",
            "kv_cache_dtype": "auto",
            "cache_cpu_gib": 0,
            "cache_disk_gib": 0,
            "speculative_method": "off",
        },
        "validation": "generated-template",
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    paths = fetch_queries(args.cache)
    groups = architecture_groups(args.cache)
    models, ranks, queries, times = {}, {}, [], []
    for path in paths:
        data = json.loads(path.read_text())
        queries.append(data["url"])
        times.append(data["fetched_at"])
        for rank, model in enumerate(data["models"], 1):
            repo = model["id"]
            models.setdefault(repo, model)
            ranks.setdefault(repo, {})[path.stem] = rank

    def score(model):
        return sum(1 / (50 + rank) for rank in ranks[model["id"]].values())

    by_category = {kind: [] for kind in QUOTAS}
    for model in models.values():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["id"]):
            continue
        if not re.fullmatch(r"[a-f0-9]{40}", model.get("sha") or ""):
            continue
        if not (model.get("safetensors") or {}).get("parameters"):
            continue
        if not (model.get("cardData") or {}).get("license"):
            continue
        if not supported_weight_format(model):
            continue
        kind = category(model, groups)
        if kind:
            by_category[kind].append(model)
    selected = []
    for kind, quota in QUOTAS.items():
        selected.extend(
            (model, kind) for model in sorted(by_category[kind], key=score, reverse=True)[:quota]
        )
    if len(selected) < 1000:
        raise SystemExit(
            f"Only {len(selected)} eligible distinct configurations; refusing to pad the catalog"
        )
    captured = datetime.datetime.fromtimestamp(max(times), datetime.UTC).isoformat()
    entries = []
    for model, kind in selected:
        recipe = make_recipe(model, kind, args.version)
        origin = {
            "kind": "generated",
            "category": kind,
            "library": model["library_name"],
            "weight_format": "safetensors",
            "task": model.get("pipeline_tag"),
            "captured_at": captured,
            "downloads": model.get("downloads", 0),
            "likes": model.get("likes", 0),
            "trending_score": model.get("trendingScore", 0),
            "rank_score": round(score(model), 8),
            "query_ranks": ranks[model["id"]],
            "gated": bool(model.get("gated")),
            "architectures": (model.get("config") or {}).get("architectures", []),
            "model_card": f"https://huggingface.co/{model['id']}/blob/{model['sha']}/README.md",
            "metadata_url": f"https://huggingface.co/api/models/{model['id']}/revision/{model['sha']}",
            "safetensors": model["safetensors"],
        }
        entries.append({"recipe": recipe, "origin": origin})
    entries.sort(key=lambda e: (-e["origin"]["rank_score"], e["recipe"]["id"]))
    payload = {
        "schema_version": "spark-cluster.recipe-catalog/v1",
        "captured_at": captured,
        "ranking": "Reciprocal rank fusion (k=50) across current trending and rolling-download "
        "lists, with per-task quotas; one recipe per distinct repository.",
        "queries": queries,
        "entries": entries,
    }
    for previous in args.output.parent.glob("*.catalog.json"):
        old = json.loads(previous.read_text())
        known = {(e["recipe"]["id"], e["recipe"]["version"]): e["recipe"] for e in old["entries"]}
        for entry in entries:
            recipe = entry["recipe"]
            prior = known.get((recipe["id"], recipe["version"]))
            if prior and prior != recipe:
                raise SystemExit(f"Bump recipe version: {recipe['id']}@{recipe['version']} changed")
    # Refuse rewriting an immutable release in an existing snapshot.
    if args.output.exists() and json.loads(args.output.read_text()) != payload:
        raise SystemExit(
            "Snapshot exists with different content; use a new output "
            "and bump changed recipe releases"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        "Templates",
        len(entries),
        "categories",
        {k: sum(e["origin"]["category"] == k for e in entries) for k in QUOTAS},
    )


if __name__ == "__main__":
    main()
