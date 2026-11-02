"""Build non-executable workload references and pin upstream recipe-library trees."""

import argparse
import datetime
import hashlib
import json
import pathlib
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
r = args.cache
dst = args.output
r.mkdir(parents=True, exist_ok=True)
for key, repo in [
    ("nvidia", "NVIDIA/local-ai-inference-recipes"),
    ("spark_vllm", "eugr/spark-vllm-docker"),
]:
    p = r / (key + "_tree.json")
    if not p.exists():
        headers = {"User-Agent": "spark-cluster-catalog/1.0"}
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/commits/HEAD", headers=headers
        )
        head = json.load(urllib.request.urlopen(req, timeout=30))["sha"]
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/git/trees/{head}?recursive=1", headers=headers
        )
        data = json.load(urllib.request.urlopen(req, timeout=30))
        if data.get("truncated"):
            raise SystemExit("Upstream tree is truncated")
        p.write_text(json.dumps(data))

models = {}
ranks = {}
times = []
for p in r.glob("*.json"):
    d = json.loads(p.read_text())
    if not isinstance(d, dict):
        continue
    if "models" not in d:
        continue
    times.append(d["fetched_at"])
    for rank, m in enumerate(d["models"], 1):
        models.setdefault(m["id"], m)
        ranks.setdefault(m["id"], []).append(rank)
quotas = {"text-to-image": 60, "image-to-image": 30, "text-to-video": 30, "text-to-speech": 40}
refs = []
for task, limit in quotas.items():
    eligible = [
        m
        for m in models.values()
        if m.get("pipeline_tag") == task and m.get("sha") and "/" in m["id"]
    ]
    eligible.sort(key=lambda m: sum(1 / (50 + n) for n in ranks[m["id"]]), reverse=True)
    for m in eligible[:limit]:
        refs.append(
            {
                "id": hashlib.sha256((m["id"] + m["sha"]).encode()).hexdigest(),
                "model_id": m["id"],
                "revision": m["sha"],
                "task": task,
                "library": m.get("library_name") or "unknown",
                "license": str((m.get("cardData") or {}).get("license") or "not specified"),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "trending_score": m.get("trendingScore", 0),
                "rank_score": sum(1 / (50 + n) for n in ranks[m["id"]]),
                "model_card": f"https://huggingface.co/{m['id']}/blob/{m['sha']}/README.md",
                "status": "reference-only",
                "blocker": "This workload needs a dedicated runtime adapter; "
                "a vLLM chat recipe would not be equivalent.",
            }
        )
sources = []
for key, repo in [
    ("nvidia", "NVIDIA/local-ai-inference-recipes"),
    ("spark_vllm", "eugr/spark-vllm-docker"),
]:
    tree = json.loads((r / (key + "_tree.json")).read_text())
    files = [
        x["path"]
        for x in tree["tree"]
        if x.get("type") == "blob"
        and (x["path"].endswith("recipe.yaml") or "/recipes/" in "/" + x["path"])
    ]
    sources.append(
        {
            "repository": repo,
            "commit": tree["sha"],
            "url": f"https://github.com/{repo}/tree/{tree['sha']}",
            "recipe_paths": files,
        }
    )
payload = {
    "schema_version": "spark-cluster.recipe-references/v1",
    "captured_at": datetime.datetime.fromtimestamp(max(times), datetime.UTC).isoformat(),
    "references": refs,
    "upstream_libraries": sources,
}
if dst.exists() and json.loads(dst.read_text()) != payload:
    raise SystemExit("Use a new snapshot path; never overwrite catalog provenance")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, indent=2) + "\n")
print(
    "reference workloads",
    len(refs),
    "source library files",
    [(s["repository"], len(s["recipe_paths"])) for s in sources],
)
