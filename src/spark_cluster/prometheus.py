from collections.abc import Iterable


def render_metrics(snapshot) -> str:
    lines = [
        "# HELP spark_cluster_cache_hit_ratio Effective multi-tier KV cache hit ratio.",
        "# TYPE spark_cluster_cache_hit_ratio gauge",
        f"spark_cluster_cache_hit_ratio {snapshot.cache_hit_rate / 100:.6f}",
        "# HELP spark_cluster_output_tokens_per_second Cluster output token throughput.",
        "# TYPE spark_cluster_output_tokens_per_second gauge",
        f"spark_cluster_output_tokens_per_second {snapshot.total_output_tokens_per_second}",
    ]
    lines.extend(_node_metrics(snapshot.nodes))
    lines.extend(_cache_metrics(snapshot.cache_tiers))
    return "\n".join(lines) + "\n"


def _node_metrics(nodes: Iterable) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        labels = f'node="{node.id}",role="{node.role}"'
        lines.append(
            f"spark_cluster_node_memory_used_bytes{{{labels}}} {node.memory.used_gib * 2**30}"
        )
        lines.append(
            f"spark_cluster_node_gpu_utilization_ratio{{{labels}}} {node.gpu_utilization / 100}"
        )
    return lines


def _cache_metrics(tiers: Iterable) -> list[str]:
    lines: list[str] = []
    for tier in tiers:
        labels = f'tier="{tier.kind}"'
        lines.append(f"spark_cluster_cache_used_bytes{{{labels}}} {tier.used_gib * 2**30}")
        lines.append(f"spark_cluster_cache_hits_total{{{labels}}} {tier.hits}")
        lines.append(f"spark_cluster_cache_misses_total{{{labels}}} {tier.misses}")
    return lines
