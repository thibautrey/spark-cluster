import math
import random
from datetime import UTC, datetime

from .models import CacheTier, ClusterSnapshot, MemoryStats, ModelRuntime, NodeSnapshot


class ClusterSimulator:
    def __init__(self, cluster_name: str, seed: int = 42):
        self.cluster_name = cluster_name
        self.random = random.Random(seed)
        self.tick = 0
        self.started_at = datetime.now(UTC)
        self.hits = [92000, 41000, 12000, 6000]
        self.misses = [11000, 9000, 3100, 2000]

    def snapshot(self) -> ClusterSnapshot:
        self.tick += 1
        wave = math.sin(self.tick / 7)
        nodes: list[NodeSnapshot] = []
        for index in range(4):
            utilization = 61 + wave * 14 + index * 3 + self.random.uniform(-3, 3)
            used = 86 + wave * 8 + index * 2 + self.random.uniform(-2, 2)
            output = 18 + wave * 3 + index * 1.2 + self.random.uniform(-1, 1)
            nodes.append(
                NodeSnapshot(
                    id=f"spark-{index + 1}",
                    hostname=f"spark-{index + 1}",
                    role="head" if index == 0 else "worker",
                    address=f"10.10.0.{index + 11}",
                    memory=MemoryStats(
                        total_gib=128,
                        used_gib=round(max(0, used), 1),
                        model_gib=68.4,
                        kv_cache_gib=round(10 + index + wave * 2, 1),
                    ),
                    gpu_utilization=round(max(0, min(100, utilization)), 1),
                    temperature_c=round(57 + utilization * 0.18 + self.random.uniform(-1, 1), 1),
                    input_tokens_per_second=round(output * 11.7, 1),
                    output_tokens_per_second=round(output, 1),
                    active_requests=max(0, round(3 + wave + index / 2)),
                )
            )

        tiers = []
        tier_data = [
            ("GPU KV", "gpu", 38 + wave * 3, 64, 0.8, False),
            ("Unified memory", "cpu", 119 + wave * 6, 220, 2.4, False),
            ("Local NVMe", "nvme", 732 + wave * 20, 2048, 6.8, True),
            ("Shared cache", "remote", 1210 + wave * 30, 4096, 14.2, True),
        ]
        for index, (name, kind, used, capacity, latency, persistent) in enumerate(tier_data):
            self.hits[index] += self.random.randint(80, 430)
            self.misses[index] += self.random.randint(5, 45)
            tiers.append(
                CacheTier(
                    name=name,
                    kind=kind,
                    used_gib=round(used, 1),
                    capacity_gib=capacity,
                    hits=self.hits[index],
                    misses=self.misses[index],
                    read_latency_ms=round(latency + self.random.uniform(-0.2, 0.2), 1),
                    persistent=persistent,
                )
            )

        total_hits = sum(t.hits for t in tiers)
        total = total_hits + sum(t.misses for t in tiers)
        return ClusterSnapshot(
            cluster_name=self.cluster_name,
            nodes=nodes,
            cache_tiers=tiers,
            runtime=ModelRuntime(
                model="deepseek-ai/DeepSeek-R1",
                quantization="NVFP4",
                context_length=524288,
                tensor_parallel=4,
                pipeline_parallel=1,
                uptime_seconds=int((datetime.now(UTC) - self.started_at).total_seconds()),
                queued_requests=max(0, round(2 + wave * 2)),
            ),
            total_input_tokens_per_second=round(sum(n.input_tokens_per_second for n in nodes), 1),
            total_output_tokens_per_second=round(sum(n.output_tokens_per_second for n in nodes), 1),
            cache_hit_rate=round(total_hits / total * 100, 2),
            prefix_reuse_tokens=84_000_000 + self.tick * 273_191,
        )
