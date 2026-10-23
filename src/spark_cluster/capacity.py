import math

from .models import CapacityPlan, CapacityRequest


def calculate_capacity(request: CapacityRequest) -> CapacityPlan:
    weights_gib = (
        request.parameter_count_billions * 1_000_000_000 * request.bits_per_weight / 8 / 2**30
    )
    kv_cache_gib = (
        request.context_length
        * request.concurrent_sequences
        * request.kv_bytes_per_token_billions
        * 1_000_000_000
        / 2**30
    )
    runtime_overhead_gib = weights_gib * 0.08 + 8
    total = weights_gib + kv_cache_gib + runtime_overhead_gib
    usable = request.node_memory_gib * request.usable_fraction
    return CapacityPlan(
        weights_gib=round(weights_gib, 2),
        kv_cache_gib=round(kv_cache_gib, 2),
        runtime_overhead_gib=round(runtime_overhead_gib, 2),
        total_required_gib=round(total, 2),
        usable_per_node_gib=round(usable, 2),
        minimum_nodes=math.ceil(total / usable),
    )
