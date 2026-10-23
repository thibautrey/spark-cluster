from spark_cluster.capacity import calculate_capacity
from spark_cluster.models import CapacityRequest


def test_capacity_plan_scales_with_context_and_concurrency():
    small = calculate_capacity(CapacityRequest(parameter_count_billions=70, context_length=131072))
    large = calculate_capacity(
        CapacityRequest(parameter_count_billions=70, context_length=524288, concurrent_sequences=4)
    )
    assert large.kv_cache_gib > small.kv_cache_gib
    assert large.minimum_nodes >= small.minimum_nodes


def test_deepseek_671b_four_bit_requires_multiple_128g_nodes():
    plan = calculate_capacity(
        CapacityRequest(parameter_count_billions=671, bits_per_weight=4, context_length=524288)
    )
    assert plan.weights_gib > 300
    assert plan.minimum_nodes >= 4
