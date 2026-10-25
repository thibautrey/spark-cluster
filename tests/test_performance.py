from spark_cluster.config import Settings
from spark_cluster.performance import PerformancePlanRequest, build_performance_plan


def test_deepseek_v41_exl3_plan_uses_pinned_tensorfold_dual_rail_findings():
    plan = build_performance_plan(
        PerformancePlanRequest(
            model_id="LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw",
            revision="04a492fc3b4d25ddbab8f2d2f07f235ec77c9010",
            node_count=4,
            topology="ring",
            context_length=262_144,
        ),
        available_nodes=4,
    )
    assert plan.engine_id == "tensorfold"
    assert plan.engine_executable is False
    assert "TensorFold native" in plan.engine
    assert plan.tensor_parallel == 4
    assert plan.context_length == 262_144
    assert 1_048_576 in plan.context_options
    assert plan.max_num_seqs == 16
    assert plan.speculative_method == "dspark"
    assert plan.collective_backend == "Dual-rail RoCE four-rank fabric"
    assert any("both rails" in item.finding.casefold() for item in plan.evidence)
    assert plan.recipe.environment["TF_DS_WINDOW"] == "2097152"


def test_glm53_plan_uses_fp8_kv_mtp_evidence_and_no_fixed_gid():
    plan = build_performance_plan(
        PerformancePlanRequest(
            model_id="zai-org/GLM-5.3-Flash",
            node_count=4,
            topology="switch",
            concurrent_sequences=2,
        ),
        available_nodes=4,
    )
    assert plan.engine == "vLLM SM121 optimized image"
    assert plan.kv_cache_dtype == "fp8_e4m3"
    assert plan.gpu_memory_utilization == 0.85
    assert plan.max_num_seqs == 6
    assert "RoCEnante" in plan.collective_backend
    assert any("GID_INDEX" in warning for warning in plan.warnings)
    assert any("non-commercial" in warning for warning in plan.warnings)


def test_portable_profiles_bound_scheduler_for_workload():
    throughput = build_performance_plan(
        PerformancePlanRequest(
            model_id="Qwen/Qwen3.8-27B",
            node_count=8,
            workload="throughput",
            concurrent_sequences=12,
        ),
        available_nodes=4,
    )
    assert throughput.tensor_parallel == 8
    assert throughput.max_num_seqs == 12
    assert throughput.max_num_batched_tokens == 16_384
    assert throughput.gpu_memory_utilization == 0.85

    context = build_performance_plan(
        PerformancePlanRequest(
            model_id="Qwen/Qwen3.8-27B",
            node_count=2,
            workload="long-context",
            concurrent_sequences=1,
        ),
        available_nodes=2,
    )
    assert context.kv_cache_dtype == "fp8_e4m3"
    assert context.max_num_seqs == 1


def test_generic_nvidia_host_does_not_receive_spark_fabric_guidance():
    plan = build_performance_plan(
        PerformancePlanRequest(model_id="Qwen/Qwen2.5-Coder-3B-Instruct-AWQ", node_count=2),
        available_nodes=2,
        settings=Settings(active_hardware="nvidia-gpu", active_compute_capability=8.6),
    )

    assert plan.collective_backend == "NCCL over local PCIe"
    assert "peer access" in plan.topology_guidance
    assert "GB10" not in plan.engine_guidance
    assert plan.evidence == []
