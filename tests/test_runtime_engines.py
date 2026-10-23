import pytest
from pydantic import ValidationError

from spark_cluster.config import Settings
from spark_cluster.models import ModelLaunchRequest
from spark_cluster.runtime_engines import build_execution_plan


def request(**updates):
    values = {
        "model_id": "acme/moe-model",
        "served_model_name": "model",
        "context_length": 262144,
        "tensor_parallel": 4,
        "pipeline_parallel": 1,
    }
    values.update(updates)
    return ModelLaunchRequest(**values)


def test_vllm_execution_plan_makes_advanced_options_executable():
    plan = build_execution_plan(
        request(
            speculative_method="dspark",
            speculative_tokens=3,
            expert_parallel=True,
            expert_load_balancing=True,
            redundant_experts=2,
            all2all_backend="allgather_reducescatter",
        ),
        "/models/acme/moe-model",
        Settings(runtime_vllm_speculative_methods="off,auto,mtp,dspark"),
    )
    assert plan.executable is True
    assert plan.engine_id == "vllm"
    assert plan.environment["SPECULATIVE_CONFIG"] == (
        '{"method":"dspark","num_speculative_tokens":3}'
    )
    assert plan.environment["EXPERT_PARALLEL"] == "1"
    assert plan.environment["EXPERT_LOAD_BALANCING"] == "1"
    assert "--speculative-config" in plan.command
    assert "expert parallelism" in plan.capabilities


def test_unvalidated_engines_are_visible_but_cannot_be_launched():
    unsupported_dspark = build_execution_plan(
        request(speculative_method="dspark", speculative_tokens=3),
        "/models/deepseek",
        Settings(),
    )
    assert unsupported_dspark.executable is False
    assert any("dspark" in blocker.casefold() for blocker in unsupported_dspark.blockers)
    tensorfold = build_execution_plan(
        request(engine_id="tensorfold"), "/models/deepseek", Settings()
    )
    assert tensorfold.executable is False
    assert any("command contract" in blocker for blocker in tensorfold.blockers)
    deepinfer = build_execution_plan(
        request(engine_id="deepinfer", speculative_method="dspark", speculative_tokens=3),
        "/models/deepseek",
        Settings(runtime_deepinfer_image="example/deepinfer:latest"),
    )
    assert deepinfer.executable is False
    assert any("SM121" in blocker for blocker in deepinfer.blockers)


def test_launch_validation_rejects_invalid_speculation_moe_and_rank_plans():
    with pytest.raises(ValidationError, match="DSpark"):
        request(speculative_method="dspark", speculative_tokens=1)
    with pytest.raises(ValidationError, match="requires expert parallelism"):
        request(expert_load_balancing=True)
    with pytest.raises(ValidationError, match="exactly one node"):
        request(rank_order=["spark-1", "spark-2"])
    with pytest.raises(ValidationError, match="duplicate"):
        request(rank_order=["spark-1"] * 4)
