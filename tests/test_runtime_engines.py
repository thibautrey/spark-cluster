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


def test_local_sglang_accepts_tensor_parallel_but_rejects_unwired_options():
    baseline = build_execution_plan(
        request(engine_id="sglang", tensor_parallel=2),
        "/models/acme/model",
        Settings(),
        image_overrides={"sglang": "lmsysorg/sglang:v0.5.22"},
        cluster_kind="local-docker",
    )
    assert baseline.executable is True
    speculative = build_execution_plan(
        request(
            engine_id="sglang",
            tensor_parallel=2,
            speculative_method="mtp",
            speculative_tokens=2,
        ),
        "/models/acme/model",
        Settings(),
        image_overrides={"sglang": "lmsysorg/sglang:v0.5.22"},
        cluster_kind="local-docker",
    )
    assert speculative.executable is False
    assert any("mtp" in blocker.casefold() for blocker in speculative.blockers)


@pytest.mark.parametrize(
    "engine_id,setting,image,evidence,tensor_parallel,pipeline_parallel,protocol",
    [
        (
            "llama-cpp",
            "runtime_llama_cpp_image",
            "llama:server-cuda",
            "llama-server",
            1,
            1,
            "completions",
        ),
        ("ollama", "runtime_ollama_image", "ollama:1.0.0", "ollama", 1, 1, "chat"),
        (
            "tensorrt-llm",
            "runtime_tensorrt_llm_image",
            "tensorrt:1.1.0",
            "trtllm-serve",
            2,
            2,
            "completions",
        ),
        ("triton", "runtime_triton_image", "triton:26.09-py3", "tritonserver", 1, 1, "triton"),
        ("tgi", "runtime_tgi_image", "tgi:3.3.7", "text-generation-launcher", 2, 1, "chat"),
        ("lmdeploy", "runtime_lmdeploy_image", "lmdeploy:0.11.0", "lmdeploy", 2, 1, "completions"),
        (
            "xinference",
            "runtime_xinference_image",
            "xinference:1.9.0",
            "xinference",
            1,
            1,
            "completions",
        ),
        ("localai", "runtime_localai_image", "localai:3.6.0", "local-ai", 1, 1, "completions"),
    ],
)
def test_catalogue_runtime_execution_plans_are_managed(
    engine_id,
    setting,
    image,
    evidence,
    tensor_parallel,
    pipeline_parallel,
    protocol,
):
    plan = build_execution_plan(
        request(
            engine_id=engine_id,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
        ),
        "/models/acme/model",
        Settings(**{setting: image}),
        cluster_kind="local-docker",
    )
    assert plan.executable is True
    assert plan.image == image
    assert evidence in " ".join(plan.command)
    assert plan.inference_protocol == protocol


def test_launch_validation_rejects_invalid_speculation_moe_and_rank_plans():
    with pytest.raises(ValidationError, match="DSpark"):
        request(speculative_method="dspark", speculative_tokens=1)
    with pytest.raises(ValidationError, match="requires expert parallelism"):
        request(expert_load_balancing=True)
    with pytest.raises(ValidationError, match="exactly one node"):
        request(rank_order=["spark-1", "spark-2"])
    with pytest.raises(ValidationError, match="duplicate"):
        request(rank_order=["spark-1"] * 4)


@pytest.mark.parametrize('engine_id', ['vllm', 'sglang'])
def test_ram_offload_is_an_executable_local_runtime_contract(engine_id):
    plan = build_execution_plan(
        request(engine_id=engine_id, tensor_parallel=1, cpu_offload_gib=3.5),
        '/models/acme/model', Settings(runtime_sglang_image='sglang:pinned'),
        cluster_kind='local-docker',
    )
    assert plan.executable
    assert plan.environment['CPU_OFFLOAD_GIB'] == '3.5'
    assert '--cpu-offload-gb' in plan.command
    assert any('slower' in warning for warning in plan.warnings)
    remote = build_execution_plan(
        request(engine_id=engine_id, tensor_parallel=1, cpu_offload_gib=3.5),
        '/models/acme/model', Settings(runtime_sglang_image='sglang:pinned'),
    )
    assert not remote.executable


def test_cpu_layers_require_llama_cpp_and_do_not_enable_other_engines():
    launch = request(tensor_parallel=1, engine_id='llama-cpp', gpu_layers=12)
    plan = build_execution_plan(launch, '/models/gguf', Settings(), cluster_kind='local-docker')
    assert '--n-gpu-layers' in plan.command
    assert plan.environment['GPU_LAYERS'] == '12'
    assert plan.environment['CPU_KV'] == '1'
    invalid = build_execution_plan(
        request(tensor_parallel=1, engine_id='vllm', gpu_layers=12),
        '/models/model', Settings(), cluster_kind='local-docker',
    )
    assert not invalid.executable
    with pytest.raises(ValidationError):
        request(cpu_offload_gib=2, gpu_layers=12)
