from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.catalog_recommendation import recommend
from spark_cluster.config import Settings
from spark_cluster.deepseek_mia import (
    KV_BYTES_PER_TOKEN_PER_RANK,
    MODEL_ID,
    MODEL_REVISION,
    RECIPE_REVISION,
    build_mia_recipe,
)
from spark_cluster.model_manager import GIB, RepositoryMetadata, build_fit_plan
from spark_cluster.models import ModelFitRequest, ModelLaunchRequest
from spark_cluster.performance import PerformancePlanRequest, build_performance_plan
from spark_cluster.runtime_engines import build_execution_plan
from spark_cluster.spark_optimization import classify_spark


def checkpoint(commit=MODEL_REVISION):
    return RepositoryMetadata(
        repo_id=MODEL_ID,
        revision="main",
        commit=commit,
        repository_bytes=int(475.25 * GIB),
        weight_bytes=int(475.25 * GIB),
        parameter_count=763_205_315_794,
        architecture="DeepseekV41ForCausalLM",
        config={},
        engram_bytes=203_073_077_576,
    )


def fit(tmp_path, monkeypatch, nodes=3, free_gib=2048, **updates):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=int(free_gib * GIB)),
    )
    metadata = updates.pop("metadata", checkpoint())
    hardware = updates.pop("hardware", "dgx-spark")
    memory = updates.pop("memory", 121.7)
    return build_fit_plan(
        ModelFitRequest(repo_id=MODEL_ID, context_length=32_768, **updates),
        metadata,
        None,
        Settings(
            model_cache_path=tmp_path,
            fallback_node_count=nodes,
            fallback_node_memory_gib=memory,
            active_hardware=hardware,
        ),
    )


def test_native_recipe_requires_three_sparks_and_is_not_an_executable_adapter(
    tmp_path, monkeypatch
):
    two = fit(tmp_path, monkeypatch, nodes=2)
    three = fit(tmp_path, monkeypatch, nodes=3)
    assert not two.runtime_fits
    assert three.minimum_nodes == 3
    assert three.runtime_fits and not three.memory_fits
    assert three.fit_status == "engine-required"
    assert not three.fits and not three.launch_supported
    assert three.resident_weights_gib == 305
    assert three.kv_cache_gib == pytest.approx(
        750_000 * KV_BYTES_PER_TOKEN_PER_RANK * 3 / GIB, abs=0.01
    )
    assert three.ssd_required_gib_per_node >= 63
    assert three.runtime_storage_required_gib > three.repository_bytes / GIB + 63
    assert three.recipe_id == "mia-deepseek-v41-tp3"
    assert three.checkpoint_commit == MODEL_REVISION


def test_capacity_does_not_transfer_to_changed_checkpoint_or_other_hardware(tmp_path, monkeypatch):
    for changes in (
        {"metadata": checkpoint("changed")},
        {"metadata": checkpoint(None)},
        {"hardware": "nvidia-gpu"},
    ):
        plan = fit(tmp_path, monkeypatch, **changes)
        assert plan.recipe_id is None
        assert not plan.launch_supported
    assert not fit(tmp_path, monkeypatch, memory=8).runtime_fits


def test_fixed_native_kv_pool_is_not_rescaled_by_generic_precision(tmp_path, monkeypatch):
    native = fit(tmp_path, monkeypatch)
    fp8 = fit(tmp_path, monkeypatch, kv_cache_dtype="fp8_e4m3")
    assert fp8.kv_cache_gib == native.kv_cache_gib
    assert fp8.kv_cache_dtype == "native-compressed"


def test_packed_head_rows_are_additional_storage(tmp_path, monkeypatch):
    plan = fit(tmp_path, monkeypatch, free_gib=476)
    assert plan.download_fits
    assert not plan.runtime_storage_fits
    assert not plan.fits


@pytest.mark.parametrize("sequences,expected", [(1, True), (7, True), (8, False), (16, False)])
def test_tp4_contexts_share_the_pinned_pool(tmp_path, monkeypatch, sequences, expected):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2048 * GIB),
    )
    plan = build_fit_plan(
        ModelFitRequest(repo_id=MODEL_ID, context_length=1_048_576, concurrent_sequences=sequences),
        checkpoint(),
        None,
        Settings(model_cache_path=tmp_path, fallback_node_count=4, fallback_node_memory_gib=121.7),
    )
    assert plan.runtime_fits is expected
    assert plan.recipe_id == "mia-deepseek-v41-tp4"
    assert not plan.launch_supported


def test_performance_plan_has_safe_tp3_defaults_and_opt_in_long_context():
    balanced = build_performance_plan(
        PerformancePlanRequest(model_id=MODEL_ID, revision=MODEL_REVISION, node_count=3), 3
    )
    assert balanced.context_length == 32_768
    assert balanced.engine_id == "sglang"
    assert not balanced.engine_executable
    assert balanced.max_num_seqs == 4
    assert balanced.speculative_method == "dspark" and balanced.speculative_tokens == 5
    assert not balanced.expert_parallel
    assert balanced.cache_cpu_gib == balanced.cache_disk_gib == 0
    assert "triangle" in balanced.collective_backend
    assert balanced.recipe.environment["EP_SIZE"] == "1"
    assert balanced.recipe.environment["CHUNKED_PREFILL_SIZE"] == "768"
    longer = build_performance_plan(
        PerformancePlanRequest(model_id=MODEL_ID, node_count=3, workload="long-context"), 3
    )
    assert longer.context_length == 262_144
    assert any("head-memory" in warning for warning in longer.warnings)


def test_tp4_plan_preserves_production_requirements_and_pool_limit():
    plan = build_performance_plan(
        PerformancePlanRequest(
            model_id=MODEL_ID,
            node_count=4,
            topology="ring",
            context_length=1_048_576,
            concurrent_sequences=16,
        ),
        4,
        Settings(runtime_sglang_image="example/stock-sglang:pinned"),
    )
    assert not plan.engine_executable
    assert plan.max_num_seqs == 16
    assert plan.gpu_memory_utilization == 0.8
    assert plan.recipe.environment["DSV41_INDEXER_CHUNKED"] == "1"
    assert any("KV pool" in item for item in plan.blockers)
    assert any("NCCL patch" in item for item in plan.blockers)
    assert any("Dockerfile.canary-roce" in item for item in plan.recipe.prerequisites)
    assert any("synthetic" in item.finding for item in plan.evidence)


@pytest.mark.parametrize("engine", ["auto", "vllm", "sglang", "tensorfold"])
def test_native_checkpoint_cannot_fall_through_to_stock_runtime(engine):
    plan = build_execution_plan(
        ModelLaunchRequest(model_id=MODEL_ID, served_model_name="deepseek", engine_id=engine),
        "/models/deepseek",
        Settings(runtime_sglang_image="example/sglang:pinned"),
    )
    assert not plan.executable
    assert any("MIA multi-node" in blocker for blocker in plan.blockers)


def test_profile_bounds_and_recipe_identity():
    with pytest.raises(ValueError, match="three or four"):
        build_mia_recipe(2, 32_768, 1)
    recipe = build_mia_recipe(3, 524_288, 5)
    assert recipe.recipe_revision == RECIPE_REVISION
    assert any("ceiling" in blocker for blocker in recipe.blockers)
    assert any("active requests" in blocker for blocker in recipe.blockers)
    assert recipe.environment["OFFLOAD_MODE"] == "nvme"
    assert recipe.environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"


def test_catalog_evidence_is_revision_bound_and_expires():
    from datetime import date

    data = {"id": MODEL_ID, "sha": MODEL_REVISION}
    current = recommend(data, classify_spark(data), date(2026, 10, 25))
    assert current.priority == 2 and current.reported_nodes == [3, 4]
    assert "SGLang" in current.runtime
    assert RECIPE_REVISION in current.sources[0]
    assert recommend(data, classify_spark(data), date(2026, 11, 25)).priority < 2
    changed = {**data, "sha": "other"}
    assert recommend(changed, classify_spark(changed), date(2026, 10, 25)).priority < 2


def test_http_plan_exposes_reviewed_recipe_without_enabling_launch(tmp_path):
    app = create_app(
        Settings(
            mode="demo", database_path=tmp_path / "test.db", model_cache_path=tmp_path / "models"
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/performance/plan",
            json={"model_id": MODEL_ID, "revision": MODEL_REVISION, "node_count": 3},
        )
        assert response.status_code == 200
        plan = response.json()
        assert plan["recipe"]["launcher"] == "./start.sh"
        assert plan["recipe"]["environment"]["CONTEXT_LENGTH"] == "32768"
        assert plan["blockers"] and not plan["engine_executable"]
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "get_performance_plan",
                    "arguments": {
                        "model_id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "node_count": 3,
                    },
                },
            },
        )
        result = response.json()["result"]
        assert result["structuredContent"]["recipe"] == plan["recipe"]
        assert result["structuredContent"]["blockers"] == plan["blockers"]
