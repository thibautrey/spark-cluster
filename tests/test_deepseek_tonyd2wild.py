from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from spark_cluster.app import create_app
from spark_cluster.catalog_recommendation import recommend
from spark_cluster.config import Settings
from spark_cluster.deepseek_tonyd2wild import (
    BASE_MODEL_REVISION,
    RECIPE_REVISION,
    TP2_ENGINE_REVISION,
    TP2_KIT_REVISION,
    TP2_MODEL_ID,
    TP2_MODEL_REVISION,
    TP2_RANK_CACHE_REVISION,
    TP4_MODEL_ID,
    TP4_MODEL_REVISION,
    build_tony_recipe,
)
from spark_cluster.model_manager import GIB, RepositoryMetadata, build_fit_plan
from spark_cluster.models import ModelFitRequest, ModelLaunchRequest
from spark_cluster.performance import PerformancePlanRequest, build_performance_plan
from spark_cluster.runtime_engines import build_execution_plan
from spark_cluster.spark_optimization import classify_spark


def checkpoint(model_id=TP2_MODEL_ID, commit=None):
    tp2 = model_id == TP2_MODEL_ID
    return RepositoryMetadata(
        repo_id=model_id,
        revision="main",
        commit=commit or (TP2_MODEL_REVISION if tp2 else TP4_MODEL_REVISION),
        repository_bytes=int((196.34 if tp2 else 239.95) * GIB),
        weight_bytes=int((196.34 if tp2 else 239.95) * GIB),
        parameter_count=105_318_449_362 if tp2 else 128_729_792_722,
        architecture="DeepseekV41ForCausalLM",
        config={
            "model_type": "deepseek_v41",
            "quantization_config": {
                "quant_method": "exl3",
                "avg_bits": 2.9 if tp2 else 3.5,
            },
        },
        tags=("exl3", "moe"),
        external_artifacts_bytes=203_073_077_576,
        external_artifacts=("base/shard-47", "base/shard-48"),
    )


def fit(tmp_path, monkeypatch, model_id=TP2_MODEL_ID, nodes=None, **updates):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2_048 * GIB),
    )
    metadata = updates.pop("metadata", checkpoint(model_id))
    hardware = updates.pop("hardware", "dgx-spark")
    memory = updates.pop("memory", 121.7)
    context_length = updates.pop("context_length", 262_144)
    nodes = nodes or (2 if model_id == TP2_MODEL_ID else 4)
    return build_fit_plan(
        ModelFitRequest(repo_id=model_id, context_length=context_length, **updates),
        metadata,
        None,
        Settings(
            model_cache_path=tmp_path,
            fallback_node_count=nodes,
            fallback_node_memory_gib=memory,
            active_hardware=hardware,
        ),
    )


def test_tp2_fit_uses_measured_tensorfold_envelope(tmp_path, monkeypatch):
    plan = fit(tmp_path, monkeypatch)
    assert plan.runtime_fits and not plan.launch_supported and not plan.fits
    assert plan.fit_status == "engine-required"
    assert plan.minimum_nodes == 2
    assert plan.resident_weights_gib == pytest.approx(210.08, abs=0.01)
    assert plan.runtime_overhead_gib == pytest.approx(19.92, abs=0.01)
    assert plan.total_required_gib == 230
    assert plan.headroom_gib == pytest.approx(13.4, abs=0.01)
    assert plan.kv_cache_gib == 0
    assert plan.kv_cache_dtype == "tensorfold-native"
    assert plan.ssd_required_gib_per_node == 290
    assert plan.runtime_storage_required_gib > 670
    assert plan.recipe_id == "tonyd2wild-deepseek-v41-tp2"
    assert plan.checkpoint_commit == TP2_MODEL_REVISION
    assert any("external run measurements" in note for note in plan.notes)


def test_tp4_fit_uses_four_rank_memory_and_storage(tmp_path, monkeypatch):
    plan = fit(tmp_path, monkeypatch, model_id=TP4_MODEL_ID)
    assert plan.runtime_fits and not plan.launch_supported
    assert plan.minimum_nodes == 4
    assert plan.resident_weights_gib == pytest.approx(287.48, abs=0.01)
    assert plan.total_required_gib == pytest.approx(374.8, abs=0.01)
    assert plan.headroom_gib == pytest.approx(112.0, abs=0.01)
    assert plan.ssd_required_gib_per_node == 255
    assert plan.recipe_id == "tonyd2wild-deepseek-v41-tp4"


def test_profile_does_not_transfer_to_changed_revision_hardware_or_small_nodes(
    tmp_path, monkeypatch
):
    changed = fit(
        tmp_path,
        monkeypatch,
        metadata=checkpoint(TP2_MODEL_ID, commit="changed"),
    )
    other_hardware = fit(tmp_path, monkeypatch, hardware="nvidia-gpu")
    too_small = fit(tmp_path, monkeypatch, memory=8)
    for plan in (changed, other_hardware, too_small):
        assert not plan.runtime_fits
        assert not plan.launch_supported
    assert changed.recipe_id is None
    assert other_hardware.recipe_id is None


def test_position_pool_blocks_multiple_full_contexts(tmp_path, monkeypatch):
    plan = fit(
        tmp_path,
        monkeypatch,
        context_length=1_048_576,
        concurrent_sequences=2,
    )
    assert not plan.runtime_fits
    assert plan.minimum_nodes == 2
    assert any("shared" in requirement for requirement in plan.requirements)


def test_tp2_performance_plan_is_revision_bound_and_exposes_native_settings():
    plan = build_performance_plan(
        PerformancePlanRequest(
            model_id=TP2_MODEL_ID,
            revision=TP2_MODEL_REVISION,
            node_count=2,
            context_length=1_048_576,
        ),
        available_nodes=2,
    )
    assert plan.engine_id == "tensorfold" and not plan.engine_executable
    assert plan.tensor_parallel == 2 and plan.max_num_seqs == 4
    assert plan.speculative_method == "dspark" and plan.speculative_tokens == 0
    assert plan.recipe.recipe_revision == RECIPE_REVISION
    assert plan.recipe.environment["ENGINE_REVISION"] == TP2_ENGINE_REVISION
    assert plan.recipe.environment["RANK_CACHE_REVISION"] == TP2_RANK_CACHE_REVISION
    assert plan.recipe.environment["KIT_REVISION"] == TP2_KIT_REVISION
    assert plan.recipe.environment["BASE_CHECKPOINT_REVISION"] == BASE_MODEL_REVISION
    assert plan.recipe.environment["TF_DS_WINDOW"] == "1048576"
    assert "Dual-rail" in plan.collective_backend
    assert any("external measurements" in warning for warning in plan.warnings)
    assert any("hidden GB10 slow state" in warning for warning in plan.warnings)
    assert any("double-encoded" in warning for warning in plan.warnings)


def test_tp4_plan_preserves_stream_pool_and_measurement_boundaries():
    plan = build_performance_plan(
        PerformancePlanRequest(
            model_id=TP4_MODEL_ID,
            revision=TP4_MODEL_REVISION,
            node_count=4,
            context_length=1_048_576,
            concurrent_sequences=16,
        ),
        available_nodes=4,
    )
    assert plan.tensor_parallel == 4 and plan.max_num_seqs == 16
    assert plan.recipe.environment["TF_DS_WINDOW"] == "2097152"
    assert plan.recipe.environment["TF_DS_2D_GU"] == "parity"
    assert any("shared pool" in blocker for blocker in plan.blockers)
    assert any("500K" in item.finding for item in plan.evidence)
    assert any("not run" in item.finding for item in plan.evidence)
    assert any("GID" in item for item in plan.recipe.prerequisites)
    assert not any("192.168" in value for value in plan.recipe.environment.values())


def test_wrong_revision_and_missing_nodes_remain_blocked():
    plan = build_performance_plan(
        PerformancePlanRequest(model_id=TP4_MODEL_ID, revision="main", node_count=2),
        available_nodes=2,
    )
    assert any("weight revision" in blocker for blocker in plan.blockers)
    assert any("only 2" in blocker for blocker in plan.blockers)


@pytest.mark.parametrize("model_id", [TP2_MODEL_ID, TP4_MODEL_ID])
def test_managed_runtime_cannot_fall_through_to_generic_engines(model_id):
    automatic = build_execution_plan(
        ModelLaunchRequest(model_id=model_id, served_model_name="deepseek"),
        "/models/deepseek",
        Settings(runtime_tensorfold_image="example/tensorfold:pinned"),
    )
    assert automatic.engine_id == "tensorfold" and not automatic.executable
    assert any("native Zig" in blocker for blocker in automatic.blockers)
    vllm = build_execution_plan(
        ModelLaunchRequest(model_id=model_id, served_model_name="deepseek", engine_id="vllm"),
        "/models/deepseek",
        Settings(),
    )
    assert not vllm.executable
    assert any("not an equivalent fallback" in blocker for blocker in vllm.blockers)


def test_catalog_evidence_uses_exact_recipe_and_weight_revisions():
    for model_id, revision, nodes in (
        (TP2_MODEL_ID, TP2_MODEL_REVISION, [2]),
        (TP4_MODEL_ID, TP4_MODEL_REVISION, [4]),
    ):
        data = {"id": model_id, "sha": revision}
        result = recommend(data, classify_spark(data), date(2026, 10, 25))
        assert result.priority == 2 and result.reported_nodes == nodes
        assert RECIPE_REVISION in result.sources[0]
        assert "TensorFold" in result.runtime
        changed = {**data, "sha": "changed"}
        assert recommend(changed, classify_spark(changed), date(2026, 10, 25)).priority < 2


def test_http_and_mcp_return_same_tp2_recipe(tmp_path):
    app = create_app(
        Settings(
            mode="demo",
            database_path=tmp_path / "test.db",
            model_cache_path=tmp_path / "models",
        )
    )
    request = {
        "model_id": TP2_MODEL_ID,
        "revision": TP2_MODEL_REVISION,
        "node_count": 2,
    }
    with TestClient(app) as client:
        api = client.post("/api/v1/performance/plan", json=request).json()
        rpc = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "get_performance_plan", "arguments": request},
            },
        ).json()["result"]["structuredContent"]
    assert api["recipe"] == rpc["recipe"]
    assert api["blockers"] == rpc["blockers"]


def test_recipe_rejects_unreviewed_model_and_bounds_context():
    with pytest.raises(ValueError, match="No reviewed"):
        build_tony_recipe("other/model", 32_768, 1)
    recipe = build_tony_recipe(TP2_MODEL_ID, 2_000_000, 8, revision="main")
    assert any("weight revision" in blocker for blocker in recipe.blockers)
    assert any("at most" in blocker for blocker in recipe.blockers)
    assert any("streams" in blocker for blocker in recipe.blockers)
    assert any("shared" in blocker for blocker in recipe.blockers)
