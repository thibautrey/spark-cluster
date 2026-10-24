import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from spark_cluster.config import Settings
from spark_cluster.model_manager import (
    GIB,
    ModelManager,
    RepositoryMetadata,
    build_fit_plan,
    safe_repo_parts,
)
from spark_cluster.models import DownloadJob, ModelDownloadRequest, ModelFitRequest, RuntimeEndpoint
from spark_cluster.simulator import ClusterSimulator
from spark_cluster.store import SnapshotStore


def metadata(
    parameters: int = 70_000_000_000,
    weight_gib: float = 35,
    **values,
) -> RepositoryMetadata:
    return RepositoryMetadata(
        repo_id="acme/model",
        revision="main",
        repository_bytes=1 * GIB,
        weight_bytes=int(weight_gib * GIB),
        parameter_count=parameters,
        architecture="DeepseekV3ForCausalLM",
        config={
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_hidden_layers": 61,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
        **values,
    )


def test_safe_repo_id_rejects_paths():
    assert safe_repo_parts("acme/model") == ("acme", "model")
    with pytest.raises(ValueError):
        safe_repo_parts("../../secret")


def test_fit_plan_uses_current_cluster_and_architecture(tmp_path):
    settings = Settings(model_cache_path=tmp_path, model_usable_memory_fraction=0.82)
    snapshot = ClusterSimulator("test").snapshot()
    plan = build_fit_plan(
        ModelFitRequest(
            repo_id="acme/model",
            bits_per_weight=4,
            context_length=524288,
            concurrent_sequences=1,
        ),
        metadata(),
        snapshot,
        settings,
    )
    assert plan.confidence == "architecture"
    assert plan.kv_cache_gib > 30
    assert plan.cluster_nodes == 4
    assert plan.memory_fits is True
    assert plan.fits is True


def test_fit_plan_rejects_model_beyond_cluster_memory(tmp_path):
    settings = Settings(model_cache_path=tmp_path)
    snapshot = ClusterSimulator("test").snapshot()
    plan = build_fit_plan(
        ModelFitRequest(repo_id="acme/model", bits_per_weight=16, context_length=524288),
        metadata(parameters=671_000_000_000, weight_gib=1250),
        snapshot,
        settings,
    )
    assert plan.memory_fits is False
    assert plan.minimum_nodes > plan.cluster_nodes
    assert plan.fits is False
    assert plan.download_fits is True


def test_deepseek_v41_engram_uses_validated_local_nvme_placement(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2 * 1024 * GIB),
    )
    engram_bytes = 190 * GIB
    repository = RepositoryMetadata(
        repo_id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        revision="main",
        repository_bytes=475 * GIB,
        weight_bytes=475 * GIB,
        parameter_count=763_000_000_000,
        architecture="DeepseekV41ForCausalLM",
        config={
            "model_type": "deepseek_v41",
            "text_config": {
                "model_type": "deepseek_v41",
                "num_hidden_layers": 61,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
            },
        },
        files=(
            ("model-00001-of-00048.safetensors", 285 * GIB),
            ("model-00047-of-00048.safetensors", 95 * GIB),
            ("model-00048-of-00048.safetensors", 95 * GIB),
        ),
        tags=("moe", "fp8"),
        library_name="transformers",
        pipeline_tag="image-text-to-text",
        engram_bytes=engram_bytes,
    )
    plan = build_fit_plan(
        ModelFitRequest(repo_id=repository.repo_id, context_length=131_072),
        repository,
        ClusterSimulator("test").snapshot(),
        Settings(model_cache_path=tmp_path),
    )
    assert plan.memory_fits is False
    assert plan.runtime_fits is True
    assert plan.launch_supported is False
    assert plan.download_fits is True
    assert plan.fit_status == "engine-required"
    assert plan.placement_strategy == "engram-nvme"
    assert plan.resident_weights_gib == pytest.approx(285, abs=0.1)
    assert plan.offloaded_weights_gib == pytest.approx(190, abs=0.1)
    assert plan.ssd_required_gib_per_node == pytest.approx(47.5, abs=0.1)
    assert any("Engram" in item for item in plan.requirements)


def test_quantized_engram_pack_accounts_for_external_base_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2 * 1024 * GIB),
    )
    repository = RepositoryMetadata(
        repo_id="LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw",
        revision="main",
        repository_bytes=240 * GIB,
        weight_bytes=240 * GIB,
        parameter_count=128_000_000_000,
        architecture="DeepseekV41ForCausalLM",
        config={
            "model_type": "deepseek_v41",
            "quantization_config": {"quant_method": "exl3", "avg_bits": 3.5},
            "text_config": {
                "model_type": "deepseek_v41",
                "num_hidden_layers": 61,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
            },
        },
        files=(("model-00001-of-00044.safetensors", 240 * GIB),),
        tags=("exl3", "moe"),
        external_artifacts_bytes=190 * GIB,
        external_artifacts=("base/shard-47", "base/shard-48"),
    )
    plan = build_fit_plan(
        ModelFitRequest(repo_id=repository.repo_id, context_length=524_288),
        repository,
        ClusterSimulator("test").snapshot(),
        Settings(model_cache_path=tmp_path),
    )
    assert plan.weights_gib == pytest.approx(430, abs=0.1)
    assert plan.resident_weights_gib == pytest.approx(240, abs=0.1)
    assert plan.additional_artifacts_bytes == 190 * GIB
    assert plan.runtime_storage_required_gib == pytest.approx(430, abs=0.1)
    assert any("two Engram shards" in item for item in plan.requirements)


def test_prefix_cache_does_not_claim_to_offload_active_decode_kv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2 * 1024 * GIB),
    )
    repository = RepositoryMetadata(
        repo_id="acme/long-context",
        revision="main",
        repository_bytes=300 * GIB,
        weight_bytes=300 * GIB,
        parameter_count=160_000_000_000,
        architecture="LlamaForCausalLM",
        config={
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "hidden_size": 8192,
        },
        files=(("model.safetensors", 300 * GIB),),
        library_name="transformers",
        pipeline_tag="text-generation",
    )
    plan = build_fit_plan(
        ModelFitRequest(repo_id=repository.repo_id, context_length=131_072),
        repository,
        ClusterSimulator("test").snapshot(),
        Settings(model_cache_path=tmp_path),
    )
    assert plan.memory_fits is False
    assert plan.runtime_fits is False
    assert plan.launch_supported is True
    assert plan.fits is False
    assert plan.fit_status == "download-only"
    assert plan.placement_strategy == "kv-offload"
    assert plan.offloaded_kv_cache_gib == 0
    assert plan.offloaded_weights_gib == 0


def test_moe_and_gguf_do_not_claim_unvalidated_fast_offload(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2 * 1024 * GIB),
    )
    snapshot = ClusterSimulator("test").snapshot()
    settings = Settings(model_cache_path=tmp_path)
    moe = RepositoryMetadata(
        repo_id="acme/huge-moe",
        revision="main",
        repository_bytes=600 * GIB,
        weight_bytes=600 * GIB,
        parameter_count=500_000_000_000,
        architecture="HugeMoeForCausalLM",
        config={"n_routed_experts": 256, "num_experts_per_tok": 8},
        files=(("model.safetensors", 600 * GIB),),
        tags=("moe",),
        library_name="transformers",
        pipeline_tag="text-generation",
    )
    moe_plan = build_fit_plan(ModelFitRequest(repo_id=moe.repo_id), moe, snapshot, settings)
    assert moe_plan.placement_strategy == "expert-offload"
    assert moe_plan.runtime_fits is False
    assert moe_plan.download_fits is True
    assert any("3.1%" in note for note in moe_plan.notes)

    gguf = RepositoryMetadata(
        repo_id="acme/model-GGUF",
        revision="main",
        repository_bytes=600 * GIB,
        weight_bytes=600 * GIB,
        parameter_count=None,
        architecture=None,
        config={},
        files=(("model-Q4_K_M.gguf", 600 * GIB),),
        tags=("gguf",),
        pipeline_tag="text-generation",
    )
    gguf_plan = build_fit_plan(ModelFitRequest(repo_id=gguf.repo_id), gguf, snapshot, settings)
    assert gguf_plan.placement_strategy == "mmap"
    assert gguf_plan.runtime_fits is False
    assert gguf_plan.launch_supported is False
    assert gguf_plan.performance_impact == "high"


def test_repository_download_is_allowed_without_a_runtime_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage",
        lambda _: SimpleNamespace(free=2 * 1024 * GIB),
    )
    plan = build_fit_plan(
        ModelFitRequest(repo_id="acme/huge"),
        metadata(weight_gib=600),
        ClusterSimulator("test").snapshot(),
        Settings(model_cache_path=tmp_path),
    )
    assert plan.runtime_fits is False
    assert plan.download_fits is True
    assert plan.fits is False


def test_interrupted_download_is_requeued_on_restart(tmp_path):
    settings = Settings(model_cache_path=tmp_path / "models")
    store = SnapshotStore(tmp_path / "state.db")
    fit = build_fit_plan(
        ModelFitRequest(repo_id="acme/model"),
        metadata(),
        ClusterSimulator("test").snapshot(),
        settings,
    )
    store.save_download(
        DownloadJob(
            id="job-1",
            repo_id="acme/model",
            revision="main",
            status="downloading",
            total_bytes=fit.repository_bytes,
            target_path=str(settings.model_cache_path / "acme" / "model"),
            staging_path=str(settings.model_cache_path / ".partial" / "job-1"),
            fit=fit,
        )
    )
    recovered = store.recover_downloads()
    assert [job.id for job in recovered] == ["job-1"]
    assert store.get_download("job-1").status == "queued"
    store.close()


def test_progress_includes_completed_and_incomplete_shards(tmp_path):
    staging = tmp_path / "partial"
    (staging / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (staging / "config.json").write_bytes(b"1234")
    (staging / ".cache" / "huggingface" / "download" / "shard.incomplete").write_bytes(b"123456")
    assert ModelManager._staging_progress(staging) == 10


@pytest.mark.asyncio
async def test_download_job_completes_and_model_is_in_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    settings = Settings(model_cache_path=tmp_path / "models")
    store = SnapshotStore(tmp_path / "state.db")
    runtime = SimpleNamespace(latest=ClusterSimulator("test").snapshot())
    manager = ModelManager(settings, store, runtime)

    class FakeCatalog:
        def inspect(self, repo_id, revision):
            return metadata()

        def download(self, repo_id, revision, target: Path):
            (target / "config.json").write_text("{}")
            (target / "model.safetensors").write_bytes(b"weights")

    manager.catalog = FakeCatalog()
    await manager.start()
    try:
        job = await manager.enqueue(ModelDownloadRequest(repo_id="acme/model"))
        await manager.queue.join()
        completed = store.get_download(job.id)
        assert completed is not None
        assert completed.status == "completed"
        models = await manager.local_models()
        assert [(model.id, model.source) for model in models] == [("acme/model", "managed")]
        manifest = json.loads(
            (settings.model_cache_path / "acme" / "model" / ".spark-cluster.json").read_text()
        )
        assert manifest["repo_id"] == "acme/model"
    finally:
        await manager.stop()
        store.close()


@pytest.mark.asyncio
async def test_download_can_pause_resume_and_cancel_without_losing_control(tmp_path):
    settings = Settings(model_cache_path=tmp_path / "models")
    store = SnapshotStore(tmp_path / "state.db")
    runtime = SimpleNamespace(latest=ClusterSimulator("test").snapshot())
    manager = ModelManager(settings, store, runtime)
    plan = build_fit_plan(
        ModelFitRequest(repo_id="acme/model"),
        metadata(),
        runtime.latest,
        settings,
    )
    staging = settings.model_cache_path / ".partial" / "job-1"
    staging.mkdir(parents=True)
    (staging / "partial.bin").write_bytes(b"partial")
    job = DownloadJob(
        id="job-1",
        repo_id="acme/model",
        revision="main",
        status="queued",
        target_path=str(settings.model_cache_path / "acme" / "model"),
        staging_path=str(staging),
        fit=plan,
    )
    store.save_download(job)

    paused = await manager.pause(job.id)
    assert paused.status == "paused"
    assert (staging / "partial.bin").exists()
    resumed = await manager.resume(job.id)
    assert resumed.status == "queued"
    assert await manager.queue.get() == job.id
    manager.queue.task_done()
    await manager.pause(job.id)
    cancelled = await manager.cancel(job.id)
    assert cancelled.status == "cancelled"
    assert not staging.exists()
    store.close()


@pytest.mark.asyncio
async def test_delete_managed_model_removes_files_and_marks_history(tmp_path):
    settings = Settings(
        model_cache_path=tmp_path / "models",
        openai_api_base="http://127.0.0.1:9/v1",
    )
    store = SnapshotStore(tmp_path / "state.db")
    runtime = SimpleNamespace(latest=ClusterSimulator("test").snapshot())
    manager = ModelManager(settings, store, runtime)
    plan = build_fit_plan(
        ModelFitRequest(repo_id="acme/model"),
        metadata(),
        runtime.latest,
        settings,
    )
    target = settings.model_cache_path / "acme" / "model"
    target.mkdir(parents=True)
    (target / "weights.bin").write_bytes(b"model-data")
    (target / ".spark-cluster.json").write_text("{}")
    store.save_download(
        DownloadJob(
            id="job-1",
            repo_id="acme/model",
            revision="main",
            status="completed",
            target_path=str(target),
            staging_path=str(settings.model_cache_path / ".partial" / "job-1"),
            fit=plan,
        )
    )

    result = await manager.delete_model("acme/model")
    assert result.status == "deleted"
    assert result.controller_deleted is True
    assert result.cluster_deleted is True
    assert result.freed_bytes > 0
    assert not target.exists()
    assert store.get_download("job-1").status == "deleted"
    store.close()


def test_kv_head_replication_is_not_divided_by_tensor_parallel(tmp_path):
    from spark_cluster.model_manager import _node_requirement

    # MLA/MQA has one KV head replicated over tensor ranks.
    fits, _, total, _ = _node_requirement([100, 100, 100, 100], 4, 200, 40, 1)
    assert fits
    assert total == pytest.approx(392)
    # Eight GQA KV heads shard across four ranks.
    _, _, total, _ = _node_requirement([100] * 4, 4, 200, 40, 8)
    assert total == pytest.approx(272)


def test_small_auxiliary_format_does_not_replace_primary_weights(tmp_path):
    from spark_cluster.model_manager import _runtime_weight_bytes

    weights, notes = _runtime_weight_bytes(
        metadata(
            files=(
                ("model.safetensors", 100 * GIB),
                ("auxiliary.bin", GIB),
            )
        )
    )
    assert weights == 100 * GIB
    assert notes


def test_repository_storage_shortage_blocks_even_download_override(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage", lambda _: SimpleNamespace(free=GIB // 2)
    )
    plan = build_fit_plan(
        ModelDownloadRequest(repo_id="acme/model", allow_oversized=True),
        metadata(),
        ClusterSimulator("test").snapshot(),
        Settings(model_cache_path=tmp_path),
    )
    assert not plan.download_fits
    assert not plan.fits
    assert plan.fit_status == "insufficient-storage"


def test_fp8_kv_halves_cache_estimate_without_converting_weights(tmp_path):
    snapshot = ClusterSimulator("test").snapshot()
    settings = Settings(model_cache_path=tmp_path)
    bf16 = build_fit_plan(
        ModelFitRequest(repo_id="acme/model", kv_cache_dtype="bf16"),
        metadata(),
        snapshot,
        settings,
    )
    fp8 = build_fit_plan(
        ModelFitRequest(repo_id="acme/model", kv_cache_dtype="fp8_e4m3"),
        metadata(),
        snapshot,
        settings,
    )
    assert bf16.kv_cache_gib == pytest.approx(fp8.kv_cache_gib * 2, abs=0.02)
    assert bf16.weights_gib == fp8.weights_gib


def test_gguf_artifacts_take_priority_over_base_model_tags(tmp_path):
    from spark_cluster.model_manager import _quantization

    assert _quantization(
        metadata(
            files=(("model-Q4_K_M.gguf", GIB),),
            tags=("fp8", "base_model:acme/fp8"),
        )
    ) == ("GGUF Q4", 4)


def test_cached_repository_does_not_need_double_disk_space(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spark_cluster.model_manager.shutil.disk_usage", lambda _: SimpleNamespace(free=GIB // 2)
    )
    plan = build_fit_plan(
        ModelFitRequest(repo_id="acme/model"),
        metadata(),
        ClusterSimulator("test").snapshot(),
        Settings(model_cache_path=tmp_path),
        cached_repository_bytes=GIB,
    )
    assert plan.runtime_storage_fits
    assert plan.fits


def test_managed_inventory_only_scans_valid_repository_depth(tmp_path):
    model_root = tmp_path / "models"
    valid = model_root / "acme" / "model"
    partial = model_root / ".partial" / "job"
    nested = model_root / "acme" / "model" / "unexpected"
    for path, repo_id in (
        (valid, "acme/model"),
        (partial, "ignored/partial"),
        (nested, "ignored/deep"),
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / ".spark-cluster.json").write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "revision": "main",
                    "downloaded_at": "2026-10-23T12:00:00+00:00",
                }
            )
        )
    manager = ModelManager(
        Settings(model_cache_path=model_root),
        SnapshotStore(tmp_path / "inventory.db"),
        SimpleNamespace(latest=None),
    )
    assert [model.id for model in manager._scan_managed_models(set())] == ["acme/model"]
    manager.store.close()


@pytest.mark.asyncio
async def test_inventory_serves_completed_jobs_while_storage_scan_runs(
    tmp_path, monkeypatch
):
    settings = Settings(model_cache_path=tmp_path / "models")
    store = SnapshotStore(tmp_path / "inventory-fast.db")
    runtime = SimpleNamespace(latest=ClusterSimulator("test").snapshot())
    manager = ModelManager(settings, store, runtime)
    fit = build_fit_plan(
        ModelFitRequest(repo_id="acme/model"),
        metadata(),
        runtime.latest,
        settings,
    ).model_copy(update={"launch_supported": False})
    store.save_download(
        DownloadJob(
            id="completed-job",
            repo_id="acme/model",
            revision="main",
            status="completed",
            total_bytes=123,
            target_path=str(settings.model_cache_path / "acme" / "model"),
            staging_path=str(settings.model_cache_path / ".partial" / "completed-job"),
            fit=fit,
        )
    )

    async def runtime_endpoint():
        return RuntimeEndpoint(
            base_url="http://runtime/v1",
            models_url="http://runtime/v1/models",
            chat_completions_url="http://runtime/v1/chat/completions",
            status="unreachable",
        )

    scan_started = threading.Event()
    release_scan = threading.Event()

    def blocked_scan(_loaded):
        scan_started.set()
        release_scan.wait(5)
        return []

    monkeypatch.setattr(manager, "runtime_endpoint", runtime_endpoint)
    monkeypatch.setattr(manager, "_scan_managed_models", blocked_scan)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    try:
        started_at = time.monotonic()
        models = await manager.local_models()
        assert time.monotonic() - started_at < 0.5
        assert [(model.id, model.size_bytes, model.launch_supported) for model in models] == [
            ("acme/model", 123, False)
        ]
        assert scan_started.wait(1)
    finally:
        release_scan.set()
        if manager._inventory_refresh_thread:
            manager._inventory_refresh_thread.join(1)
        store.close()


@pytest.mark.asyncio
async def test_local_fit_uses_recorded_metadata_without_scanning_storage(
    tmp_path, monkeypatch
):
    settings = Settings(model_cache_path=tmp_path / "models")
    store = SnapshotStore(tmp_path / "fit-fast.db")
    runtime = SimpleNamespace(latest=ClusterSimulator("test").snapshot())
    manager = ModelManager(settings, store, runtime)
    previous = build_fit_plan(
        ModelFitRequest(repo_id="acme/model", context_length=131_072),
        metadata(),
        runtime.latest,
        settings,
    )
    store.save_download(
        DownloadJob(
            id="completed-fit",
            repo_id="acme/model",
            revision="main",
            status="completed",
            total_bytes=previous.repository_bytes,
            target_path=str(settings.model_cache_path / "acme" / "model"),
            staging_path=str(settings.model_cache_path / ".partial" / "completed-fit"),
            fit=previous,
        )
    )

    monkeypatch.setattr(
        manager.catalog,
        "inspect",
        lambda *_args, **_kwargs: pytest.fail("Recorded local metadata must avoid Hub lookup"),
    )
    result = await manager.fit_for_local(
        ModelFitRequest(repo_id="acme/model", context_length=262_144)
    )

    assert result.kv_cache_gib == pytest.approx(previous.kv_cache_gib * 2, abs=0.02)
    assert result.repository_bytes == previous.repository_bytes
    store.close()


def test_nvfp4_download_is_allowed_but_launch_is_blocked_on_ampere(tmp_path):
    settings = Settings(
        model_cache_path=tmp_path,
        fallback_node_count=2,
        fallback_node_memory_gib=8,
        active_compute_capability=8.6,
    )
    repository = RepositoryMetadata(
        repo_id="acme/model-NVFP4",
        revision="main",
        repository_bytes=4 * GIB,
        weight_bytes=4 * GIB,
        parameter_count=7_000_000_000,
        architecture="LlamaForCausalLM",
        tags=("nvfp4",),
        config={
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
    )
    plan = build_fit_plan(
        ModelFitRequest(repo_id=repository.repo_id, context_length=8192),
        repository,
        None,
        settings,
    )
    assert plan.detected_format == "NVFP4"
    assert plan.download_fits is True
    assert plan.launch_supported is False
    assert plan.runtime_fits is False
    assert any("Blackwell" in item for item in plan.requirements)


def test_discrete_gpu_reserve_allows_small_model_without_overstating_capacity(tmp_path):
    settings = Settings(
        model_cache_path=tmp_path,
        active_hardware="nvidia-gpu",
        fallback_node_count=2,
        fallback_node_memory_gib=8,
    )
    request = ModelFitRequest(repo_id="acme/model", context_length=8192)
    small = build_fit_plan(
        request, metadata(parameters=135_000_000, weight_gib=0.25), None, settings
    )
    assert small.memory_fits
    assert small.minimum_nodes == 1
    assert small.runtime_overhead_gib == 2
    large = build_fit_plan(request, metadata(weight_gib=12), None, settings)
    assert not large.memory_fits
    spark = build_fit_plan(
        request,
        metadata(parameters=135_000_000, weight_gib=0.25),
        None,
        settings.model_copy(update={"active_hardware": "dgx-spark"}),
    )
    assert not spark.memory_fits
