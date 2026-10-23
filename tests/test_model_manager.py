import json
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
from spark_cluster.models import DownloadJob, ModelDownloadRequest, ModelFitRequest
from spark_cluster.simulator import ClusterSimulator
from spark_cluster.store import SnapshotStore


def metadata(parameters: int = 70_000_000_000) -> RepositoryMetadata:
    return RepositoryMetadata(
        repo_id="acme/model",
        revision="main",
        repository_bytes=1 * GIB,
        weight_bytes=35 * GIB,
        parameter_count=parameters,
        architecture="DeepseekV3ForCausalLM",
        config={
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_hidden_layers": 61,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
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
        metadata(parameters=671_000_000_000),
        snapshot,
        settings,
    )
    assert plan.memory_fits is False
    assert plan.minimum_nodes > plan.cluster_nodes
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
