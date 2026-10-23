import asyncio

import pytest

from spark_cluster.cluster_control import RuntimeDeploymentService, RuntimeLogService
from spark_cluster.config import Settings
from spark_cluster.models import LocalModel, ModelFitPlan, ModelLaunchRequest
from spark_cluster.store import SnapshotStore


@pytest.mark.asyncio
async def test_demo_log_service_streams_and_persists_runtime_logs(tmp_path):
    settings = Settings(
        mode="demo",
        database_path=tmp_path / "logs.db",
        model_cache_path=tmp_path / "models",
        runtime_logs_mode="demo",
    )
    store = SnapshotStore(settings.database_path)
    service = RuntimeLogService(settings, store)
    await service.start()
    try:
        await asyncio.sleep(0.9)
        entries = await service.history()
        assert entries
        assert entries[0].pod.startswith("spark-cluster-runtime-")
        assert entries[0].container in {"engine", "ray-worker"}
        assert await service.instances()
    finally:
        await service.stop()
        store.close()


def test_runtime_log_filters_are_persistent(tmp_path):
    settings = Settings(database_path=tmp_path / "logs.db", model_cache_path=tmp_path / "models")
    store = SnapshotStore(settings.database_path)
    from spark_cluster.models import RuntimeLog

    store.append_runtime_log(
        RuntimeLog(id="one", pod="head", container="engine", level="info", message="ready")
    )
    store.append_runtime_log(
        RuntimeLog(id="two", pod="worker", container="ray-worker", level="error", message="failed")
    )
    assert [entry.id for entry in store.list_runtime_logs(pod="head")] == ["one"]
    assert [entry.id for entry in store.list_runtime_logs(level="error")] == ["two"]
    store.close()


def test_runtime_log_sanitizer_redacts_credentials_and_signed_urls():
    value = RuntimeLogService._sanitize(
        "Authorization: Bearer secret-token hf-abcdefghijklmnop "
        "https://example.com/model?X-Amz-Signature=secret"
    )
    assert "secret-token" not in value
    assert "abcdefghijklmnop" not in value
    assert "X-Amz-Signature" not in value


@pytest.mark.asyncio
async def test_kubernetes_launch_patches_runtime_resources_and_waits_for_readiness(tmp_path):
    class FakeKubernetes:
        available = True

        def __init__(self):
            self.statefulset_patch = None
            self.deployment_patch = None

        async def patch_statefulset(self, name, payload):
            self.statefulset_patch = (name, payload)
            return {}

        async def patch_deployment(self, name, payload):
            self.deployment_patch = (name, payload)
            return {}

        async def get_statefulset(self, name):
            return {"status": {"readyReplicas": 3}}

        async def get_deployment(self, name):
            return {"status": {"availableReplicas": 1}}

    class FakeModels:
        async def managed_model(self, model_id):
            return LocalModel(
                id=model_id,
                revision="main",
                source="managed",
                path="/models/acme/model",
            )

        async def fit_for_local(self, request):
            return ModelFitPlan(
                repo_id=request.repo_id,
                revision="main",
                repository_bytes=1,
                weight_bytes=1,
                weights_gib=100,
                kv_cache_gib=10,
                runtime_overhead_gib=10,
                total_required_gib=120,
                cluster_nodes=4,
                usable_per_node_gib=104.96,
                usable_cluster_gib=419.84,
                minimum_nodes=2,
                headroom_gib=299.84,
                memory_fits=True,
                storage_available_gib=100,
                storage_fits=True,
                fits=True,
                confidence="parameters",
            )

        @staticmethod
        def safe_error(error):
            return str(error)

    class FakeLogs:
        def __init__(self, kubernetes):
            self.kubernetes = kubernetes
            self.entries = []

        async def emit(self, pod, container, level, message):
            self.entries.append((pod, container, level, message))

    settings = Settings(
        mode="live",
        runtime_control_mode="kubernetes",
        database_path=tmp_path / "state.db",
        model_cache_path=tmp_path / "models",
    )
    store = SnapshotStore(settings.database_path)
    kubernetes = FakeKubernetes()
    logs = FakeLogs(kubernetes)
    service = RuntimeDeploymentService(settings, store, FakeModels(), logs)
    await service.start()
    try:
        job = await service.launch(
            ModelLaunchRequest(
                model_id="acme/model",
                served_model_name="model",
                tensor_parallel=2,
                pipeline_parallel=2,
            )
        )
        await service.queue.join()
        completed = store.get_launch(job.id)
        assert completed.status == "ready"
        worker_spec = kubernetes.statefulset_patch[1]["spec"]
        assert worker_spec["replicas"] == 3
        assert worker_spec["template"]["spec"]["containers"][0]["image"].startswith(
            "nvcr.io/nvidia/vllm:"
        )
        containers = kubernetes.deployment_patch[1]["spec"]["template"]["spec"]["containers"]
        environment = {item["name"]: item["value"] for item in containers[0]["env"]}
        assert environment["MODEL_ID"] == "/models/acme/model"
        assert environment["TENSOR_PARALLEL_SIZE"] == "2"
        assert environment["PIPELINE_PARALLEL_SIZE"] == "2"
        assert environment["KV_CACHE_DTYPE"] == "auto"
        assert environment["MAX_NUM_SEQS"] == "4"
        assert environment["MAX_NUM_BATCHED_TOKENS"] == "8192"
        assert any("ready to serve" in entry[3] for entry in logs.entries)
    finally:
        await service.stop()
        store.close()
