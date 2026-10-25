from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from spark_cluster.config import Settings
from spark_cluster.deepseek_mia import MODEL_ID as MIA_MODEL_ID
from spark_cluster.models import (
    ModelLaunchRequest,
    RuntimeReadiness,
    RuntimeWorkflowJob,
    RuntimeWorkflowRequest,
)
from spark_cluster.runtime_catalog import RegistryError
from spark_cluster.runtime_workflows import RuntimeWorkflowService
from spark_cluster.store import SnapshotStore


class Registry:
    def __init__(self):
        self.target = SimpleNamespace(
            id="local-1",
            name="Local GPUs",
            kind="local-docker",
            gpu_count=2,
            runtime_port=30000,
            host_address="192.168.1.20",
        )

    def get(self, cluster_id):
        return self.target if cluster_id == self.target.id else None

    def active(self):
        return self.target


class Prerequisites:
    def __init__(self):
        self.values = {}

    async def _target_platform(self, target):
        return "linux/amd64"

    def activation(self, cluster_id, engine_id):
        return self.values.get((cluster_id, engine_id))

    async def activate(self, **value):
        value["id"] = f"{value['cluster_id']}:{value['engine_id']}"
        self.values[(value["cluster_id"], value["engine_id"])] = value
        return value

    async def inspect(self, model_id, engine_id, cluster_id):
        return RuntimeReadiness(
            model_id=model_id,
            runtime_id=engine_id,
            runtime_name=engine_id,
            required_version="v0.5.22",
            image="docker.io/lmsysorg/sglang:v0.5.22",
            ready=True,
            install_supported=True,
            message="Ready",
        )


class Remote:
    metadata = {
        "runtime_id": "sglang",
        "version": "v0.5.22",
        "platform": "linux/amd64",
        "repository": "docker.io/lmsysorg/sglang",
        "image": "docker.io/lmsysorg/sglang:v0.5.22",
        "digest": "sha256:" + "a" * 64,
        "config_digest": "sha256:" + "b" * 64,
        "size_bytes": 123,
    }

    async def latest_stable(self, runtime_id, platform):
        return {**self.metadata, "runtime_id": runtime_id, "platform": platform}

    async def resolve(self, runtime_id, version, platform):
        return {**self.metadata, "runtime_id": runtime_id, "version": version, "platform": platform}


class Library:
    def __init__(self):
        self.remote = Remote()
        self.tasks = {}
        self.artifact = {**Remote.metadata, "id": "c" * 32}
        self.jobs = {}

    async def verified_artifact_for_image(self, image):
        return self.artifact if image == self.artifact["image"] else None

    async def install(self, request):
        value = {
            "id": "d" * 32,
            "status": "completed",
            "message": "Installed",
            "nodes": [{"address": "192.168.1.20", "status": "ready"}],
        }
        self.jobs[value["id"]] = value
        return value

    async def job(self, job_id):
        return self.jobs[job_id]


@pytest.mark.asyncio
async def test_guided_runtime_workflow_reuses_cache_installs_and_activates(tmp_path):
    settings = Settings(database_path=tmp_path / "state.db")
    store = SnapshotStore(settings.database_path)
    prerequisites = Prerequisites()
    service = RuntimeWorkflowService(
        settings, store, Registry(), prerequisites, Library(), SimpleNamespace()
    )
    guide = await service.guide("local-1", engine_id="sglang")
    assert guide.compatible is True
    assert guide.version == "v0.5.22"
    assert [step.id for step in guide.steps] == ["resolve", "cache", "install", "activate"]

    job = await service.create(RuntimeWorkflowRequest(cluster_id="local-1", engine_id="sglang"))
    await service.tasks[job.id]
    completed = await service.get(job.id)
    assert completed.status == "completed"
    assert completed.progress == 100
    assert all(step.status == "completed" for step in completed.steps)
    assert prerequisites.activation("local-1", "sglang")["digest"] == Remote.metadata["digest"]
    store.close()


@pytest.mark.asyncio
async def test_guide_keeps_specialized_unmanaged_model_blocked(tmp_path):
    settings = Settings(database_path=tmp_path / "state.db")
    store = SnapshotStore(settings.database_path)
    service = RuntimeWorkflowService(
        settings, store, Registry(), Prerequisites(), Library(), SimpleNamespace()
    )
    guide = await service.guide("local-1", model_id=MIA_MODEL_ID)
    assert guide.compatible is False
    assert "multi-node bootstrap" in guide.blockers[0]
    assert guide.alternatives
    store.close()


@pytest.mark.asyncio
async def test_active_runtime_remains_prepared_when_registry_is_unavailable(tmp_path):
    settings = Settings(database_path=tmp_path / "state.db")
    store = SnapshotStore(settings.database_path)
    prerequisites = Prerequisites()
    prerequisites.values[("local-1", "sglang")] = {
        **Remote.metadata,
        "id": "local-1:sglang",
        "cluster_id": "local-1",
        "engine_id": "sglang",
        "artifact_id": "c" * 32,
    }
    library = Library()
    library.remote.latest_stable = AsyncMock(
        side_effect=RegistryError("Registry unavailable or rate limited")
    )
    service = RuntimeWorkflowService(
        settings, store, Registry(), prerequisites, library, SimpleNamespace()
    )
    guide = await service.guide("local-1", engine_id="sglang")
    assert guide.compatible is True
    assert guide.prepared is True
    assert guide.version == "v0.5.22"
    store.close()


@pytest.mark.asyncio
async def test_model_workflow_smoke_test_requires_a_real_completion(tmp_path):
    settings = Settings(database_path=tmp_path / "state.db")
    store = SnapshotStore(settings.database_path)
    service = RuntimeWorkflowService(
        settings, store, Registry(), Prerequisites(), Library(), SimpleNamespace()
    )

    def respond(request):
        assert request.url.path == "/v1/completions"
        assert request.method == "POST"
        return httpx.Response(200, json={"choices": [{"text": "OK"}]})

    service.http_transport = httpx.MockTransport(respond)
    request = RuntimeWorkflowRequest(
        cluster_id="local-1",
        engine_id="sglang",
        model_id="acme/model",
        launch=ModelLaunchRequest(
            model_id="acme/model", served_model_name="model", context_length=8192
        ),
    )
    job = RuntimeWorkflowJob(
        id="e" * 32,
        request=request,
        engine_id="sglang",
        engine_name="SGLang",
        version="v0.5.22",
        image=Remote.metadata["image"],
        digest=Remote.metadata["digest"],
        platform="linux/amd64",
    )
    assert await service._smoke(job) == "OK"
    store.close()
