import asyncio

import pytest

from spark_cluster.cluster_registry import ClusterRegistry, GPUDevice
from spark_cluster.cluster_setup import ClusterSetupService
from spark_cluster.config import Settings
from spark_cluster.runtime_prerequisites import RuntimePrerequisiteService
from spark_cluster.setup_models import SetupJob, SetupNode, SetupRequest
from spark_cluster.store import SnapshotStore

FINGERPRINT = "SHA256:" + "A" * 43


class RuntimeTransport:
    def __init__(self, image: str):
        self.image = image
        self.images = {
            "192.168.50.10": [image],
            "192.168.50.11": ["nvcr.io/nvidia/vllm:older"],
        }
        self.pull_gate = None

    async def run(self, node, script, timeout=60):
        if "images pull" in script or "crictl pull" in script or "docker pull" in script:
            if self.pull_gate:
                await self.pull_gate.wait()
            self.images[node.address] = [self.image]
            return "pulled"
        return "\n".join(self.images.get(node.address, []))


def ready_setup(store, settings, transport):
    setup = ClusterSetupService(settings, store, transport=transport)
    request = SetupRequest(
        name="test-cluster",
        topology="direct",
        nodes=[
            SetupNode(
                address="192.168.50.10",
                username="nvidia",
                fingerprint=FINGERPRINT,
                role="head",
            ),
            SetupNode(
                address="192.168.50.11",
                username="nvidia",
                fingerprint=FINGERPRINT,
                role="worker",
            ),
        ],
    )
    store.save_setup_job(SetupJob(request=request, mode="ssh", status="ready", steps=[]))
    return setup


@pytest.mark.asyncio
async def test_runtime_readiness_and_install_updates_every_node(tmp_path):
    image = "nvcr.io/nvidia/vllm:25.09-py3"
    settings = Settings(
        mode="live",
        database_path=tmp_path / "state.db",
        setup_state_path=tmp_path / "setup",
        runtime_vllm_image=image,
    )
    store = SnapshotStore(settings.database_path)
    transport = RuntimeTransport(image)
    service = RuntimePrerequisiteService(settings, store, ready_setup(store, settings, transport))

    report = await service.inspect("acme/model")
    assert report.ready is False
    assert [node.status for node in report.nodes] == ["ready", "outdated"]
    assert report.nodes[1].installed_version == "older"

    job = await service.install("acme/model")
    await service.tasks[job.id]
    completed = await service.get_install(job.id)
    assert completed.status == "completed"
    assert all(node.status == "ready" for node in completed.nodes)
    assert {log.node for log in completed.logs} == {"192.168.50.10", "192.168.50.11"}
    assert (await service.inspect("acme/model")).ready is True
    store.close()


@pytest.mark.asyncio
async def test_engine_overview_and_prepare_are_bound_to_explicit_pool(tmp_path):
    image = "nvcr.io/nvidia/vllm:25.09-py3"
    settings = Settings(
        mode="live",
        database_path=tmp_path / "state.db",
        setup_state_path=tmp_path / "setup",
        runtime_vllm_image=image,
    )
    store = SnapshotStore(settings.database_path)
    transport = RuntimeTransport(image)
    registry = ClusterRegistry(settings)
    pool = registry.add_remote(
        name="Spark pool",
        endpoint="http://192.168.50.10:30080",
        head_address="192.168.50.10",
        hardware="dgx-spark",
    )
    other = registry.add_remote(
        name="Other pool",
        endpoint="http://192.168.60.10:30080",
        head_address="192.168.60.10",
        hardware="other",
        activate=False,
    )
    service = RuntimePrerequisiteService(
        settings, store, ready_setup(store, settings, transport), registry=registry
    )

    overview = await service.inspect_engines(pool.id)
    assert overview.cluster_id == pool.id
    assert overview.platform == "linux/arm64"
    vllm = next(engine for engine in overview.engines if engine.id == "vllm")
    assert vllm.state == "partial"
    assert vllm.nodes_ready == 1 and vllm.nodes_total == 2
    assert vllm.prepare_supported
    assert next(engine for engine in overview.engines if engine.id == "sglang").state == "blocked"
    assert (
        next(engine for engine in overview.engines if engine.id == "llama-cpp").capability
        == "package_only"
    )

    transport.pull_gate = asyncio.Event()
    first = await service.prepare(pool.id, "vllm")
    await asyncio.sleep(0)
    duplicate = await service.prepare(pool.id, "vllm")
    assert duplicate.id == first.id
    registry.select(other.id)
    transport.pull_gate.set()
    await service.tasks[first.id]
    completed = await service.get_install(first.id)
    assert completed.cluster_id == pool.id
    assert completed.cluster_name == "Spark pool"
    assert {node.address for node in completed.nodes} == {"192.168.50.10", "192.168.50.11"}
    assert completed.status == "completed"
    store.close()


@pytest.mark.asyncio
async def test_local_docker_engine_overview_and_prepare(tmp_path):
    class FakeDocker:
        available = True

        def __init__(self):
            self.present = False

        async def image_present(self, image):
            return self.present

        async def pull_image(self, image):
            self.present = True

        async def platform(self):
            return "linux/amd64"

    settings = Settings(
        database_path=tmp_path / "state.db",
        setup_state_path=tmp_path / "setup",
    )
    store = SnapshotStore(settings.database_path)
    registry = ClusterRegistry(settings)
    pool = registry.add_local(
        name="Local GPUs",
        host_address="192.168.1.2",
        runtime_port=30000,
        gpus=[
            GPUDevice(index=0, name="GPU", uuid="GPU-0", memory_gib=24),
            GPUDevice(index=1, name="GPU", uuid="GPU-1", memory_gib=24),
        ],
        model_storage_path=str(tmp_path / "models"),
    )
    docker = FakeDocker()
    setup = ClusterSetupService(
        settings, store, transport=RuntimeTransport(settings.runtime_vllm_image)
    )
    service = RuntimePrerequisiteService(settings, store, setup, registry=registry, docker=docker)

    overview = await service.inspect_engines(pool.id)
    vllm = next(engine for engine in overview.engines if engine.id == "vllm")
    assert overview.platform == "linux/amd64"
    assert vllm.state == "needs_preparation" and vllm.nodes_total == 2
    job = await service.prepare(pool.id, "vllm")
    await service.tasks[job.id]
    assert (await service.get_install(job.id)).status == "completed"
    assert (
        next(
            engine
            for engine in (await service.inspect_engines(pool.id)).engines
            if engine.id == "vllm"
        ).state
        == "ready"
    )
    store.close()


def test_terminal_tickets_are_short_lived_and_single_use(tmp_path):
    settings = Settings(database_path=tmp_path / "state.db", setup_state_path=tmp_path / "setup")
    store = SnapshotStore(settings.database_path)
    transport = RuntimeTransport(settings.runtime_vllm_image)
    service = RuntimePrerequisiteService(settings, store, ready_setup(store, settings, transport))

    ticket = service.issue_terminal_ticket("192.168.50.10")
    assert service.consume_terminal_ticket(ticket.ticket).role == "head"
    with pytest.raises(KeyError):
        service.consume_terminal_ticket(ticket.ticket)
    with pytest.raises(KeyError):
        service.issue_terminal_ticket("192.168.50.99")
    store.close()
