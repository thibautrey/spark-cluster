import pytest

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

    async def run(self, node, script, timeout=60):
        if "images pull" in script or "crictl pull" in script or "docker pull" in script:
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
    service = RuntimePrerequisiteService(
        settings, store, ready_setup(store, settings, transport)
    )

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


def test_terminal_tickets_are_short_lived_and_single_use(tmp_path):
    settings = Settings(database_path=tmp_path / "state.db", setup_state_path=tmp_path / "setup")
    store = SnapshotStore(settings.database_path)
    transport = RuntimeTransport(settings.runtime_vllm_image)
    service = RuntimePrerequisiteService(
        settings, store, ready_setup(store, settings, transport)
    )

    ticket = service.issue_terminal_ticket("192.168.50.10")
    assert service.consume_terminal_ticket(ticket.ticket).role == "head"
    with pytest.raises(KeyError):
        service.consume_terminal_ticket(ticket.ticket)
    with pytest.raises(KeyError):
        service.issue_terminal_ticket("192.168.50.99")
    store.close()
