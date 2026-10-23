import base64
import json
import subprocess
import time
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from spark_deepseek.app import create_app
from spark_deepseek.cluster_setup import (
    ClusterSetupService,
    SetupFailure,
    SSHTransport,
    fingerprint,
)
from spark_deepseek.config import Settings
from spark_deepseek.setup_models import DiscoveryRequest, SetupRequest
from spark_deepseek.setup_scripts import chart_archive
from spark_deepseek.store import SnapshotStore

PIN = "SHA256:" + "A" * 43


def plan(count=2):
    return SetupRequest(
        name="test-cluster",
        topology="single" if count == 1 else "switch",
        nodes=[
            {
                "address": f"192.168.50.{10 + index}",
                "username": "spark",
                "fingerprint": PIN,
                "role": "head" if index == 0 else "worker",
            }
            for index in range(count)
        ],
    )


def settings(tmp_path, **kwargs):
    return Settings(
        mode="live",
        setup_enabled=True,
        model_admin_token="admin-test-secret",
        database_path=tmp_path / "state.db",
        model_cache_path=tmp_path / "models",
        setup_state_path=tmp_path / "trust",
        **kwargs,
    )


class FakeTransport:
    """A deterministic remote boundary; never invokes SSH or modifies the system."""

    def __init__(self):
        self.scripts = []
        self.trusted = []
        self.fail_once = None
        self.gpus = 1
        self.sudo = True
        self.toolkit = True
        self.architecture = "aarch64"
        self.k3s = False
        self.fabric = True
        self.uploaded = False
        self.request = plan()

    async def trust(self, nodes):
        self.trusted = deepcopy(nodes)

    async def run(self, node, script, timeout=60):
        # Validate every generated shell program without executing any of it.
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        self.scripts.append(script)
        if self.fail_once and self.fail_once in script:
            self.fail_once = None
            raise RuntimeError("sensitive-output-that-must-not-be-persisted")
        if "python3 - <<'PROBE'" in script:
            suffix = node.address.rsplit(".", 1)[-1]
            return json.dumps(
                {
                    "hostname": "spark-" + suffix,
                    "architecture": self.architecture,
                    "gpu": "NVIDIA GB10",
                    "driver": "580.0",
                    "memory_gib": 128,
                    "available_gib": 110,
                    "disk_gib": 1500,
                    "sudo": self.sudo,
                    "toolkit": self.toolkit,
                    "k3s": self.k3s,
                    "managed_cluster": None,
                    "clock": time.time(),
                    "rdma": self.fabric,
                    "interfaces": [
                        {"up": True, "speed_mbps": 200000, "addresses": [f"10.10.0.{suffix}"]}
                    ],
                }
            )
        if "server/node-token" in script:
            return "K10::server:private-join-token\n"
        if "kubectl get nodes -o json" in script:
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": f"spark-{index}"},
                            "status": {
                                "addresses": [
                                    {"type": "InternalIP", "address": f"10.10.0.{10 + index}"}
                                ],
                                "allocatable": {"nvidia.com/gpu": str(self.gpus)},
                            },
                        }
                        for index in range(len(self.request.nodes))
                    ]
                }
            )
        return ""

    async def upload(self, node, archive):
        assert archive[:2] == b"\x1f\x8b"
        self.uploaded = True


async def terminal(service, job):
    await service.tasks[job.id]
    return service.store.get_setup_job(job.id)


@pytest.mark.parametrize(
    "payload",
    [
        {"subnet": "10.0.0.0/8"},
        {"subnet": "8.8.8.0/24"},
        {"addresses": ["127.0.0.1"]},
        {"addresses": ["169.254.169.254"]},
        {"addresses": ["::1"]},
        {"addresses": ["192.168.1.2;id"]},
    ],
)
def test_discovery_rejects_unsafe_or_unbounded_targets(payload):
    with pytest.raises(ValidationError):
        DiscoveryRequest(**payload)


def test_topology_and_shell_inputs_are_validated():
    assert DiscoveryRequest(subnet="192.168.1.42/24").subnet == "192.168.1.0/24"
    payload = plan().model_dump()
    for field, value in [
        ("username", "root;id"),
        ("fingerprint", "anything"),
        ("fabric_address", "8.8.8.8"),
    ]:
        changed = deepcopy(payload)
        changed["nodes"][0][field] = value
        with pytest.raises(ValidationError):
            SetupRequest(**changed)
    payload["nodes"][1]["role"] = "head"
    with pytest.raises(ValidationError):
        SetupRequest(**payload)
    payload = plan(4).model_dump()
    payload["topology"] = "direct"
    with pytest.raises(ValidationError):
        SetupRequest(**payload)


@pytest.mark.asyncio
async def test_preflight_autodetects_fabric_and_flags_automatic_repairs(tmp_path):
    store = SnapshotStore(tmp_path / "state.db")
    remote = FakeTransport()
    remote.toolkit = False
    service = ClusterSetupService(settings(tmp_path), store, remote)
    request = plan()
    checks = await service.preflight(request)
    assert all(node.fabric_address for node in request.nodes)
    assert not any(check.status == "error" for check in checks)
    assert any(check.code == "toolkit" and check.automatic for check in checks)
    assert sum(check.code == "fabric-reachability" for check in checks) == 2
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attribute,value,code",
    [
        ("sudo", False, "sudo"),
        ("architecture", "x86_64", "gb10"),
        ("k3s", True, "ownership"),
        ("fabric", False, "fabric"),
    ],
)
async def test_preflight_blocks_unsupported_or_unmanaged_nodes(tmp_path, attribute, value, code):
    store = SnapshotStore(tmp_path / "state.db")
    remote = FakeTransport()
    setattr(remote, attribute, value)
    service = ClusterSetupService(settings(tmp_path), store, remote)
    job = await terminal(service, await service.launch(plan()))
    assert job.status == "blocked"
    assert any(check.code == code and check.status == "error" for check in job.checks)
    assert not remote.uploaded
    assert not any("apt-get" in script for script in remote.scripts)
    store.close()


@pytest.mark.asyncio
async def test_full_bootstrap_uses_valid_scripts_and_keeps_tokens_out_of_state(tmp_path):
    store = SnapshotStore(tmp_path / "state.db")
    remote = FakeTransport()
    service = ClusterSetupService(settings(tmp_path), store, remote)
    job = await terminal(service, await service.launch(plan()))
    assert job.status == "ready", job.error
    assert all(step.status == "done" for step in job.steps)
    assert remote.uploaded
    assert job.dashboard_url == "http://192.168.50.10:30080"
    assert "private-join-token" not in job.model_dump_json()
    assert "admin-test-secret" not in job.model_dump_json()
    join_script = next(script for script in remote.scripts if "K3S_TOKEN=" in script)
    assert "trap " in join_script and "join.env" in join_script
    assert any('"deferStart": true' in script for script in remote.scripts)
    store.close()


@pytest.mark.asyncio
async def test_failed_step_is_diagnosed_and_resumes_after_rechecking(tmp_path):
    store = SnapshotStore(tmp_path / "state.db")
    remote = FakeTransport()
    remote.fail_once = "apt-get update"
    service = ClusterSetupService(settings(tmp_path), store, remote)
    job = await terminal(service, await service.launch(plan()))
    assert job.status == "failed"
    assert any(check.code == "dependencies" and check.automatic for check in job.checks)
    assert "sensitive-output" not in job.model_dump_json()
    job = await terminal(service, await service.resume(job.id))
    assert job.status == "ready", job.error
    assert len(remote.trusted) == 2
    assert sum("python3 - <<'PROBE'" in script for script in remote.scripts) == 4
    store.close()


@pytest.mark.asyncio
async def test_missing_gpu_stops_before_storage_and_dashboard(tmp_path):
    store = SnapshotStore(tmp_path / "state.db")
    remote = FakeTransport()
    remote.gpus = 0
    service = ClusterSetupService(settings(tmp_path), store, remote)
    job = await terminal(service, await service.launch(plan()))
    assert job.status == "failed"
    assert "GPU resources" in job.error
    assert not remote.uploaded
    store.close()


@pytest.mark.asyncio
async def test_restart_requires_explicit_resume_and_preserves_plan(tmp_path):
    store = SnapshotStore(tmp_path / "state.db")
    remote = FakeTransport()
    service = ClusterSetupService(settings(tmp_path), store, remote)
    # Persist the state that would remain after a process exit.
    from spark_deepseek.setup_models import SetupJob, SetupStep

    job = SetupJob(
        request=plan(),
        mode="ssh",
        status="running",
        steps=[SetupStep(id="head", label="Head", status="running")],
    )
    store.save_setup_job(job)
    await service.start()
    recovered = store.get_setup_job(job.id)
    assert recovered.status == "blocked"
    assert recovered.request == job.request
    assert not service.tasks
    assert recovered.steps[0].status == "failed"
    store.close()


@pytest.mark.asyncio
async def test_changed_host_key_fails_before_trust_is_written(tmp_path, monkeypatch):
    transport = SSHTransport(settings(tmp_path))

    async def changed(address):
        return base64.b64encode(b"new-host-key").decode(), "SHA256:" + "B" * 43

    monkeypatch.setattr(transport, "host_key", changed)
    with pytest.raises(SetupFailure, match="Host key changed"):
        await transport.trust(plan().nodes)
    assert not transport.known_hosts.exists()
    assert fingerprint(base64.b64encode(b"key").decode()).startswith("SHA256:")


@pytest.mark.asyncio
async def test_concurrent_setups_are_rejected(tmp_path):
    store = SnapshotStore(tmp_path / "state.db")
    service = ClusterSetupService(settings(tmp_path), store, FakeTransport())
    job = await service.launch(plan())
    with pytest.raises(SetupFailure, match="in progress"):
        await service.launch(plan())
    await terminal(service, job)
    store.close()


def test_live_setup_requires_opt_in_and_admin_authentication(tmp_path):
    app = create_app(settings(tmp_path))
    with TestClient(app) as client:
        for path in ["/api/v1/setup/discover", "/api/v1/setup/check", "/api/v1/setup/jobs"]:
            assert client.post(path, json={}).status_code == 401
        assert client.get("/api/v1/setup").status_code == 401
        assert client.post("/api/v1/setup/jobs/missing/resume").status_code == 401
        headers = {"Authorization": "Bearer admin-test-secret"}
        assert (
            client.post(
                "/api/v1/setup/discover", headers=headers, json={"subnet": "10.0.0.0/8"}
            ).status_code
            == 422
        )
        assert client.post("/api/v1/setup/jobs/missing/resume", headers=headers).status_code == 404
    disabled = settings(tmp_path)
    disabled.setup_enabled = False
    app = create_app(disabled)
    with TestClient(app) as client:
        response = client.post("/api/v1/setup/discover", json={}, headers=headers)
        assert response.status_code == 409


def test_demo_wizard_end_to_end_without_network_or_ssh(tmp_path, monkeypatch):
    config = Settings(
        mode="demo",
        database_path=tmp_path / "demo.db",
        model_cache_path=tmp_path / "models",
        openai_api_base="http://127.0.0.1:9/v1",
    )
    app = create_app(config)

    async def forbidden(*args, **kwargs):
        pytest.fail("Demo setup must not contact a real node")

    monkeypatch.setattr(app.state.setup_service.transport, "run", forbidden)
    monkeypatch.setattr(app.state.setup_service.transport, "trust", forbidden)
    with TestClient(app) as client:
        assert "Build your Spark cluster" in client.get("/").text
        assert client.get("/assets/setup.js").status_code == 200
        discovery = client.post("/api/v1/setup/discover", json={}).json()
        assert len(discovery["nodes"]) == 4
        assert client.post("/api/v1/setup/check", json=plan().model_dump()).status_code == 200
        response = client.post("/api/v1/setup/jobs", json=plan().model_dump())
        assert response.status_code == 202
        job_id = response.json()["id"]
        for _ in range(60):
            job = client.get("/api/v1/setup").json()["jobs"][0]
            if job["status"] == "ready":
                break
            time.sleep(0.05)
        assert job["id"] == job_id and job["status"] == "ready"
        assert len(job["steps"]) == 8


def test_chart_is_packaged_in_bootstrap_archive():
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(chart_archive()), mode="r:gz") as archive:
        assert "spark-deepseek/Chart.yaml" in archive.getnames()
        assert "spark-deepseek/templates/runtime.yaml" in archive.getnames()


def test_packaged_chart_matches_development_chart():
    from pathlib import Path

    source = Path(__file__).parents[1] / "deploy/helm/spark-deepseek"
    packaged = Path(__file__).parents[1] / "src/spark_deepseek/chart"
    source_files = {path.relative_to(source) for path in source.rglob("*") if path.is_file()}
    packaged_files = {path.relative_to(packaged) for path in packaged.rglob("*") if path.is_file()}
    assert source_files == packaged_files
    for relative in source_files:
        assert source.joinpath(relative).read_bytes() == packaged.joinpath(relative).read_bytes()
