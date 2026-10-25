import json

import httpx
import pytest

from spark_cluster.cluster_registry import ClusterRegistry, GPUDevice
from spark_cluster.config import Settings
from spark_cluster.local_docker import DockerEngineClient, local_gpu_snapshot


def settings(tmp_path, **values):
    return Settings(
        mode="live",
        database_path=tmp_path / "controller.db",
        model_cache_path=tmp_path / "models",
        setup_state_path=tmp_path / "setup",
        docker_host_model_path=tmp_path / "host-models",
        **values,
    )


def gpus():
    return [
        GPUDevice(
            index=index,
            name="RTX 3060 Ti",
            uuid=f"GPU-{index}",
            memory_gib=8,
            compute_capability="8.6",
        )
        for index in range(2)
    ]


def test_registry_persists_multiple_clusters_and_applies_selection(tmp_path):
    config = settings(tmp_path)
    registry = ClusterRegistry(config)
    local = registry.add_local(
        name="Unraid GPUs",
        host_address="192.168.1.149",
        runtime_port=30000,
        gpus=gpus(),
        model_storage_path="/mnt/models",
    )
    remote = registry.add_remote(
        name="Spark lab",
        endpoint="http://192.168.1.42:30080",
        head_address="192.168.1.42",
        hardware="dgx-spark",
        activate=False,
    )
    assert len(registry.list()) == 2
    assert registry.active().id == local.id
    assert config.remote_cluster_url is None
    assert config.fallback_node_count == 2
    assert config.fallback_node_memory_gib == 8
    assert config.openai_api_base == "http://192.168.1.149:30000/v1"

    registry.select(remote.id)
    assert config.remote_cluster_url == "http://192.168.1.42:30080"
    restarted_settings = settings(tmp_path)
    restarted = ClusterRegistry(restarted_settings)
    assert restarted.active().id == remote.id
    assert len(restarted.list()) == 2
    assert restarted.path.stat().st_mode & 0o777 == 0o600
    assert "secret" not in restarted.path.read_text()


def test_registry_migrates_legacy_active_connection(tmp_path):
    config = settings(tmp_path)
    config.setup_state_path.mkdir(parents=True)
    (config.setup_state_path / "active-cluster.json").write_text(
        json.dumps(
            {
                "url": "http://192.168.1.42:30080",
                "cluster_name": "legacy-spark",
                "head_address": "192.168.1.42",
            }
        )
    )
    registry = ClusterRegistry(config)
    assert registry.active().name == "legacy-spark"
    assert registry.active().hardware == "dgx-spark"
    assert config.remote_cluster_url == "http://192.168.1.42:30080"


def test_local_snapshot_represents_each_gpu_and_storage(tmp_path):
    config = settings(tmp_path)
    config.model_cache_path.mkdir(parents=True)
    registry = ClusterRegistry(config)
    cluster = registry.add_local(
        name="Unraid GPUs",
        host_address="192.168.1.149",
        runtime_port=30000,
        gpus=gpus(),
        model_storage_path="/mnt/models",
    )
    snapshot = local_gpu_snapshot(cluster, config, {1: ["acme/model"]})
    assert snapshot.cluster_name == "Unraid GPUs"
    assert len(snapshot.nodes) == 2
    assert snapshot.nodes[0].memory.total_gib == 8
    assert snapshot.nodes[0].loaded_model_ids == []
    assert snapshot.nodes[1].loaded_model_ids == ["acme/model"]
    assert snapshot.cache_tiers[0].persistent is True


@pytest.mark.asyncio
async def test_running_models_are_mapped_to_their_assigned_gpus(tmp_path):
    socket = tmp_path / "docker.sock"
    socket.touch()
    config = settings(tmp_path, docker_runtime_enabled=True, docker_socket_path=socket)
    registry = ClusterRegistry(config)
    cluster = registry.add_local(
        name="Unraid GPUs",
        host_address="192.168.1.149",
        runtime_port=30000,
        gpus=gpus(),
        model_storage_path="/mnt/models",
    )

    def handle(request):
        if request.url.path == "/containers/json":
            return httpx.Response(200, json=[{"Id": "runtime-1"}])
        if request.url.path == "/containers/runtime-1/json":
            return httpx.Response(
                200,
                json={
                    "State": {"Running": True},
                    "Config": {
                        "Labels": {
                            "spark-cluster.model-id": "acme/model",
                            "spark-cluster.model": "model",
                        }
                    },
                    "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["1"]}]},
                },
            )
        raise AssertionError(f"Unexpected Docker request: {request.method} {request.url}")

    docker = DockerEngineClient(config)
    docker.transport = httpx.MockTransport(handle)

    assert await docker.loaded_models_by_gpu(cluster) == {1: ["acme/model"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [1, 2])
async def test_docker_runtime_launch_uses_selected_gpus_and_shared_models(tmp_path, parallel):
    socket = tmp_path / "docker.sock"
    socket.touch()
    config = settings(
        tmp_path,
        docker_runtime_enabled=True,
        docker_socket_path=socket,
    )
    config.model_cache_path.mkdir(parents=True)
    model = config.model_cache_path / "acme" / "model"
    model.mkdir(parents=True)
    calls = []

    def handle(request):
        body = request.read()
        calls.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path.startswith("/images/"):
            return httpx.Response(200, json={"Id": "image"})
        if request.method == "GET" and request.url.path.startswith("/containers/"):
            return httpx.Response(404)
        if request.method == "POST" and request.url.path == "/containers/create":
            return httpx.Response(201, json={"Id": "container123"})
        if request.method == "POST" and request.url.path.endswith("/start"):
            return httpx.Response(204)
        raise AssertionError(f"Unexpected Docker request: {request.method} {request.url}")

    docker = DockerEngineClient(config)
    docker.transport = httpx.MockTransport(handle)
    registry = ClusterRegistry(config)
    cluster = registry.add_local(
        name="Unraid GPUs",
        host_address="192.168.1.149",
        runtime_port=30000,
        gpus=gpus(),
        model_storage_path=str(config.docker_host_model_path),
    )
    result = await docker.launch(
        cluster=cluster,
        image="vllm/vllm-openai:latest",
        model_path=str(model),
        model_id="acme/model",
        served_model_name="test-model",
        context_length=8192,
        tensor_parallel=parallel,
        pipeline_parallel=1,
        gpu_memory_utilization=0.85,
        kv_cache_dtype="auto",
        max_num_seqs=2,
        max_num_batched_tokens=2048,
        expert_parallel=False,
        chunked_prefill=True,
    )
    assert result == "container123"
    create = next(item for item in calls if item[1] == "/containers/create")
    payload = json.loads(create[2])
    assert payload["Cmd"][0] == "/models/acme/model"
    assert payload["Labels"]["spark-cluster.model-id"] == "acme/model"
    assert payload["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] == [
        str(i) for i in range(parallel)
    ]
    assert payload["HostConfig"]["Binds"] == [f"{config.docker_host_model_path}:/models:ro"]


def test_sglang_payload_uses_engine_command_and_rejects_unsupported_options(tmp_path):
    config = settings(tmp_path)
    config.model_cache_path.mkdir(parents=True)
    model = config.model_cache_path / "acme" / "model"
    model.mkdir(parents=True)
    cluster = ClusterRegistry(config).add_local(
        name="Local GPUs",
        host_address="192.168.1.149",
        runtime_port=30000,
        gpus=gpus(),
        model_storage_path=str(config.docker_host_model_path),
    )
    docker = DockerEngineClient(config)
    payload = docker.runtime_payload(
        cluster=cluster,
        engine_id="sglang",
        image="lmsysorg/sglang:v0.5.22",
        model_path=str(model),
        served_model_name="test-model",
        context_length=8192,
        tensor_parallel=2,
        pipeline_parallel=1,
        gpu_memory_utilization=0.8,
        kv_cache_dtype="auto",
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        expert_parallel=False,
        chunked_prefill=True,
    )
    assert payload["Entrypoint"] == ["python3", "-m", "sglang.launch_server"]
    assert payload["Cmd"][:2] == ["--model-path", "/models/acme/model"]
    assert payload["Cmd"][payload["Cmd"].index("--tp") + 1] == "2"
    assert payload["Labels"]["spark-cluster.engine"] == "sglang"
    with pytest.raises(ValueError, match="pipeline parallelism"):
        docker.runtime_payload(
            cluster=cluster,
            engine_id="sglang",
            image="lmsysorg/sglang:v0.5.22",
            model_path=str(model),
            served_model_name="test-model",
            context_length=8192,
            tensor_parallel=1,
            pipeline_parallel=2,
            gpu_memory_utilization=0.8,
            kv_cache_dtype="auto",
            max_num_seqs=4,
            max_num_batched_tokens=2048,
            expert_parallel=False,
            chunked_prefill=True,
        )


@pytest.mark.parametrize(
    "engine_id,command",
    [
        ("llama-cpp", "llama-server"),
        ("ollama", "ollama create"),
        ("tensorrt-llm", "trtllm-serve"),
        ("triton", "tritonserver"),
        ("tgi", "text-generation-launcher"),
        ("lmdeploy", "lmdeploy serve api_server"),
        ("xinference", "xinference launch"),
        ("localai", "local-ai run"),
    ],
)
def test_catalogue_runtime_payloads_have_managed_commands(tmp_path, engine_id, command):
    config = settings(tmp_path)
    config.model_cache_path.mkdir(parents=True)
    model = config.model_cache_path / "acme" / "model"
    model.mkdir(parents=True)
    cluster = ClusterRegistry(config).add_local(
        name="Local GPUs",
        host_address="192.168.1.149",
        runtime_port=30000,
        gpus=gpus(),
        model_storage_path=str(config.docker_host_model_path),
    )
    payload = DockerEngineClient(config).runtime_payload(
        cluster=cluster,
        engine_id=engine_id,
        image=f"example/{engine_id}:stable",
        model_path=str(model),
        served_model_name="test-model",
        context_length=8192,
        tensor_parallel=1,
        pipeline_parallel=1,
        gpu_memory_utilization=0.8,
        kv_cache_dtype="auto",
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        expert_parallel=False,
        chunked_prefill=True,
    )
    assert payload["Entrypoint"] == ["sh", "-lc"]
    assert command in payload["Cmd"][0]
    assert "MODEL_ID=/models/acme/model" in payload["Env"]


def test_cluster_api_lists_and_switches_targets_with_admin_auth(tmp_path):
    from fastapi.testclient import TestClient

    from spark_cluster.app import create_app

    config = settings(tmp_path, model_admin_token="cluster-secret")
    app = create_app(config)
    local = app.state.cluster_registry.add_local(
        name="Unraid GPUs",
        host_address="192.168.1.149",
        runtime_port=30000,
        gpus=gpus(),
        model_storage_path="/mnt/models",
    )
    remote = app.state.cluster_registry.add_remote(
        name="Spark lab",
        endpoint="http://192.168.1.42:30080",
        head_address="192.168.1.42",
        hardware="dgx-spark",
        activate=False,
    )
    with TestClient(app) as client:
        listed = client.get("/api/v1/clusters").json()
        assert listed["active_cluster_id"] == local.id
        assert len(listed["clusters"]) == 2
        assert client.post(f"/api/v1/clusters/{remote.id}/select").status_code == 401
        selected = client.post(
            f"/api/v1/clusters/{remote.id}/select",
            headers={"Authorization": "Bearer cluster-secret"},
        )
        assert selected.status_code == 200
        assert selected.json()["active"] is True
        assert config.remote_cluster_url == "http://192.168.1.42:30080"
