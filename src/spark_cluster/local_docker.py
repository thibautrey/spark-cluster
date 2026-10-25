"""Local NVIDIA GPU discovery and Docker-backed inference runtime."""

import asyncio
import contextlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx

from .cluster_registry import ClusterTarget, GPUDevice
from .config import Settings
from .models import CacheTier, ClusterSnapshot, MemoryStats, ModelRuntime, NodeSnapshot
from .runtime_engines import MANAGED_ENGINE_IDS, container_launch_script

GIB = 2**30


def gpu_process_memory(devices: list[GPUDevice]) -> list[dict]:
    """Read actual device use and host PIDs; unavailable measurements fail closed."""
    try:
        import pynvml
    except ImportError:
        return []
    result = []
    try:
        pynvml.nvmlInit()
        for gpu in devices:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                processes = {}
                complete = True
                try:
                    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        if 0 <= process.usedGpuMemory <= memory.total:
                            processes[process.pid] = process.usedGpuMemory / GIB
                        else:
                            complete = False
                except pynvml.NVMLError:
                    complete = False
                result.append(
                    {
                        "id": gpu.uuid,
                        "index": gpu.index,
                        "total_gib": memory.total / GIB,
                        "used_gib": memory.used / GIB,
                        "processes": processes,
                        "accounting_complete": complete,
                    }
                )
            except pynvml.NVMLError:
                continue
    except pynvml.NVMLError:
        return []
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return result


def detect_nvidia_gpus() -> list[GPUDevice]:
    try:
        import pynvml
    except ImportError:
        return []
    devices = []
    try:
        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            devices.append(
                GPUDevice(
                    index=index,
                    name=pynvml.nvmlDeviceGetName(handle),
                    uuid=pynvml.nvmlDeviceGetUUID(handle),
                    memory_gib=round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / GIB, 2),
                    compute_capability=f"{major}.{minor}",
                )
            )
    except pynvml.NVMLError:
        return []
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
    return devices


def local_gpu_snapshot(
    cluster: ClusterTarget,
    settings: Settings,
    loaded_models_by_gpu: dict[int, list[str]] | None = None,
) -> ClusterSnapshot:
    try:
        import pynvml
    except ImportError:
        pynvml = None
    nodes = []
    if pynvml:
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlInit()
    try:
        for position, gpu in enumerate(cluster.gpus):
            used = utilization = temperature = 0.0
            measured = False
            if pynvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used = memory.used / GIB
                    measured = True
                    utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                    temperature = float(
                        pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    )
                except pynvml.NVMLError:
                    pass
            nodes.append(
                NodeSnapshot(
                    id=gpu.uuid,
                    hostname=f"{cluster.name}-gpu-{gpu.index}",
                    role="head" if position == 0 else "worker",
                    status="healthy" if measured else "offline",
                    address=cluster.host_address or settings.docker_runtime_host,
                    memory=MemoryStats(
                        total_gib=gpu.memory_gib,
                        used_gib=round(used, 2),
                        accelerator_used_gib=round(used, 2),
                    ),
                    gpu_utilization=utilization,
                    temperature_c=temperature,
                    loaded_model_ids=(loaded_models_by_gpu or {}).get(gpu.index, []),
                )
            )
    finally:
        if pynvml:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()
    storage = shutil.disk_usage(settings.model_cache_path)
    return ClusterSnapshot(
        timestamp=datetime.now(UTC),
        cluster_name=cluster.name,
        status="healthy" if nodes and all(n.status == "healthy" for n in nodes) else "offline",
        nodes=nodes,
        cache_tiers=[
            CacheTier(
                name="Shared model storage",
                kind="remote",
                used_gib=round((storage.total - storage.free) / GIB, 2),
                capacity_gib=round(storage.total / GIB, 2),
                persistent=True,
            )
        ],
        runtime=ModelRuntime(
            model="No model loaded",
            quantization="auto",
            context_length=0,
            tensor_parallel=max(1, len(nodes)),
            pipeline_parallel=1,
        ),
    )


class DockerEngineClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.socket = settings.docker_socket_path
        self.transport: httpx.AsyncBaseTransport | None = None

    @property
    def available(self) -> bool:
        return self.settings.docker_runtime_enabled and self.socket.exists()

    def _transport(self) -> httpx.AsyncBaseTransport:
        return self.transport or httpx.AsyncHTTPTransport(uds=str(self.socket))

    async def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict | None = None,
        params: dict | None = None,
        timeout: float = 60,
        allow: set[int] | None = None,
    ) -> httpx.Response:
        if not self.available:
            raise RuntimeError("Docker runtime access is not configured")
        async with httpx.AsyncClient(
            base_url="http://docker",
            transport=self._transport(),
            timeout=timeout,
        ) as client:
            response = await client.request(method, path, json=payload, params=params)
        if allow and response.status_code in allow:
            return response
        response.raise_for_status()
        return response

    async def load_runtime_archive(
        self, archive: Path, platform: str, image: str, config_digest: str
    ) -> None:
        if not self.available:
            raise ValueError("Docker runtime access is not configured")
        info = (await self.request("GET", "/info")).json()
        architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(
            info.get("Architecture"), info.get("Architecture")
        )
        if f"{info.get('OSType')}/{architecture}" != platform:
            raise ValueError("Docker host architecture does not match the controller archive")

        async def content():
            with archive.open("rb") as source:
                while chunk := await asyncio.to_thread(source.read, 1024 * 1024):
                    yield chunk

        async with httpx.AsyncClient(
            base_url="http://docker", transport=self._transport(), timeout=7200
        ) as client:
            async with client.stream(
                "POST",
                "/images/load",
                params={"quiet": "true"},
                content=content(),
                headers={"Content-Type": "application/x-tar"},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("error") or event.get("errorDetail"):
                        raise ValueError("Docker rejected the runtime archive")
        installed = await self.request("GET", f"/images/{quote(image, safe='')}/json")
        value = installed.json()
        architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(
            value.get("Architecture"), value.get("Architecture")
        )
        if f"{value.get('Os')}/{architecture}" != platform:
            raise ValueError("Docker imported an image for a different architecture")

    async def ensure_runtime_image(
        self, repository: str, digest: str, platform: str, config_digest: str
    ) -> dict:
        """Pull an immutable image directly into the local daemon and verify its identity."""
        if not self.available:
            raise ValueError("Docker runtime access is not configured")
        exact = f"{repository}@{digest}"

        async def inspect():
            response = await self.request(
                "GET", f"/images/{quote(exact, safe='')}/json", allow={404}, timeout=30
            )
            return response.json() if response.status_code == 200 else None

        value = await inspect()
        if value is None:
            async with httpx.AsyncClient(
                base_url="http://docker", transport=self._transport(), timeout=7200
            ) as client:
                async with client.stream(
                    "POST", "/images/create", params={"fromImage": exact}
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event.get("error") or event.get("errorDetail"):
                            raise ValueError("Docker failed to pull the verified runtime image")
            value = await inspect()
        if value is None:
            raise ValueError("Docker did not retain the verified runtime image")
        architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(
            value.get("Architecture"), value.get("Architecture")
        )
        if f"{value.get('Os')}/{architecture}" != platform:
            raise ValueError("Docker pulled an image for a different architecture")
        if value.get("Id") != config_digest:
            raise ValueError("Docker image configuration does not match the resolved digest")
        return value

    async def ping(self) -> bool:
        try:
            response = await self.request("GET", "/_ping", timeout=3)
            return response.text.strip() == "OK"
        except (httpx.HTTPError, RuntimeError):
            return False

    async def platform(self) -> str | None:
        try:
            info = (await self.request("GET", "/info", timeout=10)).json()
        except (httpx.HTTPError, RuntimeError):
            return None
        architecture = {"x86_64": "amd64", "aarch64": "arm64"}.get(
            info.get("Architecture"), info.get("Architecture")
        )
        system = info.get("OSType")
        return f"{system}/{architecture}" if system and architecture else None

    async def loaded_models_by_gpu(self, cluster: ClusterTarget) -> dict[int, list[str]]:
        """Read model-to-GPU placement from running Spark Cluster containers."""
        filters = json.dumps({"label": ["spark-cluster.managed=true"]})
        try:
            response = await self.request(
                "GET", "/containers/json", params={"filters": filters}, timeout=10
            )
            containers = response.json()
            placements: dict[int, set[str]] = {}
            known_devices = {str(gpu.index): gpu.index for gpu in cluster.gpus}
            known_devices.update({gpu.uuid: gpu.index for gpu in cluster.gpus})
            for container in containers:
                details = (
                    await self.request(
                        "GET", f"/containers/{quote(container['Id'], safe='')}/json", timeout=10
                    )
                ).json()
                if not details.get("State", {}).get("Running"):
                    continue
                labels = details.get("Config", {}).get("Labels", {})
                model = labels.get("spark-cluster.model-id") or labels.get("spark-cluster.model")
                if not model:
                    continue
                requests = details.get("HostConfig", {}).get("DeviceRequests") or []
                device_ids = [
                    device for request in requests for device in (request.get("DeviceIDs") or [])
                ]
                if "all" in device_ids:
                    device_ids = list(known_devices)
                for device in device_ids:
                    index = known_devices.get(str(device))
                    if index is not None:
                        placements.setdefault(index, set()).add(str(model))
            return {index: sorted(models) for index, models in placements.items()}
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return {}

    async def gpu_memory(self, profiles, cluster: ClusterTarget | None = None) -> list[dict]:
        devices = cluster.gpus if cluster else await asyncio.to_thread(detect_nvidia_gpus)
        gpus = await asyncio.to_thread(gpu_process_memory, devices)
        if not gpus:
            return []

        async def processes(profile):
            try:
                name = quote(profile.name, safe="")
                state = await self.request(
                    "GET", f"/containers/{name}/json", allow={404}, timeout=5
                )
                if state.status_code == 404 or not state.json().get("State", {}).get("Running"):
                    return profile.id, set(), True
                labels = state.json().get("Config", {}).get("Labels", {})
                if labels.get("spark-cluster.on-demand") != profile.id:
                    return profile.id, set(), False
                response = await self.request(
                    "GET", f"/containers/{name}/top", params={"ps_args": "-eo pid"}, timeout=5
                )
                table = response.json()
                column = [title.casefold() for title in table["Titles"]].index("pid")
                return profile.id, {int(row[column]) for row in table["Processes"]}, True
            except (httpx.HTTPError, RuntimeError, KeyError, TypeError, ValueError):
                return profile.id, set(), False

        owners = await asyncio.gather(*(processes(profile) for profile in profiles))
        for gpu in gpus:
            gpu["managed_gib"] = {
                owner: sum(gpu["processes"].get(pid, 0.0) for pid in pids)
                for owner, pids, _ in owners
            }
            gpu["accounting_complete"] &= all(complete for _, _, complete in owners)
            gpu.pop("processes")
        return gpus

    @staticmethod
    def container_name(cluster_id: str) -> str:
        return f"spark-cluster-runtime-{cluster_id}"

    async def _remove_existing(self, name: str) -> None:
        encoded = quote(name, safe="")
        response = await self.request("GET", f"/containers/{encoded}/json", allow={404}, timeout=10)
        if response.status_code == 404:
            return
        state = response.json().get("State", {})
        if state.get("Running"):
            await self.request(
                "POST",
                f"/containers/{encoded}/stop",
                params={"t": 20},
                allow={304},
                timeout=30,
            )
        await self.request(
            "DELETE",
            f"/containers/{encoded}",
            params={"force": "true"},
            timeout=30,
        )

    async def image_present(self, image: str) -> bool:
        response = await self.request(
            "GET", f"/images/{quote(image, safe='')}/json", allow={404}, timeout=10
        )
        return response.status_code == 200

    async def pull_image(self, image: str) -> None:
        repository, separator, tag = image.rpartition(":")
        if not separator or "/" in tag:
            repository, tag = image, "latest"
        response = await self.request(
            "POST",
            "/images/create",
            params={"fromImage": repository, "tag": tag},
            timeout=1800,
        )
        if b'"error"' in response.content:
            raise RuntimeError("Docker reported an image pull failure")

    def runtime_payload(
        self,
        *,
        cluster: ClusterTarget,
        engine_id: str = "vllm",
        image: str,
        model_path: str,
        served_model_name: str,
        context_length: int,
        tensor_parallel: int,
        pipeline_parallel: int,
        gpu_memory_utilization: float,
        kv_cache_dtype: str,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        expert_parallel: bool,
        chunked_prefill: bool,
    ) -> dict:
        if tensor_parallel * pipeline_parallel > cluster.gpu_count:
            raise ValueError("Requested parallelism exceeds GPUs assigned to this cluster")
        if engine_id not in MANAGED_ENGINE_IDS:
            raise ValueError(f"The local Docker adapter cannot launch {engine_id}")
        tensor_engines = {"vllm", "sglang", "tensorrt-llm", "tgi", "lmdeploy"}
        pipeline_engines = {"vllm", "tensorrt-llm"}
        expert_engines = {"vllm", "tensorrt-llm"}
        if tensor_parallel > 1 and engine_id not in tensor_engines:
            raise ValueError(f"The local {engine_id} adapter does not expose tensor parallelism")
        if pipeline_parallel > 1 and engine_id not in pipeline_engines:
            raise ValueError(f"The local {engine_id} adapter does not expose pipeline parallelism")
        if expert_parallel and engine_id not in expert_engines:
            raise ValueError(f"The local {engine_id} adapter does not expose expert parallelism")
        controller_root = self.settings.model_cache_path.resolve()
        resolved_model = Path(model_path).resolve()
        if controller_root not in resolved_model.parents:
            raise ValueError("Model path is outside managed model storage")
        relative = resolved_model.relative_to(controller_root).as_posix()
        runtime_model = f"/models/{relative}"
        if engine_id == "sglang":
            entrypoint = ["python3", "-m", "sglang.launch_server"]
            command = [
                "--model-path",
                runtime_model,
                "--served-model-name",
                served_model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(cluster.runtime_port),
                "--tp",
                str(tensor_parallel),
                "--context-length",
                str(context_length),
                "--mem-fraction-static",
                str(gpu_memory_utilization),
                "--max-running-requests",
                str(max_num_seqs),
            ]
            if kv_cache_dtype != "auto":
                command.extend(["--kv-cache-dtype", kv_cache_dtype])
            if chunked_prefill:
                command.extend(["--chunked-prefill-size", str(max_num_batched_tokens)])
        elif engine_id == "vllm":
            entrypoint = ["vllm", "serve"]
            command = [
                runtime_model,
                "--served-model-name",
                served_model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(cluster.runtime_port),
                "--tensor-parallel-size",
                str(tensor_parallel),
                "--pipeline-parallel-size",
                str(pipeline_parallel),
                "--max-model-len",
                str(context_length),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--max-num-seqs",
                str(max_num_seqs),
                "--max-num-batched-tokens",
                str(max_num_batched_tokens),
            ]
            if kv_cache_dtype != "auto":
                command.extend(["--kv-cache-dtype", kv_cache_dtype])
            if expert_parallel:
                command.append("--enable-expert-parallel")
            if chunked_prefill:
                command.append("--enable-chunked-prefill")
        else:
            entrypoint = ["sh", "-lc"]
            command = [container_launch_script(engine_id)]
        device_ids = [
            str(item.index) for item in cluster.gpus[: tensor_parallel * pipeline_parallel]
        ]
        payload = {
            "Image": image,
            "Entrypoint": entrypoint,
            "Cmd": command,
            "Env": [
                f"NVIDIA_VISIBLE_DEVICES={','.join(device_ids)}",
                f"MODEL_ID={runtime_model}",
                f"SERVED_MODEL_NAME={served_model_name}",
                f"RUNTIME_PORT={cluster.runtime_port}",
                f"MAX_MODEL_LEN={context_length}",
                f"TENSOR_PARALLEL_SIZE={tensor_parallel}",
                f"PIPELINE_PARALLEL_SIZE={pipeline_parallel}",
                f"GPU_MEMORY_UTILIZATION={gpu_memory_utilization}",
                f"KV_CACHE_DTYPE={kv_cache_dtype}",
                f"MAX_NUM_SEQS={max_num_seqs}",
                f"MAX_NUM_BATCHED_TOKENS={max_num_batched_tokens}",
            ],
            "HostConfig": {
                "NetworkMode": "host",
                "Runtime": "nvidia",
                "Binds": [f"{self.settings.docker_host_model_path}:/models:ro"],
                "DeviceRequests": [
                    {
                        "Driver": "nvidia",
                        "DeviceIDs": device_ids,
                        "Capabilities": [["gpu"]],
                    }
                ],
                "IpcMode": "host",
                "ShmSize": 16 * GIB,
                "RestartPolicy": {"Name": "unless-stopped"},
            },
            "Labels": {
                "spark-cluster.managed": "true",
                "spark-cluster.cluster-id": cluster.id,
                "spark-cluster.model": served_model_name,
                "spark-cluster.engine": engine_id,
            },
        }
        return payload

    async def launch(
        self,
        *,
        cluster: ClusterTarget,
        engine_id: str = "vllm",
        image: str,
        model_path: str,
        model_id: str,
        served_model_name: str,
        context_length: int,
        tensor_parallel: int,
        pipeline_parallel: int,
        gpu_memory_utilization: float,
        kv_cache_dtype: str,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        expert_parallel: bool,
        chunked_prefill: bool,
    ) -> str:
        if tensor_parallel * pipeline_parallel > cluster.gpu_count:
            raise ValueError("Requested parallelism exceeds GPUs assigned to this cluster")
        image_path = quote(image, safe="")
        image_response = await self.request(
            "GET", f"/images/{image_path}/json", allow={404}, timeout=10
        )
        if image_response.status_code == 404:
            raise RuntimeError(
                f"Runtime image {image} is not present on the Docker host; install it first"
            )
        payload = self.runtime_payload(
            cluster=cluster,
            engine_id=engine_id,
            image=image,
            model_path=model_path,
            served_model_name=served_model_name,
            context_length=context_length,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_dtype=kv_cache_dtype,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            expert_parallel=expert_parallel,
            chunked_prefill=chunked_prefill,
        )
        payload["Labels"]["spark-cluster.model-id"] = model_id
        name = self.container_name(cluster.id)
        await self._remove_existing(name)
        response = await self.request(
            "POST",
            "/containers/create",
            params={"name": name},
            payload=payload,
            timeout=30,
        )
        container_id = response.json()["Id"]
        await self.request("POST", f"/containers/{container_id}/start", timeout=30)
        return container_id

    async def wait_ready(
        self, cluster: ClusterTarget, readiness_path: str = "/v1/models", timeout: int = 900
    ) -> None:
        url = f"http://{cluster.host_address}:{cluster.runtime_port}{readiness_path}"
        deadline = asyncio.get_running_loop().time() + timeout
        async with httpx.AsyncClient(timeout=3) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(3)
        raise TimeoutError("Docker runtime did not become ready within 15 minutes")
