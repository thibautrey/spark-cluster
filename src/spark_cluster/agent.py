import os
import socket
from datetime import UTC, datetime

from .models import MemoryStats, NodeSnapshot


def _system_memory() -> tuple[float, float]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    total = os.sysconf("SC_PHYS_PAGES") * page_size / 2**30
    available = os.sysconf("SC_AVPHYS_PAGES") * page_size / 2**30
    return total, total - available


def collect_node(node_id: str | None = None, role: str = "worker") -> NodeSnapshot:
    hostname = socket.gethostname()
    total, used = _system_memory()
    gpu_utilization = 0.0
    temperature = 0.0
    gpu_used = 0.0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        temperature = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        gpu_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**30
    except (ImportError, Exception):
        pass
    return NodeSnapshot(
        id=node_id or hostname,
        hostname=hostname,
        role=role,  # type: ignore[arg-type]
        address=socket.gethostbyname(hostname),
        memory=MemoryStats(
            total_gib=round(total, 2), used_gib=round(used, 2), kv_cache_gib=round(gpu_used, 2)
        ),
        gpu_utilization=gpu_utilization,
        temperature_c=temperature,
        last_seen=datetime.now(UTC),
    )
