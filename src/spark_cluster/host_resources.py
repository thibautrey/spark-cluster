"""Measured Linux host resources, shared by GPU snapshots and node agents."""

import socket
import threading
import time
from pathlib import Path

from .models import CpuCoreLoad, HostResources


class HostSampler:
    def __init__(self, proc_root: Path = Path("/proc")):
        self.proc_root = proc_root
        self.lock = threading.Lock()
        self.previous: dict[int, tuple[int, int]] = {}
        self.cached: HostResources | None = None
        self.sampled_at = -float("inf")
        self.physical_cores: int | None = None

    def collect(self, identity: str, hostname: str) -> HostResources:
        with self.lock:
            now = time.monotonic()
            if self.cached and now - self.sampled_at < 1:
                return self.cached.model_copy(update={"id": identity, "hostname": hostname})
            result = HostResources(id=identity, hostname=hostname)
            try:
                memory = dict(
                    line.split(":", 1)
                    for line in (self.proc_root / "meminfo").read_text().splitlines()
                )
                total = int(memory["MemTotal"].split()[0]) / 1024**2
                available = int(memory["MemAvailable"].split()[0]) / 1024**2
                result.ram_total_gib = round(total, 2)
                result.ram_available_gib = round(min(total, available), 2)
                result.ram_used_gib = round(max(0, total - available), 2)
            except (OSError, ValueError, KeyError):
                pass
            counters = {}
            try:
                for line in (self.proc_root / "stat").read_text().splitlines():
                    fields = line.split()
                    if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
                        continue
                    # guest/guest_nice are already included in user/nice.
                    values = [int(v) for v in fields[1:9]]
                    if len(values) < 5:
                        continue
                    core = int(fields[0][3:])
                    total, idle = sum(values), values[3] + values[4]
                    utilization = None
                    previous = self.previous.get(core)
                    if previous and total > previous[0] and idle >= previous[1]:
                        utilization = round(
                            max(
                                0,
                                min(100, 100 * (1 - (idle - previous[1]) / (total - previous[0]))),
                            ),
                            1,
                        )
                    result.cpu_cores.append(CpuCoreLoad(id=core, utilization=utilization))
                    counters[core] = (total, idle)
                if result.cpu_cores:
                    result.cpu_logical_cores = len(result.cpu_cores)
                    measured = [
                        c.utilization for c in result.cpu_cores if c.utilization is not None
                    ]
                    result.cpu_utilization = (
                        round(sum(measured) / len(measured), 1) if measured else None
                    )
            except (OSError, ValueError):
                pass
            if self.physical_cores is None:
                try:
                    topology = []
                    for entry in (self.proc_root / "cpuinfo").read_text().split("\n\n"):
                        values = dict(
                            line.split(":", 1) for line in entry.splitlines() if ":" in line
                        )
                        values = {k.strip(): v.strip() for k, v in values.items()}
                        if "core id" in values and "physical id" in values:
                            topology.append((values["physical id"], values["core id"]))
                    self.physical_cores = len(set(topology)) or None
                except OSError:
                    pass
            result.cpu_physical_cores = self.physical_cores
            self.previous = counters
            self.cached, self.sampled_at = result, now
            return result


_sampler = HostSampler()


def collect_host_resources(
    identity: str | None = None, hostname: str | None = None
) -> HostResources:
    hostname = hostname or socket.gethostname()
    return _sampler.collect(identity or hostname, hostname)
