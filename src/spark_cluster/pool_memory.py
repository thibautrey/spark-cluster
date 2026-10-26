"""Per-device admission planning. Plans are read-only; the node owns reservations."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RuntimeResources(BaseModel):
    gpu_count: int = Field(default=1, ge=1, le=64)
    gpu_ids: list[str] = Field(default_factory=list)
    host_memory_gib: float = Field(default=0, ge=0)
    required_gib_per_gpu: float | None = Field(default=None, gt=0)
    memory_fraction: float = Field(default=0.82, gt=0, lt=1)
    mode: Literal["auto", "fixed", "exclusive"] = "exclusive"


@dataclass
class MemoryPlan:
    gpu_ids: list[str] = field(default_factory=list)
    budget_gib: dict[str, float] = field(default_factory=dict)
    fraction: float | None = None
    evict: list[str] = field(default_factory=list)
    waiting: bool = False
    reason: str | None = None
    available_gib: float = 0.0

    def as_dict(self):
        return {
            "gpu_ids": self.gpu_ids,
            "budget_gib": self.budget_gib,
            "gpu_memory_utilization": self.fraction,
            "evict": self.evict,
            "waiting": self.waiting,
            "reason": self.reason,
            "fits": bool(self.gpu_ids),
            "required_gib": round(sum(self.budget_gib.values()), 3),
            "available_gib": round(self.available_gib, 3),
        }


def host_memory_available_gib() -> float | None:
    """Host RAM with an OS reserve, bounded by the controller's cgroup when limited."""
    try:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        total = int(values["MemTotal"].split()[0]) / 1024**2
        available = int(values["MemAvailable"].split()[0]) / 1024**2
        limit = Path("/sys/fs/cgroup/memory.max")
        if limit.exists() and (value := limit.read_text().strip()) != "max":
            total = min(total, int(value) / 2**30)
            used = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
            available = min(available, int(value) / 2**30 - used)
        return max(0.0, available - max(2.0, total * 0.1))
    except (OSError, ValueError, KeyError):
        return None


def command_value(command: list, flag: str) -> str | None:
    for index, value in enumerate(command):
        if value == flag and index + 1 < len(command):
            return str(command[index + 1])
        if isinstance(value, str) and value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def imported_resources(profile, gpus: list[dict]) -> RuntimeResources:
    """Imported recipes keep their device bindings and engine memory limits."""
    host = profile.container.get("HostConfig", {})
    requests = host.get("DeviceRequests") or []
    devices = [str(d) for r in requests for d in r.get("DeviceIDs", [])]
    env = dict(item.split("=", 1) for item in profile.container.get("Env", []) if "=" in item)
    visible = env.get("NVIDIA_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES")
    if not devices and visible:
        devices = visible.split(",")
    if not devices or "all" in devices or any(r.get("Count") == -1 for r in requests):
        devices = [g["id"] for g in gpus]
    aliases = {str(alias): g["id"] for g in gpus for alias in (g["id"], g.get("index"))}
    devices = list(dict.fromkeys(aliases.get(d, d) for d in devices))
    command = profile.container.get("Cmd") or []
    for flag in ("--gpu-memory-utilization", "--mem-fraction-static"):
        value = command_value(command, flag)
        if value is not None:
            try:
                fraction = float(value)
                if math.isfinite(fraction) and 0 < fraction < 1:
                    return RuntimeResources(
                        gpu_count=max(1, len(devices)),
                        gpu_ids=devices,
                        memory_fraction=fraction,
                        mode="fixed",
                    )
            except ValueError:
                pass
    return RuntimeResources(gpu_count=max(1, len(devices)), gpu_ids=devices)


def gpu_availability(gpus, residents, usable_fraction, exclude=()):
    """Count measured external use plus max(reservation, actual managed use)."""
    result = []
    for gpu in gpus:
        owners = [r for r in residents.values() if gpu["id"] in r.gpu_ids]
        managed = gpu.get("managed_gib", {})
        external = max(0.0, gpu["used_gib"] - sum(managed.get(r.profile.id, 0) for r in owners))
        remaining = [r for r in owners if r.profile.id not in exclude]
        reserved = sum(
            max(r.budget_gib[gpu["id"]], managed.get(r.profile.id, 0)) for r in remaining
        )
        usable = gpu["total_gib"] * usable_fraction
        free = max(0.0, usable - external - reserved)
        # Missing process attribution cannot safely support sharing a live device.
        if remaining and not gpu.get("accounting_complete", True):
            free = 0.0
        result.append(
            {
                **gpu,
                "external_gib": external,
                "reserved_gib": reserved,
                "available_gib": free,
                "occupied": bool(remaining),
                "exclusive": any(r.exclusive for r in remaining),
            }
        )
    return result


def _allocate(resources, availability):
    allowed = [g for g in availability if not resources.gpu_ids or g["id"] in resources.gpu_ids]
    if resources.gpu_ids and set(resources.gpu_ids) != {g["id"] for g in allowed}:
        return None
    if resources.mode == "exclusive":
        allowed = [g for g in allowed if not g["occupied"] and g["available_gib"] > 0]
        allowed.sort(key=lambda g: (-g["available_gib"], g["id"]))
        if len(allowed) < resources.gpu_count:
            return None
        chosen = allowed[: resources.gpu_count]
        return MemoryPlan(
            [g["id"] for g in chosen],
            {g["id"]: g["available_gib"] for g in chosen},
            available_gib=sum(g["available_gib"] for g in chosen),
        )
    allowed = [g for g in allowed if not g["exclusive"]]
    fractions = [resources.memory_fraction]
    if resources.mode == "auto":
        if resources.required_gib_per_gpu is None:
            return None
        # Each possible smallest rank determines a common engine fraction. This
        # avoids a combinatorial GPU-subset search on heterogeneous hardware.
        fractions = sorted(
            {
                math.ceil(resources.required_gib_per_gpu / g["total_gib"] * 10000) / 10000
                for g in allowed
                if g["total_gib"] > 0
            }
        )
    choices = []
    for fraction in fractions:
        if fraction > resources.memory_fraction + 1e-9:
            continue
        eligible = [
            g
            for g in allowed
            if (
                g["total_gib"] * fraction <= g["available_gib"] + 1e-6
                and (
                    resources.required_gib_per_gpu is None
                    or g["total_gib"] * fraction + 1e-6 >= resources.required_gib_per_gpu
                )
            )
        ]
        eligible.sort(key=lambda g: (g["occupied"], -g["available_gib"], g["id"]))
        if len(eligible) < resources.gpu_count:
            continue
        chosen = eligible[: resources.gpu_count]
        budget = {g["id"]: g["total_gib"] * fraction for g in chosen}
        plan = MemoryPlan(
            list(budget), budget, fraction, available_gib=sum(g["available_gib"] for g in chosen)
        )
        choices.append(
            (sum(g["occupied"] for g in chosen), sum(budget.values()), plan.gpu_ids, plan)
        )
    return min(choices, key=lambda item: item[:3])[-1] if choices else None


def plan_memory(
    resources, gpus, residents, retained, usable_fraction=0.82, host_available_gib=None
):
    if resources.host_memory_gib:
        # Reserve every resident's full offload allowance as well as measured use.
        # This conservative bound also covers runtimes that are still starting.
        reserved = sum(
            r.profile.resources.host_memory_gib
            for r in residents.values() if r.profile.resources
        )
        if host_available_gib is None:
            return MemoryPlan(reason="Host RAM telemetry is unavailable for degraded loading")
        if resources.host_memory_gib + reserved > host_available_gib:
            return MemoryPlan(reason="Insufficient host RAM for degraded loading")
    if not gpus:
        return MemoryPlan(waiting=True, reason="GPU memory telemetry is unavailable")
    available = gpu_availability(gpus, residents, usable_fraction)
    plan = _allocate(resources, available)
    if plan:
        return plan
    victims = sorted(
        (
            r
            for r in residents.values()
            if r.profile.id not in retained
            and not r.active_requests
            and r.status in {"ready", "unhealthy"}
            and (not resources.gpu_ids or set(r.gpu_ids) & set(resources.gpu_ids))
        ),
        key=lambda r: (r.last_used, r.profile.id),
    )
    evict = []
    for victim in victims:
        evict.append(victim.profile.id)
        plan = _allocate(resources, gpu_availability(gpus, residents, usable_fraction, evict))
        if plan:
            plan.evict = list(evict)
            plan.reason = "Free memory by unloading idle models: " + ", ".join(evict)
            return plan
    empty = [
        {
            **g,
            "available_gib": g["total_gib"] * usable_fraction,
            "occupied": False,
            "exclusive": False,
        }
        for g in gpus
    ]
    possible = _allocate(resources, empty)
    return MemoryPlan(
        waiting=possible is not None,
        reason=(
            "Waiting for GPU memory: retained models, active requests, "
            "starting runtimes or external use"
            if possible
            else "Model budget or parallelism exceeds the assigned GPU capacity"
        ),
        available_gib=sum(g["available_gib"] for g in available),
    )
