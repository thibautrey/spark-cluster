from pathlib import Path

import pytest

from spark_cluster.host_resources import HostSampler


def write_host(root: Path, stat: str):
    (root / "stat").write_text(stat)
    (root / "meminfo").write_text(
        "MemTotal: 8388608 kB\nMemAvailable: 3145728 kB\nMemFree: 1048576 kB\n"
    )
    (root / "cpuinfo").write_text(
        "processor: 0\nphysical id: 0\ncore id: 0\n\nprocessor: 1\nphysical id: 0\ncore id: 0\n"
    )


def test_host_ram_uses_available_memory_and_cpu_uses_per_core_deltas(tmp_path, monkeypatch):
    clock = iter([0, 2])
    monkeypatch.setattr("spark_cluster.host_resources.time.monotonic", lambda: next(clock))
    write_host(tmp_path, "cpu0 100 0 0 900 0 0 0 0 100 0\ncpu1 200 0 0 800 0 0 0 0 0 0\n")
    sampler = HostSampler(tmp_path)
    first = sampler.collect("host", "Unraid")
    assert first.ram_total_gib == 8
    assert first.ram_used_gib == 5
    assert first.ram_available_gib == 3
    assert first.cpu_physical_cores == 1
    assert first.cpu_logical_cores == 2
    assert first.cpu_utilization is None
    write_host(tmp_path, "cpu0 150 0 0 950 0 0 0 0 150 0\ncpu1 220 0 0 860 20 0 0 0 0 0\n")
    measured = sampler.collect("host", "Unraid")
    assert [core.utilization for core in measured.cpu_cores] == [50, 20]
    assert measured.cpu_utilization == 35


def test_host_sampling_does_not_invent_zero_load_without_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr("spark_cluster.host_resources.time.monotonic", lambda: 0)
    host = HostSampler(tmp_path).collect("missing", "Missing")
    assert host.ram_total_gib is None
    assert host.cpu_utilization is None
    assert host.cpu_cores == []


def test_sampling_cache_and_cpu_counter_reset_are_explicit(tmp_path, monkeypatch):
    clock = iter([0, 0.1, 2, 4])
    monkeypatch.setattr("spark_cluster.host_resources.time.monotonic", lambda: next(clock))
    write_host(tmp_path, "cpu0 100 0 0 900 0\n")
    sampler = HostSampler(tmp_path)
    first = sampler.collect("host", "Unraid")
    write_host(tmp_path, "cpu0 1 0 0 1 0\n")
    assert sampler.collect("host", "Unraid").last_seen == first.last_seen
    assert sampler.collect("host", "Unraid").cpu_utilization is None
    write_host(tmp_path, "cpu0 2 0 0 4 0\n")
    assert sampler.collect("host", "Unraid").cpu_utilization == pytest.approx(25)
