"""Read-only probe executed over authenticated SSH on a candidate machine."""

import glob
import ipaddress
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path

PROBE_PEERS = globals().get("PROBE_PEERS", [])


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def read(path: Path, default=""):
    try:
        return path.read_text().strip()
    except OSError:
        return default


def rdma_details(interface: str):
    devices = []
    for device_path in Path("/sys/class/infiniband").glob("*"):
        if not (device_path / "device" / "net" / interface).exists():
            continue
        ports = []
        for port_path in (device_path / "ports").glob("*"):
            ports.append(
                {
                    "port": port_path.name,
                    "state": read(port_path / "state"),
                    "physical_state": read(port_path / "phys_state"),
                    "rate": read(port_path / "rate"),
                    "link_layer": read(port_path / "link_layer"),
                }
            )
        devices.append({"device": device_path.name, "ports": ports})
    return devices


def peer_checks(interface, peers):
    """Send bounded interface-bound pings only to selected peers on this fabric subnet."""
    results = []
    if not interface["carrier"] or not interface["rdma"]:
        return results
    networks = [
        ipaddress.ip_network(f"{item['local']}/{item['prefixlen']}", strict=False)
        for item in interface["address_info"]
    ]
    for address in peers[:8]:
        try:
            target = ipaddress.IPv4Address(address)
        except ipaddress.AddressValueError:
            continue
        if address in interface["addresses"] or not any(target in net for net in networks):
            continue
        try:
            reachable = (
                subprocess.run(
                    ["ping", "-n", "-I", interface["name"], "-c", "1", "-W", "1", address],
                    capture_output=True,
                    timeout=2,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            reachable = False
        results.append({"address": address, "reachable": reachable})
    return results


def system_memory():
    system = platform.system().lower()
    if system == "linux":
        try:
            values = dict(
                line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
            return (
                int(values["MemTotal"].split()[0]) / 1024**2,
                int(values["MemAvailable"].split()[0]) / 1024**2,
            )
        except (OSError, KeyError, ValueError):
            return 0.0, 0.0
    if system == "darwin":
        try:
            total = int(command("sysctl", "-n", "hw.memsize")) / 2**30
            return total, None
        except ValueError:
            return 0.0, 0.0
    if system == "windows":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong),
                    ("available_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 2**30, status.available / 2**30
        except (AttributeError, OSError):
            return 0.0, 0.0
    return 0.0, 0.0


def accelerator_facts():
    names = [
        item
        for item in command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()
        if item
    ]
    if names:
        return "nvidia", "cuda", names
    system = platform.system().lower()
    if system == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        name = command("sysctl", "-n", "machdep.cpu.brand_string") or "Apple Silicon"
        return "apple", "metal", [name]
    amd = command("rocm-smi", "--showproductname")
    if amd:
        names = [line.split(":", 1)[-1].strip() for line in amd.splitlines() if "Card" in line]
        return "amd", "rocm", names or ["AMD GPU"]
    if system == "windows":
        output = command(
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        )
        names = [name for name in output.splitlines() if name.strip()]
        if names:
            vendor = "amd" if "AMD" in output.upper() else "unknown"
            return vendor, "unverified", names
    return "none", "cpu", []


def machine_class(system, vendor, names, architecture):
    joined = " ".join(names).upper()
    if system == "linux" and "GB10" in joined and architecture == "aarch64":
        return "dgx-spark", "k3s-cluster", True
    if system == "linux" and vendor == "nvidia":
        return "linux-nvidia", "k3s-nvidia", True
    if system == "linux" and vendor == "amd":
        return "linux-amd", "docker-rocm", False
    if system == "darwin" and vendor == "apple":
        return "mac-apple-silicon", "metal-agent", False
    if system == "windows" and vendor != "none":
        return "windows-gpu", "windows-agent", False
    return f"{system or 'unknown'}-cpu", "unsupported", False


if __name__ == "__main__":
    interfaces = []
    addresses = {
        item["ifname"]: item.get("addr_info", [])
        for item in json.loads(command("ip", "-j", "-4", "addr") or "[]")
    }
    for item in json.loads(command("ip", "-j", "link") or "[]"):
        item["addr_info"] = addresses.get(item["ifname"], [])
        path = Path("/sys/class/net") / item["ifname"]
        try:
            speed = int((path / "speed").read_text().strip())
        except (OSError, ValueError):
            speed = 0
        details = rdma_details(item["ifname"])
        device_path = path / "device"
        try:
            pci_address = device_path.resolve().name
        except OSError:
            pci_address = None
        try:
            driver = (device_path / "driver").resolve().name
        except OSError:
            driver = None
        interfaces.append(
            {
                "name": item["ifname"],
                "speed_mbps": speed,
                "addresses": [a["local"] for a in item.get("addr_info", [])],
                "address_info": [
                    {"local": a["local"], "prefixlen": a["prefixlen"]}
                    for a in item.get("addr_info", [])
                ],
                "up": item.get("operstate") == "UP",
                "carrier": read(path / "carrier") == "1",
                "mtu": int(item.get("mtu", 0) or 0),
                "mac_address": item.get("address"),
                "pci_address": pci_address,
                "driver": driver,
                "rdma": bool(details),
                "rdma_device": details[0]["device"] if details else None,
                "rdma_ports": [port for device in details for port in device["ports"]],
                "neighbors": json.loads(
                    command("ip", "-j", "neigh", "show", "dev", item["ifname"]) or "[]"
                ),
            }
        )
        if globals().get("PROBE_PEERS"):
            interfaces[-1]["peer_checks"] = peer_checks(interfaces[-1], PROBE_PEERS)
            interfaces[-1]["neighbors"] = json.loads(
                command("ip", "-j", "neigh", "show", "dev", item["ifname"]) or "[]"
            )
    total_memory, available_memory = system_memory()
    system = platform.system().lower()
    vendor, backend, accelerators = accelerator_facts()
    machine_kind, setup_strategy, install_supported = machine_class(
        system, vendor, accelerators, platform.machine()
    )
    state_root = Path("/var/lib") if Path("/var/lib").exists() else Path.home()
    marker = Path("/var/lib/spark-cluster/cluster-name")
    role = Path("/var/lib/spark-cluster/cluster-role")
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "platform": system,
                "architecture": platform.machine(),
                "machine_kind": machine_kind,
                "setup_strategy": setup_strategy,
                "install_supported": install_supported and shutil.which("apt-get") is not None,
                "accelerator_vendor": vendor,
                "accelerator_backend": backend,
                "accelerators": accelerators,
                "gpu": "\n".join(accelerators),
                "driver": command(
                    "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
                ),
                "memory_gib": total_memory,
                "available_gib": available_memory,
                "disk_gib": shutil.disk_usage(state_root).free / 2**30,
                "sudo": (
                    subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0
                    if system != "windows" and shutil.which("sudo")
                    else False
                ),
                "toolkit": (
                    shutil.which("nvidia-container-runtime") is not None
                    if backend == "cuda"
                    else backend in {"metal", "rocm"}
                ),
                "k3s": shutil.which("k3s") is not None,
                "managed_cluster": marker.read_text().strip() if marker.exists() else None,
                "managed_role": role.read_text().strip() if role.exists() else None,
                "interfaces": interfaces,
                "rdma": bool(glob.glob("/sys/class/infiniband/*")),
                "clock": time.time(),
                "os": (
                    platform.freedesktop_os_release().get("PRETTY_NAME", "")
                    if system == "linux"
                    else platform.platform()
                ),
                "dgx_release": command("cat", "/etc/dgx-release"),
                "uid": os.getuid() if hasattr(os, "getuid") else None,
                "gid": os.getgid() if hasattr(os, "getgid") else None,
            }
        )
    )
