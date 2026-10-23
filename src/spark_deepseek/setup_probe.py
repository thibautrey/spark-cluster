"""Read-only probe executed over authenticated SSH on a candidate Spark."""

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
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    marker = Path("/var/lib/spark-deepseek/cluster-name")
    role = Path("/var/lib/spark-deepseek/cluster-role")
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "architecture": platform.machine(),
                "gpu": command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
                "driver": command(
                    "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
                ),
                "memory_gib": int(memory["MemTotal"].split()[0]) / 1024**2,
                "available_gib": int(memory["MemAvailable"].split()[0]) / 1024**2,
                "disk_gib": shutil.disk_usage("/var/lib").free / 2**30,
                "sudo": subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0,
                "toolkit": shutil.which("nvidia-container-runtime") is not None,
                "k3s": shutil.which("k3s") is not None,
                "managed_cluster": marker.read_text().strip() if marker.exists() else None,
                "managed_role": role.read_text().strip() if role.exists() else None,
                "interfaces": interfaces,
                "rdma": bool(glob.glob("/sys/class/infiniband/*")),
                "clock": time.time(),
                "os": platform.freedesktop_os_release().get("PRETTY_NAME", ""),
                "dgx_release": command("cat", "/etc/dgx-release"),
                "uid": os.getuid(),
                "gid": os.getgid(),
            }
        )
    )
