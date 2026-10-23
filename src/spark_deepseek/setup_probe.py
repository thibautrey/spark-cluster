"""Read-only probe executed over authenticated SSH on a candidate Spark."""

import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


if __name__ == "__main__":
    interfaces = []
    for item in json.loads(command("ip", "-j", "-4", "addr") or "[]"):
        path = Path("/sys/class/net") / item["ifname"]
        try:
            speed = int((path / "speed").read_text().strip())
        except (OSError, ValueError):
            speed = 0
        interfaces.append({
            "name": item["ifname"], "speed_mbps": speed,
            "addresses": [a["local"] for a in item.get("addr_info", [])],
            "up": item.get("operstate") == "UP",
        })
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    marker = Path("/var/lib/spark-deepseek/cluster-name")
    role = Path("/var/lib/spark-deepseek/cluster-role")
    print(json.dumps({
        "hostname": socket.gethostname(), "architecture": platform.machine(),
        "gpu": command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
        "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "memory_gib": int(memory["MemTotal"].split()[0]) / 1024**2,
        "available_gib": int(memory["MemAvailable"].split()[0]) / 1024**2,
        "disk_gib": shutil.disk_usage("/var/lib").free / 2**30,
        "sudo": subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0,
        "toolkit": shutil.which("nvidia-container-runtime") is not None,
        "k3s": shutil.which("k3s") is not None,
        "managed_cluster": marker.read_text().strip() if marker.exists() else None,
        "managed_role": role.read_text().strip() if role.exists() else None,
        "interfaces": interfaces, "rdma": bool(glob.glob("/sys/class/infiniband/*")),
        "clock": time.time(), "os": platform.freedesktop_os_release().get("PRETTY_NAME", ""),
        "dgx_release": command("cat", "/etc/dgx-release"),
        "uid": os.getuid(), "gid": os.getgid(),
    }))
