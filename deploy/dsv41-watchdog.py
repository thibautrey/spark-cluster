#!/usr/bin/env python3
"""Recover only a running DSV4.1 container with repeatedly confirmed fatal engine health.

Missing HTTP observations, startup latency and intentional container stops never trigger
a restart. Recovery is limited to this one container, with a cooldown and hourly cap.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def decision(running: bool, health: dict | None, now: float, state: dict) -> bool:
    fatal = running and isinstance(health, dict) and bool(health.get("fatal"))
    if not fatal:
        state["confirmed_fatal"] = 0
        return False
    state["confirmed_fatal"] = state.get("confirmed_fatal", 0) + 1
    recent = [stamp for stamp in state.get("restarts", []) if 0 <= now - stamp < 3600]
    state["restarts"] = recent
    if state["confirmed_fatal"] < 3 or len(recent) >= 3:
        return False
    return not recent or now - recent[-1] >= 300


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".watchdog-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="dsv41")
    parser.add_argument("--health-url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    try:
        state = json.loads(args.state.read_text())
    except (OSError, ValueError):
        state = {}
    state["confirmed_fatal"] = 0
    while True:
        running, health = False, None
        try:
            inspect = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", args.container],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            running = inspect.stdout.strip() == "true"
            if running:
                with urllib.request.urlopen(args.health_url, timeout=5) as response:
                    health = json.load(response)
        except (subprocess.SubprocessError, urllib.error.URLError, OSError, ValueError):
            pass
        if decision(running, health, time.time(), state):
            # Fatal CUDA state is terminal. It cannot finish admitted requests successfully.
            # Never restart the Docker daemon or any other container.
            try:
                subprocess.run(
                    ["docker", "restart", "--time", "30", args.container],
                    check=True,
                    timeout=60,
                    capture_output=True,
                )
                state.setdefault("restarts", []).append(time.time())
                state["confirmed_fatal"] = 0
                state["last_recovery"] = "confirmed-fatal-runtime"
                save(args.state, state)
                print(
                    "Recovered the inference container after confirmed fatal engine health",
                    flush=True,
                )
            except (subprocess.SubprocessError, OSError):
                # Bound failed recovery attempts too; avoid a tight retry loop.
                state.setdefault("restarts", []).append(time.time())
                save(args.state, state)
                print("Inference container recovery failed; cooldown applies", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
