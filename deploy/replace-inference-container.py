#!/usr/bin/env python3
"""Replace one idle inference container while preserving its settings and rollback copy.

Docker configuration, including credentials, travels over the local Unix socket only.
The private inspect backup is never printed. The old container remains stopped under
a timestamped name; failed startup restores that exact container automatically.
"""

import argparse
import datetime
import http.client
import json
import os
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path


class Docker(http.client.HTTPConnection):
    def __init__(self):
        super().__init__("localhost", timeout=90)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")


def api(method, path, value=None):
    connection = Docker()
    try:
        data = json.dumps(value).encode() if value is not None else None
        connection.request(method, path, body=data, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = response.read()
        if response.status >= 400:
            raise RuntimeError(f"Docker {method} failed with HTTP {response.status}")
        return json.loads(body) if body else None
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--idle-url")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--env-overrides", type=Path)
    parser.add_argument("--no-drafts", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--stop-timeout", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.stop_timeout <= 300:
        parser.error("--stop-timeout must be between 1 and 300 seconds")
    # A live idle observation is required immediately before mutation.
    health = json.load(urllib.request.urlopen(args.idle_url or args.health_url, timeout=10))
    if health.get("ok") is not True or health.get("inflight") != 0 or health.get("queued", 0):
        raise SystemExit("Runtime is not healthy and idle; existing container was preserved")
    name = urllib.parse.quote(args.name, safe="")
    old = api("GET", f"/containers/{name}/json")
    api("GET", "/images/" + urllib.parse.quote(args.image, safe="") + "/json")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_name = args.name + "-before-" + stamp
    args.backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = args.backup_dir / f"{args.name}-{stamp}.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(old, stream)
    config = old["Config"]
    config["Image"] = args.image
    config["HostConfig"] = old["HostConfig"]
    if args.env_overrides:
        overrides = json.loads(args.env_overrides.read_text())
        env = dict(entry.split("=", 1) for entry in config.get("Env", []))
        env.update({key: str(value) for key, value in overrides.items()})
        config["Env"] = [f"{key}={value}" for key, value in env.items()]
    if args.no_drafts and "--no-drafts" not in config.get("Cmd", []):
        config["Cmd"].append("--no-drafts")
    created = False
    api("POST", f"/containers/{name}/stop?t={args.stop_timeout}")
    api("POST", f"/containers/{name}/rename?name={urllib.parse.quote(backup_name)}")
    try:
        api("POST", f"/containers/create?name={name}", config)
        created = True
        api("POST", f"/containers/{name}/start")
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            try:
                state = json.load(urllib.request.urlopen(args.health_url, timeout=5))
                ready = state.get("status") == "ok" if args.controller else state.get("ok") is True
                if ready and not state.get("fatal"):
                    print(
                        json.dumps(
                            {"ready": True, "image": args.image, "rollback_container": backup_name}
                        ),
                        flush=True,
                    )
                    return
                if state.get("fatal"):
                    raise RuntimeError("Candidate runtime reported a fatal engine error")
            except (OSError, ValueError):
                pass
            time.sleep(5)
        candidate = api("GET", f"/containers/{name}/json")
        live = candidate["State"]
        if live.get("Running") and not live.get("OOMKilled"):
            # An observation deadline is not a terminal process state. The caller must
            # continue observing this same container rather than restart its initialization.
            print(
                json.dumps(
                    {
                        "ready": False,
                        "status": "still_starting",
                        "container_id": candidate["Id"],
                        "rollback_container": backup_name,
                    }
                ),
                flush=True,
            )
            return
        raise RuntimeError("Candidate process terminated before readiness")
    except BaseException:
        if created:
            api("POST", f"/containers/{name}/stop?t=10")
            api("DELETE", f"/containers/{name}")
        old_name = urllib.parse.quote(backup_name, safe="")
        api("POST", f"/containers/{old_name}/rename?name={name}")
        api("POST", f"/containers/{name}/start")
        raise


if __name__ == "__main__":
    main()
