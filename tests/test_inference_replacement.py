import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "inference_replacement", Path(__file__).parents[1] / "deploy/replace-inference-container.py"
)
replacement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replacement)


def setup_recovery(monkeypatch, tmp_path, *, running=False, fatal=False):
    commands = []
    old = {
        "State": {"Running": running},
        "Config": {"Image": "previous", "Cmd": [], "Env": []},
        "HostConfig": {},
    }

    def api(method, path, value=None):
        commands.append((method, path))
        return old if method == "GET" and path == "/containers/engine/json" else {}

    monkeypatch.setattr(replacement, "api", api)
    monkeypatch.setattr(
        replacement.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps({"ok": not fatal, "fatal": fatal}).encode()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replace",
            "--name",
            "engine",
            "--image",
            "candidate",
            "--health-url",
            "http://health",
            "--backup-dir",
            str(tmp_path),
            "--replace-stopped",
        ],
    )
    return commands


def test_recovery_never_stops_a_running_engine(monkeypatch, tmp_path):
    commands = setup_recovery(monkeypatch, tmp_path, running=True)
    with pytest.raises(SystemExit, match="running runtime was preserved"):
        replacement.main()
    assert all(method == "GET" for method, _ in commands)


def test_stopped_candidate_can_be_recovered_and_backup_is_private(monkeypatch, tmp_path):
    commands = setup_recovery(monkeypatch, tmp_path)
    replacement.main()
    assert ("POST", "/containers/engine/start") in commands
    backups = list(tmp_path.glob("*.json"))
    assert len(backups) == 1
    assert backups[0].stat().st_mode & 0o777 == 0o600


def test_failed_recovery_restores_the_original_stopped_state(monkeypatch, tmp_path):
    commands = setup_recovery(monkeypatch, tmp_path, fatal=True)
    with pytest.raises(RuntimeError, match="fatal engine error"):
        replacement.main()
    assert commands.count(("POST", "/containers/engine/start")) == 1
    assert ("DELETE", "/containers/engine") in commands
    assert commands[-1][1].endswith("/rename?name=engine")
