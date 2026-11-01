import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dsv41_watchdog", Path(__file__).parents[1] / "deploy/dsv41-watchdog.py"
)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


def test_only_repeated_explicit_fatal_health_triggers_recovery():
    state = {}
    fatal = {"ok": False, "fatal": "CudaFailed"}
    assert not watchdog.decision(True, fatal, 100, state)
    assert not watchdog.decision(
        True, None, 110, state
    )  # Observational failure resets confirmation.
    assert not watchdog.decision(True, fatal, 120, state)
    assert not watchdog.decision(True, fatal, 130, state)
    assert watchdog.decision(True, fatal, 140, state)
    assert not watchdog.decision(False, fatal, 150, state)  # Intentional stop stays stopped.
    assert not watchdog.decision(True, {"ok": True, "inflight": 4, "oldest_s": 2000}, 160, state)


def test_recovery_cooldown_and_hourly_cap():
    state = {"confirmed_fatal": 3, "restarts": [1000, 1400, 1800]}
    fatal = {"ok": False, "fatal": "CudaFailed"}
    assert not watchdog.decision(True, fatal, 2200, state)
    assert watchdog.decision(True, fatal, 4700, state)
    state = {"confirmed_fatal": 3, "restarts": [1000]}
    assert not watchdog.decision(True, fatal, 1200, state)
    assert watchdog.decision(True, fatal, 1300, state)
