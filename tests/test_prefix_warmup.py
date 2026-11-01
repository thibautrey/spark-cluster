import json
import stat

import httpx
from test_external_runtime import MODELS, remote_registry

from spark_cluster.external_runtime import ExternalRuntimeRequest, ExternalRuntimeService
from spark_cluster.prefix_warmup import MAX_PROFILES, MAX_TEXT, PrefixWarmup


def payload(app="codex", question="Question one"):
    return {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "system", "content": app + " private instructions " * 100},
            {"role": "user", "content": question},
        ],
        "tools": [{"type": "function", "function": {"name": app + "_read"}}],
        "stream": True,
    }


def test_profiles_survive_restart_are_private_and_do_not_retain_short_questions(tmp_path):
    service = PrefixWarmup(tmp_path / "profiles", None)
    service.remember(payload(question="PRIVATE_CUSTOMER_QUESTION"))
    service.remember(payload(question="A different question"))
    restored = PrefixWarmup(tmp_path / "profiles", None)
    profiles = restored.status()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["seen"] == 2
    assert "private instructions" not in json.dumps(restored.status())
    path = next((tmp_path / "profiles").glob("*.json"))
    assert "PRIVATE_CUSTOMER_QUESTION" not in path.read_text()
    assert all(m["role"] != "user" for m in json.loads(path.read_text())["body"]["messages"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_profile_storage_is_bounded_and_does_not_fetch_images(tmp_path):
    service = PrefixWarmup(tmp_path / "profiles", None)
    for i in range(MAX_PROFILES + 5):
        service.remember(payload(app=str(i)))
    assert len(service.status()["profiles"]) == MAX_PROFILES
    request = payload("long")
    request["messages"][0]["content"] = "x" * (MAX_TEXT * 2)
    service.remember(request)
    profile = next(
        p for p in service._read() if p["body"]["messages"][0]["content"].startswith("x")
    )
    assert len(profile["body"]["messages"][0]["content"]) == MAX_TEXT
    request = payload("images")
    request["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/private.png"}},
            ],
        }
    ]
    before = service.status()["profiles"]
    service.remember(request)
    assert service.status()["profiles"] == before


async def test_large_codex_tool_catalog_is_persisted_and_shutdown_drains_writes(tmp_path):
    service = PrefixWarmup(tmp_path / "profiles", None)
    request = payload()
    request["tools"][0]["function"]["description"] = "Tool documentation " * 25000
    service.observe(request)
    await service.stop()
    profiles = service.status()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["bytes"] > 262144
    assert not service.pending


async def test_warmup_waits_for_idle_runs_once_and_rearms_after_runtime_restart(tmp_path):
    config, registry, cluster = remote_registry(tmp_path)
    external = ExternalRuntimeService(config, registry)
    calls = []
    health = {"ok": True, "uptime_s": 100, "inflight": 1, "queued": 0, "idle_s": 10}

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=health)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=MODELS)
        calls.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer secret-key"
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "length"}],
                "usage": {"prompt_tokens": 2000, "prompt_tokens_details": {"cached_tokens": 1984}},
            },
        )

    external.transport = httpx.MockTransport(handler)
    await external.configure(cluster.id, ExternalRuntimeRequest(api_key="secret-key"))
    service = PrefixWarmup(tmp_path / "profiles", external)
    service.remember(payload())
    await service.sample()
    assert not calls
    health.update(inflight=0)
    await service.sample()
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 1
    assert calls[0]["stream"] is False
    assert "chat_template_kwargs" not in calls[0]  # Preserve the caller's native default.
    await service.sample()
    assert len(calls) == 1
    health.update(uptime_s=1)
    await service.sample()
    assert len(calls) == 2
    assert (
        service.status()["last_results"][0]["usage"]["prompt_tokens_details"]["cached_tokens"]
        == 1984
    )
    health.update(ok=False, fatal="CudaFailed", uptime_s=0)
    await service.sample()
    assert len(calls) == 2


def test_reasoning_modes_keep_distinct_exact_prefix_profiles(tmp_path):
    service = PrefixWarmup(tmp_path / "profiles", None)
    high = payload()
    high["reasoning_effort"] = "high"
    high["chat_template_kwargs"] = {"enable_thinking": True}
    low = payload()
    low["reasoning_effort"] = "low"
    low["chat_template_kwargs"] = {"enable_thinking": True}
    service.remember(high)
    service.remember(low)
    profiles = service._read()
    assert len(profiles) == 2
    assert {p["body"]["reasoning_effort"] for p in profiles} == {"low", "high"}
    assert all(p["body"]["chat_template_kwargs"]["enable_thinking"] for p in profiles)
