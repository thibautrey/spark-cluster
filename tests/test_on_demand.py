import asyncio
import json
import time

import httpx
import pytest

from spark_cluster.on_demand import DemandProxy, InferenceProfile, OnDemandInference, load_profiles


def profile(**values):
    return InferenceProfile(
        id="test",
        model="test-model",
        listen_port=18100,
        upstream_port=28100,
        container={"Image": "test"},
        **values,
    )


class Docker:
    def __init__(self):
        self.containers = {}
        self.calls = []

    async def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        name = path.split("/")[2] if path.startswith("/containers/") else ""
        status, body = 200, {}
        if path == "/containers/create":
            name = kwargs["params"]["name"]
            self.containers[name] = {"Config": kwargs["payload"], "State": {"Running": True}}
            body = {"Id": name}
        elif method == "DELETE":
            self.containers.pop(name, None)
        elif path.endswith("/json"):
            status = 200 if name in self.containers else 404
            body = self.containers.get(name, {})
        return httpx.Response(
            status, json=body, request=httpx.Request(method, "http://docker" + path)
        )


@pytest.mark.asyncio
async def test_switch_removes_previous_runtime_and_disables_restart(monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=transport))
    docker = Docker()
    first = profile()
    second = profile().model_copy(update={"id": "second"})
    manager = OnDemandInference(docker, [first, second])
    await manager.ensure(first)
    await manager.ensure(first)
    assert sum(path == "/containers/create" for _, path, _ in docker.calls) == 1
    await manager.ensure(second)
    assert first.name not in docker.containers
    assert manager.active.id == "second"
    config = docker.containers[second.name]["Config"]
    assert config["HostConfig"]["RestartPolicy"] == {"Name": "no"}
    assert config["Healthcheck"]["Test"] == ["NONE"]
    assert "container" not in json.dumps(manager.status())


@pytest.mark.asyncio
async def test_failed_start_cleans_up(monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda r: httpx.Response(503))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=transport))
    p = profile(startup_seconds=1)
    manager = OnDemandInference(Docker(), [p])
    with pytest.raises(TimeoutError):
        await manager.ensure(p)
    assert not manager.docker.containers
    assert manager.active is None


@pytest.mark.asyncio
async def test_never_removes_unmanaged_container():
    p = profile()
    docker = Docker()
    docker.containers[p.name] = {"Config": {}, "State": {"Running": True}}
    with pytest.raises(RuntimeError, match="unmanaged"):
        await OnDemandInference(docker, [p]).remove(p)
    assert p.name in docker.containers


@pytest.mark.asyncio
async def test_health_and_inventory_do_not_start_models():
    p = profile()
    manager = OnDemandInference(Docker(), [p])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=DemandProxy(manager, p)), base_url="http://test"
    ) as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/v1/models")).json()["data"][0]["id"] == p.model
    assert not manager.docker.calls


@pytest.mark.asyncio
async def test_lease_covers_stream_and_preserves_request(monkeypatch):
    original = httpx.AsyncClient
    p = profile()
    manager = OnDemandInference(Docker(), [p])

    async def ensure(_):
        manager.active = p

    monkeypatch.setattr(manager, "ensure", ensure)

    class Events(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: test\n\ndata: [DONE]\n\n"

    def upstream(request):
        assert manager.lock.locked()
        assert request.headers["authorization"] == "Bearer test"
        assert request.url.path == "/v1/chat/completions"
        assert request.url.query == b"x=1"
        assert request.content == b'{"stream":true}'
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Events())

    transport = httpx.MockTransport(upstream)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=transport))
    messages = []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"x=1",
        "headers": [(b"authorization", b"Bearer test")],
        "scheme": "http",
    }

    async def receive():
        return {"type": "http.request", "body": b'{"stream":true}', "more_body": False}

    async def send(message):
        assert manager.lock.locked()
        messages.append(message)

    await DemandProxy(manager, p)(scope, receive, send)
    assert b"[DONE]" in b"".join(m.get("body", b"") for m in messages)
    assert not manager.lock.locked()
    assert manager.pending == 0


@pytest.mark.asyncio
async def test_idle_reaper_never_evicts_active_request():
    p = profile(idle_seconds=1)
    docker = Docker()
    docker.containers[p.name] = {"Config": {"Labels": {"spark-cluster.on-demand": p.id}}}
    manager = OnDemandInference(docker, [p])
    manager.active = p
    manager.last_used = time.monotonic() - 10
    await manager.lock.acquire()
    task = asyncio.create_task(manager.reap())
    try:
        await asyncio.sleep(1.1)
        assert manager.active is p
        manager.lock.release()
        await asyncio.sleep(1.1)
        assert manager.active is None
        assert not docker.containers
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_private_recipes_and_unique_ports(tmp_path):
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps([profile().model_dump()]))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        load_profiles(path)
    path.chmod(0o600)
    assert len(load_profiles(path)) == 1
    path.write_text(json.dumps([profile().model_dump(), profile().model_dump()]))
    with pytest.raises(ValueError, match="unique"):
        load_profiles(path)
