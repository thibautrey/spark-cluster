import argparse
from unittest.mock import AsyncMock

import httpx
import pytest

from spark_cluster import cli


@pytest.mark.asyncio
async def test_long_running_agent_recovers_bridge_outage(monkeypatch):
    args = argparse.Namespace(server="http://bridge", node_id="node", role="head", once=False, interval=2)
    original = httpx.AsyncClient
    calls = 0

    def handle(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("Bridge restarting")
        args.once = True
        return httpx.Response(200)

    monkeypatch.setattr(cli.httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    monkeypatch.setattr(cli.asyncio, "sleep", AsyncMock())
    await cli.run_agent(args)
    assert calls == 2
    cli.asyncio.sleep.assert_awaited_once_with(2)
