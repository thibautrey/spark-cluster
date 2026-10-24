"""Persistent LAN cluster attachment and transparent API/WebSocket forwarding."""

import asyncio
import contextlib
import hashlib
import hmac
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from starlette.types import ASGIApp, Receive, Scope, Send
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import WebSocketException

from .config import Settings
from .models import ClusterSnapshot
from .setup_models import private_address

MAX_PROXY_BODY = 2 * 1024 * 1024


class RemoteClusterUnavailable(Exception):
    pass


def controller_signature(token: str, method: str, target: str, timestamp: str, body: bytes) -> str:
    digest = hashlib.sha256(body).hexdigest()
    message = f"{method.upper()}\n{target}\n{timestamp}\n{digest}".encode()
    return hmac.new(token.encode(), message, hashlib.sha256).hexdigest()


def signed_controller_headers(
    token: str, method: str, target: str, body: bytes = b""
) -> dict[str, str]:
    timestamp = str(int(time.time()))
    return {
        "x-spark-timestamp": timestamp,
        "x-spark-signature": controller_signature(token, method, target, timestamp, body),
    }


def valid_controller_signature(
    token: str,
    method: str,
    target: str,
    timestamp: str | None,
    signature: str | None,
    body: bytes,
) -> bool:
    if not timestamp or not signature:
        return False
    try:
        if abs(time.time() - int(timestamp)) > 30:
            return False
    except ValueError:
        return False
    expected = controller_signature(token, method, target, timestamp, body)
    return hmac.compare_digest(expected, signature)


def connection_path(settings: Settings) -> Path:
    return settings.setup_state_path / "active-cluster.json"


def load_active_connection(settings: Settings) -> dict[str, str] | None:
    path = connection_path(settings)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
        url = validated_cluster_url(value["url"])
        return {
            "url": url,
            "cluster_name": str(value["cluster_name"]),
            "head_address": private_address(value["head_address"]),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def current_connection(settings: Settings) -> dict[str, str] | None:
    if saved := load_active_connection(settings):
        return saved
    if not settings.remote_cluster_url:
        return None
    url = validated_cluster_url(settings.remote_cluster_url)
    return {
        "url": url,
        "cluster_name": settings.cluster_name,
        "head_address": urlparse(url).hostname or "",
    }


def save_active_connection(settings: Settings, value: dict[str, str]) -> dict[str, str]:
    connection = {
        "url": validated_cluster_url(value["url"]),
        "cluster_name": str(value["cluster_name"]),
        "head_address": private_address(value["head_address"]),
    }
    path = connection_path(settings)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(connection, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)
    settings.remote_cluster_url = connection["url"]
    return connection


def validated_cluster_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ValueError("Cluster URL must be an HTTP(S) LAN endpoint")
    private_address(parsed.hostname)
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Cluster URL must not contain a path, query or fragment")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not 1 <= port <= 65535:
        raise ValueError("Invalid cluster port")
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


class RemoteClusterConnector:
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, settings: Settings, registry=None):
        self.settings = settings
        self.registry = registry

    async def attach(self, address: str, port: int = 30080) -> dict[str, str]:
        address = private_address(address)
        if not 1024 <= port <= 65535:
            raise ValueError("Use an unprivileged LAN port between 1024 and 65535")
        url = validated_cluster_url(f"http://{address}:{port}")
        token = self._authorization()
        headers = signed_controller_headers(token, "GET", "/api/v1/controller/verify")
        try:
            async with httpx.AsyncClient(
                base_url=url, transport=self.transport, timeout=10, trust_env=False
            ) as client:
                response = await client.get("/api/v1/controller/verify", headers=headers)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise RemoteClusterUnavailable(
                "The cluster bridge could not be authenticated on this LAN address"
            ) from error
        if payload.get("status") != "ready" or not payload.get("cluster_name"):
            raise RemoteClusterUnavailable("The endpoint is not a compatible cluster bridge")
        if self.registry:
            cluster = self.registry.add_remote(
                name=payload["cluster_name"],
                endpoint=url,
                head_address=address,
                hardware="dgx-spark",
                activate=True,
            )
            return {
                **cluster.model_dump(mode="json"),
                "url": cluster.endpoint,
                "cluster_name": cluster.name,
                "head_address": cluster.host_address,
            }
        return save_active_connection(
            self.settings,
            {"url": url, "cluster_name": payload["cluster_name"], "head_address": address},
        )

    async def snapshot(self) -> ClusterSnapshot:
        if not self.settings.remote_cluster_url:
            raise RemoteClusterUnavailable("No cluster bridge is attached")
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.remote_cluster_url,
                transport=self.transport,
                timeout=10,
                trust_env=False,
            ) as client:
                response = await client.get("/api/v1/snapshot")
                response.raise_for_status()
                return ClusterSnapshot.model_validate(response.json())
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise RemoteClusterUnavailable(
                "Cluster capacity is unavailable; check the bridge before downloading a model"
            ) from error

    async def model_storage_available(self) -> int:
        if not self.settings.remote_cluster_url:
            raise RemoteClusterUnavailable("No cluster bridge is attached")
        path = "/api/v1/models/storage"
        headers = signed_controller_headers(self._authorization(), "GET", path)
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.remote_cluster_url,
                transport=self.transport,
                timeout=10,
                trust_env=False,
            ) as client:
                response = await client.get(path, headers=headers)
                response.raise_for_status()
                return int(response.json()["available_bytes"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            raise RemoteClusterUnavailable(
                "Cluster model storage is unavailable; check the bridge and shared volume"
            ) from error

    def _authorization(self) -> str:
        if not self.settings.model_admin_token:
            raise RemoteClusterUnavailable("Configure MODEL_ADMIN_TOKEN before attaching a cluster")
        return self.settings.model_admin_token


class RemoteClusterProxyMiddleware:
    """Keep the UI local while forwarding cluster state and control to its LAN bridge."""

    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, app: ASGIApp, settings: Settings, registry=None):
        self.app = app
        self.settings = settings
        self.registry = registry

    def should_forward(self, scope: Scope) -> bool:
        path = scope.get("path", "")
        controller_local = (
            path.startswith("/api/v1/controller/")
            or path.startswith("/api/v1/clusters")
            or path == "/api/v1/pool"
            or path.startswith("/api/v1/pool/")
            or path.startswith("/api/v1/catalog/")
            or path.startswith("/api/v1/model-management/")
            or path.startswith("/api/v1/runtime-management/")
            or path.startswith("/api/v1/performance/")
            or path == "/api/v1/models/fit"
            or path == "/api/v1/models/download"
            or path == "/api/v1/models/import"
            or path.startswith("/api/v1/downloads")
        )
        return bool(
            self.settings.remote_cluster_url
            and not controller_local
            and (
                (path.startswith("/api/v1/") and not path.startswith("/api/v1/setup"))
                or path == "/metrics"
            )
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.should_forward(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http":
            await self.forward_http(scope, receive, send)
        elif scope["type"] == "websocket":
            await self.forward_websocket(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    def headers(self, scope: Scope) -> dict[str, str]:
        headers = {}
        blocked = {b"host", b"content-length", b"connection", b"upgrade", b"authorization"}
        for key, value in scope.get("headers", []):
            lowered = key.lower()
            if lowered in blocked or lowered.startswith(b"sec-websocket-"):
                continue
            headers[key.decode("latin-1")] = value.decode("latin-1")
        return headers

    async def forward_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_PROXY_BODY:
                await self.error_response(send, 413, "Request body exceeds the controller limit")
                return
            if not message.get("more_body", False):
                break
        query = scope.get("query_string", b"").decode("ascii")
        path = scope["path"] + (f"?{query}" if query else "")
        headers = self.headers(scope)
        if self.settings.model_admin_token:
            headers.update(
                signed_controller_headers(
                    self.settings.model_admin_token, scope["method"], path, bytes(body)
                )
            )
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.remote_cluster_url,
                transport=self.transport,
                timeout=60,
                trust_env=False,
            ) as client:
                response = await client.request(
                    scope["method"], path, content=bytes(body), headers=headers
                )
        except httpx.HTTPError:
            await self.error_response(
                send, 502, "Cluster bridge is unreachable; check its head node and LAN route"
            )
            return
        response_headers = []
        for name in ("content-type", "cache-control", "etag"):
            if value := response.headers.get(name):
                response_headers.append((name.encode(), value.encode()))
        await send(
            {
                "type": "http.response.start",
                "status": response.status_code,
                "headers": response_headers,
            }
        )
        await send({"type": "http.response.body", "body": response.content})

    async def error_response(self, send: Send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def forward_websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        base = self.settings.remote_cluster_url
        scheme = "wss" if base.startswith("https://") else "ws"
        query = scope.get("query_string", b"").decode("ascii")
        url = f"{scheme}://{urlparse(base).netloc}{scope['path']}"
        if query:
            url += f"?{query}"
        try:
            async with websocket_connect(
                url,
                additional_headers=self.headers(scope),
                open_timeout=10,
                ping_interval=20,
                ping_timeout=20,
                max_size=4 * 1024 * 1024,
            ) as remote:
                await send({"type": "websocket.accept"})

                async def upstream():
                    while True:
                        message = await receive()
                        if message["type"] == "websocket.disconnect":
                            await remote.close()
                            return
                        if text := message.get("text"):
                            await remote.send(text)
                        elif data := message.get("bytes"):
                            await remote.send(data)

                async def downstream():
                    async for message in remote:
                        key = "text" if isinstance(message, str) else "bytes"
                        await send({"type": "websocket.send", key: message})

                tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    with contextlib.suppress(Exception):
                        task.result()
        except (OSError, TimeoutError, WebSocketException):
            await send(
                {"type": "websocket.close", "code": 1013, "reason": "Cluster bridge is unreachable"}
            )
