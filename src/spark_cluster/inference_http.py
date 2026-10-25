"""Shared bounds and HTTP semantics for inference gateways.

Requests are deliberately sent once: an upstream may have accepted a generation
even when the caller has not received response headers yet.
"""

import asyncio
import contextlib
import math

import httpx
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse

MAX_REQUEST_BYTES = 64 * 1024 * 1024
INFERENCE_TIMEOUT = httpx.Timeout(connect=5, read=900, write=30, pool=5)
HOP_HEADERS = {
    b"host",
    b"connection",
    b"keep-alive",
    b"proxy-connection",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


class RequestTooLarge(ValueError):
    pass


class RuntimeUnavailable(RuntimeError):
    def __init__(self, retry_after):
        super().__init__("Runtime recovery is backing off; retry after the indicated delay")
        self.retry_after = max(1, math.ceil(retry_after))


@contextlib.asynccontextmanager
async def admission_disconnect(receive):
    """Cancel admission on disconnect, after consuming the body and before streaming.

    The watcher stops before StreamingResponse starts, so only one consumer reads the
    ASGI receive channel at a time.
    """
    if receive is None:
        yield
        return
    owner = asyncio.current_task()
    disconnected = False

    async def watch():
        nonlocal disconnected
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
                owner.cancel()
                return
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(watch())
    try:
        yield
    except asyncio.CancelledError:
        if disconnected:
            raise ClientDisconnect from None
        raise
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


async def read_inference_body(request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise RequestTooLarge("Request exceeds 64 MiB")
        body.extend(chunk)
    return bytes(body)


def forwarded_headers(raw_headers, *, request=False):
    """Remove hop headers, including names nominated by Connection on either leg."""
    excluded = HOP_HEADERS | ({b"content-length"} if request else set())
    for name, value in raw_headers:
        if name.lower() == b"connection":
            excluded |= {token.strip().lower() for token in value.split(b",")}
    return [(name, value) for name, value in raw_headers if name.lower() not in excluded]


def inference_error(message, *, status=503, retry_after=None):
    headers = {"Retry-After": str(max(1, math.ceil(retry_after)))} if retry_after else None
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "server_error" if status >= 500 else "invalid_request_error",
                "param": None,
                "code": {413: "request_too_large", 504: "upstream_timeout"}.get(
                    status, "service_unavailable"
                ),
            }
        },
        status_code=status,
        headers=headers,
    )
