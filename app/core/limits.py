"""Reject oversized request bodies *before* they are read into memory or temp files.

FastAPI parses multipart uploads before the endpoint runs, so a size check inside
the endpoint alone would still accept (and spool) a multi-GB body first.
"""

import json
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLargeError(Exception):
    pass


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int, path_prefixes: tuple[str, ...]) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_prefixes = path_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope["path"]).startswith(self.path_prefixes):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        exceeded = False
        started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:  # chunked uploads without Content-Length
                    exceeded = True
                    raise _BodyTooLargeError
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            # Frameworks may catch our error and answer e.g. 400 "error parsing the body";
            # once the limit is exceeded, the client gets a 413 instead.
            if exceeded:
                if message["type"] == "http.response.start" and not started:
                    started = True
                    await self._reject(send)
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLargeError:
            if not started:
                started = True
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        mb = self.max_bytes // (1024 * 1024)
        body = json.dumps({"detail": f"Request body is larger than the ~{mb} MB limit."}).encode()
        headers: list[Any] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"connection", b"close"),
        ]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})
