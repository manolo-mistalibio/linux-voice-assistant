"""Authenticated local HTTP control for externally triggered music ducking.

Issue #356 needs Home Assistant to duck the smartdash music player when the
doorbell rings. HA runs off-device, so a loopback-only socket is not reachable;
the requirement is therefore a *separate*, token-authenticated control endpoint
— never the unauthenticated peripheral WebSocket exposed on the LAN.

Security model
--------------
* The server only starts when a token is configured (``LVA_DUCK_TOKEN``); a
  missing token disables the feature rather than exposing it anonymously.
* Every mutating request must present the token as ``Authorization: Bearer
  <token>`` or ``X-LVA-Token: <token>``; comparison is constant-time.
* Requests are tiny JSON bodies with a hard size cap; ``source_id`` and
  durations are validated before they reach the controller.
* ``/health`` is intentionally unauthenticated but reveals only liveness.

Endpoints
---------
``POST /duck``    ``{"source_id": "doorbell", "duration_seconds": 20}``
                  optional ``"factor"`` (0–1) overrides the duck ratio.
``POST /unduck``  ``{"source_id": "doorbell"}``
``GET  /leases``  authenticated lease snapshot (debug/observability).
``GET  /health``  ``{"status": "ok"}``

The implementation is stdlib-only (``http.server``) so the overlay adds no
dependency to the pinned image and the logic is unit-testable.
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

from .gain_controller import GainController, LeaseError, SATELLITE_REASON

_LOGGER = logging.getLogger(__name__)

_MAX_BODY_BYTES = 4096


class _Handler(BaseHTTPRequestHandler):
    server_version = "LVADuckControl/1.0"

    # Silence the default stderr access log; LVA owns logging.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        _LOGGER.debug("duck-control %s - %s", self.address_string(), fmt % args)

    # -- helpers ------------------------------------------------------

    @property
    def _controller(self) -> GainController:
        return self.server.controller  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        token = getattr(self.server, "token", "")  # type: ignore[attr-defined]
        if not token:
            return False
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            supplied = auth[len("Bearer ") :]
        else:
            supplied = self.headers.get("X-LVA-Token", "")
        return bool(supplied) and hmac.compare_digest(supplied, token)

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length < 0 or length > _MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    # -- routes -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            self._send_json(200, {"status": "ok", "service": "lva-duck-control"})
            return
        if path == "/leases":
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            self._send_json(200, {"leases": self._controller.lease_snapshot()})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        path = urlsplit(self.path).path
        data = self._read_json()
        if data is None:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        if path == "/duck":
            self._handle_duck(data)
        elif path == "/unduck":
            self._handle_unduck(data)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_duck(self, data: Dict[str, Any]) -> None:
        source_id = data.get("source_id")
        duration = data.get("duration_seconds")
        factor = data.get("factor")
        if not isinstance(source_id, str):
            self._send_json(400, {"error": "source_id is required"})
            return
        if source_id == SATELLITE_REASON:
            self._send_json(400, {"error": "source_id is reserved"})
            return
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            self._send_json(400, {"error": "duration_seconds is required and must be a number"})
            return
        if factor is not None and (not isinstance(factor, (int, float)) or isinstance(factor, bool)):
            self._send_json(400, {"error": "factor must be a number between 0 and 1"})
            return
        try:
            self._controller.duck(source_id, float(duration), factor=factor)
        except LeaseError as err:
            self._send_json(400, {"error": str(err)})
            return
        _LOGGER.info("External duck lease: source=%s duration=%.1fs", source_id, float(duration))
        self._send_json(
            200,
            {
                "status": "ducked",
                "source_id": source_id,
                "duration_seconds": float(duration),
                "music_gain": round(self._controller.music_gain(), 4),
            },
        )

    def _handle_unduck(self, data: Dict[str, Any]) -> None:
        source_id = data.get("source_id")
        if not isinstance(source_id, str):
            self._send_json(400, {"error": "source_id is required"})
            return
        if source_id == SATELLITE_REASON:
            self._send_json(400, {"error": "source_id is reserved"})
            return
        try:
            released = self._controller.unduck(source_id)
        except LeaseError as err:
            self._send_json(400, {"error": str(err)})
            return
        _LOGGER.info("External unduck: source=%s released=%s", source_id, released)
        self._send_json(
            200,
            {
                "status": "unducked",
                "source_id": source_id,
                "released": released,
                "music_gain": round(self._controller.music_gain(), 4),
            },
        )


class DuckControlServer:
    """Thin wrapper around a token-authenticated ``ThreadingHTTPServer``."""

    def __init__(
        self,
        controller: GainController,
        token: str,
        *,
        host: str = "127.0.0.1",
        port: int = 6060,
    ) -> None:
        if not token:
            raise ValueError("a non-empty token is required")
        self._controller = controller
        self._token = token
        self._host = host
        self._port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        return (self._host, self._port)

    def start(self) -> Tuple[str, int]:
        httpd = ThreadingHTTPServer((self._host, self._port), _Handler)
        httpd.daemon_threads = True
        httpd.controller = self._controller  # type: ignore[attr-defined]
        httpd.token = self._token  # type: ignore[attr-defined]
        self._server = httpd
        # Resolve the actual port (port 0 -> ephemeral, used by tests).
        self._port = httpd.server_address[1]
        self._thread = threading.Thread(target=httpd.serve_forever, name="lva-duck-control", daemon=True)
        self._thread.start()
        _LOGGER.info("Duck control listening on http://%s:%d (authenticated)", self._host, self._port)
        return (self._host, self._port)

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._server = None
        self._thread = None
