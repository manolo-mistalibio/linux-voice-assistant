"""Tests for the authenticated external-duck control API (#356)."""

import json
import urllib.error
import urllib.request

import pytest

from linux_voice_assistant.duck_control import DuckControlServer
from linux_voice_assistant.gain_controller import GainController


class FakePlayer:
    def __init__(self):
        self.volumes = []

    def set_volume(self, volume):
        self.volumes.append(int(volume))

    @property
    def last(self):
        return self.volumes[-1]


TOKEN = "s3cret-token"


def _http(method, url, body=None, token=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as err:
        raw = err.read().decode()
        try:
            return err.code, json.loads(raw or "{}")
        except json.JSONDecodeError:
            return err.code, {"raw": raw}


@pytest.fixture
def server():
    music, speech = FakePlayer(), FakePlayer()
    controller = GainController(music, speech, autostart=False)
    controller.set_media_volume(0.5)
    srv = DuckControlServer(controller, TOKEN, host="127.0.0.1", port=0)
    host, port = srv.start()
    srv.music, srv.speech, srv.controller = music, speech, controller
    srv.base = f"http://{host}:{port}"
    yield srv
    srv.stop()


def test_requires_a_token_to_construct():
    with pytest.raises(ValueError):
        DuckControlServer(GainController(FakePlayer(), FakePlayer()), "")


def test_health_is_open(server):
    status, body = _http("GET", f"{server.base}/health")
    assert status == 200
    assert body["status"] == "ok"


def test_duck_without_token_is_unauthorized(server):
    status, _ = _http("POST", f"{server.base}/duck", {"source_id": "doorbell", "duration_seconds": 20})
    assert status == 401


def test_duck_with_wrong_token_is_unauthorized(server):
    status, _ = _http("POST", f"{server.base}/duck", {"source_id": "doorbell", "duration_seconds": 20}, token="nope")
    assert status == 401


def test_duck_and_unduck_roundtrip(server):
    status, body = _http("POST", f"{server.base}/duck", {"source_id": "doorbell", "duration_seconds": 20}, token=TOKEN)
    assert status == 200
    assert body["status"] == "ducked"
    assert server.music.last == 25

    status, body = _http("GET", f"{server.base}/leases", token=TOKEN)
    assert status == 200
    assert body["leases"][0]["source_id"] == "doorbell"

    status, body = _http("POST", f"{server.base}/unduck", {"source_id": "doorbell"}, token=TOKEN)
    assert status == 200
    assert body["released"] is True
    assert server.music.last == 50


@pytest.mark.parametrize(
    "payload",
    [
        {"source_id": "doorbell"},  # duration missing
        {"source_id": "doorbell", "duration_seconds": 9999},  # too long
        {"source_id": "bad id", "duration_seconds": 5},  # invalid source_id
    ],
)
def test_invalid_duck_requests_are_rejected(server, payload):
    status, _ = _http("POST", f"{server.base}/duck", payload, token=TOKEN)
    assert status == 400


def test_unknown_path_is_404(server):
    status, _ = _http("POST", f"{server.base}/nope", {}, token=TOKEN)
    assert status == 404


def test_x_lva_token_header_is_accepted(server):
    req = urllib.request.Request(
        f"{server.base}/duck",
        data=json.dumps({"source_id": "doorbell", "duration_seconds": 5}).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("X-LVA-Token", TOKEN)
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
