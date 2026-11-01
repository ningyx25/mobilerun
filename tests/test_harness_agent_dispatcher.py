"""Agent dispatcher: selection, composite-id routing and health aggregation."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import mobilerun.harness.agent_service.episode as episode_module
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.harness.agent_service.app import create_agent_app
from mobilerun.harness.agent_service.dispatcher import create_agent_dispatcher
from mobilerun.harness.agent_service.registry import parse_entry
from mobilerun.harness.client import AgentClient
from tests.test_harness_agent_service import _fake_agent_class, _FakeDriver

# ---------------------------------------------------------------------------
# Wiring: N worker apps reachable over ASGI, one dispatcher in front of them
# ---------------------------------------------------------------------------


def _raise_connect(request: httpx.Request):
    raise httpx.ConnectError("unreachable", request=request)


class _Fleet:
    def __init__(self, monkeypatch, specs, *, worker_names=None):
        """specs: [(agent_id, agent_name, url)]; a ``*:1`` url is unreachable."""
        monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class())
        monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)
        self.apps = {}
        self.transports = {}
        self.dead_urls = set()
        for _, name, url in specs:
            if url.endswith(":1"):  # deliberately unreachable
                self.dead_urls.add(url)
                continue
            worker_name = (worker_names or {}).get(url, name)
            app = create_agent_app(MobileConfig(), llms={}, name=worker_name)
            self.apps[url] = app
            self.transports[url] = httpx.ASGITransport(app=app)

        def factory(url):
            if url in self.transports:
                return AgentClient(url, transport=self.transports[url])
            return AgentClient(url, transport=httpx.MockTransport(_raise_connect))

        self.dispatcher = create_agent_dispatcher(
            [f"{agent_id}={name}@{url}" for agent_id, name, url in specs],
            client_factory=factory,
            probe_interval=0.0,
        )

    def episodes(self, url):
        manager = self.apps[url].state.manager
        return manager._episodes  # noqa: SLF001 - test introspection


def _request(**overrides):
    body = {
        "goal": "do the thing",
        "environment": {"base_url": "http://env.test", "session_id": "s_1"},
        "limits": {"max_steps": 4, "timeout_s": 30.0},
        "evidence": {"enabled": False},
    }
    body.update(overrides)
    return body


def _wait(client, episode_id):
    import time

    for _ in range(200):
        info = client.get(f"/v1/episodes/{episode_id}").json()
        if info["state"] != "running":
            return info
        time.sleep(0.01)
    raise AssertionError("episode did not finish")


# ---------------------------------------------------------------------------
# Entry parsing
# ---------------------------------------------------------------------------


def test_parse_entry_accepts_the_documented_form():
    assert parse_entry("fast0=fast-agent@http://localhost:9201") == (
        "fast0",
        "fast-agent",
        "http://localhost:9201",
    )
    assert parse_entry("jev0=jev@https://host:9202/") == (
        "jev0",
        "jev",
        "https://host:9202",
    )


@pytest.mark.parametrize(
    "entry",
    [
        "fast0=http://localhost:9201",  # no agent_name
        "fast0=fast-agent",  # no url
        "=fast-agent@http://h:1",  # empty id
        "fast:0=fast-agent@http://h:1",  # ':' is the composite separator
        "fast0=@http://h:1",  # empty name
    ],
)
def test_parse_entry_rejects_malformed_entries(entry):
    with pytest.raises(ValueError):
        parse_entry(entry)


def test_duplicate_agent_id_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        create_agent_dispatcher(
            [
                "a0=x@http://h:9201",
                "a0=y@http://h:9202",
            ]
        )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_aggregates_slots_and_marks_dead_ones(monkeypatch):
    fleet = _Fleet(
        monkeypatch,
        [
            ("fast0", "fast-agent", "http://fast:9201"),
            ("dead0", "jev", "http://dead:1"),
        ],
    )
    with TestClient(fleet.dispatcher) as client:
        health = client.get("/v1/health").json()

    assert health["ok"] is True
    rows = {row["id"]: row for row in health["agents"]}
    assert rows["fast0"]["state"] == "ready"
    assert rows["fast0"]["implementation"] == "mobilerun"
    assert rows["dead0"]["state"] == "down"


def test_registered_label_mismatch_is_reported_but_not_fatal(monkeypatch):
    # Worker calls itself "mobilerun" (config.agent.name) but is registered as
    # "fast-agent": that is a naming choice, not an error.
    fleet = _Fleet(
        monkeypatch,
        [("fast0", "fast-agent", "http://fast:9201")],
        worker_names={"http://fast:9201": "mobilerun"},
    )
    with TestClient(fleet.dispatcher) as client:
        rows = client.get("/v1/health").json()["agents"]
        created = client.post("/v1/episodes", json=_request())

    assert rows[0]["name_mismatch"] is True
    assert created.status_code == 202


def test_registered_label_matches_the_worker_name(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "mobilerun", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        rows = client.get("/v1/health").json()["agents"]
    assert rows[0]["name_mismatch"] is False


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_explicit_agent_id_routes_to_that_worker(monkeypatch):
    fleet = _Fleet(
        monkeypatch,
        [
            ("fast0", "fast-agent", "http://fast:9201"),
            ("jev0", "jev", "http://jev:9202"),
        ],
    )
    with TestClient(fleet.dispatcher) as client:
        created = client.post("/v1/episodes", json=_request(agent_id="jev0")).json()
        assert created["episode_id"].startswith("jev0:")
        assert created["agent_id"] == "jev0"
        _wait(client, created["episode_id"])

    assert fleet.episodes("http://jev:9202")
    assert not fleet.episodes("http://fast:9201")


def test_unknown_agent_is_404(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "fast-agent", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        response = client.post("/v1/episodes", json=_request(agent_id="nope"))
    assert response.status_code == 404
    assert response.json()["code"] == "unknown_agent"


def test_omitted_agent_id_picks_a_ready_worker(monkeypatch):
    fleet = _Fleet(
        monkeypatch,
        [
            ("fast0", "fast-agent", "http://fast:9201"),
            ("dead0", "jev", "http://dead:1"),
        ],
    )
    with TestClient(fleet.dispatcher) as client:
        created = client.post("/v1/episodes", json=_request()).json()
    # The dead slot is skipped; ties among ready slots break by id.
    assert created["agent_id"] == "fast0"


def test_no_ready_agent_is_503_with_retry_after(monkeypatch):
    fleet = _Fleet(monkeypatch, [("dead0", "jev", "http://dead:1")])
    with TestClient(fleet.dispatcher) as client:
        response = client.post("/v1/episodes", json=_request())
    assert response.status_code == 503
    assert response.headers["retry-after"]
    assert response.json()["code"] == "capacity"


def test_explicit_down_agent_is_503(monkeypatch):
    fleet = _Fleet(monkeypatch, [("dead0", "jev", "http://dead:1")])
    with TestClient(fleet.dispatcher) as client:
        response = client.post("/v1/episodes", json=_request(agent_id="dead0"))
    assert response.status_code == 503
    assert response.json()["code"] == "backend_unavailable"


# ---------------------------------------------------------------------------
# Composite episode ids
# ---------------------------------------------------------------------------


def test_composite_id_routes_poll_and_result(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "fast-agent", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        created = client.post("/v1/episodes", json=_request()).json()
        info = _wait(client, created["episode_id"])

    assert info["episode_id"] == created["episode_id"]
    assert info["agent_id"] == "fast0"
    assert info["state"] == "succeeded"
    assert info["result"]["declared_done"] is True


def test_unknown_episode_prefix_is_404(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "fast-agent", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        response = client.get("/v1/episodes/nope:ep_1")
    assert response.status_code == 404
    assert response.json()["code"] == "unknown_agent"


def test_bare_episode_id_is_400(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "fast-agent", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        response = client.get("/v1/episodes/ep_1")
    assert response.status_code == 400
    assert response.json()["code"] == "bad_request"


def test_upstream_error_is_passed_through(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "fast-agent", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        response = client.get("/v1/episodes/fast0:missing")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "unknown_episode"
    assert body["detail"]["agent_id"] == "fast0"


# ---------------------------------------------------------------------------
# Events and cancellation
# ---------------------------------------------------------------------------


def test_event_stream_is_proxied(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "fast-agent", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        created = client.post("/v1/episodes", json=_request()).json()
        _wait(client, created["episode_id"])
        with client.stream(
            "GET", f"/v1/episodes/{created['episode_id']}/events"
        ) as response:
            lines = [line for line in response.iter_lines() if line.startswith("data:")]

    assert lines
    types = {json.loads(line[5:].strip())["type"] for line in lines}
    assert "ToolExecutionEvent" in types


def test_cancel_routes_to_the_right_worker(monkeypatch):
    monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class(delay=5.0))
    monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)

    app = create_agent_app(MobileConfig(), llms={}, name="fast-agent")
    transport = httpx.ASGITransport(app=app)

    def factory(url):
        return AgentClient(url, transport=transport)

    dispatcher = create_agent_dispatcher(
        ["fast0=fast-agent@http://fast:9201"],
        client_factory=factory,
        probe_interval=0.0,
    )
    with TestClient(dispatcher) as client:
        created = client.post("/v1/episodes", json=_request()).json()
        cancelled = client.post(f"/v1/episodes/{created['episode_id']}/cancel")
        info = client.get(f"/v1/episodes/{created['episode_id']}").json()

    assert cancelled.status_code == 200
    assert cancelled.json()["agent_id"] == "fast0"
    assert info["state"] == "cancelled"


def test_policy_routes_are_still_not_implemented(monkeypatch):
    fleet = _Fleet(monkeypatch, [("fast0", "fast-agent", "http://fast:9201")])
    with TestClient(fleet.dispatcher) as client:
        response = client.post("/v1/policies/fast0/act", json={})
        missing = client.post("/v1/policies/nope/act", json={})
    assert response.status_code == 501
    assert response.json()["code"] == "not_implemented"
    assert missing.status_code == 404
