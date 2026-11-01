"""Agent service surface: episodes, declared-done gating, SSE, cancellation."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import mobilerun.harness.agent_service.episode as episode_module
from mobilerun.agent.common.events import ToolExecutionEvent
from mobilerun.agent.droid.events import ManagerPlanEvent
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.harness.agent_service.app import create_agent_app


class _Handler:
    """Stands in for a llama-index WorkflowHandler."""

    def __init__(self, agen, result):
        self._agen = agen
        self._result = result

    async def stream_events(self):
        async for event in self._agen:
            yield event

    def __await__(self):
        async def _done():
            return self._result

        return _done().__await__()


class _FakeDriver:
    instances = []

    def __init__(self, url, session_id, token=None):
        self.url = url
        self.session_id = session_id
        self.calls = []
        self.closed = False
        _FakeDriver.instances.append(self)

    async def connect(self):
        self.calls.append("connect")

    async def close(self):
        self.closed = True

    async def get_ui_tree(self):
        self.calls.append("tree")
        return {
            "a11y_tree": {"children": [{"text": "Continue"}]},
            "phone_state": {"currentApp": "com.example/.Main"},
            "device_context": {"screen_bounds": {"width": 108, "height": 240}},
        }

    async def screenshot(self):
        return b"\x89PNG\r\n"

    async def tap(self, x, y):
        self.calls.append(("tap", x, y))


def _fake_agent_class(*, success=True, answer="42", delay=0.0, manager_answer=True):
    class _FakeAgent:
        def __init__(
            self, goal, config, llms, driver, timeout, runtype="benchmark", **kw
        ):
            self.goal = goal
            self.config = config
            self.driver = driver
            self.shared_state = SimpleNamespace(
                success=None, answer="", message_history=[]
            )
            self.trajectory = None

        def run(self):
            agent = self

            async def events():
                if delay:
                    await asyncio.sleep(delay)
                await agent.driver.get_ui_tree()
                await agent.driver.tap(5, 6)
                agent.shared_state.success = success
                agent.shared_state.answer = answer
                yield ToolExecutionEvent(
                    tool_name="tap",
                    tool_args={"x": 5, "y": 6},
                    success=True,
                    summary="Tapped at (5, 6)",
                )
                yield ManagerPlanEvent(
                    plan="plan",
                    current_subgoal="subgoal",
                    thought="thought",
                    answer="final" if manager_answer else "",
                )

            return _Handler(
                events(), SimpleNamespace(success=success, reason="done", steps=2)
            )

    return _FakeAgent


def _app(config: MobileConfig | None = None, **kwargs):
    return create_agent_app(config or MobileConfig(), llms={}, **kwargs)


def _request(**overrides):
    body = {
        "goal": "do the thing",
        "environment": {"base_url": "http://env.test", "session_id": "s_1"},
        "limits": {"max_steps": 4, "timeout_s": 30.0},
        "evidence": {"enabled": False},
    }
    body.update(overrides)
    return body


def _wait(client: TestClient, episode_id: str) -> dict:
    for _ in range(200):
        info = client.get(f"/v1/episodes/{episode_id}").json()
        if info["state"] != "running":
            return info
        import time

        time.sleep(0.01)
    raise AssertionError("episode did not finish")


def test_health_reports_the_configured_agent():
    with TestClient(_app()) as client:
        health = client.get("/v1/health").json()
        assert health["ok"] is True
        assert health["protocol_version"] == "1.0"
        assert health["agent"] == "mobilerun"


def test_episode_runs_against_the_environment_session(monkeypatch):
    monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class())
    monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)

    with TestClient(_app()) as client:
        started = client.post("/v1/episodes", json=_request())
        assert started.status_code == 202
        episode_id = started.json()["episode_id"]

        info = _wait(client, episode_id)
        assert info["state"] == "succeeded"
        result = info["result"]
        assert result["declared_done"] is True
        assert result["answer"] == "42"
        assert result["steps"] == 2

    driver = _FakeDriver.instances[-1]
    assert ("tap", 5, 6) in driver.calls
    assert driver.closed is True
    assert driver.session_id == "s_1"


def test_non_reasoning_declared_done_follows_shared_state(monkeypatch):
    # success=False still counts as "declared" (the agent said it is over).
    monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class(success=False))
    monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)

    with TestClient(_app()) as client:
        episode_id = client.post("/v1/episodes", json=_request()).json()["episode_id"]
        info = _wait(client, episode_id)
    assert info["result"]["declared_done"] is True
    assert info["state"] == "failed"


def test_reasoning_declared_done_follows_the_manager_answer(monkeypatch):
    config = MobileConfig()
    config.agent.reasoning = True
    monkeypatch.setattr(
        episode_module, "MobileAgent", _fake_agent_class(manager_answer=False)
    )
    monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)

    with TestClient(_app(config)) as client:
        episode_id = client.post("/v1/episodes", json=_request()).json()["episode_id"]
        info = _wait(client, episode_id)
    assert info["result"]["declared_done"] is False


def test_event_stream_replays_and_ends(monkeypatch):
    monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class())
    monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)

    with TestClient(_app()) as client:
        episode_id = client.post("/v1/episodes", json=_request()).json()["episode_id"]
        _wait(client, episode_id)

        with client.stream("GET", f"/v1/episodes/{episode_id}/events") as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line.startswith("data:")]

    assert lines, "expected at least one streamed event"
    payloads = [json.loads(line[5:].strip()) for line in lines]
    types = {payload["type"] for payload in payloads}
    assert "ToolExecutionEvent" in types
    assert "ManagerPlanEvent" in types
    tap = next(p for p in payloads if p["type"] == "ToolExecutionEvent")
    assert tap["data"]["tool_name"] == "tap"


def test_cancel_stops_a_running_episode(monkeypatch):
    monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class(delay=5.0))
    monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)

    with TestClient(_app()) as client:
        episode_id = client.post("/v1/episodes", json=_request()).json()["episode_id"]
        cancelled = client.post(f"/v1/episodes/{episode_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "cancelled"

        info = client.get(f"/v1/episodes/{episode_id}").json()
        assert info["state"] == "cancelled"


def test_unknown_episode_is_404():
    with TestClient(_app()) as client:
        assert client.get("/v1/episodes/nope").status_code == 404
        assert client.post("/v1/episodes/nope/cancel").status_code == 404


def test_evidence_bundle_is_written_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class())
    monkeypatch.setattr(episode_module, "RemoteEnvDriver", _FakeDriver)

    app = _app(evidence_root=tmp_path)
    with TestClient(app) as client:
        body = _request(evidence={"enabled": True, "dir": None})
        body["metadata"] = {"task_id": "Alpha", "task_idx": 0}
        episode_id = client.post("/v1/episodes", json=body).json()["episode_id"]
        info = _wait(client, episode_id)

    evidence_dir = info["result"]["evidence_dir"]
    assert evidence_dir
    manifest = json.loads((tmp_path / episode_id / "episode.json").read_text())
    assert manifest["episode_id"] == episode_id
    assert manifest["goal"] == "do the thing"


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/v1/policies/jev/sessions", {"goal": "g"}),
        (
            "post",
            "/v1/policies/jev/act",
            {
                "policy_session_id": "ps",
                "step": 1,
                "observation": {"screen": {"native_width": 108, "native_height": 240}},
            },
        ),
        ("delete", "/v1/policies/jev/sessions/ps", None),
    ],
)
def test_policy_routes_are_declared_but_not_implemented(method, path, payload):
    with TestClient(_app()) as client:
        call = getattr(client, method)
        response = call(path, json=payload) if payload is not None else call(path)
        assert response.status_code == 501
        assert response.json()["code"] == "not_implemented"
