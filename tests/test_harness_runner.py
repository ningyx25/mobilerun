"""Harness runner: the orchestration loop across both services, in-process."""

import asyncio
import json

import httpx
import pytest

import mobilerun.harness.agent_service.episode as episode_module
import mobilerun.harness.runner as runner_module
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.harness.agent_service.app import create_agent_app
from mobilerun.harness.client import AgentClient, EnvClient
from mobilerun.harness.drivers import RemoteEnvDriver
from mobilerun.harness.env_service.app import create_env_app
from mobilerun.harness.protocol import TaskSpec
from mobilerun.harness.runner import _select_tasks
from mobilerun.harness.testing import ScriptedBackend
from tests.test_harness_agent_service import _fake_agent_class


class _Wiring:
    """Both services in one event loop, reachable over ASGI, not sockets."""

    def __init__(
        self, backend: ScriptedBackend, monkeypatch, *, dispatcher=False
    ) -> None:
        self.backend = backend
        self.monkeypatch = monkeypatch
        self.dispatcher_enabled = dispatcher
        self.env_app = create_env_app([backend])
        self.agent_app = create_agent_app(MobileConfig(), llms={}, name="fast-agent")
        self.dispatcher_app = None

    async def __aenter__(self) -> "_Wiring":
        await self.env_app.state.registry.start()
        await self.agent_app.state.manager.start()
        env_transport = httpx.ASGITransport(app=self.env_app)
        agent_transport = httpx.ASGITransport(app=self.agent_app)

        # Optionally put a dispatcher in front of the worker, so the runner
        # exercises the real `--agent-id` path.
        public_transport = agent_transport
        if self.dispatcher_enabled:
            from mobilerun.harness.agent_service.dispatcher import (
                create_agent_dispatcher,
            )

            def factory(url):
                return AgentClient(url, transport=agent_transport)

            self.dispatcher_app = create_agent_dispatcher(
                ["fast0=fast-agent@http://worker.test"],
                client_factory=factory,
                probe_interval=0.0,
            )
            await self.dispatcher_app.state.registry.probe(force=True)
            public_transport = httpx.ASGITransport(app=self.dispatcher_app)

        def env_client(url, token=None, **kwargs):
            return EnvClient(url, token=token, transport=env_transport)

        def agent_client(url, token=None, **kwargs):
            return AgentClient(url, token=token, transport=public_transport)

        def remote_driver(url, session_id, token=None, **kwargs):
            return RemoteEnvDriver(
                url, session_id, client=EnvClient(url, transport=env_transport)
            )

        self.monkeypatch.setattr(runner_module, "EnvClient", env_client)
        self.monkeypatch.setattr(runner_module, "AgentClient", agent_client)
        self.monkeypatch.setattr(episode_module, "RemoteEnvDriver", remote_driver)
        self.monkeypatch.setattr(episode_module, "MobileAgent", _fake_agent_class())
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self.dispatcher_app is not None:
            await self.dispatcher_app.state.registry.close()
        await self.agent_app.state.manager.stop()
        await self.env_app.state.registry.stop()


def _run(coro):
    return asyncio.run(coro)


def test_suite_runs_every_task_and_writes_comparable_results(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha", "Beta"])
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return ok, output, backend

    ok, output, backend = _run(scenario())
    assert ok is True

    payload = json.loads(output.read_text())
    assert payload["benchmark"] == "harness"
    assert payload["protocol_version"] == "1.0"
    assert payload["selected_count"] == 2
    rows = {row["task_type"]: row for row in payload["tasks"]}
    assert set(rows) == {"Alpha", "Beta"}
    for row in rows.values():
        assert row["score"] == 1.0
        assert row["gated_score"] == 1.0
        assert row["finished"] is True
        assert row["steps"] == 2
        assert row["error"] is None
    assert backend.prepared == ["Alpha", "Beta"]
    assert backend.teardowns == ["Alpha#0", "Beta#0"]


def test_resume_skips_tasks_already_recorded(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha", "Beta"])
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                tasks="Alpha",
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            prepared_after_first = list(backend.prepared)
            await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                resume=True,
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return prepared_after_first, backend.prepared, output

    after_first, after_second, output = _run(scenario())
    assert after_first == ["Alpha"]
    # Resume ran only the task that was not recorded yet.
    assert after_second == ["Alpha", "Beta"]
    payload = json.loads(output.read_text())
    assert {row["task_type"] for row in payload["tasks"]} == {"Alpha", "Beta"}


def test_dry_run_lists_tasks_without_opening_sessions(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha", "Beta"])
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                dry_run=True,
                health_timeout=5.0,
            )
            return ok, output, backend

    ok, output, backend = _run(scenario())
    assert ok is True
    assert not output.exists()
    assert backend.prepared == []


def test_agent_id_flows_through_the_dispatcher(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch, dispatcher=True):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                agent_id="fast0",
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return ok, output

    ok, output = _run(scenario())
    assert ok is True
    payload = json.loads(output.read_text())
    assert payload["agent_id"] == "fast0"
    assert payload["agents"][0]["id"] == "fast0"
    row = payload["tasks"][0]
    assert row["agent_id"] == "fast0"
    assert row["gated_score"] == 1.0


def _task(task_id: str, backend: str = "aw0", difficulty: str = "easy") -> TaskSpec:
    return TaskSpec(id=task_id, backend=backend, metadata={"difficulty": difficulty})


def test_select_tasks_filters_by_backend_range_and_name():
    catalogue = [
        _task("A#0", "aw0"),
        _task("B#0", "aw0"),
        _task("C#0", "aw1"),
        _task("D#0", "aw1", difficulty="hard"),
    ]
    assert [t.id for t in _select_tasks(catalogue, backend="aw0")] == ["A#0", "B#0"]
    assert [t.id for t in _select_tasks(catalogue, task_index_range="1-2")] == [
        "A#0",
        "B#0",
    ]
    assert [t.id for t in _select_tasks(catalogue, names="C#0,A#0")] == ["C#0", "A#0"]
    assert [t.id for t in _select_tasks(catalogue, difficulty=["hard"])] == ["D#0"]


def test_select_tasks_dedupes_backends_when_none_is_pinned():
    catalogue = [
        _task("A#0", "aw0"),
        _task("B#0", "aw0"),
        _task("A#0", "aw1"),
        _task("B#0", "aw1"),
    ]
    assert [(t.id, t.backend) for t in _select_tasks(catalogue)] == [
        ("A#0", "aw0"),
        ("B#0", "aw0"),
    ]


def test_select_tasks_keeps_the_pinned_backend():
    catalogue = [_task("A#0", "aw0"), _task("B#0", "aw1")]
    assert [t.id for t in _select_tasks(catalogue, backend="aw1")] == ["B#0"]


def test_select_tasks_rejects_unknown_names_and_empty_selection():
    catalogue = [_task("A#0")]
    with pytest.raises(ValueError, match="Unknown task"):
        _select_tasks(catalogue, names="Nope#0")
    with pytest.raises(ValueError, match="empty"):
        _select_tasks(catalogue, backend="aw9")
