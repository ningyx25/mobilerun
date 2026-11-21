"""Harness runner: the orchestration loop across both services, in-process."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import mobilerun.harness.agent_service.episode as episode_module
import mobilerun.harness.client as client_module
import mobilerun.harness.runner as runner_module
from mobilerun.benchmark.android_world import TaskResult
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.harness.agent_service.app import create_agent_app
from mobilerun.harness.client import AgentClient, EnvClient
from mobilerun.harness.drivers import RemoteEnvDriver
from mobilerun.harness.env_service.app import create_env_app
from mobilerun.harness.protocol import (
    AgentOverrides,
    Capabilities,
    EpisodeInfo,
    EpisodeResult,
    EpisodeState,
    FinishResponse,
    TaskSpec,
    TokenUsage,
)
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
        # runbench rows carry the invocation id; the runs[] record counts them.
        assert row["run_id"] == payload["run_id"]
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


def _rerun_source(tmp_path, tasks, name="source.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"tasks": tasks}))
    return path


_RERUN_SOURCE_ROWS = [
    {"task_type": "Alpha", "task_idx": 0, "score": 1.0, "error": None},
    {"task_type": "Beta", "task_idx": 0, "score": 0.0, "error": None},
    {
        "task_type": "Charlie",
        "task_idx": 0,
        "score": 0.0,
        "error": "capacity: no free backend",
    },
    {"task_type": "Delta", "task_idx": 0, "score": 0.5, "error": None},
    {"task_type": "Echo", "task_idx": 0, "score": None, "error": None},
    {"task_type": "Zulu", "task_idx": 0, "score": 0.0, "error": None},
]


def test_rerun_fail_runs_only_failed_tasks(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha", "Beta", "Charlie", "Delta", "Echo"])
        source = _rerun_source(tmp_path, _RERUN_SOURCE_ROWS)
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                rerun_fail=str(source),
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return ok, output, source, backend

    ok, output, source, backend = _run(scenario())
    assert ok is True
    # Error rows (Charlie), partial scores (Delta) and passes (Alpha) are out;
    # Zulu is failed but absent from this suite's catalogue.
    assert backend.prepared == ["Beta", "Echo"]

    payload = json.loads(output.read_text())
    assert payload["selected_count"] == 2
    assert payload["rerun_fail_from"] == str(source)
    assert payload["rerun_fail_failed_count"] == 3
    assert payload["rerun_fail_unknown"] == ["Zulu#0"]
    assert {row["task_type"] for row in payload["tasks"]} == {"Beta", "Echo"}
    assert payload["summary"]["tasks"] == 2
    for row in payload["tasks"]:
        assert row["score"] == 1.0
        assert row["error"] is None
        assert row["run_id"] == payload["run_id"]


def test_rerun_fail_intersects_with_tasks_and_range(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha", "Beta", "Charlie", "Delta", "Echo"])
        source = _rerun_source(tmp_path, _RERUN_SOURCE_ROWS)
        async with _Wiring(backend, monkeypatch):
            ok_tasks = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(tmp_path / "by_tasks.json"),
                tasks="Alpha,Beta",
                rerun_fail=str(source),
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            prepared_after_tasks = list(backend.prepared)
            ok_range = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(tmp_path / "by_range.json"),
                task_index_range="2-2",
                rerun_fail=str(source),
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return ok_tasks, ok_range, backend, prepared_after_tasks

    ok_tasks, ok_range, backend, prepared_after_tasks = _run(scenario())
    # --tasks cannot widen the failed set: Alpha passed, so only Beta runs.
    assert ok_tasks is True
    assert prepared_after_tasks == ["Beta"]
    # The range is 1-based over the catalogue (2-2 is Beta), not re-numbered
    # over the failed subset.
    assert ok_range is True
    assert backend.prepared == ["Beta", "Beta"]


def test_rerun_fail_no_failures_is_a_noop(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        source = _rerun_source(
            tmp_path, [{"task_type": "Alpha", "task_idx": 0, "score": 1.0}]
        )
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                rerun_fail=str(source),
                capture_evidence=False,
                health_timeout=5.0,
            )
            return ok, output, backend

    ok, output, backend = _run(scenario())
    assert ok is True
    assert backend.prepared == []
    assert not output.exists()


def test_rerun_fail_empty_intersection_fails(tmp_path, monkeypatch):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha", "Beta"])
        unknown_source = _rerun_source(
            tmp_path,
            [{"task_type": "Zulu", "task_idx": 0, "score": 0.0}],
            name="unknown.json",
        )
        filtered_source = _rerun_source(
            tmp_path,
            [{"task_type": "Beta", "task_idx": 0, "score": 0.0}],
            name="filtered.json",
        )
        async with _Wiring(backend, monkeypatch):
            ok_unknown = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(tmp_path / "unknown_out.json"),
                rerun_fail=str(unknown_source),
                capture_evidence=False,
                health_timeout=5.0,
            )
            ok_filtered = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(tmp_path / "filtered_out.json"),
                tasks="Alpha",
                rerun_fail=str(filtered_source),
                capture_evidence=False,
                health_timeout=5.0,
            )
            return ok_unknown, ok_filtered, backend, tmp_path

    ok_unknown, ok_filtered, backend, tmp = _run(scenario())
    assert ok_unknown is False
    assert ok_filtered is False
    assert backend.prepared == []
    assert not (tmp / "unknown_out.json").exists()
    assert not (tmp / "filtered_out.json").exists()


def test_rerun_fail_dry_run_lists_the_filtered_subset(tmp_path, monkeypatch, capsys):
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha", "Beta", "Charlie", "Delta", "Echo"])
        source = _rerun_source(tmp_path, _RERUN_SOURCE_ROWS)
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                rerun_fail=str(source),
                dry_run=True,
                health_timeout=5.0,
            )
            return ok, output, backend

    ok, output, backend = _run(scenario())
    assert ok is True
    assert backend.prepared == []
    assert not output.exists()
    printed = capsys.readouterr().out
    assert "Beta" in printed
    assert "Echo" in printed
    assert "Alpha" not in printed


def test_rerun_fail_rejects_bad_source_files(tmp_path):
    import pytest as _pytest

    from mobilerun.harness.runner import _failed_task_keys

    cases = [
        ("missing.json", None, "does not exist"),
        ("bad.json", "{not json", "is not valid JSON"),
        ("array.json", "[]", "must contain a JSON object"),
        ("empty.json", "{}", "has no 'tasks' list"),
        ("row.json", '{"tasks": [1]}', "is not an object"),
        ("no_type.json", '{"tasks": [{"score": 0.0}]}', "task_type"),
        (
            "bad_score.json",
            '{"tasks": [{"task_type": "A", "score": "0"}]}',
            "non-numeric score",
        ),
    ]
    for name, content, message in cases:
        path = tmp_path / name
        if content is not None:
            path.write_text(content)
        with _pytest.raises(ValueError, match=message):
            _failed_task_keys(path)


def test_failed_task_keys_dedupes_and_defaults_idx(tmp_path):
    from mobilerun.harness.runner import _failed_task_keys

    source = _rerun_source(
        tmp_path,
        [
            # Duplicate key: the later row wins, so Beta is not failed.
            {"task_type": "Beta", "task_idx": 0, "score": 0.0},
            {"task_type": "Beta", "task_idx": 0, "score": 1.0},
            {"task_type": "Alpha", "score": 0.0},
            {"task_type": "Gamma", "task_idx": 2, "score": 0.0},
        ],
    )
    failed, meta = _failed_task_keys(source)
    assert failed == {"Alpha#0", "Gamma#2"}
    assert "tasks" not in meta


def test_eval_cli_forwards_rerun_fail(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from click.testing import CliRunner

    import mobilerun.cli.main as cli_main

    runner = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "mobilerun.harness.runner.run_harness_suite", runner, raising=False
    )

    source = _rerun_source(tmp_path, [{"task_type": "Alpha", "rank": 0}])
    result = CliRunner().invoke(
        cli_main.cli,
        [
            "eval",
            "--env",
            "http://env.test",
            "--agent",
            "http://agent.test",
            "--rerun-fail",
            str(source),
        ],
    )

    assert result.exit_code == 0, result.output
    assert runner.await_args.kwargs["rerun_fail"] == str(source)


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


def test_episode_manifest_is_written_after_scoring(tmp_path):
    from mobilerun.benchmark.android_world import TaskResult
    from mobilerun.harness.runner import _write_episode_manifest

    bundle = tmp_path / "Task_0"
    (bundle / "trajectory").mkdir(parents=True)
    (bundle / "timeline.json").write_text(
        json.dumps(
            {
                "usage_totals": {
                    "request_tokens": 1,
                    "response_tokens": 2,
                    "total_tokens": 3,
                    "requests": 4,
                }
            }
        )
    )
    (bundle / "conversation.json").write_text("[]")
    (bundle / "ui_tree_before.json").write_text("{}")

    result = TaskResult(task_type="Task", task_idx=0, difficulty="easy")
    result.evidence_dir = str(bundle)
    result.goal = "do it"
    result.score = 0.75
    result.gated_score = 0.75
    result.success = True
    result.finished = True
    result.steps = 3
    result.reason = "done"
    result.max_steps = 20
    result.usage_totals = {"total_tokens": 10}

    run_meta = {"run_id": "r1", "benchmark": "harness"}
    _write_episode_manifest(result, run_meta)

    manifest = json.loads((bundle / "episode.json").read_text())
    assert manifest["task_type"] == "Task"
    assert manifest["score"] == 0.75
    assert manifest["trajectory_dir"] == "trajectory"
    # Prefer the timeline's totals: it sums every model call, not just the
    # usage attached to episode decisions.
    assert manifest["usage_totals"] == {
        "request_tokens": 1,
        "response_tokens": 2,
        "total_tokens": 3,
        "requests": 4,
    }
    # The row carries the request count too, so the file summary can sum it.
    assert result.usage_totals["requests"] == 4
    assert manifest["conversation_path"] == "conversation.json"
    assert manifest["timeline_path"] == "timeline.json"
    assert manifest["ui_tree_paths"] == ["ui_tree_before.json"]
    assert manifest["run"] == run_meta


def test_an_empty_timeline_keeps_the_episode_usage(tmp_path):
    from mobilerun.harness.runner import _write_episode_manifest

    bundle = tmp_path / "Task_0"
    (bundle / "trajectory").mkdir(parents=True)
    (bundle / "timeline.json").write_text(
        json.dumps(
            {
                "usage_totals": {
                    "request_tokens": 0,
                    "response_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0,
                }
            }
        )
    )

    result = TaskResult(task_type="Task", task_idx=0)
    result.evidence_dir = str(bundle)
    result.usage_totals = {
        "request_tokens": 10,
        "response_tokens": 5,
        "total_tokens": 15,
        "requests": 3,
    }

    _write_episode_manifest(result, {"run_id": "r1"})

    manifest = json.loads((bundle / "episode.json").read_text())
    # jev-mobile never emits per-call usage events, so its timeline is all
    # zeros; the episode result must not be replaced by them.
    assert manifest["usage_totals"] == {
        "request_tokens": 10,
        "response_tokens": 5,
        "total_tokens": 15,
        "requests": 3,
    }
    assert result.usage_totals["requests"] == 3


def test_summary_sums_token_and_request_counts():
    from mobilerun.benchmark.common import _build_summary

    rows = {
        "A#0": TaskResult(
            task_type="A",
            task_idx=0,
            score=1.0,
            usage_totals={"total_tokens": 10, "requests": 2},
        ),
        "B#0": TaskResult(
            task_type="B",
            task_idx=0,
            score=0.0,
            usage_totals={"total_tokens": 20, "requests": 3},
        ),
    }

    summary = _build_summary(rows)

    assert summary["total_tokens"] == 30
    assert summary["total_requests"] == 5


def test_a_smaller_timeline_cannot_lower_the_request_count(tmp_path):
    from mobilerun.harness.runner import _write_episode_manifest

    bundle = tmp_path / "Task_0"
    (bundle / "trajectory").mkdir(parents=True)
    (bundle / "timeline.json").write_text(
        json.dumps(
            {
                "usage_totals": {
                    "request_tokens": 1,
                    "response_tokens": 1,
                    "total_tokens": 2,
                    "requests": 1,
                }
            }
        )
    )

    result = TaskResult(task_type="Task", task_idx=0)
    result.evidence_dir = str(bundle)
    result.usage_totals = {"total_tokens": 15, "requests": 3}

    _write_episode_manifest(result, {"run_id": "r1"})

    manifest = json.loads((bundle / "episode.json").read_text())
    assert manifest["usage_totals"]["requests"] == 3
    assert result.usage_totals["requests"] == 3


def test_episode_manifest_skips_a_missing_bundle(tmp_path):
    from mobilerun.benchmark.android_world import TaskResult
    from mobilerun.harness.runner import _write_episode_manifest

    result = TaskResult(task_type="Task", task_idx=0)
    result.evidence_dir = str(tmp_path / "missing")
    _write_episode_manifest(result, {})  # must not raise
    assert not (tmp_path / "missing" / "episode.json").exists()


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


def test_episode_events_render_like_runbench(capsys):
    from mobilerun.harness.runner import _render_episode_event

    _render_episode_event(
        "Task",
        {
            "type": "llm_response",
            "data": {"thought": "look at the screen", "code": "<function_calls/>"},
        },
    )
    _render_episode_event(
        "Task", {"type": "action_result", "data": {"ok": False, "error": "boom"}}
    )
    _render_episode_event(
        "Task", {"type": "agent_end", "data": {"success": True, "reason": "finished"}}
    )
    # The mobilerun worker forwards workflow class names instead.
    _render_episode_event(
        "Task",
        {
            "type": "FastAgentResponseEvent",
            "data": {"thought": "weigh options", "code": "<function_calls/>"},
        },
    )
    _render_episode_event(
        "Task", {"type": "FastAgentOutputEvent", "data": {"output": "Error: boom"}}
    )
    _render_episode_event(
        "Task",
        {"type": "FastAgentEndEvent", "data": {"success": False, "reason": "nope"}},
    )

    out = capsys.readouterr().out
    assert "🧠 Thinking: look at the screen" in out
    assert "💻 Executing action code" in out
    assert "❌ Action error: boom" in out
    assert "■ done: finished" in out
    assert "🧠 Thinking: weigh options" in out
    assert "❌ Action error: Error: boom" in out
    assert "■ failed: nope" in out


def test_goal_line_matches_runbench(capsys):
    from mobilerun.harness.protocol import EpisodeResult
    from mobilerun.harness.runner import _print_goal_line

    _print_goal_line("Task", EpisodeResult(success=True, reason="ok"))
    _print_goal_line("Task", EpisodeResult(success=False, reason="nope"))

    out = capsys.readouterr().out
    assert "🎉 Goal achieved: ok" in out
    assert "❌ Goal failed: nope" in out


def test_wait_healthy_requires_a_live_backend():
    from mobilerun.harness.protocol import BackendInfo, HealthResponse
    from mobilerun.harness.runner import _wait_healthy

    class _Env:
        base_url = "http://env"

        async def health(self):
            return HealthResponse(
                ok=True,
                backends=[BackendInfo(id="aw0", kind="android-world", state="down")],
            )

    class _Agent:
        base_url = "http://agent"

        async def health(self):
            return {"ok": True}

    assert asyncio.run(_wait_healthy(_Env(), _Agent(), timeout=0)) is False


def test_run_aborts_when_services_do_not_recover(tmp_path, monkeypatch):
    from mobilerun.harness.protocol import BackendInfo, HealthResponse

    class _Env:
        base_url = "http://env"

        def __init__(self):
            self.health_calls = 0
            self.closed = False

        async def health(self):
            self.health_calls += 1
            if self.health_calls == 1:
                return HealthResponse(
                    ok=True,
                    backends=[
                        BackendInfo(id="aw0", kind="android-world", state="ready")
                    ],
                )
            raise ConnectionError("down")

        async def list_tasks(self, backend=None):
            return [
                TaskSpec(id="Alpha#0", backend="aw0"),
                TaskSpec(id="Beta#0", backend="aw0"),
            ]

        async def create_session(self, *args, **kwargs):
            raise ConnectionError("down")

        async def close(self):
            self.closed = True

    class _Agent:
        base_url = "http://agent"

        async def health(self):
            return {"ok": True}

        async def close(self):
            return None

    env = _Env()
    agent = _Agent()
    monkeypatch.setattr(runner_module, "EnvClient", lambda *a, **k: env)
    monkeypatch.setattr(runner_module, "AgentClient", lambda *a, **k: agent)

    output = tmp_path / "results.json"

    async def scenario():
        return await runner_module.run_harness_suite(
            "http://env.test",
            "http://agent.test",
            output=str(output),
            parallel=1,
            health_timeout=0.0,
            task_timeout=1.0,
            capture_evidence=False,
        )

    assert asyncio.run(scenario()) is False
    payload = json.loads(output.read_text())
    # The failing task is recorded; the queued ones stay unrecorded for --resume.
    assert [row["task_type"] for row in payload["tasks"]] == ["Alpha"]
    assert "ConnectionError" in payload["tasks"][0]["error"]


# ---------------------------------------------------------------------------
# --max-retry: in-process retries for rows that end with an error
# ---------------------------------------------------------------------------


def _retry_task(name: str = "Alpha") -> TaskSpec:
    return TaskSpec(id=f"{name}#0", backend="fake")


def _retry_row(
    name: str = "Alpha", *, error: str | None = None, score: float | None = 1.0
) -> TaskResult:
    return TaskResult(task_type=name, task_idx=0, score=score, error=error)


def test_run_with_retries_recovers_from_transient_errors(monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    responses = [
        (_retry_row(error="boom", score=None), False),
        (_retry_row(error="boom", score=None), False),
        (_retry_row(), False),
    ]
    calls = []

    async def fake_run(task, *, env, **kwargs):
        calls.append(task.id)
        return responses[len(calls) - 1]

    monkeypatch.setattr(runner_module, "_run_with_capacity_retry", fake_run)

    result = asyncio.run(
        runner_module._run_with_retries(
            _retry_task(),
            env=object(),
            agent=object(),
            max_retry=3,
            abort=asyncio.Event(),
            health_timeout=0.0,
        )
    )

    assert calls == ["Alpha#0"] * 3
    assert result.error is None
    assert result.attempts == 3


def test_run_with_retries_stops_after_max_retry(monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    calls = []

    async def fake_run(task, *, env, **kwargs):
        calls.append(task.id)
        return _retry_row(error="always boom", score=None), False

    monkeypatch.setattr(runner_module, "_run_with_capacity_retry", fake_run)

    result = asyncio.run(
        runner_module._run_with_retries(
            _retry_task(),
            env=object(),
            agent=object(),
            max_retry=2,
            abort=asyncio.Event(),
            health_timeout=0.0,
        )
    )

    assert calls == ["Alpha#0"] * 3
    assert result.error == "always boom"
    assert result.attempts == 3


def test_max_retry_zero_disables_retries(monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    calls = []

    async def fake_run(task, *, env, **kwargs):
        calls.append(task.id)
        return _retry_row(error="always boom", score=None), False

    monkeypatch.setattr(runner_module, "_run_with_capacity_retry", fake_run)

    result = asyncio.run(
        runner_module._run_with_retries(
            _retry_task(),
            env=object(),
            agent=object(),
            max_retry=0,
            abort=asyncio.Event(),
            health_timeout=0.0,
        )
    )

    assert calls == ["Alpha#0"]
    assert result.attempts == 1


def test_capacity_exhaustion_is_not_retried_again(monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    calls = []

    async def fake_run(task, *, env, **kwargs):
        calls.append(task.id)
        return _retry_row(error="429 capacity: no free backend", score=None), True

    async def fake_services_up(env, agent):
        return True

    monkeypatch.setattr(runner_module, "_run_with_capacity_retry", fake_run)
    monkeypatch.setattr(runner_module, "_services_up", fake_services_up)

    result = asyncio.run(
        runner_module._run_with_retries(
            _retry_task(),
            env=object(),
            agent=object(),
            max_retry=10,
            abort=asyncio.Event(),
            health_timeout=0.0,
        )
    )

    # `_run_with_capacity_retry` already spent a whole task budget polling.
    assert calls == ["Alpha#0"]
    assert result.attempts == 1
    assert "capacity" in result.error


def test_retry_waits_for_services_recovery_first(monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    responses = [
        (_retry_row(error="ConnectionError: down", score=None), True),
        (_retry_row(), False),
    ]
    calls = []

    async def fake_run(task, *, env, **kwargs):
        calls.append(task.id)
        return responses[len(calls) - 1]

    async def fake_services_up(env, agent):
        return False

    async def fake_wait_healthy(env, agent, timeout):
        return True

    monkeypatch.setattr(runner_module, "_run_with_capacity_retry", fake_run)
    monkeypatch.setattr(runner_module, "_services_up", fake_services_up)
    monkeypatch.setattr(runner_module, "_wait_healthy", fake_wait_healthy)

    result = asyncio.run(
        runner_module._run_with_retries(
            _retry_task(),
            env=object(),
            agent=object(),
            max_retry=2,
            abort=asyncio.Event(),
            health_timeout=5.0,
        )
    )

    assert calls == ["Alpha#0", "Alpha#0"]
    assert result.error is None
    assert result.attempts == 2


def test_retry_aborts_when_services_never_recover(monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    calls = []

    async def fake_run(task, *, env, **kwargs):
        calls.append(task.id)
        return _retry_row(error="ConnectionError: down", score=None), True

    async def fake_services_up(env, agent):
        return False

    async def fake_wait_healthy(env, agent, timeout):
        return False

    monkeypatch.setattr(runner_module, "_run_with_capacity_retry", fake_run)
    monkeypatch.setattr(runner_module, "_services_up", fake_services_up)
    monkeypatch.setattr(runner_module, "_wait_healthy", fake_wait_healthy)

    abort = asyncio.Event()
    result = asyncio.run(
        runner_module._run_with_retries(
            _retry_task(),
            env=object(),
            agent=object(),
            max_retry=10,
            abort=abort,
            health_timeout=0.0,
        )
    )

    assert calls == ["Alpha#0"]
    assert abort.is_set() is True
    assert result.error is not None


def test_retry_stops_when_another_worker_aborts(monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    calls = []
    abort = asyncio.Event()

    async def fake_run(task, *, env, **kwargs):
        calls.append(task.id)
        abort.set()
        return _retry_row(error="boom", score=None), False

    monkeypatch.setattr(runner_module, "_run_with_capacity_retry", fake_run)

    result = asyncio.run(
        runner_module._run_with_retries(
            _retry_task(),
            env=object(),
            agent=object(),
            max_retry=10,
            abort=abort,
            health_timeout=0.0,
        )
    )

    assert calls == ["Alpha#0"]
    assert result.error == "boom"


def test_transient_scoring_error_is_retried_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(client_module, "DEFAULT_BACKOFF_S", 0.0)
    failures = {"left": 2}

    def score_fn(backend):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("transient boom")
        return 1.0

    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"], score_fn=score_fn)
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                max_retry=3,
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return ok, output, backend

    ok, output, backend = _run(scenario())
    assert ok is True
    assert backend.prepared == ["Alpha", "Alpha", "Alpha"]

    payload = json.loads(output.read_text())
    assert payload["max_retry"] == 3
    assert payload["runs"][-1]["max_retry"] == 3
    assert len(payload["tasks"]) == 1
    row = payload["tasks"][0]
    assert row["error"] is None
    assert row["score"] == 1.0
    assert row["attempts"] == 3


@pytest.mark.parametrize(("max_retry", "expected"), [(0, 1), (2, 3)])
def test_retry_exhaustion_records_one_final_row(
    tmp_path, monkeypatch, max_retry, expected
):
    monkeypatch.setattr(runner_module, "_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(client_module, "DEFAULT_BACKOFF_S", 0.0)

    def score_fn(backend):
        raise RuntimeError("always boom")

    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"], score_fn=score_fn)
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                max_retry=max_retry,
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return ok, output, backend

    ok, output, backend = _run(scenario())
    assert ok is False
    assert backend.prepared == ["Alpha"] * expected

    payload = json.loads(output.read_text())
    assert len(payload["tasks"]) == 1
    row = payload["tasks"][0]
    assert row["error"] is not None
    assert row["attempts"] == expected


def test_legacy_results_default_to_one_attempt(tmp_path):
    from mobilerun.benchmark.android_world import _load_results

    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "Alpha",
                        "task_idx": 0,
                        "score": 1.0,
                        "error": None,
                    }
                ]
            }
        )
    )

    loaded = _load_results(path)
    assert loaded["Alpha#0"].attempts == 1


def _invoke_eval_cli(args, monkeypatch):
    from unittest.mock import AsyncMock

    from click.testing import CliRunner

    import mobilerun.cli.main as cli_main

    runner = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "mobilerun.harness.runner.run_harness_suite", runner, raising=False
    )
    result = CliRunner().invoke(
        cli_main.cli,
        ["eval", "--env", "http://env.test", "--agent", "http://agent.test", *args],
    )
    return result, runner


def test_eval_cli_forwards_max_retry_default(monkeypatch):
    result, runner = _invoke_eval_cli([], monkeypatch)

    assert result.exit_code == 0, result.output
    assert runner.await_args.kwargs["max_retry"] == 10


def test_eval_cli_forwards_max_retry_explicit(monkeypatch):
    result, runner = _invoke_eval_cli(["--max-retry", "3"], monkeypatch)

    assert result.exit_code == 0, result.output
    assert runner.await_args.kwargs["max_retry"] == 3


@pytest.mark.parametrize("value", ["-1", "abc"])
def test_eval_cli_rejects_invalid_max_retry(monkeypatch, value):
    result, runner = _invoke_eval_cli([f"--max-retry={value}"], monkeypatch)

    assert result.exit_code == 2
    assert runner.await_count == 0


def test_overrides_force_vision_only_for_tree_less_backends():
    from mobilerun.harness.protocol import AgentOverrides, Capabilities

    overrides = AgentOverrides()
    forced = runner_module._overrides_for_session(
        overrides, Capabilities(has_ui_tree=False)
    )
    assert forced.vision_only is True
    assert overrides.vision_only is None

    explicit = AgentOverrides(vision_only=False)
    kept = runner_module._overrides_for_session(
        explicit, Capabilities(has_ui_tree=False)
    )
    assert kept.vision_only is False

    untouched = runner_module._overrides_for_session(
        overrides, Capabilities(has_ui_tree=True)
    )
    assert untouched is overrides


# ---------------------------------------------------------------------------
# Abnormal episode endings must carry an error
# ---------------------------------------------------------------------------


class _RunOneEnv:
    base_url = "http://env.test"

    def __init__(self):
        self.finish_calls = 0
        self.finish_declared_done = []
        self.task = TaskSpec(
            id="Alpha#0", backend="fake", instruction="do Alpha", max_steps=5
        )
        self.capabilities = Capabilities()

    async def create_session(self, *args, **kwargs):
        return SimpleNamespace(
            session_id="s1", task=self.task, capabilities=self.capabilities
        )

    async def finish(self, session_id, *, declared_done, answer=None):
        self.finish_calls += 1
        self.finish_declared_done.append(declared_done)
        return FinishResponse(score=0.0, gated_score=0.0, declared_done=declared_done)

    async def delete_session(self, session_id):
        pass


class _RunOneAgent:
    base_url = "http://agent.test"

    def __init__(self, *, final, running_info=None):
        self.final = final
        self.running_info = running_info or final
        self.cancelled = False

    async def start_episode(self, request):
        return EpisodeInfo(
            episode_id="ep1",
            state=EpisodeState.RUNNING,
            created_at="2026-01-01T00:00:00Z",
        )

    async def wait_episode(self, episode_id, *, timeout_s, poll_s=1.0):
        return self.running_info

    async def cancel_episode(self, episode_id):
        self.cancelled = True

    async def get_episode(self, episode_id):
        return self.final


def _run_one_case(final, *, running_info=None, task_timeout=30.0):
    env, agent = _RunOneEnv(), _RunOneAgent(final=final, running_info=running_info)
    result, _ = asyncio.run(
        runner_module._run_one(
            _task("Alpha#0"),
            env=env,
            agent=agent,
            agent_env_url=None,
            overrides=AgentOverrides(),
            steps=None,
            task_timeout=task_timeout,
            seed=None,
            evidence=False,
            evidence_root=None,
            backend=None,
            run_id="r-test",
        )
    )
    return result, env, agent


def _episode(state, *, result=None):
    return EpisodeInfo(
        episode_id="ep1",
        state=state,
        created_at="2026-01-01T00:00:00Z",
        result=result,
    )


def test_agent_episode_timeout_sets_error_and_is_not_scored():
    final = _episode(
        EpisodeState.FAILED,
        result=EpisodeResult(
            success=False,
            declared_done=False,
            reason="timed out after 600s",
            steps=3,
        ),
    )
    result, env, _ = _run_one_case(final)

    assert result.error.startswith("InfraError:")
    assert "timed out after 600s" in result.error
    assert result.reason == "timed out after 600s"
    assert result.score is None
    assert result.gated_score is None
    assert env.finish_calls == 0


def test_workflow_timeout_reason_sets_error():
    final = _episode(
        EpisodeState.FAILED,
        result=EpisodeResult(
            success=False,
            declared_done=False,
            reason="WorkflowTimeoutError: Operation timed out after 30 seconds",
        ),
    )
    result, env, _ = _run_one_case(final)

    assert result.error.startswith("InfraError:")
    assert "WorkflowTimeoutError" in result.error
    assert env.finish_calls == 0


def test_agent_declared_max_steps_failure_stays_error_free():
    final = _episode(
        EpisodeState.FAILED,
        result=EpisodeResult(
            success=False,
            declared_done=False,
            reason="max_steps (50) reached",
            steps=50,
        ),
    )
    result, env, _ = _run_one_case(final)

    assert result.error is None
    assert env.finish_calls == 1
    assert env.finish_declared_done == [False]
    assert result.score == 0.0
    assert result.gated_score == 0.0


def test_agent_declared_failure_stays_error_free():
    final = _episode(
        EpisodeState.FAILED,
        result=EpisodeResult(
            success=False,
            declared_done=True,
            reason="could not find the screen",
        ),
    )
    result, env, _ = _run_one_case(final)

    assert result.error is None
    assert env.finish_calls == 1
    assert env.finish_declared_done == [True]


def test_usage_requests_are_recorded_from_the_agent_result():
    final = _episode(
        EpisodeState.SUCCEEDED,
        result=EpisodeResult(
            success=True,
            declared_done=True,
            steps=4,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                requests=3,
            ),
        ),
    )
    result, _, _ = _run_one_case(final)

    # Same key names the benchmark runner writes, including the request count
    # the summary sums into total_requests.
    assert result.usage_totals == {
        "request_tokens": 10,
        "response_tokens": 5,
        "total_tokens": 15,
        "requests": 3,
    }


def test_cancelled_episode_sets_error_and_is_not_scored():
    final = _episode(
        EpisodeState.CANCELLED,
        result=EpisodeResult(success=False, reason="cancelled"),
    )
    result, env, _ = _run_one_case(final)

    assert result.error == "agent episode was cancelled"
    assert env.finish_calls == 0


@pytest.mark.parametrize("state", [EpisodeState.FAILED, EpisodeState.SUCCEEDED])
def test_terminal_state_without_result_sets_error(state):
    result, env, _ = _run_one_case(_episode(state, result=None))

    assert "without a result" in result.error
    assert env.finish_calls == 0


def test_runner_wait_timeout_is_not_overwritten_by_cancel():
    running = _episode(EpisodeState.RUNNING)
    final = _episode(
        EpisodeState.CANCELLED,
        result=EpisodeResult(success=False, reason="cancelled"),
    )
    result, env, agent = _run_one_case(final, running_info=running, task_timeout=30.0)

    assert agent.cancelled is True
    assert result.error == "agent episode timed out after 30s"
    assert env.finish_calls == 0


def test_agent_absorbed_provider_failure_is_an_infra_error():
    final = _episode(
        EpisodeState.FAILED,
        result=EpisodeResult(
            success=False,
            declared_done=False,
            reason=("Error: Error code: 429 - {'error': {'code': 'model_cooldown'}}"),
        ),
    )
    result, env, _ = _run_one_case(final)

    assert result.error.startswith("InfraError:")
    assert env.finish_calls == 0


def test_timeout_reason_flows_through_the_agent_service(tmp_path, monkeypatch):
    import sys

    original = _fake_agent_class
    monkeypatch.setattr(
        sys.modules[__name__],
        "_fake_agent_class",
        lambda: original(success=False, reason="timed out after 600s"),
    )

    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        output = tmp_path / "results.json"
        async with _Wiring(backend, monkeypatch):
            ok = await runner_module.run_harness_suite(
                "http://env.test",
                "http://agent.test",
                output=str(output),
                max_retry=0,
                capture_evidence=False,
                health_timeout=5.0,
                task_timeout=30.0,
            )
            return ok, output

    ok, output = _run(scenario())
    payload = json.loads(output.read_text())
    row = payload["tasks"][0]

    assert ok is False
    assert row["error"].startswith("InfraError:")
    assert "timed out after 600s" in row["error"]
    assert row["score"] is None
    assert row["gated_score"] is None
    assert payload["summary"]["errors"] == 1


def test_unknown_failed_reason_is_an_error_by_default():
    final = _episode(
        EpisodeState.FAILED,
        result=EpisodeResult(
            success=False,
            declared_done=False,
            reason="watchdog aborted the episode",
        ),
    )
    result, env, _ = _run_one_case(final)

    assert result.error.startswith("agent episode failed without declaring completion:")
    assert "watchdog aborted the episode" in result.error
    assert env.finish_calls == 0


def test_unknown_episode_state_is_an_error_by_default():
    final = SimpleNamespace(state="aborted", result=None, error=None)

    error = runner_module._abnormal_end_reason(final)

    assert error is not None
    assert "unexpected state" in error
