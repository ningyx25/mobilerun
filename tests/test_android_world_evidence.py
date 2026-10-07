"""Tests for the benchmark's rich per-episode evidence capture.

Drives the real ``run_androidworld_benchmark`` with a fake client/agent (as in
test_android_world_benchmark.py) and asserts that a complete evidence bundle is
written: trajectory artifacts, full conversation, real UI-tree snapshots, a
distilled per-step timeline, token usage, and a manifest, all referenced from
the result row.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mobilerun.benchmark import android_world as bench
from mobilerun.benchmark.evidence import EpisodeEvidence
from mobilerun.config_manager.config_manager import MobileConfig


def run(coro):
    return asyncio.run(coro)


# The autouse ``restore_mobilerun_logger`` fixture lives in tests/conftest.py.


# ---------------------------------------------------------------------------
# EpisodeEvidence unit tests
# ---------------------------------------------------------------------------


def _usage_event(thought="think", code="<c/>", request=10, response=3):
    from mobilerun.agent.fast_agent.events import FastAgentResponseEvent
    from mobilerun.agent.usage import UsageResult

    return FastAgentResponseEvent(
        thought=thought,
        code=code,
        usage=UsageResult(
            request_tokens=request,
            response_tokens=response,
            total_tokens=request + response,
            requests=1,
        ),
    )


def test_record_event_builds_timeline_and_sums_usage(tmp_path):
    ev = EpisodeEvidence(tmp_path / "ep", "ClockStopWatchRunning", 0)
    ev.record_event(_usage_event(request=10, response=3))
    ev.record_event(_usage_event(request=20, response=7))

    assert len(ev.timeline) == 2
    assert ev.timeline[0]["kind"] == "llm"
    assert ev.timeline[0]["thought"] == "think"
    assert ev.usage_totals == {
        "request_tokens": 30,
        "response_tokens": 10,
        "total_tokens": 40,
        "requests": 2,
    }


def test_record_event_distills_tool_execution(tmp_path):
    from mobilerun.agent.common.events import ToolExecutionEvent

    ev = EpisodeEvidence(tmp_path / "ep", "T", 0)
    ev.record_event(
        ToolExecutionEvent(
            tool_name="tap", tool_args={"x": 1, "y": 2}, success=True, summary="ok"
        )
    )

    entry = ev.timeline[0]
    assert entry["kind"] == "tool"
    assert entry["tool"] == "tap"
    assert entry["args"] == {"x": 1, "y": 2}
    assert entry["success"] is True


def test_record_event_skips_screenshots(tmp_path):
    from mobilerun.agent.common.events import ScreenshotEvent

    ev = EpisodeEvidence(tmp_path / "ep", "T", 0)
    ev.record_event(ScreenshotEvent(screenshot=b"\x89PNG..."))
    assert ev.timeline == []


def test_write_ui_tree_skips_empty_tree(tmp_path):
    ev = EpisodeEvidence(tmp_path / "ep", "T", 0)

    assert ev.write_ui_tree("ui_tree_after", {"a11y_tree": {"children": []}}) is None
    assert not (ev.dir / "ui_tree_after.json").exists()

    tree = {"a11y_tree": {"children": [{"text": "hi"}]}, "phone_state": {}}
    assert ev.write_ui_tree("ui_tree_after", tree) == "ui_tree_after.json"
    assert json.loads((ev.dir / "ui_tree_after.json").read_text()) == tree


def test_adopt_trajectory_moves_folder(tmp_path):
    src = tmp_path / "trajectories" / "20260101_000000_abcd"
    src.mkdir(parents=True)
    (src / "trajectory.json").write_text("[]")

    ev = EpisodeEvidence(tmp_path / "ep", "T", 0)
    assert ev.adopt_trajectory(src) == "trajectory"
    assert (ev.dir / "trajectory" / "trajectory.json").exists()
    assert not src.exists()


def test_write_conversation_serializes_chat_messages(tmp_path):
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    ev = EpisodeEvidence(tmp_path / "ep", "T", 0)
    messages = [ChatMessage(role=MessageRole.USER, content="open settings")]

    assert ev.write_conversation(messages) == "conversation.json"
    data = json.loads((ev.dir / "conversation.json").read_text())
    assert data == [{"role": "user", "content": "open settings"}]
    assert ev.write_conversation([]) is None


# ---------------------------------------------------------------------------
# End-to-end capture through run_androidworld_benchmark
# ---------------------------------------------------------------------------


class FakeTaskClient:
    def __init__(self, tasks=("ClockStopWatchRunning",), score=1.0):
        self.tasks = list(tasks)
        self.score = score
        self.calls = []

    async def wait_healthy(self, timeout, poll_interval=5.0):
        return True

    async def health(self):
        return True

    async def task_list(self):
        return list(self.tasks)

    async def task_length(self, task_type):
        return 1

    async def initialize_task(self, task_type, task_idx):
        self.calls.append(("initialize", task_type, task_idx))

    async def get_goal(self, task_type, task_idx):
        return f"goal for {task_type}"

    async def get_complexity(self, task_type, task_idx):
        return 1.0

    async def start_on_home_screen(self, task_type, task_idx):
        return True

    async def stabilize(self):
        pass

    async def get_score(self, task_type, task_idx):
        return self.score

    async def tear_down(self, task_type, task_idx):
        pass

    async def reset(self, go_home):
        pass

    async def close(self):
        pass


class FakeDriver:
    def __init__(self, url=None, **kwargs):
        self.trajectory = None

    async def connect(self):
        pass

    async def close(self):
        pass

    async def get_ui_tree(self):
        return {
            "a11y_tree": {"children": [{"text": "Stopwatch"}]},
            "phone_state": {"currentApp": "com.google.android.deskclock"},
        }


class FakeHandler:
    def __init__(self, result, events):
        self._result = result
        self._events = events

    async def stream_events(self):
        for event in self._events:
            yield event

    def __await__(self):
        async def _finish():
            return self._result

        return _finish().__await__()


def _fake_agent_class(tmp_path):
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    class FakeAgent:
        instances = []

        def __init__(
            self, goal=None, config=None, llms=None, driver=None, timeout=None, **kw
        ):
            self.shared_state = SimpleNamespace(
                success=True,
                message_history=[
                    ChatMessage(role=MessageRole.USER, content="system + state"),
                    ChatMessage(role=MessageRole.ASSISTANT, content="thought"),
                ],
            )
            # Point trajectory at a real folder so adopt_trajectory can move it.
            folder = tmp_path / "trajectories" / "20260101_000000_dead"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "trajectory.json").write_text("[]")
            (folder / "screenshots").mkdir(exist_ok=True)
            self.trajectory = SimpleNamespace(trajectory_folder=folder)
            FakeAgent.instances.append(self)

        def run(self):
            from mobilerun.agent.common.events import ToolExecutionEvent

            events = [
                _usage_event(request=100, response=20),
                ToolExecutionEvent(
                    tool_name="tap",
                    tool_args={"x": 5, "y": 6},
                    success=True,
                    summary="ok",
                ),
            ]
            return FakeHandler(
                SimpleNamespace(success=True, steps=2, reason="done"), events
            )

    return FakeAgent


def _patch(monkeypatch, client, agent_class):
    monkeypatch.setattr(bench, "AndroidWorldTaskClient", lambda *a, **k: client)
    monkeypatch.setattr(bench, "AndroidWorldHttpDriver", FakeDriver)
    monkeypatch.setattr(bench, "MobileAgent", agent_class)
    monkeypatch.setattr(
        bench.ConfigLoader,
        "load",
        lambda _path=None: MobileConfig.from_dict(
            {
                "agent": {"name": "mobilerun", "reasoning": False},
                "telemetry": {"enabled": False},
            }
        ),
    )
    monkeypatch.setattr(bench, "load_agent_llms", lambda config, **kwargs: {})


def test_evidence_bundle_is_written_and_referenced(monkeypatch, tmp_path):
    client = FakeTaskClient()
    agent_class = _fake_agent_class(tmp_path)
    _patch(monkeypatch, client, agent_class)
    output = tmp_path / "results.json"

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(output)
        )
    )
    assert success is True

    data = json.loads(output.read_text())
    assert data["capture_evidence"] is True
    entry = data["tasks"][0]
    assert entry["complexity"] == 1.0
    assert entry["max_steps"] == 10

    ev = Path(entry["evidence_dir"])
    assert ev.is_dir()
    assert (ev / "episode.json").is_file()
    assert (ev / "timeline.json").is_file()
    assert (ev / "conversation.json").is_file()
    assert (ev / "ui_tree_before.json").is_file()
    assert (ev / "ui_tree_after.json").is_file()
    assert (ev / "trajectory" / "trajectory.json").is_file()

    # The episode manifest carries run-level provenance so evidence and results
    # can be cross-checked in both directions.
    episode = json.loads((ev / "episode.json").read_text())
    run_block = episode["run"]
    assert run_block["run_id"] == entry["run_id"]
    assert run_block["server_url"] == "http://localhost:5000"
    assert run_block["benchmark"] == "androidworld"

    # result row references
    assert entry["trajectory_dir"] == "trajectory"
    assert entry["conversation_path"] == "conversation.json"
    assert entry["timeline_path"] == "timeline.json"
    assert set(entry["ui_tree_paths"]) == {"ui_tree_before.json", "ui_tree_after.json"}
    assert entry["usage_totals"]["total_tokens"] == 120

    # timeline captured the LLM turn + the tool call
    timeline = json.loads((ev / "timeline.json").read_text())
    kinds = [e["kind"] for e in timeline["events"]]
    assert "llm" in kinds and "tool" in kinds
    assert timeline["usage_totals"]["total_tokens"] == 120

    # conversation is the serialized message history
    conversation = json.loads((ev / "conversation.json").read_text())
    assert conversation[0] == {"role": "user", "content": "system + state"}

    # run manifest aggregates metadata + per-task + summary
    manifest = json.loads(output.read_text())
    assert manifest["summary"]["tasks"] == 1
    assert manifest["summary"]["total_tokens"] == 120


def test_evidence_can_be_disabled(monkeypatch, tmp_path):
    client = FakeTaskClient()
    agent_class = _fake_agent_class(tmp_path)
    _patch(monkeypatch, client, agent_class)
    output = tmp_path / "results.json"

    run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(output),
            capture_evidence=False,
        )
    )

    data = json.loads(output.read_text())
    assert data["capture_evidence"] is False
    assert data["evidence_root"] is None
    entry = data["tasks"][0]
    assert entry["evidence_dir"] == ""
    assert not (tmp_path / "results_evidence").exists()


if __name__ == "__main__":
    pytest.main([__file__])
