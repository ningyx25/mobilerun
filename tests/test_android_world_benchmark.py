"""Tests for the AndroidWorld benchmark runner.

The environment server is replaced by an ``httpx.MockTransport`` handler map
(for the task client) and by fake client/agent classes (for the orchestration
loop), so the tests exercise the real episode ordering, scoring/gating, task
selection, resume, and results IO without a docker container or an LLM.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from mobilerun.benchmark import android_world as bench
from mobilerun.cli import main as cli_main
from mobilerun.config_manager.config_manager import MobileConfig


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Task client (against httpx.MockTransport)
# ---------------------------------------------------------------------------


def _client_handlers() -> dict:
    return {
        ("GET", "/health"): lambda r: httpx.Response(200, json={"status": "success"}),
        ("GET", "/suite/task_list"): lambda r: httpx.Response(
            200, json={"task_list": ["A", "B"]}
        ),
        ("GET", "/suite/task_length"): lambda r: httpx.Response(
            200, json={"length": 2}
        ),
        ("POST", "/suite/reinitialize"): lambda r: httpx.Response(
            200, json={"status": "success"}
        ),
        ("POST", "/task/initialize"): lambda r: httpx.Response(
            200, json={"status": "success"}
        ),
        ("GET", "/task/goal"): lambda r: httpx.Response(
            200, json={"goal": "do the thing"}
        ),
        ("POST", "/task/complexity"): lambda r: httpx.Response(
            200, json={"complexity": 1.5}
        ),
        ("POST", "/task/start_on_home_screen"): lambda r: httpx.Response(
            200, json={"start_on_home_screen": False}
        ),
        ("GET", "/task/score"): lambda r: httpx.Response(200, json={"score": 0.75}),
        ("POST", "/task/tear_down"): lambda r: httpx.Response(
            200, json={"status": "success"}
        ),
        ("POST", "/reset"): lambda r: httpx.Response(200, json={"status": "success"}),
    }


def make_client(handlers=None):
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        responder = (handlers or _client_handlers()).get(
            (request.method, request.url.path)
        )
        if responder is None:
            return httpx.Response(
                404, json={"detail": f"no handler for {request.url.path}"}
            )
        return responder(request)

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )
    return client, recorded


def test_task_client_endpoints_and_http_methods():
    client, recorded = make_client()

    assert run(client.task_list()) == ["A", "B"]
    assert run(client.task_length("A")) == 2
    run(client.initialize_task("A", 0))
    assert run(client.get_goal("A", 0)) == "do the thing"
    assert run(client.get_complexity("A", 0)) == 1.5
    assert run(client.start_on_home_screen("A", 0)) is False
    assert run(client.get_score("A", 0)) == 0.75
    run(client.tear_down("A", 0))
    run(client.reset(go_home=True))

    methods = {(r.method, r.url.path) for r in recorded}
    # goal/score are GETs; lifecycle calls are POSTs.
    assert ("GET", "/task/goal") in methods
    assert ("GET", "/task/score") in methods
    assert ("POST", "/task/initialize") in methods
    assert ("POST", "/task/tear_down") in methods
    assert ("POST", "/reset") in methods
    reset_request = next(r for r in recorded if r.url.path == "/reset")
    assert reset_request.url.params["go_home"] == "true"


def test_task_client_retries_transient_5xx(monkeypatch):
    monkeypatch.setattr(bench, "_RETRY_BACKOFF_S", 0.0)
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        if len(attempts) < 3:
            return httpx.Response(503, json={"detail": "busy"})
        return httpx.Response(200, json={"task_list": ["A"]})

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )

    assert run(client.task_list()) == ["A"]
    assert len(attempts) == 3


def test_task_client_connection_error_raises_connection_error(monkeypatch):
    monkeypatch.setattr(bench, "_RETRY_BACKOFF_S", 0.0)

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ConnectionError):
        run(client.task_list())


def test_task_client_http_error_includes_detail(monkeypatch):
    monkeypatch.setattr(bench, "_RETRY_BACKOFF_S", 0.0)

    def handler(request):
        return httpx.Response(422, json={"detail": "task_idx out of range"})

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(RuntimeError, match="task_idx out of range"):
        run(client.initialize_task("A", 99))


def test_wait_healthy_polls_until_ready():
    responses = [503, 503, 200]

    def handler(request):
        return httpx.Response(responses.pop(0), json={"status": "error"})

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )

    assert run(client.wait_healthy(timeout=5, poll_interval=0)) is True


def test_wait_healthy_times_out():
    def handler(request):
        return httpx.Response(503, json={"status": "error"})

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )

    assert run(client.wait_healthy(timeout=0, poll_interval=0)) is False


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


def test_select_tasks_unknown_name_suggests_close_match():
    with pytest.raises(ValueError, match="did you mean"):
        bench._select_tasks(
            ["ClockStopWatchRunning", "SystemBluetoothTurnOn"],
            tasks="ClockStopWatchRunn",
        )


def test_select_tasks_range_is_one_based_inclusive():
    available = ["A", "B", "C", "D", "E"]
    assert bench._select_tasks(available, task_index_range="2-4") == ["B", "C", "D"]
    with pytest.raises(ValueError, match="exceeds"):
        bench._select_tasks(available, task_index_range="1-9")
    with pytest.raises(ValueError, match="START-END"):
        bench._select_tasks(available, task_index_range="abc")


def test_select_tasks_range_intersects_with_tasks():
    available = ["A", "B", "C", "D", "E"]
    assert bench._select_tasks(available, tasks="A,B,C", task_index_range="3-5") == [
        "C"
    ]


def test_select_tasks_difficulty_filter():
    available = ["A", "B", "C"]
    metadata = {
        "A": {"difficulty": "easy"},
        "B": {"difficulty": "hard"},
        "C": {"difficulty": "easy"},
    }
    assert bench._select_tasks(available, difficulty=["easy"], metadata=metadata) == [
        "A",
        "C",
    ]


def test_bundled_metadata_matches_androidworld_task_list():
    metadata = bench.load_task_metadata()
    assert len(metadata) == 116
    assert sum(1 for entry in metadata.values() if entry["difficulty"] == "easy") == 61
    assert (
        sum(1 for entry in metadata.values() if entry["difficulty"] == "medium") == 36
    )
    assert sum(1 for entry in metadata.values() if entry["difficulty"] == "hard") == 19


# ---------------------------------------------------------------------------
# Results IO
# ---------------------------------------------------------------------------


def test_results_round_trip(tmp_path):
    path = tmp_path / "results.json"
    results = {
        "A#0": bench.TaskResult(
            task_type="A", score=1.0, gated_score=1.0, finished=True
        ),
        "B#0": bench.TaskResult(task_type="B", error="boom"),
    }

    bench._write_results(path, {"benchmark": "androidworld"}, results)
    loaded = bench._load_results(path)

    assert set(loaded) == {"A#0", "B#0"}
    assert loaded["A#0"].score == 1.0
    assert loaded["B#0"].error == "boom"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class FakeTaskClient:
    """Records lifecycle calls; serves canned task data."""

    def __init__(self, tasks=("ClockStopWatchRunning",), score=1.0, healthy=True):
        self.tasks = list(tasks)
        self.score = score
        self.healthy = healthy
        self.calls = []

    async def wait_healthy(self, timeout, poll_interval=5.0):
        self.calls.append(("wait_healthy",))
        return self.healthy

    async def health(self):
        return self.healthy

    async def task_list(self):
        return list(self.tasks)

    async def task_length(self, task_type):
        return 2

    async def reinitialize_suite(self, **kwargs):
        self.calls.append(("reinitialize", kwargs))

    async def initialize_task(self, task_type, task_idx):
        self.calls.append(("initialize", task_type, task_idx))

    async def get_goal(self, task_type, task_idx):
        self.calls.append(("goal", task_type, task_idx))
        return f"goal for {task_type}"

    async def get_complexity(self, task_type, task_idx):
        self.calls.append(("complexity", task_type, task_idx))
        return 1.0

    async def start_on_home_screen(self, task_type, task_idx):
        self.calls.append(("start_on_home_screen", task_type, task_idx))
        return True

    async def get_score(self, task_type, task_idx):
        self.calls.append(("score", task_type, task_idx))
        return self.score

    async def tear_down(self, task_type, task_idx):
        self.calls.append(("tear_down", task_type, task_idx))

    async def reset(self, go_home):
        self.calls.append(("reset", go_home))

    async def close(self):
        self.calls.append(("close",))


class FakeDriver:
    instances = []

    def __init__(self, url=None, **kwargs):
        self.url = url
        self.connected = False
        self.closed = False
        FakeDriver.instances.append(self)

    async def connect(self):
        self.connected = True

    async def close(self):
        self.closed = True


class FakeHandler:
    def __init__(self, result, events=()):
        self._result = result
        self._events = list(events)

    async def stream_events(self):
        for event in self._events:
            yield event

    def __await__(self):
        async def _finish():
            return self._result

        return _finish().__await__()


def make_fake_agent_class(
    *, shared_success=True, success=True, steps=5, reason="done", events=()
):
    """Fake MobileAgent.

    ``shared_success`` models ``shared_state.success`` — the direct-mode
    completion signal (None means complete() never ran); ``events`` are
    yielded from the stream (reasoning mode's manager-answer signal).
    """

    class FakeAgent:
        instances = []

        def __init__(
            self, goal=None, config=None, llms=None, driver=None, timeout=None, **kwargs
        ):
            self.goal = goal
            self.config = config
            self.driver = driver
            self.timeout = timeout
            self.shared_state = SimpleNamespace(success=shared_success)
            FakeAgent.instances.append(self)

        def run(self):
            return FakeHandler(
                SimpleNamespace(success=success, steps=steps, reason=reason),
                events=events,
            )

    return FakeAgent


def patch_benchmark(monkeypatch, client, agent_class, *, llms=None, reasoning=False):
    monkeypatch.setattr(bench, "AndroidWorldTaskClient", lambda *a, **k: client)
    monkeypatch.setattr(bench, "AndroidWorldHttpDriver", FakeDriver)
    monkeypatch.setattr(bench, "MobileAgent", agent_class)
    monkeypatch.setattr(
        bench.ConfigLoader,
        "load",
        lambda _path=None: MobileConfig.from_dict(
            {
                "agent": {"name": "external-agent", "reasoning": reasoning},
                "telemetry": {"enabled": False},
            }
        ),
    )
    monkeypatch.setattr(bench, "load_agent_llms", lambda config, **kwargs: llms or {})
    FakeDriver.instances.clear()


def test_episode_order_and_gated_score(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",), score=1.0)
    agent_class = make_fake_agent_class(shared_success=True)
    patch_benchmark(monkeypatch, client, agent_class)
    output = tmp_path / "results.json"

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(output)
        )
    )

    assert success is True
    lifecycle = [call for call in client.calls if call[0] != "close"]
    assert lifecycle == [
        ("wait_healthy",),
        ("initialize", "ClockStopWatchRunning", 0),
        ("goal", "ClockStopWatchRunning", 0),
        ("complexity", "ClockStopWatchRunning", 0),
        ("start_on_home_screen", "ClockStopWatchRunning", 0),
        ("reset", True),
        ("score", "ClockStopWatchRunning", 0),
        ("tear_down", "ClockStopWatchRunning", 0),
    ]
    # score must be read before tear_down
    assert lifecycle.index(("score", "ClockStopWatchRunning", 0)) < lifecycle.index(
        ("tear_down", "ClockStopWatchRunning", 0)
    )

    # The agent ran with the canonical step budget (10 x complexity).
    agent = agent_class.instances[0]
    assert agent.goal == "goal for ClockStopWatchRunning"
    assert agent.config.agent.max_steps == 10
    assert agent.driver is FakeDriver.instances[0]

    data = json.loads(output.read_text())
    entry = data["tasks"][0]
    assert entry["task_type"] == "ClockStopWatchRunning"
    assert entry["score"] == 1.0
    assert entry["gated_score"] == 1.0
    assert entry["finished"] is True
    assert entry["steps"] == 5
    assert entry["error"] is None
    # shared driver closed at the end
    assert FakeDriver.instances[0].closed is True


def test_gated_score_is_zero_when_agent_did_not_finish(monkeypatch, tmp_path):
    """Direct mode: complete() never ran (shared_state.success stays None)."""
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",), score=1.0)
    agent_class = make_fake_agent_class(
        shared_success=None, success=False, reason="Reached maximum steps (10)"
    )
    patch_benchmark(monkeypatch, client, agent_class)
    output = tmp_path / "results.json"

    run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(output)
        )
    )

    entry = json.loads(output.read_text())["tasks"][0]
    assert entry["score"] == 1.0
    assert entry["gated_score"] == 0.0
    assert entry["finished"] is False


def test_reasoning_mode_gating_uses_manager_answer(monkeypatch, tmp_path):
    """Reasoning mode: the manager finalizing with an answer marks the
    episode done; shared_state.success (set by the executor's complete) is
    not the completion signal there."""
    from mobilerun.agent.droid.events import ManagerPlanEvent

    plan_event = ManagerPlanEvent(
        plan="", current_subgoal="", thought="", answer="Task done", success=True
    )
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",), score=1.0)
    agent_class = make_fake_agent_class(shared_success=None, events=(plan_event,))
    patch_benchmark(monkeypatch, client, agent_class, reasoning=True)

    run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(tmp_path / "r.json")
        )
    )

    entry = json.loads((tmp_path / "r.json").read_text())["tasks"][0]
    assert entry["finished"] is True
    assert entry["gated_score"] == 1.0


def test_reasoning_mode_without_manager_answer_is_not_done(monkeypatch, tmp_path):
    from mobilerun.agent.droid.events import ManagerPlanEvent

    # An intermediate plan (empty answer) is not a completion signal.
    plan_event = ManagerPlanEvent(
        plan="open settings", current_subgoal="tap", thought="", answer="", success=None
    )
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",), score=1.0)
    agent_class = make_fake_agent_class(shared_success=None, events=(plan_event,))
    patch_benchmark(monkeypatch, client, agent_class, reasoning=True)

    run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(tmp_path / "r.json")
        )
    )

    entry = json.loads((tmp_path / "r.json").read_text())["tasks"][0]
    assert entry["finished"] is False
    assert entry["gated_score"] == 0.0


def test_steps_zero_is_honored(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    agent_class = make_fake_agent_class()
    patch_benchmark(monkeypatch, client, agent_class)

    run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(tmp_path / "r.json"),
            steps=0,
        )
    )

    assert agent_class.instances[0].config.agent.max_steps == 0


def test_invalid_server_url_fails_before_health_wait(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    patch_benchmark(monkeypatch, client, make_fake_agent_class())

    success = run(
        bench.run_androidworld_benchmark(
            server_url="localhost:5000", output=str(tmp_path / "r.json")
        )
    )

    assert success is False
    assert client.calls == []


def test_external_agent_config_is_rejected(monkeypatch, tmp_path):
    """External agents bypass the injected driver and drive adb directly, so
    runbench must refuse them up front instead of failing every task."""
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    patch_benchmark(monkeypatch, client, make_fake_agent_class())
    # patch_benchmark builds the config with agent name "external-agent".
    monkeypatch.setattr(bench, "list_agents", lambda: ["external-agent"])

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(tmp_path / "r.json")
        )
    )

    assert success is False
    assert client.calls == []


def test_invalid_task_index_range_fails_before_health_wait(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    patch_benchmark(monkeypatch, client, make_fake_agent_class())

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(tmp_path / "r.json"),
            task_index_range="59-",
        )
    )

    assert success is False
    assert client.calls == []


def test_wrapped_agent_failure_probes_health_and_aborts_when_unhealthy(
    monkeypatch, tmp_path
):
    """A non-ConnectionError task failure (e.g. the state provider's wrapped
    'Failed to get state' exception) still triggers the health probe; an
    unhealthy server aborts the run."""

    class UnhealthyAfterStart(FakeTaskClient):
        def __init__(self):
            super().__init__(tasks=("ClockStopWatchRunning", "SystemBluetoothTurnOn"))
            self.health_calls = 0

        async def health(self):
            self.health_calls += 1
            return False

        async def wait_healthy(self, timeout, poll_interval=5.0):
            self.health_calls += 1
            return self.health_calls == 1  # startup wait passes, recovery fails

    class ExplodingAgent:
        def __init__(self, **kwargs):
            self.shared_state = SimpleNamespace(success=None)

        def run(self):
            raise RuntimeError("Failed to get state after 7 attempts")

    client = UnhealthyAfterStart()
    patch_benchmark(monkeypatch, client, ExplodingAgent)
    output = tmp_path / "r.json"

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(output), health_timeout=0
        )
    )

    assert success is False
    assert len(json.loads(output.read_text())["tasks"]) == 1


def test_steps_override_replaces_complexity_budget(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    agent_class = make_fake_agent_class()
    patch_benchmark(monkeypatch, client, agent_class)

    run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(tmp_path / "r.json"),
            steps=3,
        )
    )

    assert agent_class.instances[0].config.agent.max_steps == 3


def test_unhealthy_server_returns_false(monkeypatch, tmp_path):
    client = FakeTaskClient(healthy=False)
    patch_benchmark(monkeypatch, client, make_fake_agent_class())

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(tmp_path / "r.json"),
            health_timeout=0,
        )
    )

    assert success is False
    assert ("close",) in client.calls


def test_unknown_task_name_returns_false(monkeypatch, tmp_path, capsys):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    patch_benchmark(monkeypatch, client, make_fake_agent_class())

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(tmp_path / "r.json"),
            tasks="NotARealTask",
        )
    )

    assert success is False


def test_agent_exception_records_error_and_still_tears_down(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))

    class ExplodingAgent:
        instances = []

        def __init__(self, **kwargs):
            self.shared_state = SimpleNamespace(finished=False)
            ExplodingAgent.instances.append(self)

        def run(self):
            raise RuntimeError("agent blew up")

    patch_benchmark(monkeypatch, client, ExplodingAgent)

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(tmp_path / "r.json")
        )
    )

    # A single task failure does not fail the run; teardown still happens.
    assert success is True
    assert ("tear_down", "ClockStopWatchRunning", 0) in client.calls
    entry = json.loads((tmp_path / "r.json").read_text())["tasks"][0]
    assert "agent blew up" in entry["error"]
    assert entry["score"] is None


def test_connection_failure_waits_then_aborts(monkeypatch, tmp_path):
    class FlakyClient(FakeTaskClient):
        def __init__(self):
            super().__init__(tasks=("ClockStopWatchRunning", "SystemBluetoothTurnOn"))
            self.health_calls = 0

        async def get_score(self, task_type, task_idx):
            raise ConnectionError("server went away")

        async def wait_healthy(self, timeout, poll_interval=5.0):
            self.health_calls += 1
            # First call is the startup wait; the recovery wait fails.
            return self.health_calls == 1

    client = FlakyClient()
    patch_benchmark(monkeypatch, client, make_fake_agent_class())
    output = tmp_path / "r.json"

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(output), health_timeout=0
        )
    )

    # The run aborts on the first unrecoverable connection failure and keeps
    # the partial results.
    assert success is False
    data = json.loads(output.read_text())
    assert len(data["tasks"]) == 1
    assert "ConnectionError" in data["tasks"][0]["error"]


def test_quiet_logging_raises_threshold(monkeypatch):
    import logging

    bench._setup_benchmark_logging(debug=False, verbose=False)
    assert logging.getLogger("mobilerun").level == logging.ERROR
    bench._setup_benchmark_logging(debug=False, verbose=True)
    assert logging.getLogger("mobilerun").level == logging.INFO


def test_resume_skips_recorded_and_reruns_failed(monkeypatch, tmp_path):
    output = tmp_path / "results.json"
    bench._write_results(
        output,
        {"benchmark": "androidworld"},
        {
            "ClockStopWatchRunning#0": bench.TaskResult(
                task_type="ClockStopWatchRunning", score=1.0, gated_score=1.0
            ),
            "SystemBluetoothTurnOn#0": bench.TaskResult(
                task_type="SystemBluetoothTurnOn", error="boom"
            ),
        },
    )
    client = FakeTaskClient(tasks=("ClockStopWatchRunning", "SystemBluetoothTurnOn"))
    patch_benchmark(monkeypatch, client, make_fake_agent_class())

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(output),
            resume=True,
        )
    )

    assert success is True
    initialized = [call for call in client.calls if call[0] == "initialize"]
    assert initialized == [("initialize", "SystemBluetoothTurnOn", 0)]


def test_dry_run_lists_tasks_without_running_agent(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    agent_class = make_fake_agent_class()
    patch_benchmark(monkeypatch, client, agent_class)
    output = tmp_path / "nested" / "results.json"

    success = run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000", output=str(output), dry_run=True
        )
    )

    assert success is True
    assert agent_class.instances == []
    assert not output.exists()
    assert not output.parent.exists()
    assert [call for call in client.calls if call[0] == "initialize"] == []


def test_suite_reinitialize_called_when_seed_given(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("ClockStopWatchRunning",))
    patch_benchmark(monkeypatch, client, make_fake_agent_class())

    run(
        bench.run_androidworld_benchmark(
            server_url="http://localhost:5000",
            output=str(tmp_path / "r.json"),
            seed=30,
            n_task_combinations=1,
        )
    )

    reinit = [call for call in client.calls if call[0] == "reinitialize"]
    assert reinit == [("reinitialize", {"n_task_combinations": 1, "seed": 30})]


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_runbench_cli_passes_options(monkeypatch):
    captured = {}

    async def fake_runner(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(cli_main, "run_androidworld_benchmark", fake_runner)

    from click.testing import CliRunner

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "runbench",
            "--benchmark-name",
            "androidworld",
            "-d",
            "http://localhost:5001",
            "--tasks",
            "A,B",
            "--steps",
            "7",
            "--difficulty",
            "easy",
            "--resume",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["server_url"] == "http://localhost:5001"
    assert captured["tasks"] == "A,B"
    assert captured["steps"] == 7
    assert captured["difficulty"] == ("easy",)
    assert captured["resume"] is True
    assert captured["dry_run"] is True


def test_runbench_cli_mobileworld_not_implemented():
    from click.testing import CliRunner

    result = CliRunner().invoke(
        cli_main.cli, ["runbench", "--benchmark-name", "mobileworld"]
    )

    assert result.exit_code == 1
    assert "not implemented" in result.output


def test_runbench_cli_help_lists_options():
    from click.testing import CliRunner

    result = CliRunner().invoke(cli_main.cli, ["runbench", "--help"])

    assert result.exit_code == 0
    for option in ("--task-index-range", "--difficulty", "--resume", "--dry-run"):
        assert option in result.output


if __name__ == "__main__":
    pytest.main([__file__])
