"""Tests for the MobileWorld benchmark runner.

The MobileWorld server is replaced by an ``httpx.MockTransport`` handler map
(for the task client) and by fake client/driver/agent classes (for the
orchestration loop), so the tests exercise the real task protocol, the
pure-GUI filtering, and the episode ordering without a docker container or an
LLM.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from mobilerun.benchmark import mobile_world as bench
from mobilerun.config_manager.config_manager import MobileConfig


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Task client (against httpx.MockTransport)
# ---------------------------------------------------------------------------


def _client_handlers():
    return {
        ("GET", "/health"): lambda r: httpx.Response(200, json={"ok": True}),
        ("GET", "/task/list"): lambda r: httpx.Response(
            200,
            json=[
                {"name": "GuiTask", "tags": ["lang-en"], "apps": ["Files"]},
                {"name": "McpTask", "tags": ["agent-mcp"], "apps": ["Messages"]},
            ],
        ),
        ("GET", "/task/goal"): lambda r: httpx.Response(200, json="do the thing"),
        ("POST", "/task/init"): lambda r: httpx.Response(200, json="OK"),
        ("GET", "/task/eval"): lambda r: httpx.Response(
            200, json={"score": 0.75, "reason": "close enough"}
        ),
        ("POST", "/task/tear_down"): lambda r: httpx.Response(200, json="OK"),
    }


def make_client(handlers=None):
    recorded = []

    def handler(request):
        recorded.append(request)
        responder = (handlers or _client_handlers()).get(
            (request.method, request.url.path)
        )
        if responder is None:
            return httpx.Response(
                404, json={"detail": f"no handler {request.url.path}"}
            )
        return responder(request)

    client = bench.MobileWorldTaskClient(
        "http://localhost:6800", "emulator-5554", transport=httpx.MockTransport(handler)
    )
    return client, recorded


def test_task_client_endpoints_and_methods():
    client, recorded = make_client()

    assert run(client.task_list())[0]["name"] == "GuiTask"
    assert run(client.get_goal("GuiTask")) == "do the thing"
    run(client.initialize("GuiTask"))
    assert run(client.evaluate("GuiTask")) == (0.75, "close enough")
    run(client.tear_down("GuiTask"))

    methods = {(r.method, r.url.path) for r in recorded}
    assert ("GET", "/task/list") in methods
    assert ("POST", "/task/init") in methods
    assert ("POST", "/task/tear_down") in methods
    # /task/eval is a GET (the server binds the request model from its body).
    assert ("GET", "/task/eval") in methods
    assert ("POST", "/task/eval") not in methods

    init_body = json.loads(
        next(r for r in recorded if r.url.path == "/task/init").content
    )
    assert init_body == {"task_name": "GuiTask", "req_device": "emulator-5554"}
    eval_body = json.loads(
        next(r for r in recorded if r.url.path == "/task/eval").content
    )
    assert eval_body == {"task_name": "GuiTask", "req_device": "emulator-5554"}


def test_evaluate_defaults_reason_when_absent():
    handlers = _client_handlers()
    handlers[("GET", "/task/eval")] = lambda r: httpx.Response(200, json={"score": 1.0})
    client, _ = make_client(handlers)

    score, reason = run(client.evaluate("GuiTask"))
    assert score == 1.0
    assert "GuiTask" in reason


def test_health_and_wait_healthy():
    handlers = _client_handlers()
    handlers[("GET", "/health")] = lambda r: httpx.Response(
        503, json={"ok": False, "device_status": {"emulator-5554": False}}
    )
    client, _ = make_client(handlers)

    assert run(client.health()) is False
    assert run(client.wait_healthy(timeout=0, poll_interval=0)) is False


def test_client_connection_error_raises(monkeypatch):
    monkeypatch.setattr(bench, "_RETRY_BACKOFF_S", 0.0)

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = bench.MobileWorldTaskClient(
        "http://localhost:6800", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ConnectionError):
        run(client.task_list())


# ---------------------------------------------------------------------------
# Task selection (pure-GUI default)
# ---------------------------------------------------------------------------


def _entries():
    return [
        {"name": "GuiA", "tags": ["lang-en"], "apps": []},
        {"name": "McpB", "tags": ["agent-mcp"], "apps": []},
        {"name": "UserC", "tags": ["agent-user-interaction"], "apps": []},
        {"name": "BothD", "tags": ["agent-mcp", "agent-user-interaction"], "apps": []},
        {"name": "GuiE", "tags": ["lang-en"], "apps": []},
    ]


def test_default_selection_is_pure_gui_only():
    assert bench._select_mobile_world_tasks(_entries()) == ["GuiA", "GuiE"]


def test_enable_flags_admit_the_corresponding_groups():
    # --enable-mcp alone admits agent-mcp tasks; BothD stays out because it is
    # also agent-user-interaction.
    assert bench._select_mobile_world_tasks(_entries(), enable_mcp=True) == [
        "GuiA",
        "McpB",
        "GuiE",
    ]
    assert bench._select_mobile_world_tasks(
        _entries(), enable_mcp=True, enable_user_interaction=True
    ) == ["GuiA", "McpB", "UserC", "BothD", "GuiE"]


def test_selection_range_is_one_based_over_the_filtered_list():
    assert bench._select_mobile_world_tasks(_entries(), task_index_range="2-2") == [
        "GuiE"
    ]


def test_selection_explicit_tasks_and_unknown_names():
    assert bench._select_mobile_world_tasks(_entries(), tasks="GuiE") == ["GuiE"]
    with pytest.raises(ValueError, match="did you mean"):
        bench._select_mobile_world_tasks(_entries(), tasks="GuiF")


def test_selection_rejects_gated_tasks_with_a_helpful_error():
    with pytest.raises(ValueError, match="--enable-mcp"):
        bench._select_mobile_world_tasks(_entries(), tasks="McpB")


def test_selection_ignores_difficulty():
    # MobileWorld has no difficulty field; passing one must not raise.
    assert bench._select_mobile_world_tasks(_entries(), difficulty=["easy"]) == [
        "GuiA",
        "GuiE",
    ]


def test_selection_empty_raises():
    # GuiA is in the GUI base but excluded by the range -> empty intersection.
    with pytest.raises(ValueError, match="empty"):
        bench._select_mobile_world_tasks(
            _entries(), tasks="GuiA", task_index_range="2-2"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class FakeTaskClient:
    def __init__(self, tasks=("GuiTask",), score=1.0, reason="ok"):
        self.tasks = [{"name": name, "tags": ["lang-en"], "apps": []} for name in tasks]
        self.score = score
        self.reason = reason
        self.calls = []

    async def wait_healthy(self, timeout, poll_interval=5.0):
        self.calls.append(("wait_healthy",))
        return True

    async def health(self):
        return True

    async def task_list(self):
        return list(self.tasks)

    async def get_goal(self, task_name):
        self.calls.append(("goal", task_name))
        return f"goal for {task_name}"

    async def initialize(self, task_name):
        self.calls.append(("initialize", task_name))

    async def evaluate(self, task_name):
        self.calls.append(("evaluate", task_name))
        return self.score, self.reason

    async def tear_down(self, task_name):
        self.calls.append(("tear_down", task_name))

    async def close(self):
        self.calls.append(("close",))


class FakeDriver:
    instances = []

    def __init__(self, url=None, device=None, **kwargs):
        self.url = url
        self.device = device
        self.answers = []
        self.closed = False
        FakeDriver.instances.append(self)

    async def connect(self):
        pass

    async def close(self):
        self.closed = True

    async def get_ui_tree(self):
        return {"a11y_tree": {"children": [{"text": "x"}]}, "phone_state": {}}

    async def submit_answer(self, text):
        self.answers.append(text)


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
    *, shared_success=True, success=True, steps=3, reason="done", answer="313"
):
    class FakeAgent:
        instances = []

        def __init__(
            self, goal=None, config=None, llms=None, driver=None, timeout=None, **kw
        ):
            self.goal = goal
            self.config = config
            self.driver = driver
            self.shared_state = SimpleNamespace(success=shared_success, answer=answer)
            FakeAgent.instances.append(self)

        def run(self):
            return FakeHandler(
                SimpleNamespace(success=success, steps=steps, reason=reason)
            )

    return FakeAgent


def patch_benchmark(monkeypatch, client, agent_class, *, reasoning=False):
    monkeypatch.setattr(bench, "MobileWorldTaskClient", lambda *a, **k: client)
    monkeypatch.setattr(bench, "MobileWorldHttpDriver", FakeDriver)
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
    monkeypatch.setattr(bench, "load_agent_llms", lambda config, **kwargs: {})
    FakeDriver.instances.clear()


def test_episode_order_and_answer_before_evaluate(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("GuiTask",), score=1.0)
    agent_class = make_fake_agent_class(shared_success=True, answer="313")
    patch_benchmark(monkeypatch, client, agent_class)
    output = tmp_path / "results.json"

    success = run(
        bench.run_mobileworld_benchmark(
            server_url="http://localhost:6800", output=str(output)
        )
    )

    assert success is True
    lifecycle = [c for c in client.calls if c[0] != "close"]
    assert lifecycle == [
        ("wait_healthy",),
        ("initialize", "GuiTask"),
        ("goal", "GuiTask"),
        ("evaluate", "GuiTask"),
        ("tear_down", "GuiTask"),
    ]

    driver = FakeDriver.instances[0]
    assert driver.answers == ["313"]
    # Answer must be submitted (via the driver) before the score is read.
    assert driver.answers  # submitted
    assert lifecycle.index(("evaluate", "GuiTask")) > 1  # after init/goal

    data = json.loads(output.read_text())
    assert data["benchmark"] == "mobileworld"
    assert data["server_url"] == "http://localhost:6800"
    assert data["device_id"] == "emulator-5554"
    assert len(data["runs"]) == 1
    entry = data["tasks"][0]
    assert entry["task_type"] == "GuiTask"
    assert entry["score"] == 1.0
    assert entry["gated_score"] == 1.0
    assert entry["finished"] is True
    assert entry["eval_reason"] == "ok"
    assert entry["server_url"] == "http://localhost:6800"
    assert entry["run_id"] == data["runs"][0]["run_id"]
    assert "summary" in data


def test_unfinished_episode_submits_no_answer_and_gates_to_zero(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("GuiTask",), score=1.0)
    agent_class = make_fake_agent_class(
        shared_success=None, success=False, reason="max steps"
    )
    patch_benchmark(monkeypatch, client, agent_class)

    run(
        bench.run_mobileworld_benchmark(
            server_url="http://localhost:6800", output=str(tmp_path / "r.json")
        )
    )

    assert FakeDriver.instances[0].answers == []
    entry = json.loads((tmp_path / "r.json").read_text())["tasks"][0]
    assert entry["score"] == 1.0
    assert entry["gated_score"] == 0.0
    assert entry["finished"] is False


def test_steps_override_sets_max_steps(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("GuiTask",))
    agent_class = make_fake_agent_class()
    patch_benchmark(monkeypatch, client, agent_class)

    run(
        bench.run_mobileworld_benchmark(
            server_url="http://localhost:6800", output=str(tmp_path / "r.json"), steps=7
        )
    )
    assert agent_class.instances[0].config.agent.max_steps == 7


def test_dry_run_lists_without_running(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("GuiTask",))
    agent_class = make_fake_agent_class()
    patch_benchmark(monkeypatch, client, agent_class)
    output = tmp_path / "nested" / "r.json"

    success = run(
        bench.run_mobileworld_benchmark(
            server_url="http://localhost:6800", output=str(output), dry_run=True
        )
    )
    assert success is True
    assert agent_class.instances == []
    assert not output.exists()
    assert ("initialize", "GuiTask") not in client.calls


def test_default_filter_runs_only_gui_tasks(monkeypatch, tmp_path):
    client = FakeTaskClient(tasks=("GuiTask", "McpTask"))
    client.tasks[1]["tags"] = ["agent-mcp"]
    agent_class = make_fake_agent_class()
    patch_benchmark(monkeypatch, client, agent_class)

    run(
        bench.run_mobileworld_benchmark(
            server_url="http://localhost:6800", output=str(tmp_path / "r.json")
        )
    )
    initialized = [c for c in client.calls if c[0] == "initialize"]
    assert initialized == [("initialize", "GuiTask")]


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_runbench_routes_mobileworld(monkeypatch):
    captured = {}

    async def fake_runner(**kwargs):
        captured.update(kwargs)
        return True

    from mobilerun.cli import main as cli_main

    monkeypatch.setattr(cli_main, "run_mobileworld_benchmark", fake_runner)

    from click.testing import CliRunner

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "runbench",
            "--benchmark-name",
            "mobileworld",
            "-d",
            "http://localhost:6800",
            "--tasks",
            "GuiTask",
            "--steps",
            "9",
            "--enable-mcp",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["server_url"] == "http://localhost:6800"
    assert captured["tasks"] == "GuiTask"
    assert captured["steps"] == 9
    assert captured["enable_mcp"] is True
    assert captured["dry_run"] is True


def test_runbench_mobileworld_help_lists_options():
    from click.testing import CliRunner

    from mobilerun.cli import main as cli_main

    result = CliRunner().invoke(cli_main.cli, ["runbench", "--help"])

    assert result.exit_code == 0
    for option in ("--enable-mcp", "--enable-user-interaction", "--device-id"):
        assert option in result.output


if __name__ == "__main__":
    pytest.main([__file__])
