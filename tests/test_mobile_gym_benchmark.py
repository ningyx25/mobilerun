"""Tests for the MobileGym benchmark runner and browser driver.

Everything external is faked: ``bench_env`` is injected as a fake
``SimpleNamespace``, the Playwright env is a stub, and the LLM-backed
``MobileAgent`` is replaced with a scripted workflow handler. No simulator,
no Chromium and no LLM are needed.
"""

import asyncio
import json
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

from mobilerun.benchmark import mobile_gym as bench
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.tools.driver.mobile_gym import (
    MOBILE_GYM_CONNECTION,
    MobileGymBrowserDriver,
    validate_mobile_gym_url,
)


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# URL / constants
# ---------------------------------------------------------------------------


def test_validate_mobile_gym_url():
    assert validate_mobile_gym_url("https://localhost:4180/") == (
        "https://localhost:4180"
    )
    assert validate_mobile_gym_url("http://localhost:4173") == ("http://localhost:4173")
    with pytest.raises(ValueError):
        validate_mobile_gym_url("localhost:4180")
    with pytest.raises(ValueError):
        validate_mobile_gym_url("ftp://localhost:4180")


def test_connection_constant_is_registered():
    from mobilerun.agent.droid import droid_agent

    assert MOBILE_GYM_CONNECTION == "mobile-gym"
    assert droid_agent.MOBILE_GYM_CONNECTION == MOBILE_GYM_CONNECTION


# ---------------------------------------------------------------------------
# bench_env Action construction (fake module injected into sys.modules)
# ---------------------------------------------------------------------------


def _install_fake_bench_env_base(monkeypatch):
    class ActionType(str, Enum):
        CLICK = "CLICK"

        SWIPE = "SWIPE"
        DRAG = "DRAG"
        LONG_PRESS = "LONG_PRESS"
        TYPE = "TYPE"
        BACK = "BACK"
        HOME = "HOME"
        ENTER = "ENTER"
        AWAKE = "AWAKE"

    @dataclass
    class Action:
        action_type: ActionType
        data: dict

        def __post_init__(self):
            if not isinstance(self.action_type, ActionType):
                raise TypeError("action_type must be ActionType")

    base = types.ModuleType("bench_env.env.base")
    base.Action = Action
    base.ActionType = ActionType
    env = types.ModuleType("bench_env.env")
    env.base = base
    pkg = types.ModuleType("bench_env")
    pkg.env = env
    monkeypatch.setitem(sys.modules, "bench_env", pkg)
    monkeypatch.setitem(sys.modules, "bench_env.env", env)
    monkeypatch.setitem(sys.modules, "bench_env.env.base", base)
    return ActionType


class RecordingEnv:
    def __init__(self):
        self.actions = []
        self.url = "https://localhost:4180"
        self.APP_NAME_MAP = {"微信": "wechat", "WeChat": "wechat", "设置": "settings"}

    async def step(self, action):
        self.actions.append(action)

    def get_device_size(self):
        return (1080, 2400)


def _make_driver(monkeypatch, env=None):
    action_types = _install_fake_bench_env_base(monkeypatch)
    env = env or RecordingEnv()
    return MobileGymBrowserDriver(env=env), env, action_types


def test_driver_translates_actions(monkeypatch):
    driver, env, at = _make_driver(monkeypatch)
    run(driver.connect())

    run(driver.tap(120, 240))
    run(driver.swipe(10, 20, 30, 40, duration_ms=300))
    run(driver.swipe(50, 60, 50, 60, duration_ms=1000))  # long_press
    run(driver.drag(1, 2, 3, 4, duration=1.5))
    run(driver.input_text("hello", clear=True))
    run(driver.press_button("back"))
    run(driver.press_key_code(66))
    run(driver.start_app("wechat"))

    assert [(a.action_type, a.data) for a in env.actions] == [
        (at.CLICK, {"point": [120, 240]}),
        (at.SWIPE, {"point1": [10, 20], "point2": [30, 40], "duration": 300}),
        (at.LONG_PRESS, {"point": [50, 60], "duration": 1000}),
        (at.DRAG, {"point1": [1, 2], "point2": [3, 4], "duration": 1500}),
        (at.TYPE, {"value": "hello", "clear": True}),
        (at.BACK, {}),
        (at.ENTER, {}),
        (at.AWAKE, {"value": "wechat"}),
    ]


def test_driver_apps_and_unsupported_verbs(monkeypatch):
    from mobilerun_core_local.driver.base import PlatformUnsupportedError

    driver, _env, _at = _make_driver(monkeypatch)
    run(driver.connect())

    apps = run(driver.get_apps())
    assert {"package": "wechat", "label": "微信"} in apps
    assert run(driver.list_packages()) == ["settings", "wechat"]
    assert run(driver.get_ui_tree())["a11y_tree"]["children"] == []

    with pytest.raises(PlatformUnsupportedError):
        run(driver.stop_app("wechat"))
    with pytest.raises(PlatformUnsupportedError):
        run(driver.install_app("app.apk"))
    with pytest.raises(ValueError):
        run(driver.press_button("volume_up"))


def test_driver_submit_answer_stores_text(monkeypatch):
    driver, _env, _at = _make_driver(monkeypatch)
    run(driver.submit_answer("42"))
    assert driver._answer == "42"


# ---------------------------------------------------------------------------
# Fake bench_env + env + agent for the runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeObservation:
    screenshot_bytes: bytes = b"jpeg"
    route: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    step_idx: int = 0


class FakeTask:
    def __init__(self, task_id, difficulty="L2", max_steps=30, apps=("wechat",)):
        self.id = task_id
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.apps = list(apps)
        self.description = f"goal for {task_id}"
        self.torn_down = False

    def teardown(self, env):
        self.torn_down = True


class FakeEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.url = kwargs.get("url")
        self.page = SimpleNamespace(evaluate=self._evaluate)
        self.agent_answer = None
        self.closed = False
        self.state_reads = []

    async def _evaluate(self, _script):
        return True

    async def start(self):
        return self

    async def wait_ready(self, timeout_ms=60000):
        return None

    async def get_state(self, *, required_apps=None):
        self.state_reads.append(required_apps)
        return {"os": {}, "apps": {"wechat": {"done": True}}}

    async def get_observation(self):
        return FakeObservation(state={})

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


class FakeJudge:
    def __init__(self, passed=True, judge_error=None):
        self.passed = passed
        self.judge_error = judge_error
        self.success = passed
        self.clean = True
        self.progress = 1.0 if passed else 0.0
        self.issues = [] if passed else [{"reason": "not done"}]
        self.warnings = []
        self.calls = []

    async def evaluate(self, task, init_obs, last_obs, exec_result, episode=None):
        self.calls.append((task, init_obs, last_obs, exec_result))
        return SimpleNamespace(
            passed=self.passed,
            success=self.success,
            clean=self.clean,
            progress=self.progress,
            issues=self.issues,
            warnings=self.warnings,
            judge_error=self.judge_error,
        )


class FakeEvaluator:
    judge = FakeJudge()

    def __init__(self, judge_mode="state", eval_mode="grounded"):
        self.judge_mode = judge_mode
        self.eval_mode = eval_mode

    async def evaluate(self, task, init_obs, last_obs, exec_result, episode=None):
        return await FakeEvaluator.judge.evaluate(
            task, init_obs, last_obs, exec_result, episode
        )


class FakeExecutionResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeController:
    @staticmethod
    async def setup(env, task, eval_mode="grounded"):
        FakeController.calls.append((env, task, eval_mode))
        return FakeObservation(state={"os": {}, "apps": {"wechat": {}}}), {}

    calls = []


class FakeRegistry:
    tasks = []

    def discover_all(self):
        return list(FakeRegistry.tasks)


class FakeDriver:
    instances = []

    def __init__(self, env):
        self.env = env
        self._answer = ""
        self.closed = False
        FakeDriver.instances.append(self)

    async def connect(self):
        pass

    async def close(self):
        self.closed = True

    async def submit_answer(self, text):
        self._answer = text

    async def get_ui_tree(self):
        return {"a11y_tree": {"children": []}}


def make_bench_env_namespace(*, tasks, split_ids=None, judge=None):
    FakeRegistry.tasks = list(tasks)
    FakeController.calls = []
    FakeEvaluator.judge = judge or FakeJudge()
    return SimpleNamespace(
        MobileGymEnv=FakeEnv,
        Controller=FakeController,
        Evaluator=FakeEvaluator,
        ExecutionResult=FakeExecutionResult,
        base_task_id=lambda task_id: task_id,
        resolve_split=lambda spec: set(split_ids if split_ids is not None else []),
        TaskRegistry=FakeRegistry,
    )


def patch_runner(monkeypatch, be, agent_class):
    monkeypatch.setattr(bench, "_import_bench_env", lambda: be)
    monkeypatch.setattr(bench, "MobileGymBrowserDriver", FakeDriver)
    monkeypatch.setattr(bench, "MobileAgent", agent_class)
    monkeypatch.setattr(bench, "_setup_benchmark_logging", lambda debug, verbose: None)
    monkeypatch.setattr(bench, "load_agent_llms", lambda config, **kwargs: {})
    monkeypatch.setattr(bench, "load_llm", lambda *a, **k: {})
    monkeypatch.setattr(
        bench.ConfigLoader,
        "load",
        lambda _path=None: MobileConfig.from_dict(
            {
                "agent": {"name": "external-agent", "reasoning": False},
                "telemetry": {"enabled": False},
            }
        ),
    )
    FakeDriver.instances.clear()


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


def test_load_task_list_applies_split_then_filters():
    tasks = [
        FakeTask("a.One", "L1"),
        FakeTask("b.Two", "L2"),
        FakeTask("c.Three", "L3"),
        FakeTask("d.Four", "L4"),
    ]
    be = make_bench_env_namespace(tasks=tasks, split_ids={"a.One", "b.Two", "c.Three"})
    selected, metadata = bench._load_task_list(
        be, split="test", tasks=None, task_index_range=None, difficulty=()
    )
    assert [t.id for t in selected] == ["a.One", "b.Two", "c.Three"]
    assert metadata["a.One"]["difficulty"] == "easy"
    assert metadata["c.Three"]["difficulty"] == "hard"

    selected, _ = bench._load_task_list(
        be, split="test", tasks="b.Two", task_index_range=None, difficulty=("medium",)
    )
    assert [t.id for t in selected] == ["b.Two"]

    selected, _ = bench._load_task_list(
        be, split="test", tasks=None, task_index_range="2-3", difficulty=()
    )
    assert [t.id for t in selected] == ["b.Two", "c.Three"]


def test_load_task_list_empty_split_raises():
    be = make_bench_env_namespace(tasks=[FakeTask("a.One")], split_ids=set())
    with pytest.raises(ValueError, match="No MobileGym tasks"):
        bench._load_task_list(
            be, split="test", tasks=None, task_index_range=None, difficulty=()
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_missing_bench_env_returns_false(monkeypatch):
    def boom():
        raise ImportError("no bench_env")

    monkeypatch.setattr(bench, "_import_bench_env", boom)
    assert run(bench.run_mobilegym_benchmark()) is False


def test_provider_model_validation():
    assert run(bench.run_mobilegym_benchmark(provider="OpenAILike")) is False


def test_dry_run_does_not_start_browser(monkeypatch, tmp_path):
    tasks = [FakeTask("a.One"), FakeTask("b.Two")]
    be = make_bench_env_namespace(tasks=tasks, split_ids={"a.One", "b.Two"})
    agent_class = make_fake_agent_class()
    patch_runner(monkeypatch, be, agent_class)

    success = run(
        bench.run_mobilegym_benchmark(
            sim_url="https://localhost:4180",
            output=str(tmp_path / "results.json"),
            capture_evidence=False,
            dry_run=True,
        )
    )
    assert success is True
    assert FakeDriver.instances == []
    assert agent_class.instances == []
    assert not (tmp_path / "results.json").exists()


def test_episode_judges_with_full_final_state(monkeypatch, tmp_path):
    tasks = [FakeTask("a.One")]
    judge = FakeJudge(passed=True)
    be = make_bench_env_namespace(tasks=tasks, split_ids={"a.One"}, judge=judge)
    agent_class = make_fake_agent_class(shared_success=True, answer="313")
    patch_runner(monkeypatch, be, agent_class)
    output = tmp_path / "results.json"

    success = run(
        bench.run_mobilegym_benchmark(
            sim_url="https://localhost:4180",
            output=str(output),
            capture_evidence=False,
            steps=7,
        )
    )
    assert success is True

    # env lifecycle
    env = FakeDriver.instances[0].env
    assert env.kwargs["coord_space"] == "physical"
    assert env.kwargs["viewport_size"] == (360, 800)
    assert env.kwargs["physical_size"] == (1080, 2400)
    assert env.closed is True
    assert FakeController.calls[0][2] == "grounded"

    # coordinator: final state re-read before judging (step() state is empty)
    assert env.state_reads == [["wechat"]]
    _task, init_obs, last_obs, exec_result = judge.calls[0]
    assert last_obs.state == {"os": {}, "apps": {"wechat": {"done": True}}}
    assert init_obs.state["apps"]["wechat"] == {}
    assert exec_result.agent_answer == "313"
    assert exec_result.steps == 3

    # answer + teardown
    assert FakeDriver.instances[0]._answer == "313"
    assert tasks[0].torn_down is True

    # results file
    data = json.loads(output.read_text())
    assert data["benchmark"] == "mobilegym"
    assert data["tasks"][0]["score"] == 1.0
    assert data["tasks"][0]["gated_score"] == 1.0
    assert data["tasks"][0]["finished"] is True
    assert data["tasks"][0]["server_url"] == "https://localhost:4180"
    assert data["summary"]["tasks"] == 1


def test_undeclared_done_gates_score_to_zero(monkeypatch, tmp_path):
    tasks = [FakeTask("a.One")]
    be = make_bench_env_namespace(
        tasks=tasks, split_ids={"a.One"}, judge=FakeJudge(passed=True)
    )
    agent_class = make_fake_agent_class(shared_success=None)
    patch_runner(monkeypatch, be, agent_class)

    output = tmp_path / "results.json"
    run(bench.run_mobilegym_benchmark(output=str(output), capture_evidence=False))
    row = json.loads(output.read_text())["tasks"][0]
    assert row["score"] == 1.0
    assert row["gated_score"] == 0.0
    assert row["finished"] is False


def test_judge_error_is_recorded(monkeypatch, tmp_path):
    tasks = [FakeTask("a.One")]
    be = make_bench_env_namespace(
        tasks=tasks,
        split_ids={"a.One"},
        judge=FakeJudge(passed=False, judge_error="state diff crashed"),
    )
    patch_runner(monkeypatch, be, make_fake_agent_class())

    output = tmp_path / "results.json"
    run(bench.run_mobilegym_benchmark(output=str(output), capture_evidence=False))
    row = json.loads(output.read_text())["tasks"][0]
    assert "JudgeError" in row["error"]
    assert row["score"] == 0.0


def test_infra_failure_is_retriable_on_resume(monkeypatch, tmp_path):
    """A 429-absorbed episode must record an error so --resume re-runs it."""
    tasks = [FakeTask("a.One")]
    be = make_bench_env_namespace(tasks=tasks, split_ids={"a.One"})
    agent_class = make_fake_agent_class(
        shared_success=None,
        success=False,
        reason=(
            "Error: Error code: 429 - {'error': {'code': 'model_cooldown', "
            "'reset_seconds': 10}}"
        ),
    )
    patch_runner(monkeypatch, be, agent_class)
    output = tmp_path / "results.json"

    run(bench.run_mobilegym_benchmark(output=str(output), capture_evidence=False))
    row = json.loads(output.read_text())["tasks"][0]
    assert row["error"] is not None
    assert row["error"].startswith("InfraError")
    assert row["gated_score"] == 0.0
    assert row["finished"] is False

    # The recorded error makes --resume retry the episode instead of
    # keeping the poisoned 0-score record.
    agent_class.instances.clear()
    run(
        bench.run_mobilegym_benchmark(
            output=str(output), capture_evidence=False, resume=True
        )
    )
    assert len(agent_class.instances) == 1


def test_genuine_failure_is_not_marked_infra(monkeypatch, tmp_path):
    """A model-level failure stays error=None so resume skips it."""
    tasks = [FakeTask("a.One")]
    be = make_bench_env_namespace(tasks=tasks, split_ids={"a.One"})
    agent_class = make_fake_agent_class(
        success=False, reason="I could not find the settings screen"
    )
    patch_runner(monkeypatch, be, agent_class)
    output = tmp_path / "results.json"

    run(bench.run_mobilegym_benchmark(output=str(output), capture_evidence=False))
    row = json.loads(output.read_text())["tasks"][0]
    assert row["error"] is None


def test_resume_skips_recorded_task(monkeypatch, tmp_path):
    tasks = [FakeTask("a.One")]
    be = make_bench_env_namespace(tasks=tasks, split_ids={"a.One"})
    agent_class = make_fake_agent_class()
    patch_runner(monkeypatch, be, agent_class)
    output = tmp_path / "results.json"

    run(
        bench.run_mobilegym_benchmark(
            output=str(output), capture_evidence=False, resume=False
        )
    )
    assert len(agent_class.instances) == 1

    agent_class.instances.clear()
    run(
        bench.run_mobilegym_benchmark(
            output=str(output), capture_evidence=False, resume=True
        )
    )
    assert agent_class.instances == []


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_runbench_routes_mobilegym(monkeypatch):
    captured = {}

    async def fake_runner(**kwargs):
        captured.update(kwargs)
        return True

    from mobilerun.cli import main as cli_main

    monkeypatch.setattr(cli_main, "run_mobilegym_benchmark", fake_runner)
    from click.testing import CliRunner

    result = CliRunner().invoke(
        cli_main.cli,
        [
            "runbench",
            "--benchmark-name",
            "mobilegym",
            "-d",
            "https://localhost:4180",
            "--split",
            "test+payment",
            "--no-headless",
            "--startup-timeout",
            "42",
            "--steps",
            "9",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["sim_url"] == "https://localhost:4180"
    assert captured["split"] == "test+payment"
    assert captured["headless"] is False
    assert captured["startup_timeout"] == 42.0
    assert captured["steps"] == 9
    assert captured["dry_run"] is True


def test_runbench_help_lists_mobilegym_options():
    from click.testing import CliRunner

    from mobilerun.cli import main as cli_main

    result = CliRunner().invoke(cli_main.cli, ["runbench", "--help"])
    assert result.exit_code == 0
    for option in ("mobilegym", "--split", "--headless", "--startup-timeout"):
        assert option in result.output


if __name__ == "__main__":
    pytest.main([__file__])
