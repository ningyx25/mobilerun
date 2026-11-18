"""MobileGym environment backend: catalogue, setup, scoring and ASGI wiring.

The real simulator/Playwright is never started; ``bench_env`` is replaced by a
small fake namespace so the backend contract can be exercised deterministically.
"""

import asyncio
import io
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from PIL import Image

from mobilerun.harness.env_service import mobile_gym_backend as backend_module
from mobilerun.harness.env_service.backend import create_backend
from mobilerun.harness.env_service.app import create_env_app
from mobilerun.harness.protocol import Action, ErrorCode, ProtocolError
from mobilerun.tools.driver.mobile_gym import MobileGymBrowserDriver


def run(coro):
    return asyncio.run(coro)


def _jpeg(width=1080, height=2400):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 30, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


@dataclass(frozen=True)
class FakeObservation:
    screenshot_bytes: bytes = b""
    route: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    step_idx: int = 0


class FakePage:
    def __init__(self):
        self.screenshot_bytes = _jpeg()
        self.fail = False
        self.evaluate_calls = 0

    async def evaluate(self, _script):
        self.evaluate_calls += 1
        return True

    async def screenshot(self, type="jpeg", quality=80):
        if self.fail:
            raise RuntimeError("page is dead")
        return self.screenshot_bytes


class FakeMobileGymEnv:
    instances = []
    APP_NAME_MAP = {
        "微信": "wechat",
        "WeChat": "wechat",
        "设置": "settings",
        "Settings": "settings",
    }

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.url = kwargs.get("url")
        self.page = FakePage()
        self.start_calls = 0
        self.close_calls = 0
        self.step_calls = []
        self.agent_answer = None
        self.required_apps = []
        self.state = {
            "os": {},
            "apps": {"answer_sheet": {"submitted": True, "answers": {"0": "42"}}},
        }
        FakeMobileGymEnv.instances.append(self)

    async def start(self):
        self.start_calls += 1
        return self

    async def wait_ready(self, timeout_ms=60000):
        return None

    async def close(self):
        self.close_calls += 1

    async def get_state(self, *, required_apps=None):
        self.required_apps.append(required_apps)
        return dict(self.state)

    async def get_observation(self):
        return FakeObservation(state={})

    async def step(self, action):
        self.step_calls.append(action)

    def get_device_size(self):
        return (1080, 2400)

    @staticmethod
    def get_launch_args(headless, proxy=None, browser_type="chromium"):
        return {"headless": headless}


class FakeTask:
    def __init__(
        self,
        task_id="Alpha",
        *,
        difficulty="L2",
        max_steps=30,
        apps=("wechat",),
        answer_fields=None,
    ):
        self.id = task_id
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.apps = list(apps)
        self.answer_fields = answer_fields
        self.task_name = ""
        self.teardown_calls = 0

    @property
    def description(self):
        return self.task_name or f"goal for {self.id}"

    def teardown(self, env):
        self.teardown_calls += 1


class FakeTaskRegistry:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.discover_calls = 0
        self.created = []

    def discover_all(self):
        self.discover_calls += 1
        return list(self.tasks)

    def create_task(self, task_id, **params):
        self.created.append((task_id, params))
        template = next(task for task in self.tasks if task.id == task_id)
        return FakeTask(
            task_id,
            difficulty=template.difficulty,
            max_steps=template.max_steps,
            apps=tuple(template.apps),
            answer_fields=template.answer_fields,
        )


class FakeController:
    calls = []

    @staticmethod
    async def setup(env, task, eval_mode="grounded"):
        FakeController.calls.append((env, task, eval_mode))
        task.task_name = task.description + " 然后打开 答题卡 APP 在里面回答问题并提交"
        return FakeObservation(state={"os": {}, "apps": {"wechat": {}}}), {}


class FakeJudge:
    def __init__(self, passed=True, judge_error=None):
        self.passed = passed
        self.success = passed
        self.clean = True
        self.progress = 1.0 if passed else 0.0
        self.issues = [] if passed else [{"reason": "not done"}]
        self.warnings = []
        self.judge_error = judge_error


class FakeEvaluator:
    judge = FakeJudge()
    calls = []

    def __init__(self, judge_mode="state", eval_mode="grounded"):
        self.judge_mode = judge_mode
        self.eval_mode = eval_mode

    async def evaluate(self, task, init_obs, last_obs, exec_result, episode=None):
        FakeEvaluator.calls.append((task, init_obs, last_obs, exec_result))
        return FakeEvaluator.judge


class FakeExecutionResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_bench_env(tasks, *, split_ids=None):
    registry = FakeTaskRegistry(tasks)
    FakeMobileGymEnv.instances = []
    FakeController.calls = []
    FakeEvaluator.calls = []
    FakeEvaluator.judge = FakeJudge()
    namespace = SimpleNamespace(
        MobileGymEnv=FakeMobileGymEnv,
        Controller=FakeController,
        Evaluator=FakeEvaluator,
        ExecutionResult=FakeExecutionResult,
        base_task_id=lambda task_id: task_id.split("_i")[0],
        resolve_split=lambda spec: set(
            split_ids if split_ids is not None else {task.id for task in tasks}
        ),
        TaskRegistry=lambda: registry,
    )
    return namespace, registry


@pytest.fixture(autouse=True)
def _patch_driver_step(monkeypatch):
    async def fake_step(driver, action_type, data):
        driver.env.step_calls.append((action_type, data))

    monkeypatch.setattr(MobileGymBrowserDriver, "_step", fake_step)


def make_backend(
    monkeypatch,
    tasks=None,
    *,
    split_ids=None,
    url="https://localhost:4180?split=test",
):
    tasks = tasks or [
        FakeTask("Alpha"),
        FakeTask("Beta", difficulty="L4", max_steps=60, apps=("settings",)),
    ]
    namespace, registry = make_bench_env(tasks, split_ids=split_ids)
    monkeypatch.setattr(backend_module, "load_bench_env", lambda: namespace)
    backend = backend_module.MobileGymBackend("mg0", url)
    return backend, namespace, registry


def test_parse_backend_url_defaults_and_options():
    defaults = backend_module.parse_mobile_gym_backend_url("https://localhost:4180")
    assert defaults.url == "https://localhost:4180"
    assert defaults.split == "test"
    assert defaults.headless is True
    assert defaults.startup_timeout == 120.0

    options = backend_module.parse_mobile_gym_backend_url(
        "https://localhost:4180?split=train&headless=false&startup_timeout=30"
    )
    assert options.url == "https://localhost:4180"
    assert options.split == "train"
    assert options.headless is False
    assert options.startup_timeout == 30.0


def test_parse_backend_url_preserves_plus_in_split():
    options = backend_module.parse_mobile_gym_backend_url(
        "https://localhost:4180?split=test+payment"
    )
    assert options.split == "test+payment"


def test_parse_backend_url_accepts_browser_path(tmp_path):
    binary = tmp_path / "chrome"
    binary.write_text("")
    options = backend_module.parse_mobile_gym_backend_url(
        f"https://localhost:4180?browser_path={binary}"
    )
    assert options.browser_path == str(binary)


def test_browser_path_overrides_launch_args(tmp_path, monkeypatch):
    binary = tmp_path / "chrome"
    binary.write_text("")
    backend, _, _ = make_backend(
        monkeypatch,
        url=f"https://localhost:4180?split=test&browser_path={binary}",
    )
    args = backend._env.get_launch_args(True)
    assert args["executable_path"] == str(binary)
    assert args["headless"] is True


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost:4180?unknown=1",
        "https://localhost:4180?split=a&split=b",
        "https://localhost:4180?split=",
        "https://localhost:4180?headless=maybe",
        "https://localhost:4180?startup_timeout=abc",
        "https://localhost:4180?startup_timeout=0",
        "https://localhost:4180?browser_path=/no/such/chrome",
        "https://localhost:4180#fragment",
    ],
)
def test_parse_backend_url_rejects_bad_input(url):
    with pytest.raises(ValueError):
        backend_module.parse_mobile_gym_backend_url(url)


def test_create_backend_dispatches_mobile_gym(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    assert isinstance(backend, backend_module.MobileGymBackend)

    created = create_backend("mobile-gym", "mg1", "https://localhost:4180?split=test")
    assert isinstance(created, backend_module.MobileGymBackend)
    assert created.id == "mg1"

    with pytest.raises(ValueError, match="mobile-gym"):
        create_backend("nope", "x", "https://localhost:4180")


def test_missing_bench_env_propagates_import_error(monkeypatch):
    def boom():
        raise ImportError("bench_env is missing; set PYTHONPATH")

    monkeypatch.setattr(backend_module, "load_bench_env", boom)
    with pytest.raises(ImportError, match="PYTHONPATH"):
        backend_module.MobileGymBackend("mg0", "https://localhost:4180")


def test_list_tasks_applies_split_and_reports_metadata(monkeypatch):
    backend, _, registry = make_backend(
        monkeypatch,
        split_ids={"Alpha", "Beta"},
    )
    tasks = run(backend.list_tasks())
    by_id = {task.id: task for task in tasks}
    assert set(by_id) == {"Alpha", "Beta"}
    assert by_id["Alpha"].instruction == ""
    assert by_id["Alpha"].max_steps == 30
    assert by_id["Alpha"].metadata["difficulty"] == "medium"
    assert by_id["Alpha"].metadata["apps"] == "wechat"
    assert by_id["Beta"].max_steps == 60
    assert by_id["Beta"].metadata["difficulty"] == "hard"

    run(backend.list_tasks())
    assert registry.discover_calls == 1


def test_prepare_uses_fresh_task_instances_and_reads_instruction_after_setup(
    monkeypatch,
):
    backend, _, registry = make_backend(monkeypatch)
    run(backend.connect())

    first = run(backend.prepare("Alpha"))
    first_task = backend._active_task_obj
    second = run(backend.prepare("Alpha"))
    second_task = backend._active_task_obj

    assert first.id == "Alpha#0"
    assert first_task is not second_task
    assert [entry[0] for entry in registry.created] == ["Alpha", "Alpha"]
    assert "然后打开 答题卡 APP" in first.instruction
    assert first.instruction.count("然后打开 答题卡 APP") == 1
    assert second.instruction == first.instruction
    assert first.max_steps == 30
    assert first.metadata["task_name"] == "Alpha"


def test_prepare_unknown_task_is_404(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    with pytest.raises(ProtocolError) as excinfo:
        run(backend.prepare("Nope"))
    assert excinfo.value.status_code == 404
    assert excinfo.value.code is ErrorCode.UNKNOWN_TASK


def test_capabilities_are_vision_only(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    capabilities = backend.capabilities()
    assert capabilities.has_ui_tree is False
    assert capabilities.supports_stabilize is False
    assert capabilities.supports_answer is True
    assert "open_app" in capabilities.ops
    assert "stop_app" not in capabilities.ops
    assert "get_ui_tree" not in capabilities.verbs


def test_observe_returns_jpeg_without_ui_tree(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    run(backend.connect())
    observation = run(backend.observe(seq=1))
    assert observation.ui_tree is None
    assert observation.screenshot_bytes()[:2] == b"\xff\xd8"
    assert observation.screen.native_width == 1080
    assert observation.screen.native_height == 2400


def test_observe_failure_maps_to_503_and_marks_disconnected(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    run(backend.connect())
    backend._env.page.fail = True
    with pytest.raises(ProtocolError) as excinfo:
        run(backend.observe(seq=1))
    assert excinfo.value.status_code == 503
    assert backend._connected is False


def test_execute_translates_open_app_labels_and_counts_device_steps(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    run(backend.connect())
    observation = run(backend.observe(seq=1))

    action = Action(op="open_app", args={"app": "微信"})
    run(backend.execute(action, screen=observation.screen))
    assert backend._env.step_calls[-1] == ("AWAKE", {"value": "wechat"})
    assert backend._steps == 1

    run(
        backend.execute(
            Action(op="wait", args={"duration_s": 0.0}), screen=observation.screen
        )
    )
    assert backend._steps == 1


def test_evaluate_uses_finish_context(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    run(backend.connect())
    run(backend.prepare("Alpha"))
    backend.set_finish_context(declared_done=True, answer="42")

    score, reason = run(backend.evaluate())

    assert score == 1.0
    task, init_obs, last_obs, exec_result = FakeEvaluator.calls[-1]
    assert task.id == "Alpha"
    assert init_obs is not None
    assert last_obs.state["apps"]["answer_sheet"]["submitted"] is True
    assert exec_result.finished is True
    assert exec_result.agent_answer == "42"
    assert backend._env.required_apps[-1] == ["wechat"]
    payload = json.loads(reason)
    assert set(payload) == {"success", "clean", "progress", "issues", "warnings"}


def test_evaluate_judge_error_scores_zero_and_reports(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    FakeEvaluator.judge = FakeJudge(passed=False, judge_error="boom")
    run(backend.connect())
    run(backend.prepare("Alpha"))

    score, reason = run(backend.evaluate())

    assert score == 0.0
    assert reason.startswith("JudgeError: boom;")


def test_evaluate_without_active_task_is_409(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    with pytest.raises(ProtocolError) as excinfo:
        run(backend.evaluate())
    assert excinfo.value.status_code == 409
    assert excinfo.value.code is ErrorCode.SESSION_STATE


def test_teardown_calls_task_teardown_and_clears_state(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    run(backend.connect())
    spec = run(backend.prepare("Alpha"))
    task = backend._active_task_obj

    run(backend.teardown(spec))
    assert task.teardown_calls == 1
    assert backend._active_task_obj is None

    run(backend.teardown(spec))
    assert task.teardown_calls == 1


def test_connect_is_idempotent_and_close_releases_env(monkeypatch):
    backend, _, _ = make_backend(monkeypatch)
    run(backend.connect())
    run(backend.connect())
    env = FakeMobileGymEnv.instances[0]
    assert env.start_calls == 1

    run(backend.close())
    assert env.close_calls >= 1


def test_env_service_session_lifecycle(monkeypatch):
    from fastapi.testclient import TestClient

    backend, _, _ = make_backend(monkeypatch)
    with TestClient(create_env_app([backend])) as client:
        health = client.get("/v1/health").json()
        assert health["backends"][0]["kind"] == "mobile-gym"
        assert health["backends"][0]["capabilities"]["has_ui_tree"] is False

        tasks = client.get("/v1/tasks").json()
        assert {task["id"] for task in tasks} == {"Alpha", "Beta"}

        created = client.post(
            "/v1/sessions", json={"backend": "mg0", "task_id": "Alpha"}
        ).json()
        session_id = created["session_id"]
        assert created["task"]["id"] == "Alpha#0"
        assert created["observation"]["ui_tree"] is None

        stepped = client.post(
            f"/v1/sessions/{session_id}/step",
            json={
                "action": {"op": "tap", "args": {"x": 1, "y": 2}},
                "step_index": 1,
            },
        ).json()
        assert stepped["ok"] is True

        finished = client.post(
            f"/v1/sessions/{session_id}/finish",
            json={"declared_done": True, "answer": "42"},
        ).json()
        assert finished["score"] == 1.0
        assert finished["gated_score"] == 1.0

        client.delete(f"/v1/sessions/{session_id}")
    assert FakeMobileGymEnv.instances[0].close_calls >= 1


def test_env_service_undeclared_finish_is_gated(monkeypatch):
    from fastapi.testclient import TestClient

    backend, _, _ = make_backend(monkeypatch)
    with TestClient(create_env_app([backend])) as client:
        session_id = client.post(
            "/v1/sessions", json={"backend": "mg0", "task_id": "Alpha"}
        ).json()["session_id"]
        finished = client.post(
            f"/v1/sessions/{session_id}/finish",
            json={"declared_done": False},
        ).json()
        assert finished["score"] == 1.0
        assert finished["gated_score"] == 0.0
        assert finished["declared_done"] is False
