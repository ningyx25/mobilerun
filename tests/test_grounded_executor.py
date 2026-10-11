"""Tests for the grounded executor: request construction and execution.

Covers the operation-menu offering rules, the request/state shapes, the action
label renderer, the coordinate conversion, the operation -> action mapping and
the failure-to-feedback policy of :class:`GroundedExecutorAgent`. The executor
is screenshot-only (``vision_only`` is enforced), so no accessibility tree is
ever built or read.
"""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.executor import jev_prompt
from mobilerun.agent.executor.grounded import GroundedExecutorAgent
from mobilerun.agent.utils.systemone import SystemOneError
from mobilerun.config_manager.config_manager import (
    AgentConfig,
    ExecutorConfig,
    SystemOneConfig,
)

SCREEN = (1080, 2400)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class FakeDriver:
    def __init__(self, apps=None, screenshot=b"\x89PNG-fake"):
        self.apps = apps if apps is not None else [{"package": "com.foo", "label": "Foo"}]
        self.screenshot_bytes = screenshot

    async def get_apps(self, include_system=True):
        return self.apps

    async def screenshot(self):
        return self.screenshot_bytes


_FAKE_LLM = object()  # stands in for the app_opener LLM the oneflows run on


def fake_context(driver=None, *, use_normalized=False, scale=1.0, llm=_FAKE_LLM):
    ui = SimpleNamespace(
        screen_width=SCREEN[0],
        screen_height=SCREEN[1],
        use_normalized=use_normalized,
        coordinate_scale_x=scale,
        coordinate_scale_y=scale,
        coordinate_contract_active=True,
        model_screenshot_width=None,
        model_screenshot_height=None,
    )
    return SimpleNamespace(
        driver=driver or FakeDriver(),
        ui=ui,
        macro_recorder=None,
        app_opener_llm=llm,
        streaming=False,
    )


def make_agent(ctx=None, **systemone_overrides):
    systemone_kwargs = {"base_url": "http://systemone.invalid"}
    systemone_kwargs.update(systemone_overrides)
    config = AgentConfig(
        reasoning=True,
        vision_only=True,
        after_sleep_action=0.0,
        executor=ExecutorConfig(kind="grounded", systemone=SystemOneConfig(**systemone_kwargs)),
    )
    state = MobileAgentState(instruction="Open the settings")
    state.width, state.height = SCREEN
    return GroundedExecutorAgent(
        action_ctx=ctx or fake_context(),
        shared_state=state,
        agent_config=config,
        timeout=30,
    )


def run_agent(agent, subgoal="Search for WiFi"):
    async def _run():
        return await agent.run(subgoal=subgoal)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# jev_prompt: operation menu offering
# ---------------------------------------------------------------------------


def test_grounded_response_event_constructs_with_usage():
    # Regression: UsageResult must be a runtime import in grounded_events --
    # a TYPE_CHECKING-only import left the pydantic model undefined and the
    # event raised at construction, which only fired when a oneflow returned
    # real usage (the fakes returned usage=None, so tests missed it).
    from mobilerun.agent.executor.grounded_events import GroundedResponseEvent
    from mobilerun.agent.usage import UsageResult

    usage = UsageResult(request_tokens=10, response_tokens=5, total_tokens=15, requests=1)
    event = GroundedResponseEvent(operation="SWIPE", usage=usage)
    assert event.usage.total_tokens == 15
    assert GroundedResponseEvent(operation="TAP", usage=None).usage is None


def test_menu_offers_the_full_operation_set():
    req = jev_prompt.build_request("Search for WiFi", apps=["Foo", "Bar"])
    assert list(req.criteria_of("operation")) == [
        "OPEN_APP",
        "TAP",
        "LONG_PRESS",
        "TYPE_TEXT",
        "SWIPE",
        "BACK",
        "HOME",
    ]
    # The operation's arguments come from the oneflows, not from questions.
    assert list(req.questions) == ["operation", "app_target"]


def test_menu_without_apps_drops_open_app():
    req = jev_prompt.build_request("goal", apps=[])
    operations = list(req.criteria_of("operation"))
    assert "OPEN_APP" not in operations
    assert "app_target" not in req.questions
    # The screen-independent operations remain.
    assert operations == [
        "TAP",
        "LONG_PRESS",
        "TYPE_TEXT",
        "SWIPE",
        "BACK",
        "HOME",
    ]


def test_app_target_prefers_named_apps_and_drops_foreground():
    # Both named apps are offered; the foreground package is never one of them.
    req = jev_prompt.build_request(
        "Open Kitchen Stories then Chrome",
        apps=["com.example.app", "Kitchen Stories", "Chrome", "Clock"],
        foreground_package="com.example.app",
    )
    assert list(req.criteria_of("app_target").values()) == ["Kitchen Stories", "Chrome"]

    # No app named: the full inventory is offered (minus the foreground app).
    req = jev_prompt.build_request(
        "Do the thing", apps=["com.example.app", "Clock"],
        foreground_package="com.example.app",
    )
    assert list(req.criteria_of("app_target").values()) == ["Clock"]


def test_state_carries_goal_and_five_recent_actions():
    history = [
        jev_prompt.HistoryEntry(operation=f"OP{i}", label=f"label {i}") for i in range(7)
    ]
    state = jev_prompt.build_state("do it", history)
    assert state["goal"] == "do it"
    assert len(state["recentActions"]) == 5
    assert state["recentActions"][0]["operation"] == "OP2"
    assert state["recentActions"][-1] == {
        "operation": "OP6",
        "label": "label 6",
        "screenChanged": None,
    }


# ---------------------------------------------------------------------------
# jev_prompt: action labels (rendered without a tree)
# ---------------------------------------------------------------------------


def test_operation_labels_render_without_a_tree():
    assert json.loads(
        jev_prompt.operation_label("TAP", tap_point=(540, 60))
    ) == {"action": "click", "coordinate": [540, 60]}
    assert json.loads(
        jev_prompt.operation_label("LONG_PRESS", tap_point=(540, 200))
    ) == {"action": "long_press", "coordinate": [540, 200]}
    # No point -> an empty coordinate list, never a guessed one.
    assert json.loads(jev_prompt.operation_label("TAP")) == {
        "action": "click",
        "coordinate": [],
    }
    assert (
        jev_prompt.operation_label("SWIPE", swipe=((540, 2040), (540, 360)))
        == "Swipe from (540, 2040) to (540, 360)."
    )
    assert json.loads(jev_prompt.operation_label("BACK")) == {
        "name": "back",
        "type": "global",
    }
    assert jev_prompt.operation_label("OPEN_APP", app_name="Chrome") == "Open Chrome"


# ---------------------------------------------------------------------------
# GroundedExecutorAgent: decision parsing
# ---------------------------------------------------------------------------


def test_parse_decision_maps_answers_onto_operations():
    agent = make_agent()
    req = jev_prompt.build_request("Open Chrome", apps=["Chrome", "Clock"])

    event = agent._parse_decision(
        {"answers": {"operation": {"choice": "OPEN_APP", "confidence": 0.75},
                     "app_target": {"choice": "1"}}},
        req,
    )
    assert (event.operation, event.app_target, event.error) == ("OPEN_APP", "Chrome", "")

    # SWIPE and TYPE_TEXT carry no arguments in the answer: the oneflows
    # resolve them in the ground step.
    event = agent._parse_decision(
        {"answers": {"operation": {"choice": "SWIPE", "confidence": 0.6}}}, req
    )
    assert (event.operation, event.error) == ("SWIPE", "")

    event = agent._parse_decision(
        {"answers": {"operation": {"choice": "TYPE_TEXT"}}}, req
    )
    assert (event.operation, event.error) == ("TYPE_TEXT", "")

    event = agent._parse_decision(
        {"answers": {"operation": {"choice": "HOP"}}}, req
    )
    assert "no execution path" in event.error

    # WAIT and ENTER are not offered (nor executable): a stray answer is
    # refused like any other unknown operation.
    for stray in ("WAIT", "ENTER"):
        event = agent._parse_decision(
            {"answers": {"operation": {"choice": stray}}}, req
        )
        assert "no execution path" in event.error

    event = agent._parse_decision({"answers": {}}, req)
    assert event.error == "The operation model returned no operation."


def test_grounding_point_falls_back_to_patch_index():
    agent = make_agent()
    fx, fy, conf = agent._grounding_point(
        {"answers": {"tap_target": {"choice": "25", "n_width": 10, "n_height": 20,
                                    "confidence": 0.4}}}
    )
    # idx 25 -> row 2, col 5 -> patch centres at (5.5/10, 2.5/20)
    assert (fx, fy) == (0.55, 0.125)
    with pytest.raises(ValueError):
        agent._grounding_point({"answers": {"tap_target": {"choice": "1"}}})


def test_fraction_to_action_space_normalized_and_scaled():
    agent = make_agent(fake_context(use_normalized=True))
    assert agent._fraction_to_action_space(0.25, 0.5) == (250, 500)

    # No declared model size -> the tree provider convention (native / scale).
    agent = make_agent(fake_context(scale=2.0))
    assert agent._fraction_to_action_space(0.5, 0.5) == (270, 600)

    # A declared model-facing size wins (screenshot-only UIState).
    ctx = fake_context()
    ctx.ui.model_screenshot_width = 800
    ctx.ui.model_screenshot_height = 2000
    agent = make_agent(ctx)
    assert agent._fraction_to_action_space(0.5, 0.5) == (400, 1000)


# ---------------------------------------------------------------------------
# GroundedExecutorAgent: end-to-end runs
# ---------------------------------------------------------------------------


def install_systemone(monkeypatch, responses):
    """Patch systemone_post with a scripted responder keyed by request model."""
    calls = []

    async def fake_post(base_url, body, *, timeout=300.0):
        calls.append(body)
        return responses[body["model"]]

    monkeypatch.setattr("mobilerun.agent.executor.grounded.systemone_post", fake_post)
    return calls


def install_actions(monkeypatch):
    """Patch the action functions with recorders."""
    calls = {}

    async def recorder(name):
        async def _call(*args, **kwargs):
            calls[name] = (args, kwargs)
            return ActionResult(success=True, summary=f"{name} ok")

        return _call

    for name in ("click_at", "long_press_at", "swipe", "system_button", "open_app", "open_bundle_id", "type_text"):
        monkeypatch.setattr(
            f"mobilerun.agent.executor.grounded.{name}", asyncio.run(recorder(name))
        )
    return calls


def install_oneflows(monkeypatch, *, swipe=None, text=None):
    """Patch the executor's oneflow classes with scripted fakes.

    ``swipe`` is the dict the fake SwipeAgent resolves (or an Exception to
    raise); ``text`` is the string the fake TextInputAgent returns. Returns a
    dict capturing the ``run`` kwargs each fake received.
    """
    captured: dict = {}

    if swipe is not None:

        class FakeSwipeAgent:
            def __init__(self, **kwargs):
                pass

            async def run(self, **kwargs):
                captured["swipe"] = kwargs
                if isinstance(swipe, Exception):
                    raise swipe
                return swipe

        monkeypatch.setattr(
            "mobilerun.agent.executor.grounded.SwipeAgent", FakeSwipeAgent
        )

    if text is not None:

        class FakeTextInputAgent:
            def __init__(self, **kwargs):
                pass

            async def run(self, **kwargs):
                captured["text"] = kwargs
                if isinstance(text, Exception):
                    raise text
                return {"text": text, "usage": None}

        monkeypatch.setattr(
            "mobilerun.agent.executor.grounded.TextInputAgent", FakeTextInputAgent
        )
    return captured


def test_app_inventory_prefers_launchable_packages(monkeypatch):
    class LaunchableDriver(FakeDriver):
        async def list_launchable_packages(self):
            return ["com.launch.only"]

        async def get_apps(self, include_system=True):
            raise AssertionError("must not fall back to the full package list")

    responses = {"clef-flash": {"answers": {"operation": {"choice": "WAIT"}}}}
    calls = install_systemone(monkeypatch, responses)
    install_actions(monkeypatch)

    agent = make_agent(fake_context(LaunchableDriver()))
    run_agent(agent)

    criteria = calls[0]["questions"]["app_target"]["criteria"]
    assert list(criteria.values()) == ["com.launch.only"]


def test_open_app_launches_the_exact_package(monkeypatch):
    responses = {
        "clef-flash": {
            "answers": {
                "operation": {"choice": "OPEN_APP"},
                "app_target": {"choice": "1"},
            }
        }
    }
    install_systemone(monkeypatch, responses)
    actions = install_actions(monkeypatch)

    # FakeDriver's inventory is [{"package": "com.foo", "label": "Foo"}]; the
    # answer is a label, and the exact package rides along to start_app.
    result = run_agent(make_agent())
    assert result["outcome"] is True
    assert result["action"] == {"action": "open_app", "text": "Foo", "app_id": "com.foo"}
    _args, kwargs = actions["open_bundle_id"]
    assert kwargs.get("app_id") == "com.foo"


def test_tap_runs_clef_then_grounding_then_click(monkeypatch):
    responses = {
        "clef-flash": {"answers": {"operation": {"choice": "TAP", "confidence": 0.8}}},
        "gui-actor": {
            "answers": {
                "tap_target": {"choice": "5", "point": [0.5, 0.025], "confidence": 0.9}
            }
        },
    }
    calls = install_systemone(monkeypatch, responses)
    actions = install_actions(monkeypatch)
    driver = FakeDriver()

    agent = make_agent(fake_context(driver))
    result = run_agent(agent)

    assert result["outcome"] is True
    assert result["action"] == {"action": "click_at", "x": 540, "y": 60}
    assert "click_at" in actions
    args, _ = actions["click_at"]
    assert args == (540, 60)

    # Both requests carried the raw screenshot, unannotated.
    expected_image = base64.b64encode(driver.screenshot_bytes).decode()
    assert calls[0]["images"] == [expected_image]
    assert calls[1]["images"] == [expected_image]
    assert calls[1]["questions"]["tap_target"]["instructions"] == "Search for WiFi"
    # The request state carries the step instruction as the goal.
    assert calls[0]["state"]["goal"] == "Search for WiFi"

    # The executed decision is remembered for the next step's recentActions,
    # with its coordinate (no tree -> the point-based label).
    entry = agent._pending
    assert entry.operation == "TAP"
    assert json.loads(entry.label) == {"action": "click", "coordinate": [540, 60]}


def test_swipe_uses_the_swipe_workflows_points(monkeypatch):
    responses = {"clef-flash": {"answers": {"operation": {"choice": "SWIPE"}}}}
    install_systemone(monkeypatch, responses)
    actions = install_actions(monkeypatch)
    captured = install_oneflows(
        monkeypatch,
        swipe={"start": [0.5, 0.8], "end": [0.5, 0.2]},
    )

    result = run_agent(make_agent())
    assert result["outcome"] is True
    (start, end, _duration), _ = actions["swipe"]
    # 0..1 screen fractions -> action space (1080x2400, not normalized).
    assert start == [540, 1920] and end == [540, 480]
    assert result["action"]["action"] == "swipe"

    # The workflow received the step instruction and the current raw screenshot.
    run_kwargs = captured["swipe"]
    assert run_kwargs["subgoal"] == "Search for WiFi"
    assert run_kwargs["image"] == b"\x89PNG-fake"


def test_swipe_workflow_failure_is_feedback(monkeypatch):
    responses = {"clef-flash": {"answers": {"operation": {"choice": "SWIPE"}}}}
    install_systemone(monkeypatch, responses)
    install_actions(monkeypatch)
    install_oneflows(monkeypatch, swipe=RuntimeError("llm down"))

    result = run_agent(make_agent())
    assert result["outcome"] is False
    assert "Swipe resolution failed" in result["error"]


def test_systemone_failure_is_feedback_not_an_abort(monkeypatch):
    async def failing_post(base_url, body, *, timeout=300.0):
        raise SystemOneError("HTTP 422: bad state")

    monkeypatch.setattr("mobilerun.agent.executor.grounded.systemone_post", failing_post)
    agent = make_agent()

    result = run_agent(agent)
    assert result["outcome"] is False
    assert "Operation model call failed" in result["error"]


def test_complete_file_extension_only_extends_task_tokens():
    agent = make_agent()
    agent.shared_state.instruction = 'Save it as "2023_06_24_training.m4a" now.'
    assert (
        agent._complete_file_extension("2023_06_24_training")
        == "2023_06_24_training.m4a"
    )
    # A mid-token fragment is not the task's token; an already-complete value
    # and non-file values pass through untouched.
    assert agent._complete_file_extension("training") == "training"
    assert (
        agent._complete_file_extension("2023_06_24_training.m4a")
        == "2023_06_24_training.m4a"
    )
    assert agent._complete_file_extension("Welsh cakes") == "Welsh cakes"
    agent.shared_state.instruction = "Open the settings app"
    assert agent._complete_file_extension("settings") == "settings"


def test_type_text_uses_the_text_workflow_value(monkeypatch):
    responses = {"clef-flash": {"answers": {"operation": {"choice": "TYPE_TEXT"}}}}
    install_systemone(monkeypatch, responses)
    actions = install_actions(monkeypatch)
    captured = install_oneflows(monkeypatch, text="Welsh cakes")

    agent = make_agent()
    run_agent(agent, subgoal="Enter the search query")

    args, _kwargs = actions["type_text"]
    assert args[0] == "Welsh cakes"
    assert captured["text"]["subgoal"] == "Enter the search query"
    assert captured["text"]["image"] == b"\x89PNG-fake"


def test_type_text_completes_a_file_extension_from_the_task(monkeypatch):
    responses = {"clef-flash": {"answers": {"operation": {"choice": "TYPE_TEXT"}}}}
    install_systemone(monkeypatch, responses)
    actions = install_actions(monkeypatch)
    # The workflow's value lacks the extension the task names; the executor's
    # deterministic net restores it before typing.
    install_oneflows(monkeypatch, text="2023_06_24_training")

    agent = make_agent()
    agent.shared_state.instruction = 'Save the clip as "2023_06_24_training.m4a".'
    run_agent(agent, subgoal='Type "2023_06_24_training" into the name field')

    args, _kwargs = actions["type_text"]
    assert args[0] == "2023_06_24_training.m4a"


def test_type_text_without_a_value_is_feedback(monkeypatch):
    responses = {"clef-flash": {"answers": {"operation": {"choice": "TYPE_TEXT"}}}}
    install_systemone(monkeypatch, responses)
    install_actions(monkeypatch)
    install_oneflows(monkeypatch, text="")

    result = run_agent(make_agent())
    assert result["outcome"] is False
    assert "without a suitable text value" in result["error"]


def test_missing_oneflow_llm_is_feedback(monkeypatch):
    responses = {"clef-flash": {"answers": {"operation": {"choice": "SWIPE"}}}}
    install_systemone(monkeypatch, responses)
    install_actions(monkeypatch)

    result = run_agent(make_agent(fake_context(llm=None)))
    assert result["outcome"] is False
    assert "app_opener LLM" in result["error"]


def test_requests_carry_only_the_current_raw_frame(monkeypatch):
    responses = {"clef-flash": {"answers": {"operation": {"choice": "BACK"}}}}
    calls = install_systemone(monkeypatch, responses)
    install_actions(monkeypatch)

    tick = {"n": 0}

    class TickingDriver(FakeDriver):
        async def screenshot(self):
            tick["n"] += 1
            return f"frame-{tick['n']}".encode()

    agent = make_agent(fake_context(TickingDriver()))
    run_agent(agent)
    run_agent(agent)
    # The first action is absorbed into history at the end of run 1; run 2
    # sends it as recentActions text alongside its own fresh frame.
    run_agent(agent)

    # One image per request, the current frame, byte-for-byte (no history, no
    # scaling); past steps travel as recentActions text only.
    assert calls[0]["images"] == [base64.b64encode(b"frame-1").decode()]
    assert calls[1]["images"] == [base64.b64encode(b"frame-2").decode()]
    assert calls[2]["state"]["recentActions"][0]["operation"] == "BACK"
    assert json.loads(calls[2]["state"]["recentActions"][0]["label"]) == {
        "type": "global",
        "name": "back",
    }


def test_screenshot_failure_does_not_abort_the_step(monkeypatch):
    # A screenshot failure (with no manager-captured frame) degrades to a
    # text-only decision request; it must not abort the step.
    class NoShotDriver(FakeDriver):
        async def screenshot(self):
            raise RuntimeError("capture exploded")

    responses = {"clef-flash": {"answers": {"operation": {"choice": "BACK"}}}}
    calls = install_systemone(monkeypatch, responses)
    install_actions(monkeypatch)

    agent = make_agent(fake_context(NoShotDriver()))
    result = run_agent(agent)

    assert result["outcome"] is True
    assert "images" not in calls[0]  # no frame to attach


def test_vision_only_never_reads_the_accessibility_tree(monkeypatch):
    # The executor never calls get_ui_tree; a driver that forbids it proves it.
    class TreeForbiddenDriver(FakeDriver):
        async def get_ui_tree(self):
            raise AssertionError("vision_only must not read the a11y tree")

    responses = {"clef-flash": {"answers": {"operation": {"choice": "SWIPE"}}}}
    calls = install_systemone(monkeypatch, responses)
    install_actions(monkeypatch)
    captured = install_oneflows(
        monkeypatch, swipe={"start": [0.5, 0.8], "end": [0.5, 0.2]}
    )

    agent = make_agent(fake_context(TreeForbiddenDriver()))
    result = run_agent(agent)

    assert result["outcome"] is True
    assert "SWIPE" in calls[0]["questions"]["operation"]["criteria"]
    assert captured["swipe"]["image"] == b"\x89PNG-fake"
    assert agent._pending.operation == "SWIPE"


def test_mobile_agent_rejects_grounded_without_vision_only():
    # The mode only supports vision_only=true: it grounds every decision on
    # the screenshot and reads no accessibility tree.
    from llama_index.core.llms.mock import MockLLM

    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.config_manager.config_manager import MobileConfig, TelemetryConfig

    config = MobileConfig(
        agent=AgentConfig(
            reasoning=True,
            vision_only=False,
            executor=ExecutorConfig(kind="grounded"),
        ),
        telemetry=TelemetryConfig(enabled=False),
    )
    with pytest.raises(ValueError, match="only supports vision_only=true"):
        MobileAgent("goal", config=config, llms=MockLLM())
