"""Wire-contract tests for the evaluation harness (protocol v1)."""

import json

import pytest
from pydantic import ValidationError

from mobilerun.harness.actions import supported_ops
from mobilerun.harness.protocol import (
    PROTOCOL_VERSION,
    Action,
    Capabilities,
    Observation,
    ScreenInfo,
    StepRequest,
    TaskSpec,
)


def test_protocol_version_is_exposed():
    assert PROTOCOL_VERSION == "1.0"


@pytest.mark.parametrize(
    "op,args",
    [
        ("tap", {"x": 1, "y": 2}),
        ("long_press", {"x": 1, "y": 2}),
        ("swipe", {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "duration_ms": 500}),
        ("drag", {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "duration_s": 1.0}),
        ("type", {"text": "hello", "clear": True}),
        ("key", {"button": "back"}),
        ("key", {"keycode": 4}),
        ("open_app", {"app": "com.example"}),
        ("stop_app", {"app": "com.example"}),
        ("wait", {"duration_s": 1.0}),
        ("complete", {"success": True, "message": "done"}),
    ],
)
def test_action_accepts_every_documented_op(op, args):
    action = Action(op=op, args=args)
    assert action.op == op
    assert Action.model_validate(action.model_dump()) == action


@pytest.mark.parametrize(
    "op,args,fragment",
    [
        ("tap", {"x": 1}, "requires 'y'"),
        ("tap", {"x": "left", "y": 2}, "must be number"),
        ("tap", {"x": True, "y": 2}, "must be number"),
        ("tap", {"x": 1, "y": 2, "z": 3}, "does not take"),
        ("key", {}, "requires 'button' or 'keycode'"),
        ("type", {"text": 5}, "must be string"),
    ],
)
def test_action_rejects_malformed_arguments(op, args, fragment):
    with pytest.raises(ValidationError) as excinfo:
        Action(op=op, args=args)
    assert fragment in str(excinfo.value)


def test_action_rejects_unknown_op():
    with pytest.raises(ValidationError):
        Action(op="teleport", args={})


def test_action_flags():
    assert Action(op="complete", args={"success": True, "message": ""}).is_terminal
    assert not Action(op="wait", args={"duration_s": 1}).touches_device
    assert Action(op="tap", args={"x": 1, "y": 1}).touches_device


def test_step_request_requires_step_index():
    with pytest.raises(ValidationError):
        StepRequest(action=Action(op="tap", args={"x": 1, "y": 1}))


def test_observation_state_dict_mirrors_the_driver_contract():
    state = {
        "a11y_tree": {"children": [], "className": "root"},
        "phone_state": {"currentApp": "com.example/.Main"},
        "device_context": {"screen_bounds": {"width": 10, "height": 20}},
    }
    observation = Observation.build(
        seq=3,
        screenshot=b"\x89PNG\r\n",
        state=state,
        screen=ScreenInfo(native_width=10, native_height=20),
    )
    combined = observation.state_dict()
    assert combined["a11y_tree"] == state["a11y_tree"]
    assert combined["phone_state"] == state["phone_state"]
    assert combined["device_context"]["screen_bounds"] == {"width": 10, "height": 20}
    assert observation.screenshot_bytes() == b"\x89PNG\r\n"
    assert observation.foreground == "com.example/.Main"


def test_observation_survives_json_round_trip():
    observation = Observation.build(
        seq=1,
        screenshot=None,
        state=None,
        screen=ScreenInfo(native_width=108, native_height=240),
    )
    restored = Observation.model_validate_json(
        json.dumps(observation.model_dump(mode="json"))
    )
    assert restored.seq == 1
    assert restored.screen.native_width == 108


def test_supported_ops_follows_driver_verbs():
    class Driver:
        supported = {"tap", "input_text", "screenshot", "get_ui_tree"}

    assert supported_ops(Driver()) == [
        "complete",
        "long_press",
        "tap",
        "type",
        "wait",
    ]


def test_task_spec_and_capabilities_round_trip():
    task = TaskSpec(id="A#0", backend="fake", instruction="do it", metadata={"x": 1})
    assert TaskSpec.model_validate(task.model_dump()).id == "A#0"
    caps = Capabilities(ops=["tap"], verbs=["tap"], platform="Android")
    assert Capabilities.model_validate(caps.model_dump()).platform == "Android"
