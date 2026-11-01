"""Action -> DeviceDriver mapping and coordinate conversion."""

import asyncio

import pytest

from mobilerun.harness.actions import (
    _resolve_package,
    execute_action,
    screen_from_state,
)
from mobilerun.harness.protocol import (
    Action,
    ErrorCode,
    ProtocolError,
    ScreenInfo,
)


class _Driver:
    """Records driver calls; capability set is what backends derive ops from."""

    supported = {
        "tap",
        "swipe",
        "drag",
        "input_text",
        "press_button",
        "press_key_code",
        "start_app",
        "stop_app",
        "screenshot",
        "get_ui_tree",
    }

    def __init__(self) -> None:
        self.calls = []

    async def tap(self, x, y):
        self.calls.append(("tap", x, y))

    async def swipe(self, x1, y1, x2, y2, duration_ms=1000):
        self.calls.append(("swipe", x1, y1, x2, y2, duration_ms))

    async def drag(self, x1, y1, x2, y2, duration=3.0):
        self.calls.append(("drag", x1, y1, x2, y2, duration))

    async def input_text(self, text, clear=False, stealth=False, wpm=0):
        self.calls.append(("input_text", text, clear))
        return True

    async def press_button(self, button):
        self.calls.append(("press_button", button))

    async def press_key_code(self, code):
        self.calls.append(("press_key_code", code))

    async def start_app(self, package, activity=None):
        self.calls.append(("start_app", package, activity))
        return f"Started {package}"

    async def stop_app(self, package):
        self.calls.append(("stop_app", package))
        return f"Stopped {package}"


SCREEN = ScreenInfo(native_width=1080, native_height=2400)


def run(action: Action, *, driver=None, screen=SCREEN, space="screenshot_pixels"):
    driver = driver or _Driver()
    summary = asyncio.run(
        execute_action(driver, action, screen=screen, default_space=space)
    )
    return driver, summary


def test_tap_is_identity_in_screenshot_pixels():
    driver, summary = run(Action(op="tap", args={"x": 12, "y": 34}))
    assert driver.calls == [("tap", 12, 34)]
    assert "Tapped at (12, 34)" in summary


def test_normalized_points_convert_to_device_pixels():
    driver, _ = run(
        Action(op="tap", args={"x": 500, "y": 500}, coordinate_space="normalized_1000")
    )
    assert driver.calls == [("tap", 540, 1200)]


def test_default_space_applies_when_action_does_not_override():
    driver, _ = run(Action(op="tap", args={"x": 0, "y": 1000}), space="normalized_1000")
    assert driver.calls == [("tap", 0, 2400)]


def test_screenshot_pixels_scale_when_input_space_differs():
    # iOS-style: screenshots are physical pixels, taps are XCTest points.
    screen = ScreenInfo(
        native_width=1080, native_height=2400, input_width=540, input_height=1200
    )
    driver, _ = run(Action(op="tap", args={"x": 108, "y": 240}), screen=screen)
    assert driver.calls == [("tap", 54, 120)]


def test_long_press_is_a_zero_length_swipe():
    driver, _ = run(Action(op="long_press", args={"x": 10, "y": 20}))
    assert driver.calls == [("swipe", 10, 20, 10, 20, 1000.0)]


def test_swipe_passes_duration_through():
    driver, _ = run(
        Action(
            op="swipe",
            args={"x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration_ms": 250},
        )
    )
    assert driver.calls == [("swipe", 1, 2, 3, 4, 250.0)]


def test_type_and_key_map_to_input_verbs():
    driver, _ = run(Action(op="type", args={"text": "hi", "clear": True}))
    assert driver.calls == [("input_text", "hi", True)]

    driver, _ = run(Action(op="key", args={"button": "back"}))
    assert driver.calls == [("press_button", "back")]

    driver, _ = run(Action(op="key", args={"keycode": 66}))
    assert driver.calls == [("press_key_code", 66)]


def test_open_app_forwards_activity():
    driver, _ = run(
        Action(op="open_app", args={"app": "com.example", "activity": ".Main"})
    )
    assert driver.calls == [("start_app", "com.example", ".Main")]


def test_wait_and_complete_never_touch_the_device():
    driver, summary = run(Action(op="wait", args={"duration_s": 0.01}))
    assert driver.calls == []
    assert "Waited" in summary

    driver, summary = run(
        Action(op="complete", args={"success": False, "message": "gave up"})
    )
    assert driver.calls == []
    assert "Declared failure" in summary


def test_unsupported_op_raises_a_protocol_error():
    class TapOnly(_Driver):
        supported = {"tap"}

    with pytest.raises(ProtocolError) as excinfo:
        run(
            Action(op="drag", args={"x1": 0, "y1": 0, "x2": 1, "y2": 1}),
            driver=TapOnly(),
        )
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_ACTION


def test_screen_from_state_prefers_declared_bounds():
    screen = screen_from_state(
        {"device_context": {"screen_bounds": {"width": 720, "height": 1280}}},
        native_width=1080,
        native_height=2400,
    )
    assert (screen.native_width, screen.native_height) == (720, 1280)

    fallback = screen_from_state(None, native_width=1080, native_height=2400)
    assert (fallback.native_width, fallback.native_height) == (1080, 2400)


class _AppDriver:
    supported = {"start_app"}

    def __init__(self, packages, *, start_result=None):
        self.packages = packages
        self.start_result = start_result
        self.started = []

    async def list_launchable_packages(self):
        return list(self.packages)

    async def start_app(self, package, activity=None):
        self.started.append((package, activity))
        return self.start_result or f"Started {package}"


def test_open_app_resolves_a_description_to_a_package():
    driver = _AppDriver(
        ["com.dimowner.audiorecorder", "com.android.chrome", "com.android.settings"]
    )

    async def scenario():
        return await execute_action(
            driver,
            Action(op="open_app", args={"app": "Audio Recorder"}),
            screen=ScreenInfo(native_width=1080, native_height=2400),
        )

    assert "Started com.dimowner.audiorecorder" in asyncio.run(scenario())
    assert driver.started == [("com.dimowner.audiorecorder", None)]


def test_open_app_passes_a_package_id_through():
    driver = _AppDriver(["com.android.chrome"])
    assert asyncio.run(_resolve_package(driver, "com.android.chrome")) == (
        "com.android.chrome"
    )


def test_open_app_rejects_an_unresolvable_name():
    driver = _AppDriver(["com.android.chrome"])
    with pytest.raises(ValueError, match="Unknown app"):
        asyncio.run(_resolve_package(driver, "Files"))


def test_open_app_surfaces_a_failed_launch():
    driver = _AppDriver(
        ["com.android.chrome"], start_result="Failed to start com.android.chrome"
    )
    with pytest.raises(RuntimeError, match="Failed to start"):
        asyncio.run(
            execute_action(
                driver,
                Action(op="open_app", args={"app": "com.android.chrome"}),
                screen=ScreenInfo(native_width=1080, native_height=2400),
            )
        )
