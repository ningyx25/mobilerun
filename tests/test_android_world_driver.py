"""Tests for the AndroidWorld docker-server driver and its CLI/agent wiring.

The driver talks to an AndroidWorld environment server over HTTP; here the
server is replaced by an in-process ``httpx.MockTransport`` handler map so the
tests exercise the real request/response contract without a docker container.
"""

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.cli import device_commands
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.tools.driver.android_world import AndroidWorldHttpDriver


def run(coro):
    return asyncio.run(coro)


def _jpeg_b64(width: int = 1080, height: int = 2400) -> str:
    output = io.BytesIO()
    Image.new("RGB", (width, height), color=(16, 20, 24)).save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode()


def _ui_element(**overrides) -> dict:
    element = {
        "text": None,
        "content_description": None,
        "class_name": "android.widget.TextView",
        "bbox": {"x_min": 0.1, "x_max": 0.5, "y_min": 0.1, "y_max": 0.2},
        "bbox_pixels": {"x_min": 100, "x_max": 500, "y_min": 200, "y_max": 300},
        "hint_text": None,
        "is_checked": None,
        "is_checkable": False,
        "is_clickable": True,
        "is_editable": False,
        "is_enabled": True,
        "is_focused": False,
        "is_focusable": False,
        "is_long_clickable": False,
        "is_scrollable": False,
        "is_selected": False,
        "is_visible": True,
        "package_name": "com.example.app",
        "resource_name": None,
        "tooltip": None,
        "resource_id": None,
        "metadata": None,
    }
    element.update(overrides)
    return element


def _ok(**json_body) -> httpx.Response:
    return httpx.Response(200, json=json_body)


def _base_handlers(
    elements: list | None = None,
    activity: str = "com.example.app/.MainActivity",
) -> dict:
    """Handler map covering the endpoints every test path needs."""
    return {
        ("GET", "/health"): lambda r: _ok(status="success"),
        ("POST", "/hide_automation_ui"): lambda r: _ok(status="success"),
        ("POST", "/state"): lambda r: _ok(
            image_b64=_jpeg_b64(), ui_elements=elements or []
        ),
        ("GET", "/adb/current_activity"): lambda r: _ok(activity=activity),
        ("POST", "/screenshot"): lambda r: _ok(image_b64=_jpeg_b64()),
        ("POST", "/adb/tap"): lambda r: _ok(status="success"),
    }


def make_driver(handlers: dict, url: str = "http://localhost:5000"):
    """Build a driver over an httpx.MockTransport backed by *handlers*."""
    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        responder = handlers.get((request.method, request.url.path))
        if responder is None:
            return httpx.Response(
                404,
                json={"detail": f"no handler for {request.method} {request.url.path}"},
            )
        return responder(request)

    driver = AndroidWorldHttpDriver(url=url, transport=httpx.MockTransport(handler))
    return driver, recorded


def _paths(recorded: list[httpx.Request]) -> list[str]:
    return [request.url.path for request in recorded]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_connect_succeeds_and_hides_automation_ui():
    driver, recorded = make_driver(_base_handlers())

    run(driver.connect())

    assert _paths(recorded) == ["/health", "/hide_automation_ui"]


def test_connect_health_failure_reports_server_reason():
    handlers = _base_handlers()
    handlers[("GET", "/health")] = lambda r: httpx.Response(
        503, json={"status": "error", "reason": "qemu process not found"}
    )
    driver, _ = make_driver(handlers)

    with pytest.raises(ConnectionError, match="qemu process not found"):
        run(driver.connect())


def test_connect_connection_error_becomes_connection_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    driver = AndroidWorldHttpDriver(
        url="http://localhost:5000", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ConnectionError, match="Could not connect"):
        run(driver.connect())


def test_invalid_url_rejected():
    with pytest.raises(ValueError):
        AndroidWorldHttpDriver(url="emulator-5554")


# ---------------------------------------------------------------------------
# Input actions
# ---------------------------------------------------------------------------


def test_tap_uses_adb_tap_endpoint():
    driver, recorded = make_driver(_base_handlers())

    run(driver.tap(540, 1200))

    request = recorded[-1]
    assert request.url.path == "/adb/tap"
    assert request.url.params["x"] == "540"
    assert request.url.params["y"] == "1200"


def test_swipe_and_drag_use_generic_request():
    handlers = _base_handlers()
    handlers[("POST", "/adb/generic_request")] = lambda r: _ok(status=0, output="")
    driver, recorded = make_driver(handlers)

    run(driver.swipe(10, 20, 30, 40, duration_ms=500))
    assert json.loads(recorded[-1].content) == [
        "shell",
        "input",
        "swipe",
        "10",
        "20",
        "30",
        "40",
        "500",
    ]

    run(driver.drag(1, 2, 3, 4, duration=1.5))
    assert json.loads(recorded[-1].content) == [
        "shell",
        "input",
        "draganddrop",
        "1",
        "2",
        "3",
        "4",
        "1500",
    ]


def test_input_text_clear_selects_all_then_deletes_then_types():
    handlers = _base_handlers()
    handlers[("POST", "/adb/generic_request")] = lambda r: _ok(status=0, output="")
    handlers[("POST", "/adb/type_text")] = lambda r: _ok(status="success")
    driver, recorded = make_driver(handlers)

    assert run(driver.input_text("hello world", clear=True)) is True

    generic_bodies = [
        json.loads(request.content)
        for request in recorded
        if request.url.path == "/adb/generic_request"
    ]
    assert generic_bodies == [
        ["shell", "input", "keycombination", "113", "29"],
        ["shell", "input", "keyevent", "67"],
    ]
    type_request = recorded[-1]
    assert type_request.url.path == "/adb/type_text"
    assert type_request.url.params["text"] == "hello world"


def test_press_button_maps_to_android_keycode():
    handlers = _base_handlers()
    handlers[("POST", "/adb/press_key")] = lambda r: _ok(status="success")
    driver, recorded = make_driver(handlers)

    run(driver.press_button("back"))
    assert recorded[-1].url.path == "/adb/press_key"
    assert recorded[-1].url.params["keycode"] == "4"

    with pytest.raises(ValueError, match="not supported"):
        run(driver.press_button("nope"))


# ---------------------------------------------------------------------------
# App management
# ---------------------------------------------------------------------------


def test_start_app_uses_monkey_and_reports_failure_output():
    handlers = _base_handlers()
    handlers[("POST", "/adb/generic_request")] = lambda r: _ok(
        status=0, output="Events injected: 1"
    )
    driver, recorded = make_driver(handlers)

    assert run(driver.start_app("com.example.app")) == "Started com.example.app"
    assert json.loads(recorded[-1].content) == [
        "shell",
        "monkey",
        "-p",
        "com.example.app",
        "-c",
        "android.intent.category.LAUNCHER",
        "1",
    ]

    handlers[("POST", "/adb/generic_request")] = lambda r: _ok(
        status=0, output="** No activities found to run, monkey aborted."
    )
    assert run(driver.start_app("com.missing.app")).startswith("Failed to start")


def test_get_apps_include_system_switches_endpoint():
    handlers = _base_handlers()
    handlers[("GET", "/adb/all_packages")] = lambda r: _ok(
        packages=["com.foo", "com.systemui"]
    )
    handlers[("POST", "/adb/generic_request")] = lambda r: _ok(
        status=0, output="package:com.foo\npackage:com.bar\n"
    )
    driver, recorded = make_driver(handlers)

    assert run(driver.list_packages(include_system=False)) == ["com.bar", "com.foo"]
    assert run(driver.get_apps(include_system=True)) == [
        {"package": "com.foo", "label": "com.foo"},
        {"package": "com.systemui", "label": "com.systemui"},
    ]
    pm_request = next(
        request for request in recorded if request.url.path == "/adb/generic_request"
    )
    assert json.loads(pm_request.content) == ["shell", "pm", "list", "packages", "-3"]


def test_get_date_reads_generic_request_output():
    handlers = _base_handlers()
    handlers[("POST", "/adb/generic_request")] = lambda r: _ok(
        status=0, output="2026-10-05T12:00:00\n"
    )
    driver, _ = make_driver(handlers)

    assert run(driver.get_date()) == "2026-10-05T12:00:00"


# ---------------------------------------------------------------------------
# State / observation
# ---------------------------------------------------------------------------


def test_screenshot_returns_png_bytes():
    driver, _ = make_driver(_base_handlers())

    data = run(driver.screenshot())

    image = Image.open(io.BytesIO(data))
    assert image.format == "PNG"
    assert image.size == (1080, 2400)


def test_get_ui_tree_converts_to_portal_shape_and_feeds_android_provider():
    elements = [
        _ui_element(
            text="VLC",
            bbox_pixels={"x_min": 491, "x_max": 588, "y_min": 808, "y_max": 879},
        ),
        _ui_element(
            content_description="Navigate up",
            is_clickable=True,
            bbox_pixels={"x_min": 0, "x_max": 147, "y_min": 128, "y_max": 275},
        ),
        _ui_element(
            text="Search",
            is_editable=True,
            is_focused=True,
            bbox_pixels={"x_min": 100, "x_max": 900, "y_min": 300, "y_max": 400},
        ),
    ]
    driver, _ = make_driver(_base_handlers(elements=elements))

    tree = run(driver.get_ui_tree())

    assert tree["device_context"]["screen_bounds"] == {"width": 1080, "height": 2400}
    assert tree["phone_state"]["packageName"] == "com.example.app"
    # currentApp keeps the full activity (agents record it as the current
    # activity); packageName stays the package.
    assert tree["phone_state"]["currentApp"] == "com.example.app/.MainActivity"
    assert tree["phone_state"]["focusedElement"] == {"text": "Search"}
    assert tree["phone_state"]["isEditable"] is True
    first = tree["a11y_tree"]["children"][0]
    assert first["boundsInScreen"] == {
        "left": 491,
        "top": 808,
        "right": 588,
        "bottom": 879,
    }
    assert first["text"] == "VLC"

    # The converted state must be consumable by the standard Android pipeline.
    from mobilerun.tools.filters import DetailedFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    provider = AndroidStateProvider(
        driver, tree_filter=DetailedFilter(), tree_formatter=IndexedFormatter()
    )
    state = run(provider.get_state())

    assert state.screen_width == 1080
    assert state.screen_height == 2400
    assert "VLC" in state.formatted_text
    assert "Navigate up" in state.formatted_text
    assert "**Keyboard:** Visible" in state.formatted_text
    assert state.get_element_coords(1)


def test_get_ui_tree_maps_resource_name_when_resource_id_missing():
    """The server's default a11y path leaves resource_id unset and carries
    the view id in resource_name; resourceId must fall back to it so keyboard
    filtering and the formatter's text fallback keep working."""
    elements = [
        _ui_element(
            text="q",
            class_name="android.widget.Button",
            resource_id=None,
            resource_name="com.google.android.inputmethod.latin:id/key_pos_q",
        )
    ]
    driver, _ = make_driver(_base_handlers(elements=elements))

    tree = run(driver.get_ui_tree())

    node = tree["a11y_tree"]["children"][0]
    assert node["resourceId"] == "com.google.android.inputmethod.latin:id/key_pos_q"

    # The keyboard filter now recognizes the Gboard element and drops it.
    from mobilerun.tools.filters import DetailedFilter

    filtered = DetailedFilter().filter(tree["a11y_tree"], tree["device_context"])
    assert filtered is not None
    assert filtered["children"] == []


def test_get_ui_tree_falls_back_to_logical_screen_size():
    handlers = _base_handlers()
    handlers[("POST", "/state")] = lambda r: _ok(image_b64="", ui_elements=[])
    handlers[("GET", "/adb/logical_screen_size")] = lambda r: _ok(
        width=720, height=1280
    )
    driver, _ = make_driver(handlers)

    tree = run(driver.get_ui_tree())

    assert tree["device_context"]["screen_bounds"] == {"width": 720, "height": 1280}


# ---------------------------------------------------------------------------
# Wiring: CLI driver factory + agent config
# ---------------------------------------------------------------------------


def _fake_config(**device_overrides) -> SimpleNamespace:
    device = SimpleNamespace(
        serial=None,
        use_tcp=False,
        platform="android",
        auto_setup=True,
        portal_mode="auto",
        control_backend=None,
        resolve_auth_token=lambda _url: None,
    )
    for key, value in device_overrides.items():
        setattr(device, key, value)
    return SimpleNamespace(device=device)


def test_create_driver_routes_url_serial_to_android_world(monkeypatch):
    calls = []

    class FakeAndroidWorldDriver:
        def __init__(self, url):
            calls.append(url)

        async def connect(self):
            pass

    monkeypatch.setattr(
        device_commands, "AndroidWorldHttpDriver", FakeAndroidWorldDriver
    )
    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _path: _fake_config(serial="http://localhost:5000"),
    )

    driver, is_ios = run(device_commands._create_driver(None, None, None, False))

    assert isinstance(driver, FakeAndroidWorldDriver)
    assert calls == ["http://localhost:5000"]
    assert is_ios is False


def test_create_driver_respects_android_world_backend_with_default_url(monkeypatch):
    calls = []

    class FakeAndroidWorldDriver:
        def __init__(self, url):
            calls.append(url)

        async def connect(self):
            pass

    monkeypatch.setattr(
        device_commands, "AndroidWorldHttpDriver", FakeAndroidWorldDriver
    )
    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _path: _fake_config(control_backend="android-world"),
    )

    driver, _ = run(device_commands._create_driver(None, None, None, False))

    assert isinstance(driver, FakeAndroidWorldDriver)
    assert calls == ["http://localhost:5000"]


def test_create_driver_does_not_reroute_other_backends(monkeypatch):
    """A visual-remote URL in the serial must not be hijacked by the
    android-world auto-routing: only a bare URL or an explicit android-world
    backend selects the new driver."""
    android_calls = []

    class FakeAndroidDriver:
        def __init__(self, serial, use_tcp, portal_mode):
            android_calls.append(serial)

        async def connect(self):
            pass

    monkeypatch.setattr(device_commands, "AndroidDriver", FakeAndroidDriver)
    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _path: _fake_config(
            serial="http://localhost:8090",
            control_backend="visual-remote",
            auto_setup=False,
        ),
    )

    driver, _ = run(device_commands._create_driver(None, None, None, False))

    assert isinstance(driver, FakeAndroidDriver)
    assert android_calls == ["http://localhost:8090"]


def test_mobile_config_accepts_android_world_backend():
    config = MobileConfig.from_dict(
        {
            "agent": {"name": "external-agent"},
            "device": {
                "control_backend": "android-world",
                "serial": "http://localhost:5000",
            },
        }
    )

    assert config.device.control_backend == "android-world"
    assert config.device.serial == "http://localhost:5000"


def test_mobile_agent_does_not_force_vision_for_android_world():
    """Unlike visual-remote, the android-world backend serves a full a11y
    tree, so screenshot-only vision must not be forced on."""
    config = MobileConfig.from_dict(
        {
            "agent": {"name": "external-agent"},
            "device": {"control_backend": "android-world"},
        }
    )

    agent = MobileAgent("Check Wi-Fi", config=config)

    assert agent.config.agent.vision_only is False


def test_run_cli_offers_android_world_control_backend():
    from click.testing import CliRunner

    from mobilerun.cli.main import cli

    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0
    assert "android-world" in result.output


if __name__ == "__main__":
    pytest.main([__file__])
