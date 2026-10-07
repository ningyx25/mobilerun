"""Tests for the MobileWorld docker-server driver and its wiring.

The MobileWorld server is replaced by an ``httpx.MockTransport`` handler map so
the tests exercise the real request/response contract of ``/step``, ``/xml``,
``/screenshot``, ``/state`` and ``/health`` without a docker container.
"""

import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from mobilerun.tools.driver.android_world import (
    looks_like_android_world_url,
)
from mobilerun.tools.driver.mobile_world import (
    MobileWorldHttpDriver,
    looks_like_mobile_world_url,
    validate_mobile_world_url,
)


def run(coro):
    return asyncio.run(coro)


def _png_b64(width=1080, height=2400):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


UI_XML = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>"
    '<hierarchy rotation="0">'
    '<node index="0" text="" resource-id="" class="android.widget.FrameLayout" '
    'package="com.android.settings" content-desc="" clickable="false" '
    'bounds="[0,0][1080,2400]">'
    '<node index="0" text="Wi-Fi" resource-id="com.android.settings:id/wifi" '
    'class="android.widget.TextView" package="com.android.settings" '
    'content-desc="" checkable="false" checked="false" clickable="true" '
    'enabled="true" focusable="true" focused="false" scrollable="false" '
    'long-clickable="false" selected="false" bounds="[40,300][400,380]"/>'
    "</node></hierarchy>"
)


def _handlers():
    return {
        ("GET", "/health"): lambda r: httpx.Response(
            200, json={"ok": True, "devices": ["emulator-5554"], "device_status": {}}
        ),
        ("GET", "/init"): lambda r: httpx.Response(
            200, json={"device": "emulator-5554", "viewport_size": [1080, 2400]}
        ),
        ("GET", "/state"): lambda r: httpx.Response(
            200,
            json={
                "device": "emulator-5554",
                "viewport_size": [1080, 2400],
                "current_activity": "com.android.settings/.Settings",
                "current_app": "settings",
            },
        ),
        ("GET", "/screenshot"): lambda r: httpx.Response(
            200, json={"device": "emulator-5554", "b64_png": _png_b64()}
        ),
        ("GET", "/xml"): lambda r: httpx.Response(
            200, json={"device": "emulator-5554", "mode": "uia", "content": UI_XML}
        ),
        ("POST", "/step"): lambda r: httpx.Response(
            200, json={"device": "emulator-5554", "result": "OK"}
        ),
    }


def make_driver(handlers=None, url="http://localhost:6800"):
    recorded = []

    def handler(request):
        recorded.append(request)
        responder = (handlers or _handlers()).get((request.method, request.url.path))
        if responder is None:
            return httpx.Response(
                404, json={"detail": f"no handler {request.url.path}"}
            )
        return responder(request)

    driver = MobileWorldHttpDriver(url=url, transport=httpx.MockTransport(handler))
    return driver, recorded


def _step_bodies(recorded):
    return [json.loads(r.content) for r in recorded if r.url.path == "/step"]


# ---------------------------------------------------------------------------
# URL / port convention
# ---------------------------------------------------------------------------


def test_port_convention_and_validation():
    assert looks_like_mobile_world_url("http://localhost:6800") is True
    assert looks_like_mobile_world_url("http://localhost:6801") is True
    assert looks_like_mobile_world_url("http://localhost:5000") is False
    assert looks_like_mobile_world_url("emulator-5554") is False
    assert looks_like_android_world_url("http://localhost:5000") is True
    assert looks_like_android_world_url("http://localhost:6800") is False
    assert (
        validate_mobile_world_url("http://localhost:6800/") == "http://localhost:6800"
    )
    with pytest.raises(ValueError):
        validate_mobile_world_url("emulator-5554")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_connect_health_and_init():
    driver, recorded = make_driver()
    run(driver.connect())
    assert [r.url.path for r in recorded] == ["/health", "/init"]


def test_connect_unhealthy_raises_connection_error():
    handlers = _handlers()
    handlers[("GET", "/health")] = lambda r: httpx.Response(
        503, json={"ok": False, "device_status": {"emulator-5554": False}}
    )
    driver, _ = make_driver(handlers)
    with pytest.raises(ConnectionError, match="not healthy"):
        run(driver.connect())


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------


def test_tap_maps_to_click():
    driver, recorded = make_driver()
    run(driver.tap(540, 1200))
    body = _step_bodies(recorded)[-1]
    assert body["action"] == {"action_type": "click", "x": 540, "y": 1200}
    assert body["device"] == "emulator-5554"


def test_swipe_and_drag_map_to_drag():
    driver, recorded = make_driver()
    run(driver.swipe(1, 2, 3, 4, duration_ms=500))
    assert _step_bodies(recorded)[-1]["action"] == {
        "action_type": "drag",
        "start_x": 1,
        "start_y": 2,
        "end_x": 3,
        "end_y": 4,
    }
    run(driver.drag(5, 6, 7, 8, duration=1.0))
    assert _step_bodies(recorded)[-1]["action"]["action_type"] == "drag"


def test_press_button_maps_to_nav_actions():
    driver, recorded = make_driver()
    run(driver.press_button("home"))
    assert _step_bodies(recorded)[-1]["action"]["action_type"] == "navigate_home"
    run(driver.press_button("enter"))
    assert _step_bodies(recorded)[-1]["action"]["action_type"] == "keyboard_enter"
    run(driver.press_button("\nback\n"))
    assert _step_bodies(recorded)[-1]["action"]["action_type"] == "navigate_back"
    with pytest.raises(ValueError, match="not supported"):
        run(driver.press_button("menu"))


def test_press_key_code_allowlist():
    driver, recorded = make_driver()
    run(driver.press_key_code(3))
    assert _step_bodies(recorded)[-1]["action"]["action_type"] == "navigate_home"
    with pytest.raises(ValueError, match="not supported"):
        run(driver.press_key_code(82))


def test_input_text_maps_to_input_text():
    driver, recorded = make_driver()
    assert run(driver.input_text("hello world")) is True
    assert _step_bodies(recorded)[-1]["action"] == {
        "action_type": "input_text",
        "text": "hello world",
    }


def test_start_app_resolves_package_to_app_name():
    driver, recorded = make_driver()
    result = run(driver.start_app("com.android.settings"))
    assert "app_name=Settings" in result
    assert _step_bodies(recorded)[-1]["action"] == {
        "action_type": "open_app",
        "app_name": "Settings",
    }


def test_start_app_reports_launch_failure_without_raising():
    handlers = _handlers()
    handlers[("POST", "/step")] = lambda r: httpx.Response(
        500, json={"detail": "1 validation error for AdbResponse"}
    )
    driver, _ = make_driver(handlers)
    result = run(driver.start_app("com.android.settings"))
    assert result.startswith("Failed to start com.android.settings")


def test_get_apps_uses_bundled_table():
    driver, _ = make_driver()
    apps = run(driver.get_apps())
    assert {"package": "com.android.settings", "label": "Settings"} in apps
    assert "com.android.settings" in run(driver.list_packages())


def test_capability_set_is_pruned():
    assert MobileWorldHttpDriver.supported_buttons == {"back", "home", "enter"}
    assert "stop_app" not in MobileWorldHttpDriver.supported
    assert "install_app" not in MobileWorldHttpDriver.supported


# ---------------------------------------------------------------------------
# State / observation
# ---------------------------------------------------------------------------


def test_get_ui_tree_parses_uiautomator_into_portal_shape():
    driver, _ = make_driver()
    tree = run(driver.get_ui_tree())

    assert tree["device_context"]["screen_bounds"] == {"width": 1080, "height": 2400}
    assert tree["phone_state"]["packageName"] == "com.android.settings"
    node = tree["a11y_tree"]["children"][0]["children"][0]
    assert node["text"] == "Wi-Fi"
    assert node["resourceId"] == "com.android.settings:id/wifi"
    assert node["isClickable"] is True
    assert node["boundsInScreen"] == {
        "left": 40,
        "top": 300,
        "right": 400,
        "bottom": 380,
    }


def test_get_ui_tree_feeds_the_android_state_provider():
    from mobilerun.tools.filters import DetailedFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    driver, _ = make_driver()
    provider = AndroidStateProvider(
        driver, tree_filter=DetailedFilter(), tree_formatter=IndexedFormatter()
    )
    state = run(provider.get_state())

    assert state.screen_width == 1080 and state.screen_height == 2400
    assert "Wi-Fi" in state.formatted_text
    assert state.get_element_coords(2)  # the element is indexable


def test_screenshot_decodes_b64_png():
    driver, _ = make_driver()
    data = run(driver.screenshot())
    image = Image.open(io.BytesIO(data))
    assert image.format == "PNG"
    assert image.size == (1080, 2400)


# ---------------------------------------------------------------------------
# Wiring: port-based driver selection
# ---------------------------------------------------------------------------


def _fake_config(serial, control_backend=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        device=SimpleNamespace(
            serial=serial,
            use_tcp=False,
            platform="android",
            auto_setup=False,
            portal_mode="disabled",
            control_backend=control_backend,
        )
    )


def test_create_driver_routes_6800_to_mobile_world(monkeypatch):
    from mobilerun.cli import device_commands

    calls = []

    class FakeMobileWorld:
        def __init__(self, url):
            calls.append(url)

        async def connect(self):
            pass

    monkeypatch.setattr(device_commands, "MobileWorldHttpDriver", FakeMobileWorld)
    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _p=None: _fake_config("http://localhost:6800"),
    )

    driver, is_ios = run(device_commands._create_driver(None, None, None, False))
    assert isinstance(driver, FakeMobileWorld)
    assert calls == ["http://localhost:6800"]
    assert is_ios is False


def test_create_driver_routes_5000_to_android_world(monkeypatch):
    from mobilerun.cli import device_commands

    calls = []

    class FakeAndroidWorld:
        def __init__(self, url):
            calls.append(url)

        async def connect(self):
            pass

    monkeypatch.setattr(device_commands, "AndroidWorldHttpDriver", FakeAndroidWorld)
    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _p=None: _fake_config("http://localhost:5000"),
    )

    driver, _ = run(device_commands._create_driver(None, None, None, False))
    assert isinstance(driver, FakeAndroidWorld)
    assert calls == ["http://localhost:5000"]


def test_explicit_mobile_world_backend_uses_default_url(monkeypatch):
    from mobilerun.cli import device_commands

    calls = []

    class FakeMobileWorld:
        def __init__(self, url):
            calls.append(url)

        async def connect(self):
            pass

    monkeypatch.setattr(device_commands, "MobileWorldHttpDriver", FakeMobileWorld)
    monkeypatch.setattr(
        device_commands.ConfigLoader,
        "load",
        lambda _p=None: _fake_config(None, control_backend="mobile-world"),
    )

    driver, _ = run(device_commands._create_driver(None, None, None, False))
    assert isinstance(driver, FakeMobileWorld)
    assert calls == ["http://localhost:6800"]


if __name__ == "__main__":
    pytest.main([__file__])
