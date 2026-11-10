"""Backend adapters: the task catalogue must report real per-task budgets."""

from __future__ import annotations

import asyncio
import io

import httpx
import pytest
from PIL import Image

import mobilerun.harness.env_service.backend as backend_module
from mobilerun.benchmark import android_world as aw
from mobilerun.harness.protocol import ErrorCode, ProtocolError


class _FakeDriver:
    def __init__(self, url: str) -> None:
        self.url = url


class _FakeTaskClient:
    def __init__(self, url: str) -> None:
        self.url = url

    async def task_list(self):
        return ["Easy", "Hard", "Broken"]

    async def get_complexity(self, name: str, task_idx: int) -> float:
        if name == "Broken":
            raise RuntimeError("complexity endpoint failed")
        return {"Easy": 1.2, "Hard": 2.8}[name]

    async def close(self) -> None:
        return None


def test_android_world_catalogue_uses_each_task_complexity(monkeypatch):
    monkeypatch.setattr(backend_module, "AndroidWorldHttpDriver", _FakeDriver)
    monkeypatch.setattr(aw, "AndroidWorldTaskClient", _FakeTaskClient)

    backend = backend_module.AndroidWorldBackend("aw0", "http://aw.test")
    tasks = asyncio.run(backend.list_tasks())

    # AndroidWorld's budget is int(10 * complexity) per task, not a flat 50.
    # A task whose complexity cannot be read falls back to the old placeholder.
    assert [(t.id, t.complexity, t.max_steps) for t in tasks] == [
        ("Easy", 1.2, 12),
        ("Hard", 2.8, 28),
        ("Broken", None, 50),
    ]


class _DisconnectingDriver:
    """A driver whose device server drops the connection."""

    supported = {"tap"}

    async def screenshot(self):
        raise httpx.RemoteProtocolError(
            "Server disconnected without sending a response"
        )


class _ObserveBackend(backend_module.DriverBackend):
    async def list_tasks(self):
        return []

    async def prepare(self, task_id, **kwargs):
        raise NotImplementedError

    async def evaluate(self):
        return 0.0, ""


def test_observe_maps_a_transport_error_to_a_retryable_503():
    backend = _ObserveBackend("aw0", _DisconnectingDriver())
    with pytest.raises(ProtocolError) as caught:
        asyncio.run(backend.observe(seq=1))
    assert caught.value.status_code == 503
    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE


class _BoundsDriver:
    """A PNG whose declared a11y bounds disagree with the real screenshot."""

    supported = {"get_ui_tree"}

    def __init__(self, bounds):
        self._bounds = bounds

    async def screenshot(self):
        buf = io.BytesIO()
        Image.new("RGB", (1080, 2400), color=(10, 20, 30)).save(buf, format="PNG")
        return buf.getvalue()

    async def get_ui_tree(self):
        return {
            "device_context": {"screen_bounds": dict(self._bounds)},
            "a11y_tree": {"children": []},
        }


def test_observe_pins_screen_bounds_to_png_dims():
    backend = _ObserveBackend(
        "mw0", _BoundsDriver({"width": 1080, "height": 2268})
    )
    observation = asyncio.run(backend.observe(seq=1))

    assert (observation.screen.native_width, observation.screen.native_height) == (
        1080,
        2400,
    )
    assert observation.device_context["screen_bounds"] == {
        "width": 1080,
        "height": 2400,
    }


def test_observe_keeps_declared_bounds_with_input_space():
    driver = _BoundsDriver({"width": 540, "height": 1200})

    async def _input_coordinate_size(width, height):
        return (540, 1200)

    driver.input_coordinate_size = _input_coordinate_size
    backend = _ObserveBackend("mw0", driver)
    observation = asyncio.run(backend.observe(seq=1))

    # A separate input space (iOS points) must keep its declared bounds.
    assert observation.device_context["screen_bounds"] == {
        "width": 540,
        "height": 1200,
    }
    assert (observation.screen.action_width, observation.screen.action_height) == (
        540,
        1200,
    )


# ---------------------------------------------------------------------------
# MobileWorld post-action stabilization
# ---------------------------------------------------------------------------


def _settle_frame(body, bar=0):
    """A tiny screenshot whose status-bar strip can vary without changing UI."""
    image = Image.new("RGB", (108, 240), color=(body, body, body))
    strip = int(240 * backend_module.MW_STATUS_BAR_FRACTION)
    for y in range(strip):
        for x in range(108):
            image.putpixel((x, y), (bar, bar, bar))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class _SettleDriver:
    supported = {"screenshot", "tap"}

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    async def screenshot(self):
        frame = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return frame


def _mw_backend(driver):
    # Bypass __init__ (it builds httpx clients); stabilize/capabilities only
    # need the driver.
    backend = object.__new__(backend_module.MobileWorldBackend)
    backend.driver = driver
    return backend


def test_settle_signature_ignores_status_bar():
    a = _settle_frame(60, bar=10)
    b = _settle_frame(60, bar=200)
    different = _settle_frame(200, bar=10)
    assert backend_module._frames_settled(
        backend_module._settle_signature(a), backend_module._settle_signature(b)
    )
    assert not backend_module._frames_settled(
        backend_module._settle_signature(a), backend_module._settle_signature(different)
    )


def test_mobile_world_stabilize_waits_for_settled_frame(monkeypatch):
    monkeypatch.setattr(backend_module, "MW_STABILITY_SLEEP_S", 0.0)
    settled = _settle_frame(180)
    driver = _SettleDriver(
        [
            _settle_frame(20),
            _settle_frame(90),
            settled,
            settled,
            settled,
            settled,
        ]
    )
    asyncio.run(_mw_backend(driver).stabilize())
    # Two transition frames, then three consecutive identical reads.
    assert driver.calls == 5


def test_mobile_world_stabilize_bounded_by_timeout(monkeypatch):
    monkeypatch.setattr(backend_module, "MW_STABILITY_SLEEP_S", 0.0)
    monkeypatch.setattr(backend_module, "MW_STABILITY_TIMEOUT_S", 0.08)

    class _NeverSettles:
        supported = {"screenshot"}

        def __init__(self):
            self.calls = 0

        async def screenshot(self):
            self.calls += 1
            await asyncio.sleep(0.02)
            return _settle_frame(20 if self.calls % 2 else 160)

    driver = _NeverSettles()
    asyncio.run(_mw_backend(driver).stabilize())
    assert 1 <= driver.calls <= 8


def test_mobile_world_capabilities_still_reports_no_stabilize():
    backend = _mw_backend(_SettleDriver([_settle_frame(50)]))
    # Scoring relies on this: flipping it enables a zero-score recheck.
    assert backend.capabilities().supports_stabilize is False
