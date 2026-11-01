"""Action execution shared by every environment backend.

The environment service is the only place in the harness that is allowed to
touch a device. Everything it does is a mapping onto the ``DeviceDriver``
contract, so no backend re-implements input handling:

=====================  ============================================
``Action.op``          ``DeviceDriver`` call
=====================  ============================================
``tap``                ``tap(x, y)``
``long_press``         ``swipe(x, y, x, y, 1000)``  (the long-press idiom)
``swipe``              ``swipe(x1, y1, x2, y2, duration_ms)``
``drag``               ``drag(x1, y1, x2, y2, duration_s)``
``type``               ``input_text(text, clear)``
``key``                ``press_button(button)`` / ``press_key_code(code)``
``open_app``           ``start_app(app, activity)``
``stop_app``           ``stop_app(app)``
``wait``               ``asyncio.sleep`` (the environment owns the clock)
``complete``           terminal marker, no device I/O
=====================  ============================================
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from mobilerun.harness.protocol import (
    Action,
    CoordinateSpace,
    ErrorCode,
    ProtocolError,
    ScreenInfo,
)

logger = logging.getLogger("mobilerun")

# Long press is expressed to drivers as a zero-length swipe, which every
# backend already interprets as "hold here" (see the AndroidWorld driver).
_LONG_PRESS_MS = 1000.0


def _to_device_point(
    x: float,
    y: float,
    *,
    screen: ScreenInfo,
    space: CoordinateSpace,
) -> tuple[int, int]:
    """Convert an action-space point into the driver's input pixel space."""
    if space == "normalized_1000":
        fx = max(0.0, min(1000.0, float(x))) / 1000.0
        fy = max(0.0, min(1000.0, float(y))) / 1000.0
        return (
            int(round(fx * screen.action_width)),
            int(round(fy * screen.action_height)),
        )
    # screenshot_pixels: scale only when the input layer uses another size
    # (iOS points, downscaled frames). Identity otherwise, which is the
    # Android case for both docker backends.
    if (
        screen.action_width == screen.native_width
        and screen.action_height == screen.native_height
    ):
        return int(round(float(x))), int(round(float(y)))
    sx = screen.action_width / max(1, screen.native_width)
    sy = screen.action_height / max(1, screen.native_height)
    return int(round(float(x) * sx)), int(round(float(y) * sy))


async def execute_action(
    driver: Any,
    action: Action,
    *,
    screen: ScreenInfo,
    default_space: CoordinateSpace = "screenshot_pixels",
) -> str:
    """Run one protocol action against *driver*; return a human summary.

    Device failures are returned as a failed :class:`ActionResult`-style string
    rather than raised: the harness treats a rejected action as feedback for
    the agent, never as an aborted episode (the same rule the tool registry
    follows -- see ``agent/CLAUDE.md`` §7.1).
    """
    op = action.op
    args: Dict[str, Any] = action.args
    space: CoordinateSpace = action.coordinate_space or default_space

    # Capabilities are advertised up front (``Capabilities.ops``); re-check at
    # execution time so a client acting on a stale capability snapshot gets a
    # precise error instead of an AttributeError deep in a driver.
    if op not in supported_ops(driver):
        raise ProtocolError(
            400,
            ErrorCode.UNSUPPORTED_ACTION,
            f"This environment cannot execute '{op}'",
            {"supported": supported_ops(driver)},
        )

    if op == "complete":
        message = str(args.get("message") or "")
        ok = bool(args.get("success", True))
        return f"{'Completed' if ok else 'Declared failure'}: {message}".strip()

    if op == "wait":
        duration = float(args.get("duration_s", 1.0))
        await asyncio.sleep(max(0.0, duration))
        return f"Waited {duration:.1f}s"

    if op in ("tap", "long_press"):
        x, y = _to_device_point(args["x"], args["y"], screen=screen, space=space)
        if op == "tap":
            await driver.tap(x, y)
            return f"Tapped at ({x}, {y})"
        await driver.swipe(x, y, x, y, _LONG_PRESS_MS)
        return f"Long-pressed at ({x}, {y})"

    if op == "swipe":
        x1, y1 = _to_device_point(args["x1"], args["y1"], screen=screen, space=space)
        x2, y2 = _to_device_point(args["x2"], args["y2"], screen=screen, space=space)
        duration = float(args.get("duration_ms", 1000.0))
        await driver.swipe(x1, y1, x2, y2, duration)
        return f"Swiped ({x1}, {y1}) -> ({x2}, {y2}) in {duration:.0f}ms"

    if op == "drag":
        x1, y1 = _to_device_point(args["x1"], args["y1"], screen=screen, space=space)
        x2, y2 = _to_device_point(args["x2"], args["y2"], screen=screen, space=space)
        duration_s = float(args.get("duration_s", 3.0))
        await driver.drag(x1, y1, x2, y2, duration_s)
        return f"Dragged ({x1}, {y1}) -> ({x2}, {y2}) in {duration_s:.1f}s"

    if op == "type":
        text = str(args.get("text", ""))
        clear = bool(args.get("clear", False))
        await driver.input_text(text, clear=clear)
        return f"Typed {len(text)} character(s)" + (" after clearing" if clear else "")

    if op == "key":
        keycode = args.get("keycode")
        if keycode is not None:
            await driver.press_key_code(int(keycode))
            return f"Pressed key code {int(keycode)}"
        button = str(args["button"])
        await driver.press_button(button)
        return f"Pressed {button}"

    if op == "open_app":
        activity = args.get("activity")
        result = await driver.start_app(str(args["app"]), activity)
        return str(result)

    if op == "stop_app":
        result = await driver.stop_app(str(args["app"]))
        return str(result)

    raise ProtocolError(
        400,
        ErrorCode.UNSUPPORTED_ACTION,
        f"Unsupported action '{op}'",
    )


def supported_ops(driver: Any) -> list[str]:
    """Protocol ops an environment backend can execute.

    Derived from ``DeviceDriver.supported`` so a backend's advertised
    capabilities can never drift from what it can really do.
    """
    verbs: set[str] = set(getattr(driver, "supported", set()) or set())
    ops: list[str] = ["wait", "complete"]
    if {"tap"} <= verbs:
        ops += ["tap", "long_press"]
    if {"swipe"} <= verbs:
        ops += ["swipe"]
    if {"drag"} <= verbs:
        ops += ["drag"]
    if {"input_text"} <= verbs:
        ops += ["type"]
    if {"press_button"} <= verbs or {"press_key_code"} <= verbs:
        ops += ["key"]
    if {"start_app"} <= verbs:
        ops += ["open_app"]
    if {"stop_app"} <= verbs:
        ops += ["stop_app"]
    return sorted(ops)


def screen_from_state(
    state: Optional[Dict[str, Any]],
    *,
    native_width: int,
    native_height: int,
    input_size: Optional[tuple[int, int]] = None,
) -> ScreenInfo:
    """Build :class:`ScreenInfo`, preferring the state's declared bounds."""
    bounds = ((state or {}).get("device_context") or {}).get("screen_bounds") or {}
    try:
        width = int(bounds.get("width") or native_width)
        height = int(bounds.get("height") or native_height)
    except (TypeError, ValueError):
        width, height = native_width, native_height
    input_width, input_height = input_size or (None, None)
    return ScreenInfo(
        native_width=width,
        native_height=height,
        input_width=input_width,
        input_height=input_height,
    )
