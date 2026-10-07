"""MobileWorld docker-server driver.

Talks to a MobileWorld environment server (``src/mobile_world/core/server.py``)
that runs the emulator inside Docker, typically on ``http://localhost:6800``:

- ``POST /step``  -- execute one JSONAction (click / drag / input_text / ...),
- ``GET /screenshot`` -- PNG (base64 or file path),
- ``GET /xml`` -- the uiautomator hierarchy,
- ``GET /state`` -- viewport size + current activity/app,
- ``GET /health`` / ``GET /init`` -- liveness + controller warm-up.

This backend is independent of the AndroidWorld driver (they are sibling
integrations, not layered): it maps the ``DeviceDriver`` verbs onto
MobileWorld's ``/step`` action vocabulary and converts the uiautomator XML
into the same portal-shaped state dict the Android state pipeline consumes,
so the tree-based toolset (indexed click/type, element filtering) works.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from mobilerun_core_local.driver.base import DeviceDisconnectedError, DeviceDriver, unsupported

from mobilerun.tools.driver.android_world import looks_like_http_url
from mobilerun.tools.helpers.images import ensure_complete_image

logger = logging.getLogger("mobilerun")

# ``device.control_backend`` value selecting this driver; the base URL travels
# in ``device.serial`` (mirrors android-world / visual-remote).
MOBILE_WORLD_CONNECTION = "mobile-world"
MOBILE_WORLD_DEFAULT_URL = "http://localhost:6800"
MOBILE_WORLD_DEFAULT_DEVICE = "emulator-5554"

# Port convention: 680* = MobileWorld servers (500* = AndroidWorld servers).
MOBILE_WORLD_PORT_PREFIX = "680"

_DEFAULT_TIMEOUT = 30.0
_STATE_TIMEOUT = 60.0

# A screenshot fetch is retried when the frame comes back incomplete (the
# server's exec-out capture can exit 0 while writing a truncated PNG). Keep
# this small: the server already retries capture paths internally, so the
# client layer only covers the residual rate; 2 x 3 rounds is the worst case.
_SCREENSHOT_ATTEMPTS = 2
_SCREENSHOT_RETRY_SLEEP = 0.5

# MobileWorld's /step only implements these three key events; everything else
# has no server-side action, so the driver advertises exactly this set and the
# tool registry prunes the rest.
_NAV_BUTTONS = {
    "back": "navigate_back",
    "home": "navigate_home",
    "enter": "keyboard_enter",
}
_KEYCODE_TO_BUTTON = {"4": "back", "3": "home", "66": "enter"}

_APPS_PATH = Path(__file__).with_name("mobile_world_apps.json")


def looks_like_mobile_world_url(value: str | None) -> bool:
    """True for an http(s) URL on a 680* port (a MobileWorld server)."""
    if not looks_like_http_url(value):
        return False
    port = urlparse(value.strip()).port
    return port is not None and str(port).startswith(MOBILE_WORLD_PORT_PREFIX)


def validate_mobile_world_url(url: str) -> str:
    """Validate and normalize a MobileWorld server base URL."""
    normalized = url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "mobile-world device must be the environment server base URL, "
            "e.g. http://localhost:6800"
        )
    return normalized


def load_app_table() -> List[Dict[str, str]]:
    """Bundled MobileWorld app table: [{package, label (app name)}]."""
    with open(_APPS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


# uiautomator bounds look like "[left,top][right,bottom]".
def _parse_bounds(value: str | None) -> Optional[Dict[str, int]]:
    if not value:
        return None
    value = value.strip()
    try:
        left, top = value[value.index("[") + 1 : value.index("]")].split(",")
        rest = value[value.index("]") + 1 :]
        right, bottom = rest[rest.index("[") + 1 : rest.index("]")].split(",")
        return {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        }
    except (ValueError, IndexError):
        return None


def _bool_attr(value: str | None) -> bool:
    return value == "true"


def _node_to_dict(element: ET.Element) -> Dict[str, Any]:
    """One uiautomator node -> a portal-shaped a11y-tree node."""
    attrs = element.attrib
    node: Dict[str, Any] = {
        "text": attrs.get("text") or "",
        "contentDescription": attrs.get("content-desc") or "",
        "resourceId": attrs.get("resource-id") or "",
        "className": attrs.get("class") or "",
        "packageName": attrs.get("package") or "",
        "isCheckable": _bool_attr(attrs.get("checkable")),
        "isChecked": _bool_attr(attrs.get("checked")),
        "isClickable": _bool_attr(attrs.get("clickable")),
        "isLongClickable": _bool_attr(attrs.get("long-clickable")),
        "isFocused": _bool_attr(attrs.get("focused")),
        "isScrollable": _bool_attr(attrs.get("scrollable")),
        "isSelected": _bool_attr(attrs.get("selected")),
        "isVisibleToUser": True,
    }
    bounds = _parse_bounds(attrs.get("bounds"))
    if bounds is not None:
        node["boundsInScreen"] = bounds
    children = [_node_to_dict(child) for child in list(element)]
    if children:
        node["children"] = children
    return node


def _root_bounds(children: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Full-screen bounds = the union of the top-level nodes' bounds."""
    nodes = [c for c in children if isinstance(c.get("boundsInScreen"), dict)]
    if not nodes:
        return None
    return {
        "left": min(c["boundsInScreen"]["left"] for c in nodes),
        "top": min(c["boundsInScreen"]["top"] for c in nodes),
        "right": max(c["boundsInScreen"]["right"] for c in nodes),
        "bottom": max(c["boundsInScreen"]["bottom"] for c in nodes),
    }


def _parse_hierarchy(xml: str) -> Dict[str, Any]:
    """Parse a uiautomator hierarchy string into a portal-shaped root node."""
    start = xml.find("<hierarchy")
    if start == -1:
        start = xml.find("<?xml")
    if start == -1:
        raise ValueError("uiautomator output did not contain XML")
    end = xml.find("</hierarchy>", start)
    end = end + len("</hierarchy>") if end != -1 else len(xml)
    root = ET.fromstring(xml[start:end].strip())
    children = [_node_to_dict(child) for child in list(root)]
    # The <hierarchy> element itself carries no bounds; synthesize a full-screen
    # root from the top-level nodes so the pipeline sees a sane extent.
    bounds = _parse_bounds(root.attrib.get("bounds")) or _root_bounds(children)
    device_context: Dict[str, Any] = {}
    if bounds is not None:
        device_context["screen_bounds"] = {
            "width": bounds["right"] - bounds["left"],
            "height": bounds["bottom"] - bounds["top"],
        }
    return {
        "a11y_tree": {
            "children": children,
            "className": "",
            "boundsInScreen": bounds or {},
        },
        "device_context": device_context,
    }


class MobileWorldHttpDriver(DeviceDriver):
    """Driver for a MobileWorld docker environment server."""

    platform = "Android"
    _server_label = "MobileWorld server"

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_button",
        "drag",
        "start_app",
        "get_apps",
        "list_packages",
        "screenshot",
        "get_ui_tree",
    }
    supported_buttons = set(_NAV_BUTTONS)

    def __init__(
        self,
        url: str = MOBILE_WORLD_DEFAULT_URL,
        device: str = MOBILE_WORLD_DEFAULT_DEVICE,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = validate_mobile_world_url(url)
        self.device = device or MOBILE_WORLD_DEFAULT_DEVICE
        self.timeout = timeout
        self.transport = transport
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._apps = load_app_table()
        self._by_package = {entry["package"]: entry["label"] for entry in self._apps}
        self._by_label = {
            entry["label"].lower(): entry["package"] for entry in self._apps
        }

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return
        self._client = httpx.AsyncClient(
            base_url=self.url, timeout=self.timeout, transport=self.transport
        )
        try:
            response = await self._client.get("/health")
            body = response.json() if response.content else {}
            if response.status_code >= 400 or not (
                isinstance(body, dict) and body.get("ok")
            ):
                status = body.get("device_status") if isinstance(body, dict) else None
                raise ConnectionError(
                    f"{self._server_label} at {self.url} is not healthy"
                    + (f": {status}" if status else f" (HTTP {response.status_code})")
                )
            # /init warms the controller for this device (idempotent).
            await self._client.get("/init", params={"device": self.device})
        except ConnectionError:
            await self._client.aclose()
            self._client = None
            raise
        except Exception as exc:
            await self._client.aclose()
            self._client = None
            raise ConnectionError(
                f"Could not connect to {self._server_label} at {self.url}"
            ) from exc
        self._connected = True
        logger.info("Connected to %s at %s", self._server_label, self.url)

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def close(self) -> None:
        """Close the underlying HTTP client (idempotent)."""
        if self._client is not None:
            await self._client.aclose()
        self._client = None
        self._connected = False

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise ConnectionError(f"{type(self).__name__} is not connected")
        return self._client

    # -- HTTP helpers ------------------------------------------------------

    async def _get(
        self, path: str, *, params: dict, timeout: float | None = None
    ) -> Any:
        await self.ensure_connected()
        response = await self._http.get(
            path, params=params, timeout=timeout or self.timeout
        )
        self._raise_for_status(response)
        return response.json()

    async def _step(
        self, action_type: str, *, timeout: float | None = None, **fields: Any
    ) -> Any:
        """Execute one MobileWorld /step action."""
        await self.ensure_connected()
        payload = {
            "device": self.device,
            "action": {"action_type": action_type, **fields},
        }
        response = await self._http.post(
            "/step", json=payload, timeout=timeout or self.timeout
        )
        self._raise_for_status(response)
        return response.json()

    @classmethod
    def _raise_for_status(cls, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = ""
        try:
            body = response.json()
            detail = (
                str(body.get("detail") or body) if isinstance(body, dict) else str(body)
            )
        except Exception:
            detail = response.text[:200]
        raise RuntimeError(
            f"{response.request.method} {response.request.url} failed with "
            f"{response.status_code}" + (f": {detail}" if detail else "")
        )

    # -- input actions -----------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self._step("click", x=int(x), y=int(y))

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        # /step has a true arbitrary-point drag; use it (a same-point drag is
        # what the long_press tool routes through swipe for).
        await self._step(
            "drag",
            start_x=int(x1),
            start_y=int(y1),
            end_x=int(x2),
            end_y=int(y2),
            timeout=max(self.timeout, duration_ms / 1000 + 5),
        )

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        await self._step(
            "drag",
            start_x=int(x1),
            start_y=int(y1),
            end_x=int(x2),
            end_y=int(y2),
            timeout=max(self.timeout, duration + 5),
        )

    async def input_text(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        # MobileWorld's /step has no "clear field" action and its text input is
        # an ADB_INPUT_B64 broadcast to the focused field, so clear=True cannot
        # be honored server-side; the field must be emptied another way. We
        # append nothing and just type -- callers that need clearing should tap
        # a field-select-all affordance first.
        await self._step(
            "input_text", text=text, timeout=max(self.timeout, 10 + len(text))
        )
        return True

    async def press_button(self, button: str) -> None:
        button_lower = str(button or "").strip().lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(
                f"Button '{button}' not supported. "
                f"Supported: {', '.join(sorted(self.supported_buttons))}"
            )
        await self._step(_NAV_BUTTONS[button_lower])

    async def press_key_code(self, key_code: int) -> None:
        button = _KEYCODE_TO_BUTTON.get(str(int(key_code)))
        if button is None:
            raise ValueError(
                f"Key code {key_code} is not supported by the MobileWorld server "
                f"(only back=4, home=3, enter=66)."
            )
        await self.press_button(button)

    async def submit_answer(self, text: str) -> None:
        """Submit an answer into the server's interaction_cache.

        Answer-style tasks grade purely from this value (set by the ``answer``
        action), so a benchmark must submit before scoring. Not part of the
        DeviceDriver verb set -- it is a scoring-side action on /step.
        """
        await self._step("answer", text=str(text))

    # -- app management ----------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        # /step's open_app takes an app *name*, not a package; resolve either.
        identifier = str(package or "").strip()
        app_name = self._by_package.get(identifier) or self._by_label.get(
            identifier.lower(), identifier
        )
        try:
            result = await self._step("open_app", app_name=app_name, timeout=30)
        except Exception as exc:  # noqa: BLE001 - report, do not crash the agent
            # MobileWorld returns 500 (an AdbResponse validation error) when the
            # package behind the app name is not installed; surface it as a
            # normal launch failure so the agent can recover.
            return f"Failed to start {identifier} (app_name={app_name}): {exc}"
        text = str(result.get("result") or "")
        if text and text.lower().startswith(("failed", "error")):
            return f"Failed to start {identifier}: {text}"
        return f"Started {identifier} (app_name={app_name})"

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        return [
            {"package": entry["package"], "label": entry["label"]}
            for entry in self._apps
        ]

    async def list_packages(self, include_system: bool = False) -> List[str]:
        return sorted(self._by_package)

    @unsupported("MobileWorld server has no stop-app endpoint")
    async def stop_app(self, package: str) -> str: ...

    @unsupported("MobileWorld server cannot install client-local APKs")
    async def install_app(self, path: str, **kwargs: Any) -> str: ...

    # -- state / observation -----------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        """Fetch a screenshot, retrying an incomplete frame.

        The server's ``exec-out screencap`` path can exit 0 while writing a
        truncated PNG (a strict prefix of a valid file). Such a frame must
        never reach the agent's image pipeline: a decode failure there used to
        abort the whole episode. Retry once, then raise -- every call site
        already degrades to "skip this frame" on a failed capture.
        """
        last_error: Exception | None = None
        for attempt in range(1, _SCREENSHOT_ATTEMPTS + 1):
            try:
                data = await self._get(
                    "/screenshot",
                    params={"device": self.device, "return_b64": True},
                    timeout=_STATE_TIMEOUT,
                )
                b64 = str(data.get("b64_png") or "")
                if "," in b64:  # tolerate a data-URL prefix
                    b64 = b64.split(",")[-1]
                image = base64.b64decode(b64)
                return ensure_complete_image(image, source="/screenshot b64_png")
            except DeviceDisconnectedError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                last_error = exc
                logger.warning(
                    "Screenshot attempt %d/%d failed: %s",
                    attempt,
                    _SCREENSHOT_ATTEMPTS,
                    exc,
                )
                if attempt < _SCREENSHOT_ATTEMPTS:
                    await asyncio.sleep(_SCREENSHOT_RETRY_SLEEP)
        raise RuntimeError(
            f"Failed to capture a complete screenshot from {self.url} after "
            f"{_SCREENSHOT_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    async def get_ui_tree(self) -> Dict[str, Any]:
        """Fetch /xml (uiautomator) and convert it to the portal state dict."""
        data = await self._get(
            "/xml",
            params={"device": self.device, "mode": "uia", "return_content": True},
            timeout=_STATE_TIMEOUT,
        )
        parsed = _parse_hierarchy(str(data.get("content") or ""))
        if "screen_bounds" not in parsed["device_context"]:
            parsed["device_context"]["screen_bounds"] = await self._screen_size()
        parsed["phone_state"] = await self._phone_state()
        return parsed

    async def _phone_state(self) -> Dict[str, Any]:
        try:
            state = await self._get("/state", params={"device": self.device})
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.debug("MobileWorld /state failed: %s", exc)
            return {}
        activity = str(state.get("current_activity") or "")
        package = activity.split("/", 1)[0] if "/" in activity else activity
        return {
            "currentApp": activity or package,
            "packageName": package,
            "focusedElement": None,
            "isEditable": False,
        }

    async def _screen_size(self) -> Dict[str, int]:
        state = await self._get("/state", params={"device": self.device})
        width, height = state.get("viewport_size") or (1080, 2400)
        return {"width": int(width), "height": int(height)}
