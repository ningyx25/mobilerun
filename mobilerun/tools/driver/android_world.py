"""AndroidWorld docker-server driver.

Talks to an AndroidWorld environment server (``android_world/server/
android_server.py``) that runs the emulator inside Docker and exposes its own
HTTP API, typically on ``http://localhost:5000``:

- ``/state`` -- JPEG screenshot + the server-side UI element list,
- ``/screenshot`` -- screenshot only,
- ``/adb/*`` -- input, key events, app management, device info.

This driver maps the ``DeviceDriver`` verbs onto those endpoints and converts
the server's flat UI element list into the portal-shaped state dict
(``a11y_tree`` / ``phone_state`` / ``device_context``) that the Android state
pipeline (``mobilerun.tools.ui.provider.AndroidStateProvider``) consumes, so
the full tree-based toolset (indexed click/type, element filtering) works.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from mobilerun_core_local.driver.base import DeviceDriver, unsupported
from PIL import Image

logger = logging.getLogger("mobilerun")

# ``device.control_backend`` value selecting this driver, mirroring the
# ``visual-remote`` convention: the base URL travels in ``device.serial``.
ANDROID_WORLD_CONNECTION = "android-world"
ANDROID_WORLD_DEFAULT_URL = "http://localhost:5000"

# The server's screenshot/state path can be slow while the UI settles; input
# actions use the client default.
_DEFAULT_TIMEOUT = 30.0
_STATE_TIMEOUT = 60.0

# Same button -> Android key code table as AndroidPortalHttpDriver.
_ANDROID_KEY_CODES = {
    "back": 4,
    "home": 3,
    "menu": 82,
    "enter": 66,
    "delete": 67,
    "escape": 111,
    "tab": 61,
    "space": 62,
    "search": 84,
    "page_up": 92,
    "page_down": 93,
    "volume_up": 24,
    "volume_down": 25,
    "power": 26,
    "media_play_pause": 85,
    "media_next": 87,
    "media_prev": 88,
}

# Ctrl+A select-all + DEL, mirroring the server's own clear_text sequence
# (android_world/env/actuation.py).
_SELECT_ALL = ["shell", "input", "keycombination", "113", "29"]
_DELETE = ["shell", "input", "keyevent", "67"]


# Port convention shared with the mobile-world backend: 500* = AndroidWorld
# servers, 680* = MobileWorld servers.
ANDROID_WORLD_PORT_PREFIX = "500"


def looks_like_http_url(value: str | None) -> bool:
    """True when *value* is an http(s) URL (never a valid ADB serial)."""
    return bool(value and value.strip().lower().startswith(("http://", "https://")))


def looks_like_android_world_url(value: str | None) -> bool:
    """True for an http(s) URL on a 500* port (an AndroidWorld server)."""
    if not looks_like_http_url(value):
        return False
    port = urlparse(value.strip()).port
    return port is not None and str(port).startswith(ANDROID_WORLD_PORT_PREFIX)


def validate_android_world_url(url: str) -> str:
    """Validate and normalize an AndroidWorld server base URL."""
    normalized = url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "android-world device must be the environment server base URL, "
            "e.g. http://localhost:5000"
        )
    return normalized


class AndroidWorldHttpDriver(DeviceDriver):
    """Driver for an AndroidWorld docker environment server."""

    platform = "Android"
    _server_label = "AndroidWorld server"

    supported = {
        "tap",
        "swipe",
        "input_text",
        "press_button",
        "press_key_code",
        "drag",
        "start_app",
        "stop_app",
        "get_apps",
        "list_packages",
        "screenshot",
        "get_ui_tree",
        "get_date",
    }
    supported_buttons = set(_ANDROID_KEY_CODES)

    def __init__(
        self,
        url: str = ANDROID_WORLD_DEFAULT_URL,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = validate_android_world_url(url)
        self.timeout = timeout
        self.transport = transport
        self._client: httpx.AsyncClient | None = None
        self._connected = False

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return
        self._client = httpx.AsyncClient(
            base_url=self.url, timeout=self.timeout, transport=self.transport
        )
        try:
            response = await self._client.get("/health")
            if response.status_code >= 400:
                reason = self._error_detail(response)
                raise ConnectionError(
                    f"{self._server_label} at {self.url} is not healthy"
                    + (f": {reason}" if reason else f" (HTTP {response.status_code})")
                )
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
        # Best-effort: hide the on-screen pointer/automation overlay so
        # screenshots stay clean. Older servers may not expose the endpoint.
        try:
            await self._client.post("/hide_automation_ui")
        except Exception as exc:
            logger.debug("hide_automation_ui failed (ignored): %s", exc)

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

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except Exception:
            return ""
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("reason") or body.get("message")
            if detail:
                return str(detail)
        return ""

    @classmethod
    def _raise_for_status(cls, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = cls._error_detail(response)
        raise RuntimeError(
            f"{response.request.method} {response.request.url} failed with "
            f"{response.status_code}" + (f": {detail}" if detail else "")
        )

    async def _get(
        self, path: str, *, params: dict | None = None, timeout: float | None = None
    ) -> Any:
        await self.ensure_connected()
        response = await self._http.get(
            path, params=params, timeout=timeout or self.timeout
        )
        self._raise_for_status(response)
        return response.json()

    async def _post(
        self,
        path: str,
        *,
        params: dict | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> Any:
        await self.ensure_connected()
        response = await self._http.post(
            path, params=params, json=json, timeout=timeout or self.timeout
        )
        self._raise_for_status(response)
        return response.json()

    async def _generic(
        self, args: list, *, server_timeout: float | None = None
    ) -> dict:
        """Issue an adb command through ``/adb/generic_request``."""
        params = {"timeout_sec": server_timeout} if server_timeout is not None else None
        result = await self._post(
            "/adb/generic_request",
            params=params,
            json=[str(arg) for arg in args],
            timeout=max(self.timeout, (server_timeout or 0) + 10),
        )
        return result if isinstance(result, dict) else {}

    # -- input actions -----------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self._post("/adb/tap", params={"x": int(x), "y": int(y)})

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        # The long_press tool routes through swipe(x, y, x, y, 1000), which
        # ``input swipe`` with identical endpoints interprets as a long press.
        await self._generic(
            [
                "shell",
                "input",
                "swipe",
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                int(duration_ms),
            ],
            server_timeout=max(10, duration_ms / 1000 + 5),
        )

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        await self._generic(
            [
                "shell",
                "input",
                "draganddrop",
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                int(duration * 1000),
            ],
            server_timeout=max(10, duration + 5),
        )

    async def input_text(
        self,
        text: str,
        clear: bool = False,
        stealth: bool = False,
        wpm: int = 0,
    ) -> bool:
        if clear:
            await self._generic(_SELECT_ALL)
            await self._generic(_DELETE)
        # Word-by-word typing with adb escaping happens server-side. Not the
        # /execute_action input_text action: that one also presses ENTER.
        await self._post(
            "/adb/type_text",
            params={"text": text},
            timeout=max(self.timeout, 10 + len(text)),
        )
        return True

    async def press_button(self, button: str) -> None:
        # Tolerate values wrapped in stray whitespace/newlines.
        button_lower = str(button or "").strip().lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(
                f"Button '{button}' not supported. "
                f"Supported: {', '.join(sorted(self.supported_buttons))}"
            )
        await self.press_key_code(_ANDROID_KEY_CODES[button_lower])

    async def press_key_code(self, key_code: int) -> None:
        await self._post("/adb/press_key", params={"keycode": str(int(key_code))})

    # -- app management ----------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        if activity:
            await self._post(
                "/adb/start_activity", params={"activity": f"{package}/{activity}"}
            )
            return f"Started {package}"
        result = await self._generic(
            [
                "shell",
                "monkey",
                "-p",
                package,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
            server_timeout=30,
        )
        output = str(result.get("output") or "")
        if "Events injected" in output:
            return f"Started {package}"
        return f"Failed to start {package}: {output.strip()[:200] or 'no output'}"

    async def stop_app(self, package: str) -> str:
        await self._generic(["shell", "am", "force-stop", package])
        return f"Stopped {package}"

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        packages = await self._packages(include_system)
        # The server exposes package names only; use them as labels too.
        return [{"package": package, "label": package} for package in packages]

    async def list_packages(self, include_system: bool = False) -> List[str]:
        return await self._packages(include_system)

    async def list_launchable_packages(self) -> List[str]:
        """Packages with a LAUNCHER activity -- what ``start_app`` can open.

        ``/adb/all_packages`` also carries providers, viewers and overlays that
        ``monkey -c LAUNCHER`` cannot start, so an open_app target chosen from
        that list can fail for reasons the caller cannot see. The launcher
        query returns exactly the apps that can be started.
        """
        result = await self._generic(
            [
                "shell",
                "cmd",
                "package",
                "query-activities",
                "--brief",
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.LAUNCHER",
            ],
            server_timeout=30,
        )
        output = str(result.get("output") or "")
        packages = set()
        for line in output.splitlines():
            if "/" not in line:
                continue
            # Activity rows read "pkg/activity"; the header and attribute
            # lines do not carry a package-shaped prefix.
            candidate = line.strip().split("/", 1)[0].strip()
            if candidate and " " not in candidate and "." in candidate:
                packages.add(candidate)
        return sorted(packages)

    async def _packages(self, include_system: bool) -> List[str]:
        if include_system:
            data = await self._get("/adb/all_packages")
            packages = data.get("packages") or []
        else:
            result = await self._generic(
                ["shell", "pm", "list", "packages", "-3"], server_timeout=30
            )
            output = str(result.get("output") or "")
            packages = [
                line.split("package:", 1)[1].strip()
                for line in output.splitlines()
                if line.startswith("package:")
            ]
        return sorted(str(package) for package in packages if package)

    @unsupported("android-world server cannot install client-local APKs")
    async def install_app(self, path: str, **kwargs: Any) -> str: ...

    # -- state / observation -----------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        data = await self._post(
            "/screenshot", params={"wait_to_stabilize": False}, timeout=_STATE_TIMEOUT
        )
        image_bytes = base64.b64decode(data["image_b64"])
        with Image.open(io.BytesIO(image_bytes)) as image:
            output = io.BytesIO()
            image.save(output, format="PNG")
        return output.getvalue()

    async def get_ui_tree(self) -> Dict[str, Any]:
        """Fetch /state and convert it to the portal-shaped state dict."""
        data = await self._post(
            "/state", params={"wait_to_stabilize": False}, timeout=_STATE_TIMEOUT
        )
        elements = [e for e in (data.get("ui_elements") or []) if isinstance(e, dict)]
        image_bytes = base64.b64decode(data.get("image_b64") or "")
        screen_width, screen_height = await self._screen_size(image_bytes)

        root = {
            "boundsInScreen": {
                "left": 0,
                "top": 0,
                "right": screen_width,
                "bottom": screen_height,
            },
            "className": "",
            "text": "",
            "contentDescription": "",
            "resourceId": "",
            "packageName": "",
            "children": [self._element_to_node(element) for element in elements],
        }
        return {
            "a11y_tree": root,
            "phone_state": await self._phone_state(elements),
            "device_context": {
                "screen_bounds": {"width": screen_width, "height": screen_height}
            },
        }

    @staticmethod
    def _element_to_node(element: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one server UI element into a portal a11y-tree node."""
        bbox = element.get("bbox_pixels") or {}
        return {
            "boundsInScreen": {
                "left": int(bbox.get("x_min", 0)),
                "top": int(bbox.get("y_min", 0)),
                "right": int(bbox.get("x_max", 0)),
                "bottom": int(bbox.get("y_max", 0)),
            },
            "className": element.get("class_name") or "",
            "text": element.get("text") or "",
            "contentDescription": element.get("content_description") or "",
            # The server's default a11y path leaves resource_id unset and
            # carries the view id in resource_name (e.g. "pkg:id/view"); the
            # portal-style resourceId feeds keyboard filtering and the
            # formatter's text fallback, so prefer whichever is populated.
            "resourceId": element.get("resource_id")
            or element.get("resource_name")
            or "",
            "packageName": element.get("package_name") or "",
            "hintText": element.get("hint_text") or "",
            "isCheckable": bool(element.get("is_checkable")),
            "isChecked": bool(element.get("is_checked")),
            "isClickable": bool(element.get("is_clickable")),
            "isLongClickable": bool(element.get("is_long_clickable")),
            "isEditable": bool(element.get("is_editable")),
            "isFocused": bool(element.get("is_focused")),
            "isVisibleToUser": bool(element.get("is_visible")),
            # The server reports these; pass them through so scroll-aware
            # consumers (the grounded executor's observation) can rely on the
            # flag instead of class-name heuristics.
            "isScrollable": bool(element.get("is_scrollable")),
            "isSelected": bool(element.get("is_selected")),
            "isEnabled": element.get("is_enabled", True) is not False,
            "children": [],
        }

    async def _phone_state(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the phone_state section from the current activity + focus."""
        activity = ""
        try:
            activity = (await self._get("/adb/current_activity")).get("activity") or ""
        except Exception as exc:
            logger.debug("Could not fetch current activity: %s", exc)

        package_name = activity.split("/", 1)[0] if activity else ""
        focused_text = ""
        is_editable = False
        for element in elements:
            if element.get("is_focused"):
                focused_text = (
                    element.get("text") or element.get("content_description") or ""
                )
                is_editable = bool(element.get("is_editable"))
                if not package_name:
                    package_name = element.get("package_name") or ""
                break
        return {
            # currentApp carries the full "pkg/Activity" (the agent records it
            # as the current activity); packageName stays the package.
            "currentApp": activity or package_name,
            "packageName": package_name,
            "focusedElement": {"text": focused_text} if focused_text else None,
            "isEditable": is_editable,
        }

    async def _screen_size(self, image_bytes: bytes) -> tuple[int, int]:
        """Screen bounds for the state, in screenshot pixel space.

        The /state image header gives the exact capture size (cheap to read,
        no full decode); it is the coordinate space both the element bounds
        and the model-facing screenshot live in. Falls back to the device's
        logical screen size.
        """
        if image_bytes:
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    width, height = image.size
                if width and height:
                    return int(width), int(height)
            except Exception as exc:
                logger.debug("Could not read /state image dimensions: %s", exc)
        data = await self._get("/adb/logical_screen_size")
        return int(data.get("width", 1080)), int(data.get("height", 2400))

    async def get_date(self) -> str:
        # No spaces in the format: the server joins args through adb shell,
        # which would split an unquoted space-containing argument.
        result = await self._generic(["shell", "date", "+%Y-%m-%dT%H:%M:%S"])
        return str(result.get("output") or "").strip()
