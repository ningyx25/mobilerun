"""MobileGym browser-simulator driver.

MobileGym has no HTTP control API: the nginx gateway on :4180 serves the
static simulator SPA and ``/api/gw/*`` is only a CORS proxy. Every device verb
is therefore executed in-process against ``bench_env.env.mobile_gym.MobileGymEnv``,
which drives the page through Playwright ``page.evaluate()`` of the
``window.__SIM__`` / ``__SIM_INPUT__`` globals.

``bench_env`` lives outside this repository (in ``mobilegym/``) and is imported
lazily so that importing this module never requires Playwright or a PYTHONPATH.
The benchmark runner owns the env lifecycle (``start``/``wait_ready``/``close``)
and injects it here.

Coordinates: the driver receives absolute screenshot pixels (1080x2400 at
device_scale_factor=3), which is exactly the ``physical`` coordinate space
``MobileGymEnv`` expects. Do not rescale here -- mobilerun's ``UIState`` already
converted the agent's 0-1000 grid into native pixels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from mobilerun_core_local.driver.base import DeviceDriver, unsupported

logger = logging.getLogger("mobilerun")

# ``device.control_backend`` value selecting this driver; the simulator URL
# travels in ``device.serial`` (mirrors android-world / mobile-world).
MOBILE_GYM_CONNECTION = "mobile-gym"
MOBILE_GYM_DEFAULT_URL = "https://localhost:4180"

# The simulator only exposes three system buttons. ``press_key_code`` maps the
# matching Android keycodes onto them.
_NAV_BUTTONS = {"back": "BACK", "home": "HOME", "enter": "ENTER"}
_KEYCODE_TO_BUTTON = {"4": "back", "3": "home", "66": "enter"}

# ``long_press`` in the agent toolset is routed through a same-point swipe with
# a 1000 ms duration; anything shorter is a regular swipe.
_LONG_PRESS_MIN_MS = 800


def validate_mobile_gym_url(url: str) -> str:
    """Validate and normalize a MobileGym simulator base URL."""
    normalized = url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "mobile-gym device must be the simulator URL, "
            "e.g. https://localhost:4180"
        )
    return normalized


class MobileGymBrowserDriver(DeviceDriver):
    """DeviceDriver facade over an in-process ``MobileGymEnv``.

    The simulator is a React DOM app with no accessibility tree, so
    ``get_ui_tree`` returns an empty portal-shaped stub and the benchmark must
    run with ``vision_only: true``.
    """

    platform = "Android"

    supported = {
        "tap",
        "swipe",
        "drag",
        "input_text",
        "press_button",
        "press_key_code",
        "start_app",
        "get_apps",
        "list_packages",
        "screenshot",
        "get_ui_tree",
    }
    supported_buttons = set(_NAV_BUTTONS)

    def __init__(self, env: Any) -> None:
        self.env = env
        self._answer: str = ""
        self._connected = False

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return
        if self.env is None:
            raise ConnectionError("MobileGymBrowserDriver has no env attached")
        self._connected = True
        logger.info("MobileGym browser driver attached to %s", self.env.url)

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def close(self) -> None:
        """Detach only; the runner owns the env/browser lifecycle."""
        self._connected = False

    # -- bench_env bridge ---------------------------------------------------

    @staticmethod
    def _make_action(action_type: str, data: Dict[str, Any]) -> Any:
        """Build a ``bench_env`` Action lazily (optional cross-repo import)."""
        from bench_env.env.base import Action, ActionType

        return Action(ActionType(action_type), data)

    async def _step(self, action_type: str, data: Dict[str, Any]) -> None:
        await self.ensure_connected()
        await self.env.step(self._make_action(action_type, data))

    # -- input actions ------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        await self._step("CLICK", {"point": [int(x), int(y)]})

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        p1, p2 = [int(x1), int(y1)], [int(x2), int(y2)]
        if p1 == p2 and duration_ms >= _LONG_PRESS_MIN_MS:
            await self._step("LONG_PRESS", {"point": p1, "duration": int(duration_ms)})
            return
        await self._step(
            "SWIPE",
            {"point1": p1, "point2": p2, "duration": int(duration_ms)},
        )

    async def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 3.0,
    ) -> None:
        await self._step(
            "DRAG",
            {
                "point1": [int(x1), int(y1)],
                "point2": [int(x2), int(y2)],
                "duration": int(duration * 1000),
            },
        )

    async def input_text(
        self,
        text: str,
        clear: bool = False,
        stealth: bool = False,
        wpm: int = 0,
    ) -> bool:
        await self._step("TYPE", {"value": str(text), "clear": bool(clear)})
        return True

    async def press_button(self, button: str) -> None:
        button_lower = str(button or "").strip().lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(
                f"Button '{button}' not supported. "
                f"Supported: {', '.join(sorted(self.supported_buttons))}"
            )
        await self._step(_NAV_BUTTONS[button_lower], {})

    async def press_key_code(self, key_code: int) -> None:
        button = _KEYCODE_TO_BUTTON.get(str(int(key_code)))
        if button is None:
            raise ValueError(
                f"Key code {key_code} is not supported by MobileGym "
                "(only back=4, home=3, enter=66)."
            )
        await self.press_button(button)

    # -- app management -----------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        identifier = str(package or "").strip()
        # AwakeHandler -> env._open_app() resolves both appIds and the Chinese /
        # English names in APP_NAME_MAP, so pass the identifier through as-is.
        # It resolves silently: an unknown name is skipped without an error,
        # so validate here to avoid reporting success for a no-op launch.
        name_map: Dict[str, str] = getattr(self.env, "APP_NAME_MAP", {}) or {}
        known = {str(k).lower() for k in name_map} | {
            str(v).lower() for v in name_map.values()
        }
        if identifier and identifier.lower() not in known:
            return (
                f"Failed to start '{identifier}': not a known MobileGym app. "
                "Open it from the home screen instead."
            )
        await self._step("AWAKE", {"value": identifier})
        return f"Started {identifier}"

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        name_map: Dict[str, str] = getattr(self.env, "APP_NAME_MAP", {}) or {}
        seen: Dict[str, str] = {}
        for label, app_id in name_map.items():
            seen.setdefault(app_id, label)
        return [
            {"package": app_id, "label": label}
            for app_id, label in sorted(seen.items())
        ]

    async def list_packages(self, include_system: bool = False) -> List[str]:
        return [entry["package"] for entry in await self.get_apps()]

    @unsupported("MobileGym simulator cannot force-stop an app")
    async def stop_app(self, package: str) -> str: ...

    @unsupported("MobileGym simulator cannot install APKs")
    async def install_app(self, path: str, **kwargs: Any) -> str: ...

    # -- state / observation ------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        """Return the simulator's native JPEG frame (1080x2400)."""
        await self.ensure_connected()
        return await self.env.page.screenshot(type="jpeg", quality=80)

    async def get_ui_tree(self) -> Dict[str, Any]:
        """Empty portal-shaped stub: the simulator is DOM, not uiautomator."""
        width, height = 1080, 2400
        get_size = getattr(self.env, "get_device_size", None)
        if get_size is not None:
            try:
                width, height = get_size()
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.debug("get_device_size failed: %s", exc)
        return {
            "a11y_tree": {"children": [], "className": "", "boundsInScreen": {}},
            "device_context": {"screen_bounds": {"width": width, "height": height}},
            "phone_state": {},
        }

    async def get_date(self) -> str:
        """The simulator's virtual clock as ``YYYY-MM-DDTHH:MM:SS``.

        MobileGym runs on a synthetic date (the OS state's ``os.time``), so
        the real host clock must not be used for date-sensitive prompts.
        """
        await self.ensure_connected()
        try:
            state = await self.env.get_state()
            os_state = (state or {}).get("os") or {}
            ts = os_state.get("time")
            if isinstance(ts, dict):
                ts = ts.get("timestamp")
            if not isinstance(ts, (int, float)) or ts <= 0:
                raise NotImplementedError("no os.time in MobileGym state")
            from datetime import datetime, timezone

            return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001 - caller falls back to ""
            raise NotImplementedError(f"MobileGym get_date failed: {exc}") from exc

    # -- scoring side -------------------------------------------------------

    async def submit_answer(self, text: str) -> None:
        """Store the answer for the grounded judge.

        Not a ``DeviceDriver`` verb -- same scoring-side hook as
        ``MobileWorldHttpDriver.submit_answer``. Grounded mode reads the
        AnswerSheet app state instead, but keeping the text answer lets the
        runner populate ``ExecutionResult.agent_answer`` for text-mode tasks.
        """
        self._answer = str(text)
