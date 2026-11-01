"""``DeviceDriver`` implementation that talks to an environment service.

This is what makes the split cheap on the agent side: ``MobileAgent`` accepts
an injected driver and never connects or closes it
(``agent/droid/droid_agent.py:326``/``:865``), and every agent -- FastAgent,
Manager+Executor, grounded, Jev -- reaches the device exclusively through the
``DeviceDriver`` verbs. Swapping the concrete driver for this one changes the
transport, not a single line of agent logic.

Relationship to the existing drivers: ``AndroidWorldHttpDriver`` speaks the
vendor's ``/adb/*`` API directly; this driver speaks *our* normalized
``/v1/sessions/*`` API. The environment service runs the former internally, so
the same device capability set is advertised either way.

Observation cache: one capture per agent step. The first state read after an
action fetches ``/observe``; ``screenshot()`` and ``get_ui_tree()`` then share
that frame until the next action invalidates it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mobilerun_core_local.driver.base import (
    DeviceDisconnectedError,
    DeviceDriver,
    unsupported,
)

from mobilerun.harness.client import EnvClient, ServiceError
from mobilerun.harness.protocol import (
    Action,
    Observation,
    ScreenInfo,
    SessionInfo,
    StepRequest,
    StepResponse,
)

logger = logging.getLogger("mobilerun")

# Protocol op -> driver verbs it enables. Used to rebuild ``supported`` from
# the environment's advertised capabilities so the tool registry prunes the
# same tools it would prune against the native driver.
_OP_VERBS: Dict[str, set[str]] = {
    "tap": {"tap"},
    "long_press": {"swipe"},
    "swipe": {"swipe"},
    "drag": {"drag"},
    "type": {"input_text"},
    "key": {"press_button", "press_key_code"},
    "open_app": {"start_app"},
    "stop_app": {"stop_app"},
}


class RemoteEnvDriver(DeviceDriver):
    """Device driver backed by one open environment session."""

    platform = "Android"

    def __init__(
        self,
        env_url: str,
        session_id: str,
        *,
        token: Optional[str] = None,
        client: Optional[EnvClient] = None,
        timeout: float = 60.0,
    ) -> None:
        self.env_url = env_url.rstrip("/")
        self.session_id = session_id
        self._client = client or EnvClient(env_url, token=token, timeout=timeout)
        self._owns_client = client is None
        self._connected = False
        self._observation: Optional[Observation] = None
        # Last seq the server showed us; used as the optional stale guard even
        # after the cached frame is invalidated by an action.
        self._last_seq: Optional[int] = None
        self._step_index = 0
        # Device actions are serialized: parallel tool calls would otherwise
        # race the step counter and the device itself.
        self._action_lock = asyncio.Lock()
        self.supported: set[str] = set()
        self.supported_buttons: set[str] = set()
        self.capabilities: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            info = await self._client.get_session(self.session_id)
        except ServiceError as exc:
            raise ConnectionError(
                f"Environment session '{self.session_id}' at {self.env_url}: {exc}"
            ) from exc
        self._configure(info)
        # Warm the cache so the agent's very first state read is free.
        self._remember(await self._client.observe(self.session_id))
        self._connected = True
        logger.info(
            "RemoteEnvDriver connected to %s session %s (%s)",
            self.env_url,
            self.session_id,
            ",".join(sorted(self.supported)),
        )

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def close(self) -> None:
        """Close the transport only; the session belongs to the harness."""
        self._connected = False
        if self._owns_client:
            await self._client.close()

    def _configure(self, info: SessionInfo) -> None:
        caps = info.capabilities
        verbs: set[str] = set(caps.verbs)
        for op in caps.ops:
            verbs |= _OP_VERBS.get(op, set())
        if caps.has_screenshot:
            verbs.add("screenshot")
        if caps.has_ui_tree:
            verbs.add("get_ui_tree")
        self.supported = verbs
        self.supported_buttons = set(caps.buttons)
        self.platform = caps.platform or "Android"
        self.capabilities = caps.model_dump(mode="json")

    # -- state / observation -----------------------------------------------

    @property
    def observation(self) -> Optional[Observation]:
        return self._observation

    def _remember(self, observation: Optional[Observation]) -> None:
        self._observation = observation
        if observation is not None:
            self._last_seq = observation.seq

    async def _current_observation(self) -> Observation:
        """Return the cached frame, capturing one if the cache is cold."""
        await self.ensure_connected()
        if self._observation is not None:
            return self._observation
        self._remember(await self._client.observe(self.session_id))
        assert self._observation is not None  # for type checkers
        return self._observation

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        observation = await self._current_observation()
        data = observation.screenshot_bytes()
        if not data:
            raise RuntimeError(
                f"Environment backend for session {self.session_id} returned no screenshot"
            )
        return data

    async def get_ui_tree(self) -> Dict[str, Any]:
        observation = await self._current_observation()
        return observation.state_dict()

    async def input_coordinate_size(
        self, screenshot_width: int, screenshot_height: int
    ) -> tuple[int, int]:
        screen = self._screen()
        if screen and screen.input_width and screen.input_height:
            return screen.input_width, screen.input_height
        return screenshot_width, screenshot_height

    async def get_date(self) -> str:
        await self.ensure_connected()
        try:
            return await self._client.device_date(self.session_id)
        except Exception as exc:  # noqa: BLE001 - informational only
            logger.debug("device_date failed: %s", exc)
            return ""

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        await self.ensure_connected()
        response = await self._client.device_apps(
            self.session_id, include_system=include_system
        )
        return list(response.apps)

    async def list_packages(self, include_system: bool = False) -> List[str]:
        await self.ensure_connected()
        response = await self._client.device_apps(
            self.session_id, include_system=include_system
        )
        return list(response.packages)

    # -- input actions -----------------------------------------------------

    async def _step(self, action: Action) -> StepResponse:
        await self.ensure_connected()
        async with self._action_lock:
            self._step_index += 1
            request = StepRequest(
                action=action,
                step_index=self._step_index,
                expect_seq=self._last_seq,
            )
            try:
                response = await self._client.step(self.session_id, request)
            except ServiceError as exc:
                if exc.code == "stale_observation":
                    # Our cached frame was superseded; drop it so the agent's
                    # next read resynchronizes instead of replaying stale seqs.
                    self._remember(None)
                raise RuntimeError(
                    f"Environment rejected {action.op}: {exc.message}"
                ) from exc
            self._remember(response.observation)
            if response.observation is None:
                self._last_seq = response.seq
            if not response.ok:
                raise RuntimeError(response.error or response.summary)
            return response

    async def _act(self, op: str, **args: Any) -> None:
        await self._step(Action(op=op, args=args))

    async def tap(self, x: int, y: int) -> None:
        await self._act("tap", x=int(x), y=int(y))

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: float = 1000,
    ) -> None:
        await self._act(
            "swipe",
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            duration_ms=float(duration_ms),
        )

    async def drag(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0
    ) -> None:
        await self._act(
            "drag",
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            duration_s=float(duration),
        )

    async def input_text(
        self,
        text: str,
        clear: bool = False,
        stealth: bool = False,
        wpm: int = 0,
    ) -> bool:
        await self._act("type", text=str(text), clear=bool(clear))
        return True

    async def press_button(self, button: str) -> None:
        button = str(button or "").strip().lower()
        if self.supported_buttons and button not in self.supported_buttons:
            raise ValueError(
                f"Button '{button}' not supported. "
                f"Supported: {', '.join(sorted(self.supported_buttons))}"
            )
        await self._act("key", button=button)

    async def press_key_code(self, key_code: int) -> None:
        await self._act("key", keycode=int(key_code))

    # -- app management ----------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        args: Dict[str, Any] = {"app": str(package)}
        if activity:
            args["activity"] = str(activity)
        response = await self._step(Action(op="open_app", args=args))
        if response.summary:
            return response.summary
        return f"Started {package}"

    async def stop_app(self, package: str) -> str:
        response = await self._step(Action(op="stop_app", args={"app": str(package)}))
        return response.summary or f"Stopped {package}"

    @unsupported("the environment service owns app installation")
    async def install_app(self, path: str, **kwargs: Any) -> str: ...

    @unsupported("the environment service owns app installation")
    async def uninstall_app(self, package: str) -> str: ...

    # -- helpers -----------------------------------------------------------

    def _screen(self) -> Optional[ScreenInfo]:
        return self._observation.screen if self._observation else None


__all__ = ["RemoteEnvDriver", "DeviceDisconnectedError"]
