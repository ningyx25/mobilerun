"""Test doubles for the harness services.

``ScriptedBackend`` is a complete, deterministic environment backend with no
device behind it: it serves a fixed task catalogue, synthesises screenshots and
accessibility states, records every action and returns a scripted score.

Besides making the harness testable in CI, it doubles as the reference for
implementing a new backend: it implements exactly the :class:`EnvBackend`
surface and nothing else.
"""

from __future__ import annotations

import io
from typing import Any, Callable, Dict, List, Optional, Sequence

from PIL import Image

from mobilerun.harness.actions import supported_ops
from mobilerun.harness.env_service.backend import EnvBackend
from mobilerun.harness.protocol import (
    Action,
    Capabilities,
    DeviceAppsResponse,
    Observation,
    ScreenInfo,
    TaskSpec,
)

_SCREEN_WIDTH = 108
_SCREEN_HEIGHT = 240


def _png(width: int = _SCREEN_WIDTH, height: int = _SCREEN_HEIGHT) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(32, 36, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _state() -> Dict[str, Any]:
    """A minimal portal-shaped accessibility state (one clickable element)."""
    return {
        "a11y_tree": {
            "boundsInScreen": {
                "left": 0,
                "top": 0,
                "right": _SCREEN_WIDTH,
                "bottom": _SCREEN_HEIGHT,
            },
            "className": "",
            "text": "",
            "contentDescription": "",
            "resourceId": "",
            "packageName": "",
            "children": [
                {
                    "boundsInScreen": {
                        "left": 10,
                        "top": 20,
                        "right": 90,
                        "bottom": 60,
                    },
                    "className": "android.widget.Button",
                    "text": "Continue",
                    "contentDescription": "",
                    "resourceId": "com.example:id/continue",
                    "packageName": "com.example",
                    "isClickable": True,
                    "isVisibleToUser": True,
                    "children": [],
                }
            ],
        },
        "phone_state": {
            "currentApp": "com.example/.MainActivity",
            "packageName": "com.example",
            "focusedElement": None,
            "isEditable": False,
        },
        "device_context": {
            "screen_bounds": {"width": _SCREEN_WIDTH, "height": _SCREEN_HEIGHT}
        },
    }


class ScriptedBackend(EnvBackend):
    """In-memory environment backend for tests."""

    kind = "scripted"

    def __init__(
        self,
        backend_id: str = "fake",
        tasks: Optional[Sequence[str]] = None,
        *,
        score: float = 1.0,
        score_fn: Optional[Callable[["ScriptedBackend"], float]] = None,
        max_steps: int = 5,
        supports_ui_tree: bool = True,
    ) -> None:
        super().__init__(backend_id)
        self.task_names = list(tasks or ["FakeTask"])
        self._score = score
        self._score_fn = score_fn
        self._max_steps = max_steps
        self._supports_ui_tree = supports_ui_tree
        self.actions: List[Action] = []
        self.prepared: List[str] = []
        self.answers: List[str] = []
        self.stabilize_count = 0
        self.teardowns: List[str] = []
        self.closed = False
        self.score_reads = 0
        self.observe_count = 0

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    # -- tasks -------------------------------------------------------------

    async def list_tasks(self) -> List[TaskSpec]:
        return [
            TaskSpec(
                id=name,
                backend=self.id,
                instruction=f"do {name}",
                max_steps=self._max_steps,
                metadata={"task_name": name, "difficulty": "easy"},
            )
            for name in self.task_names
        ]

    async def prepare(
        self,
        task_id: str,
        *,
        task_idx: int = 0,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TaskSpec:
        if task_id not in self.task_names:
            from mobilerun.harness.protocol import ErrorCode, ProtocolError

            raise ProtocolError(
                404, ErrorCode.UNKNOWN_TASK, f"Unknown task '{task_id}'"
            )
        self.prepared.append(task_id)
        task = TaskSpec(
            id=f"{task_id}#{task_idx}",
            backend=self.id,
            instruction=f"do {task_id}",
            max_steps=self._max_steps,
            complexity=2.0,
            metadata={"task_name": task_id, "task_idx": task_idx},
        )
        self._active = task
        return task

    # -- episode I/O -------------------------------------------------------

    async def observe(self, *, seq: int) -> Observation:
        self.observe_count += 1
        return Observation.build(
            seq=seq,
            screenshot=_png(),
            state=_state() if self._supports_ui_tree else None,
            screen=ScreenInfo(native_width=_SCREEN_WIDTH, native_height=_SCREEN_HEIGHT),
        )

    async def execute(self, action: Any, *, screen: ScreenInfo) -> str:
        self.actions.append(action)
        if action.op == "open_app" and action.args.get("app") == "explode":
            raise RuntimeError("device exploded")
        return f"{action.op} ok"

    # -- scoring -----------------------------------------------------------

    async def submit_answer(self, text: str) -> None:
        self.answers.append(text)

    async def evaluate(self) -> tuple[float, str]:
        self.score_reads += 1
        if self._score_fn is not None:
            return float(self._score_fn(self)), "scripted"
        return float(self._score), "scripted"

    async def stabilize(self) -> None:
        self.stabilize_count += 1

    async def teardown(self, task: TaskSpec) -> None:
        self.teardowns.append(task.id)
        self._active = None

    # -- device queries ----------------------------------------------------

    async def get_date(self) -> str:
        return "2026-01-01T00:00:00"

    async def get_apps(self, include_system: bool = True) -> DeviceAppsResponse:
        return DeviceAppsResponse(
            apps=[{"package": "com.example", "label": "Example"}],
            packages=["com.example"],
        )

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            ops=supported_ops(_FullDriver()),
            verbs=sorted(_FullDriver().supported),
            buttons=["back", "home", "enter"],
            platform="Android",
            has_ui_tree=self._supports_ui_tree,
            has_screenshot=True,
            supports_stabilize=True,
            supports_answer=True,
        )


class _FullDriver:
    """Capability probe: the verb set a fully-featured backend advertises."""

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
        "get_apps",
        "list_packages",
    }


__all__ = ["ScriptedBackend"]
