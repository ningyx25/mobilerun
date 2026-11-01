"""Serialize workflow events onto the agent service's SSE stream.

The agent runtime already funnels every child workflow's events through one
bus (``MobileAgent.handle_stream_event``); the CLI renders them with
``cli/event_handler.py`` and the benchmark records them with
``benchmark/evidence.py``. This module is the third consumer: a compact,
JSON-safe projection for remote observers.

Large or unserializable payloads (screenshots, deep trees) are reduced to
sizes rather than dropped silently, so a consumer can still tell that a frame
was captured.
"""

from __future__ import annotations

from typing import Any, Dict

from mobilerun.harness.protocol import EpisodeEvent

_MAX_TEXT = 4000
# Events worth forwarding; everything else is internal coordination traffic.
_INTERESTING = {
    "ScreenshotEvent",
    "RecordUIStateEvent",
    "ToolExecutionEvent",
    "ManagerPlanEvent",
    "ManagerPlanDetailsEvent",
    "ManagerResponseEvent",
    "ExecutorActionEvent",
    "ExecutorActionResultEvent",
    "ExecutorResponseEvent",
    "FastAgentResponseEvent",
    "FastAgentToolCallEvent",
    "FastAgentEndEvent",
    "GroundedDecisionEvent",
    "GroundedPointEvent",
    "GroundedActionResultEvent",
    "JevDecisionEvent",
    "JevStepEvent",
    "JevEndEvent",
    "FinalizeEvent",
    "ResultEvent",
}


def to_episode_event(seq: int, event: Any) -> EpisodeEvent | None:
    """Project one workflow event; ``None`` when it is not worth streaming."""
    name = type(event).__name__
    if name not in _INTERESTING:
        return None

    data: Dict[str, Any] = {}
    try:
        if name == "ScreenshotEvent":
            shot = getattr(event, "screenshot", b"") or b""
            data = {"bytes": len(shot)}
        elif name == "RecordUIStateEvent":
            elements = getattr(event, "ui_state", None) or []
            data = {"elements": len(elements)}
        elif hasattr(event, "model_dump"):
            data = _jsonable(event.model_dump())
        else:  # pragma: no cover - every current event is pydantic
            data = {"repr": _truncate(repr(event))}
    except Exception:  # noqa: BLE001 - telemetry must never break an episode
        data = {}

    step = data.get("step")
    if not isinstance(step, int):
        step = None
    return EpisodeEvent(
        seq=seq,
        type=name,
        step=step,
        message=_describe(name, data),
        data=data,
    )


def _describe(name: str, data: Dict[str, Any]) -> str:
    """One-line human summary, mirroring the CLI's phrasing."""
    if name == "ToolExecutionEvent":
        status = "✅" if data.get("success") else "❌"
        return f"{status} {data.get('tool_name', '?')}: {_truncate(str(data.get('summary', '')))}"
    if name == "ManagerPlanEvent":
        subgoal = data.get("current_subgoal") or data.get("plan") or ""
        return _truncate(str(subgoal))
    if name in (
        "ManagerResponseEvent",
        "ExecutorResponseEvent",
        "FastAgentResponseEvent",
    ):
        for key in ("response", "thought", "reason"):
            if data.get(key):
                return _truncate(str(data[key]))
    if name == "JevDecisionEvent":
        return f"{data.get('operation', '?')} ({data.get('status', '?')})"
    if name in ("FinalizeEvent", "ResultEvent"):
        return f"{'success' if data.get('success') else 'failure'}: {_truncate(str(data.get('reason', '')))}"
    return name


def _jsonable(value: Any) -> Any:
    """Make a pydantic dump JSON-safe (bytes, sets, long strings)."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return {"bytes": len(value)}
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate(str(value))


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "…"
