"""Rich per-episode evidence capture for the AndroidWorld benchmark.

The framework's ``TrajectoryWriter`` already persists a per-run folder
(``runs/trajectories/<ts>_<uuid>/``) with the agent's raw events, per-step
screenshots, UI-state snapshots and macro actions. This module complements it
so the benchmark records "everything recordable" per episode:

- a stable per-episode directory the result row can point at,
- the full LLM conversation (system/user messages with injected device state),
- real accessibility-tree snapshots (the tree is empty under ``vision_only``),
- a distilled per-step timeline (thought / raw response / tool / observation),
- cumulative token usage per episode,
- the agent's own trajectory artifacts, relocated under the episode directory.

Everything is best-effort: evidence capture must never fail an episode. Each
helper swallows its own errors and logs, so a broken serializer or a missing
trajectory folder degrades to a missing file, not a lost score.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from mobilerun.agent.trajectory import make_serializable

logger = logging.getLogger("mobilerun")


async def gather_ui_tree(
    driver: Any, tries: int = 4, delay: float = 0.5
) -> Optional[Dict[str, Any]]:
    """Read the driver's real accessibility tree, waiting out empty frames.

    The tree is empty mid-transition (right after a reset, or just after a
    tap), so this polls briefly for a populated tree. Best-effort: returns the
    last (possibly empty) tree if it never fills.
    """
    tree: Optional[Dict[str, Any]] = None
    for _ in range(max(1, tries)):
        try:
            tree = await driver.get_ui_tree()
        except Exception as exc:  # noqa: BLE001 - evidence must not fail a run
            logger.debug("get_ui_tree for evidence failed: %s", exc)
            tree = None
        if tree and tree.get("a11y_tree", {}).get("children"):
            return tree
        await asyncio.sleep(delay)
    return tree


# Event-class names whose payload is a distilled LLM turn.
_LLM_EVENT_TYPES = {
    "FastAgentResponseEvent",
    "ManagerResponseEvent",
    "ExecutorResponseEvent",
}
# Class names handled elsewhere (screenshots -> trajectory writer, UI state ->
# dedicated tree snapshots); the distilled timeline skips them.
_SKIPPED_EVENT_TYPES = {"ScreenshotEvent", "RecordUIStateEvent", "FastAgentInputEvent"}

# Cap on a recorded oneflow reply: a model can dump a whole file into its
# answer, and the timeline must stay bounded while the failure stays readable.
_MAX_RESPONSE_CHARS = 4000


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write *payload* as indented JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temp.replace(path)
    except Exception as exc:  # noqa: BLE001 - evidence must never fail a run
        logger.warning("Failed to write evidence file %s: %s", path, exc)


class EpisodeEvidence:
    """Collects and writes all recordable data for one benchmark episode."""

    def __init__(self, episode_dir: Path, task_type: str, task_idx: int) -> None:
        self.dir = Path(episode_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task_type = task_type
        self.task_idx = task_idx
        # Distilled per-step timeline built from the streamed events.
        self.timeline: List[Dict[str, Any]] = []
        # Per-call token usage, summed into totals at finalize.
        self.usage_calls: List[Dict[str, Any]] = []
        self.usage_totals: Dict[str, int] = {
            "request_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }
        self._seq = 0

    # -- live collection (called from the benchmark's stream loop) ----------

    def record_event(self, event: Any) -> None:
        """Distill one streamed event into the timeline + usage totals."""
        name = type(event).__name__
        if name in _SKIPPED_EVENT_TYPES:
            return

        usage = getattr(event, "usage", None)
        usage_dict: Optional[Dict[str, Any]] = None
        if usage is not None:
            usage_dict = make_serializable(usage)
            if isinstance(usage_dict, dict):
                self.usage_calls.append({"seq": self._seq, "type": name, **usage_dict})
                for key in self.usage_totals:
                    self.usage_totals[key] += int(usage_dict.get(key) or 0)

        entry: Dict[str, Any] = {"seq": self._seq, "type": name}
        if usage_dict:
            entry["usage"] = usage_dict

        if name in _LLM_EVENT_TYPES:
            entry["kind"] = "llm"
            for key in ("thought", "code", "response", "tool_call_status"):
                value = getattr(event, key, None)
                if value not in (None, ""):
                    entry[key] = make_serializable(value)
        elif name == "ToolExecutionEvent":
            entry["kind"] = "tool"
            entry["tool"] = getattr(event, "tool_name", None)
            entry["args"] = make_serializable(getattr(event, "tool_args", None))
            entry["success"] = getattr(event, "success", None)
            entry["summary"] = getattr(event, "summary", None)
        elif name == "FastAgentOutputEvent":
            entry["kind"] = "observation"
            entry["output"] = getattr(event, "output", None)
        elif name in ("FastAgentToolCallEvent",):
            entry["kind"] = "tool_call"
            entry["tool_calls_repr"] = getattr(event, "tool_calls_repr", None)
        elif name.endswith("EndEvent") or name == "FastAgentResultEvent":
            entry["kind"] = "end"
            entry["success"] = getattr(event, "success", None)
            entry["reason"] = getattr(event, "reason", None)
        elif name == "ManagerPlanEvent":
            entry["kind"] = "plan"
            entry["plan"] = getattr(event, "plan", None)
            entry["subgoal"] = getattr(event, "current_subgoal", None)
            entry["thought"] = getattr(event, "thought", None)
            entry["answer"] = getattr(event, "answer", None)
        elif name in ("ExecutorActionResultEvent", "GroundedActionResultEvent"):
            entry["kind"] = "executor_result"
            entry["action"] = make_serializable(getattr(event, "action", None))
            entry["success"] = getattr(event, "success", None)
            entry["error"] = getattr(event, "error", None)
            entry["summary"] = getattr(event, "summary", None)
        elif name == "GroundedDecisionEvent":
            entry["kind"] = "executor_decision"
            entry["operation"] = getattr(event, "operation", None)
            entry["app_target"] = getattr(event, "app_target", None)
            entry["confidence"] = getattr(event, "confidence", None)
        elif name == "GroundedResponseEvent":
            entry["kind"] = "llm"
            entry["operation"] = getattr(event, "operation", None)
            response = getattr(event, "response", None)
            # Truncate a runaway reply (a model can dump a whole file) so the
            # timeline stays bounded while the failure stays diagnosable.
            if isinstance(response, str) and len(response) > _MAX_RESPONSE_CHARS:
                response = response[:_MAX_RESPONSE_CHARS] + "…"
            entry["response"] = response
        elif name == "GroundedPointEvent":
            entry["kind"] = "executor_ground"
            entry["operation"] = getattr(event, "operation", None)
            entry["x"] = getattr(event, "x", None)
            entry["y"] = getattr(event, "y", None)
            entry["swipe_start"] = make_serializable(
                getattr(event, "swipe_start", None)
            )
            entry["swipe_end"] = make_serializable(getattr(event, "swipe_end", None))
            entry["text_value"] = getattr(event, "text_value", None)
        else:
            entry["kind"] = "event"

        self.timeline.append(entry)
        self._seq += 1

    # -- finalize (after the episode) --------------------------------------

    def write_conversation(self, messages: Optional[List[Any]]) -> Optional[str]:
        """Persist the full LLM conversation (message_history)."""
        if not messages:
            return None
        path = self.dir / "conversation.json"
        _write_json(path, make_serializable(messages))
        return "conversation.json"

    def write_ui_tree(self, name: str, tree: Optional[Dict[str, Any]]) -> Optional[str]:
        """Persist a real accessibility-tree snapshot (bypasses vision_only)."""
        if not tree or not tree.get("a11y_tree", {}).get("children"):
            return None
        path = self.dir / f"{name}.json"
        _write_json(path, tree)
        return path.name

    def write_timeline(self) -> str:
        """Persist the distilled per-step timeline + cumulative usage."""
        _write_json(
            self.dir / "timeline.json",
            {
                "task_type": self.task_type,
                "task_idx": self.task_idx,
                "events": self.timeline,
                "usage_totals": self.usage_totals,
                "usage_calls": self.usage_calls,
            },
        )
        return "timeline.json"

    def adopt_trajectory(self, trajectory_folder: Any) -> Optional[str]:
        """Move the agent's trajectory artifacts under the episode directory."""
        if not trajectory_folder:
            return None
        src = Path(trajectory_folder)
        if not src.exists():
            return None
        dest = self.dir / "trajectory"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))
            return "trajectory"
        except Exception as exc:  # noqa: BLE001 - evidence must never fail a run
            logger.warning("Failed to move trajectory %s: %s", src, exc)
            return None

    def write_manifest(self, payload: Dict[str, Any]) -> str:
        """Persist the episode manifest (score, goal, steps, paths, …)."""
        _write_json(self.dir / "episode.json", payload)
        return "episode.json"
