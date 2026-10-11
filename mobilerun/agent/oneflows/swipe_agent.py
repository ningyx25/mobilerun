"""
SwipeAgent - resolve one SWIPE step to concrete start/end coordinates.

Replaces the operation model's former ``scroll_direct`` question: the LLM
reads the step instruction together with the screenshot and outputs the
swipe's start and end points as normalized 0-1000 coordinates. The points are
returned as 0..1 screen fractions so the caller maps them into its own
coordinate space.
"""

import logging
from typing import Optional, Tuple

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun.agent.oneflows.common import acomplete_text, extract_json_object
from mobilerun.agent.usage import UsageResult
from mobilerun.tools.helpers.coordinate import NORMALIZED_MAX

logger = logging.getLogger("mobilerun")

_NORMALIZED_MAX = float(NORMALIZED_MAX)

# A swipe shorter than this fraction of the normalized diagonal is a tap-like
# gesture, not a swipe; the answer is treated as unusable.
_MIN_SWIPE_FRACTION = 0.02


class SwipeAgent(Workflow):
    """A small workflow that resolves one swipe to start/end coordinates."""

    def __init__(self, llm, timeout: int = 60, stream: bool = False, **kwargs):
        """
        Args:
            llm: An LLM instance used to choose the swipe coordinates.
            timeout: Workflow timeout in seconds (default: 60).
            stream: If True, stream the LLM response to the console.
            **kwargs: Additional arguments passed to Workflow.
        """
        super().__init__(timeout=timeout, **kwargs)
        self.llm = llm
        self.stream = stream

    @step
    async def resolve_swipe(self, ev: StartEvent, ctx: Context) -> StopEvent:
        """
        Expected StartEvent attributes:
            - goal (str): the full task goal
            - subgoal (str): the current step instruction
            - image (bytes | None): the current raw screenshot the points are
              measured on

        Returns:
            StopEvent whose result is a dict with ``start``/``end`` (0..1
            screen fractions) and ``usage``, or ``{"error": ...}`` when the
            swipe cannot be resolved (no screenshot, an LLM failure, or an
            unusable answer). The executor turns that into one step of
            feedback.
        """
        goal = str(ev.get("goal") or "").strip()
        subgoal = str(ev.get("subgoal") or "").strip()
        image = ev.get("image") or None

        if not image:
            return StopEvent(result={"error": "no screenshot to ground on"})

        points, usage = await self._choose_points(goal, subgoal, image)
        if points is None:
            return StopEvent(result={"error": "the model did not return a usable swipe"})
        start, end = points
        start_f = (start[0] / _NORMALIZED_MAX, start[1] / _NORMALIZED_MAX)
        end_f = (end[0] / _NORMALIZED_MAX, end[1] / _NORMALIZED_MAX)

        logger.info(
            f"↔️ Swipe resolved: ({start[0]:.0f}, {start[1]:.0f}) -> "
            f"({end[0]:.0f}, {end[1]:.0f}) normalized",
            extra={"color": "blue"},
        )
        return StopEvent(
            result={"start": list(start_f), "end": list(end_f), "usage": usage}
        )

    async def _choose_points(
        self,
        goal: str,
        subgoal: str,
        image: Optional[bytes] = None,
    ) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float]]], Optional[UsageResult]]:
        """The (start, end) normalized points the LLM picks and its token usage,
        or (None, usage) when the call fails or the answer is unusable -- a
        failed swipe stays a failed step instead of a blind gesture."""
        prompt = f"""Given one step of a mobile automation task and a screenshot of the current screen, determine the swipe gesture that advances the step.

Task Goal: {goal or "(not provided)"}
Current Step: {subgoal or "(not provided)"}

Return ONLY a JSON object with the swipe's start and end points as normalized 0-1000 coordinates (0,0 is the top-left, 1000,1000 the bottom-right):
{{
    "start": [x, y],
    "end": [x, y]
}}

To reveal content further down the screen, move the finger up (start near the bottom, end near the top). Example: {{"start": [500, 830], "end": [500, 200]}}"""

        logger.info("↔️ SwipeAgent response:", extra={"color": "blue"})
        try:
            response_text, usage = await acomplete_text(
                self.llm, prompt, image=image, stream=self.stream
            )
        except Exception as exc:  # noqa: BLE001 - a failure is feedback, not a guess
            logger.warning(f"SwipeAgent LLM call failed: {exc}")
            return None, None
        parsed = self._parse_points(response_text)
        if parsed is None:
            logger.warning("SwipeAgent answer unusable")
        return parsed, usage

    @staticmethod
    def _parse_points(
        response_text: str,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """The validated (start, end) normalized points of an LLM response, or
        None.

        Points are clamped to the 0-1000 space; a degenerate (tap-like) swipe
        is refused so the caller reports a failed step.
        """
        payload = extract_json_object(response_text)
        if payload is None:
            return None
        try:
            start_xy = [float(payload["start"][0]), float(payload["start"][1])]
            end_xy = [float(payload["end"][0]), float(payload["end"][1])]
        except (ValueError, KeyError, TypeError, IndexError):
            return None
        start_xy = [
            min(max(start_xy[0], 0.0), _NORMALIZED_MAX),
            min(max(start_xy[1], 0.0), _NORMALIZED_MAX),
        ]
        end_xy = [
            min(max(end_xy[0], 0.0), _NORMALIZED_MAX),
            min(max(end_xy[1], 0.0), _NORMALIZED_MAX),
        ]
        distance = (
            (start_xy[0] - end_xy[0]) ** 2 + (start_xy[1] - end_xy[1]) ** 2
        ) ** 0.5
        if distance < _MIN_SWIPE_FRACTION * (_NORMALIZED_MAX**2 * 2) ** 0.5:
            return None
        return (start_xy[0], start_xy[1]), (end_xy[0], end_xy[1])
