"""
TextInputAgent - decide the exact text one step must type into the field.

Replaces the operation model's ``text_value`` question: instead of selecting a
span from a fixed candidate list (which cannot carry values the instruction
paraphrased away, values longer than the span cap, or values with edge
punctuation), an LLM reads the task goal and the step instruction and produces
the exact value to enter -- copied verbatim when the task names it, derived
when the task implies it.
"""

import logging
from typing import Optional

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun.agent.oneflows.common import acomplete_text, extract_json_object
from mobilerun.agent.usage import UsageResult

logger = logging.getLogger("mobilerun")


class TextInputAgent(Workflow):
    """A small workflow that resolves the exact text for one TYPE_TEXT step."""

    def __init__(self, llm, timeout: int = 60, stream: bool = False, **kwargs):
        """
        Args:
            llm: An LLM instance used to resolve the text.
            timeout: Workflow timeout in seconds (default: 60).
            stream: If True, stream the LLM response to the console.
            **kwargs: Additional arguments passed to Workflow.
        """
        super().__init__(timeout=timeout, **kwargs)
        self.llm = llm
        self.stream = stream

    @step
    async def resolve_text(self, ev: StartEvent, ctx: Context) -> StopEvent:
        """
        Expected StartEvent attributes:
            - goal (str): the full task goal
            - subgoal (str): the current step instruction
            - focused_text (str): what the focused field currently contains
            - image (bytes | None): the current raw screenshot, attached to the
              prompt so the model sees the field it types into

        Returns:
            StopEvent whose result is the exact text to type ("" when the step
            has no suitable text).
        """
        goal = str(ev.get("goal") or "").strip()
        subgoal = str(ev.get("subgoal") or "").strip()
        focused_text = str(ev.get("focused_text") or "").strip()
        image = ev.get("image") or None

        prompt = f"""Given one step of a mobile automation task and the currently focused input field, determine the exact text to type into it.

Task Goal: {goal or "(not provided)"}
Current Step: {subgoal or "(not provided)"}
Focused Field Currently Contains: "{focused_text}"

Return ONLY a JSON object with the exact text:
{{"text": "the text to type"}}

Copy any value the task names exactly as written -- same characters, punctuation and case, keeping file extensions. If no suitable value can be determined, return an empty string. Example: {{"text": "Welsh cakes"}}"""

        logger.info("⌨️ TextInput response:", extra={"color": "blue"})
        usage: Optional[UsageResult] = None
        try:
            response_text, usage = await acomplete_text(
                self.llm, prompt, image=image, stream=self.stream
            )
        except Exception as exc:  # noqa: BLE001 - a failure is feedback, not a guess
            logger.warning(f"TextInput LLM call failed: {exc}")
            return StopEvent(result={"text": "", "usage": None})

        text = self._parse_text(response_text)
        logger.info(f"⌨️ TextInput resolved: {text!r}", extra={"color": "blue"})
        return StopEvent(result={"text": text, "usage": usage})

    @staticmethod
    def _parse_text(response_text: str) -> str:
        """The resolved text from an LLM response, or "" when unusable.

        A JSON object wins; a bare single-token answer (``"Welsh cakes"`` /
        ``Welsh cakes``) is taken as the value. Anything else -- a sentence the
        model wrapped around the answer -- is refused, so prose never gets
        typed into the device.
        """
        payload = extract_json_object(response_text)
        if payload is not None:
            return str(payload.get("text") or "").strip()
        # No JSON: accept only a bare single-line, few-word answer as the
        # value; a sentence of prose is refused so it never reaches the device.
        stripped = response_text.strip().strip('"').strip()
        if stripped and "\n" not in stripped and len(stripped.split()) <= 8:
            return stripped
        return ""
