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

# A parse failure gets exactly one stricter retry: a JSON-only instruction is
# appended so a model that wrapped the value in prose is told, in the same
# turn, exactly what is wanted. Never more than one retry.
_STRICTER_SUFFIX = (
    '\n\nReply with ONLY the JSON object {"text": "..."}; '
    "no prose, no analysis, no explanation."
)
_MAX_ATTEMPTS = 2


def _sum_usage(
    a: Optional[UsageResult], b: Optional[UsageResult]
) -> Optional[UsageResult]:
    """Field-wise sum of two usage results so a retry's cost is not lost."""
    if a is None:
        return b
    if b is None:
        return a
    return UsageResult(
        request_tokens=a.request_tokens + b.request_tokens,
        response_tokens=a.response_tokens + b.response_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
        requests=a.requests + b.requests,
    )


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
            - image (bytes | None): the current raw screenshot, attached to the
              prompt so the model sees the field it types into

        Returns:
            StopEvent whose result is the exact text to type ("" when the step
            has no suitable text), the summed usage, and the model's raw reply.
        """
        goal = str(ev.get("goal") or "").strip()
        subgoal = str(ev.get("subgoal") or "").strip()
        image = ev.get("image") or None

        prompt = f"""Given one step of a mobile automation task, determine the exact text to type into the currently focused input field.

Task Goal: {goal or "(not provided)"}
Current Step: {subgoal or "(not provided)"}

Return ONLY a JSON object with the exact text:
{{"text": "the text to type"}}

Copy any value the task names exactly as written -- same characters, punctuation and case, keeping file extensions. If no suitable value can be determined, return an empty string. Example: {{"text": "Welsh cakes"}}"""

        # A parse failure is retried once with a stricter instruction; the
        # accepted value still has to pass _parse_text, so prose never lands on
        # the device. Usage of both attempts is summed (the benchmark totals
        # must reconcile with the requests actually made).
        logger.info("⌨️ TextInput response:", extra={"color": "blue"})
        usage_total: Optional[UsageResult] = None
        last_response = ""
        first_response = ""
        text = ""
        for attempt in range(_MAX_ATTEMPTS):
            attempt_prompt = prompt if attempt == 0 else prompt + _STRICTER_SUFFIX
            try:
                response_text, usage = await acomplete_text(
                    self.llm, attempt_prompt, image=image, stream=self.stream
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - a failure is feedback, not a guess
                logger.warning(
                    f"TextInput LLM call failed (attempt {attempt + 1}): {exc}"
                )
                continue
            last_response = response_text
            if attempt == 0:
                first_response = response_text
            usage_total = _sum_usage(usage_total, usage)
            text = self._parse_text(response_text)
            if text:
                break

        logger.info(f"⌨️ TextInput resolved: {text!r}", extra={"color": "blue"})
        return StopEvent(
            result={
                "text": text,
                "usage": usage_total,
                # The first (unprimed) reply -- the diagnostically useful one
                # when a resolution failed; the retry only fires on failure.
                "response": first_response or last_response,
            }
        )

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
