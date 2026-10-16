"""
Events for the GroundedExecutorAgent workflow.

The grounded executor decides the operation with the local operation model
(clef-flash) and grounds coordinate-bearing operations with the grounding model
(gui-actor), both over ``POST /v1/systemone``. These events carry what each
stage produced so the stream, trajectory, CLI and Langfuse stay as complete as
the LLM executor's.

For MobileAgent coordination events, see droid/events.py.
"""

from typing import Dict, Optional, Tuple

from llama_index.core.workflow import Event

from mobilerun.agent.usage import UsageResult


class GroundedContextEvent(Event):
    """The step's subgoal and its captured screenshot, ready for the request."""

    subgoal: str
    error: str = ""


class GroundedDecisionEvent(Event):
    """The operation model's decision, ready for argument resolution."""

    operation: str
    app_target: Optional[str] = None
    confidence: Optional[float] = None
    error: str = ""


class GroundedPointEvent(Event):
    """A decision with its arguments resolved, ready to execute.

    For taps, ``x``/``y`` carry the grounded point; for swipes,
    ``swipe_start``/``swipe_end`` carry the 0..1 screen fractions the swipe
    workflow resolved; ``text_value`` carries the text workflow's exact value.
    ``operation`` and the argument fields are repeated so this event alone
    describes the action.
    """

    operation: str
    app_target: Optional[str] = None
    text_value: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    swipe_start: Optional[Tuple[float, float]] = None
    swipe_end: Optional[Tuple[float, float]] = None
    confidence: Optional[float] = None
    error: str = ""


class GroundedResponseEvent(Event):
    """The oneflow LLM call that resolved a SWIPE / TYPE_TEXT argument, with
    its usage so the benchmark's totals include it, and the model's raw reply
    so a failed resolution stays diagnosable from the evidence."""

    operation: str
    usage: Optional[UsageResult] = None
    response: str = ""


class GroundedActionResultEvent(Event):
    """Action execution result (internal event with full details)."""

    action: Dict
    success: bool
    error: str
    summary: str
