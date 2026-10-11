"""GroundedExecutorAgent — operation model + grounding model executor.

The third cognitive mode's execution half: the Manager still plans and emits a
single-step instruction, but the Executor no longer calls an LLM tool loop.
Each step is

    decide   operation model (``executor.systemone.operation_model``) answers
             the mobile-jev operation question built by :mod:`jev_prompt` --
             the operation type, plus the app target when it is OPEN_APP.
             The mode only supports ``vision_only: true`` (``MobileAgent``
             rejects the other combinations): the accessibility tree is never
             read, the full operation menu is offered, and every decision is
             grounded on the raw screenshot alone;
    ground   the operation's arguments are resolved: TAP / LONG_PRESS get a
             grounded point from the grounding model
             (``executor.systemone.grounding_model``) on the current raw
             screenshot, SWIPE gets start/end points from
             ``oneflows/swipe_agent`` (the model picks the start/end
             points itself), TYPE_TEXT gets the exact value from
             ``oneflows/text_input_agent``;
    execute  the resolved operation runs through ``mobilerun.agent.utils``
             action functions -- the same ActionContext / macro / driver path
             every other agent uses.

Contract parity with :class:`ExecutorAgent`: ``run(subgoal=...)`` returns a
StopEvent whose result is ``{action, outcome, error, summary, thought}``, so
``MobileAgent.run_executor`` and the error-escalation loop are untouched.

Failure policy (the package's §7.1 rule): a failed decision, a failed
grounding call or a failed action becomes one step of feedback -- never an
aborted episode. ``DeviceDisconnectedError`` is the only exception that
propagates.

Every request to either decision model carries a single image: the current,
raw, unannotated screenshot (no history frames, no resizing -- the frames are
deliberately dropped rather than downscaled). Past steps reach the operation
model as ``state.recentActions`` text only.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from mobilerun_core_local.driver.base import DeviceDisconnectedError

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.executor import jev_prompt
from mobilerun.agent.executor.grounded_events import (
    GroundedActionResultEvent,
    GroundedContextEvent,
    GroundedDecisionEvent,
    GroundedPointEvent,
    GroundedResponseEvent,
)
from mobilerun.agent.oneflows.swipe_agent import SwipeAgent
from mobilerun.agent.oneflows.text_input_agent import TextInputAgent
from mobilerun.agent.utils.actions import (
    click_at,
    long_press_at,
    open_app,
    open_bundle_id,
    swipe,
    system_button,
    type_text,
)
from mobilerun.agent.utils.errors import describe_error
from mobilerun.agent.utils.systemone import choice_of, encode_images, systemone_post
from mobilerun.tools.helpers.coordinate import NORMALIZED_MAX

if TYPE_CHECKING:
    from mobilerun.agent.action_context import ActionContext
    from mobilerun.agent.droid import MobileAgentState
    from mobilerun.agent.tool_registry import ToolRegistry
    from mobilerun.config_manager.config_manager import AgentConfig

logger = logging.getLogger("mobilerun")

# Operations that resolve to a screen point via the grounding model.
_COORDINATE_OPERATIONS = frozenset({"TAP", "LONG_PRESS"})

# Every operation this executor can execute. Anything else the operation model
# answers is refused as one step of feedback (the menu never offers it). WAIT
# is not here: the menu does not offer it, and a stray answer is refused.
_EXECUTABLE_OPERATIONS = frozenset(
    {"TAP", "LONG_PRESS", "SWIPE", "BACK", "HOME", "ENTER", "OPEN_APP", "TYPE_TEXT"}
)

_SCROLL_DURATION_S = 1.0
_GROUNDING_QUESTION_ID = "tap_target"
_ONEFLOW_TIMEOUT_S = 60.0


class GroundedExecutorAgent(Workflow):
    """Executor whose actions come from local decision models, not an LLM tool loop."""

    def __init__(
        self,
        *,
        action_ctx: "ActionContext | None",
        shared_state: "MobileAgentState",
        agent_config: "AgentConfig",
        registry: "ToolRegistry | None" = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.action_ctx = action_ctx
        self.shared_state = shared_state
        self.agent_config = agent_config
        self.config = agent_config.executor
        self.systemone = agent_config.executor.systemone
        # Accepted for wiring parity with ExecutorAgent; execution calls the
        # action functions directly (the registry is the LLM-facing tool table).
        self.registry = registry

        # Cross-run state: one run per outer step, so history survives on self.
        self._history: List[jev_prompt.HistoryEntry] = []
        self._pending: Optional[jev_prompt.HistoryEntry] = None
        self._apps: Optional[List[str]] = None
        self._app_packages: Dict[str, str] = {}

        # Per-run state.
        self._subgoal: str = ""
        self._screenshot: Optional[bytes] = None
        self._screenshot_b64: Optional[str] = None  # encoded once per step
        self._ground_fraction: Optional[Tuple[float, float]] = None
        self._oneflow_usage = None  # UsageResult from this step's oneflow call
        self._thought: str = ""

        logger.debug(
            "GroundedExecutorAgent initialized (systemone %s).", self.systemone.base_url
        )

    # ========================================================================
    # Steps
    # ========================================================================

    @step
    async def prepare(self, ctx: Context, ev: StartEvent) -> GroundedContextEvent:
        """Capture the step's raw screenshot (no accessibility tree)."""
        self._subgoal = ev.get("subgoal", "")
        logger.debug(f"🧭 Grounded executor deciding action for: {self._subgoal}")

        self._screenshot = None
        self._screenshot_b64 = None
        self._ground_fraction = None
        self._oneflow_usage = None
        self._thought = ""
        error = ""

        try:
            self._screenshot = await self._current_screenshot()
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001 - one bad read is feedback, not a crash
            error = f"Failed to read the device state: {describe_error(e)}"
            logger.warning("⚠️ %s", error)

        event = GroundedContextEvent(subgoal=self._subgoal, error=error)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def decide(self, ctx: Context, ev: GroundedContextEvent) -> GroundedDecisionEvent:
        """Ask the operation model what to do with this screen."""
        event = await self._decide(ev)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def ground(self, ctx: Context, ev: GroundedDecisionEvent) -> GroundedPointEvent:
        """Resolve the operation's arguments (grounding model / oneflows)."""
        event = await self._ground(ev)
        if self._oneflow_usage is not None:
            ctx.write_event_to_stream(
                GroundedResponseEvent(operation=ev.operation, usage=self._oneflow_usage)
            )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def execute(self, ctx: Context, ev: GroundedPointEvent) -> GroundedActionResultEvent:
        """Execute the resolved operation through the action layer."""
        event = await self._execute(ev)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def finalize(self, ctx: Context, ev: GroundedActionResultEvent) -> StopEvent:
        """Return executor results to the parent workflow (ExecutorAgent's shape)."""
        return StopEvent(
            result={
                "action": ev.action,
                "outcome": ev.success,
                "error": ev.error,
                "summary": ev.summary,
                "thought": self._thought,
            }
        )

    # ========================================================================
    # Decide / ground / execute (plain coroutines; the steps wrap them)
    # ========================================================================

    async def _decide(self, ev: GroundedContextEvent) -> GroundedDecisionEvent:
        if ev.error:
            return self._decision_event(ev.error)

        try:
            apps = await self._app_inventory()
            request = jev_prompt.build_request(
                self._subgoal,
                self._history,
                apps,
                foreground_package=self.shared_state.current_package_name or "",
            )
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001
            error = f"Failed to build the operation request: {describe_error(e)}"
            logger.warning("⚠️ %s", error)
            return self._decision_event(error)

        body: Dict[str, Any] = {**request.as_dict(), "model": self.systemone.operation_model}
        images = self._encoded_frame()
        if images:
            body["images"] = images

        logger.info(
            "🧭 Operation model request (last op: %s)",
            self._history[-1].operation if self._history else "none",
            extra={"color": "yellow"},
        )
        try:
            response = await systemone_post(
                self.systemone.base_url, body, timeout=self.systemone.timeout_s
            )
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001 - SystemOneError and friends are feedback
            error = f"Operation model call failed: {describe_error(e)}"
            logger.warning("⚠️ %s", error)
            return self._decision_event(error)

        return self._parse_decision(response, request)

    async def _ground(self, ev: GroundedDecisionEvent) -> GroundedPointEvent:
        """Resolve the operation's arguments: a grounded point for taps, the
        swipe workflow's start/end points for SWIPE, the text workflow's exact
        value for TYPE_TEXT."""
        if ev.error:
            return self._point_event(ev, error=ev.error)
        if ev.operation in _COORDINATE_OPERATIONS:
            return await self._ground_point(ev)
        if ev.operation == "SWIPE":
            return await self._resolve_swipe(ev)
        if ev.operation == "TYPE_TEXT":
            return await self._resolve_text(ev)
        return self._point_event(ev)

    async def _ground_point(self, ev: GroundedDecisionEvent) -> GroundedPointEvent:
        if not self._screenshot:
            return self._point_event(ev, error="No screenshot available to ground on.")

        body: Dict[str, Any] = {
            "model": self.systemone.grounding_model,
            "state": "",
            "questions": {
                _GROUNDING_QUESTION_ID: {
                    "type": "choice",
                    "instructions": self._subgoal,
                    "criteria": {"1": None},
                }
            },
            "images": self._encoded_frame(),
        }
        try:
            logger.info("🎯 Grounding %r", self._subgoal, extra={"color": "yellow"})
            response = await systemone_post(
                self.systemone.base_url, body, timeout=self.systemone.timeout_s
            )
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001
            error = f"Grounding model call failed: {describe_error(e)}"
            logger.warning("⚠️ %s", error)
            return self._point_event(ev, error=error)

        try:
            fx, fy, confidence = self._grounding_point(response)
            x, y = self._fraction_to_action_space(fx, fy)
        except Exception as e:  # noqa: BLE001
            error = f"Unusable grounding answer: {describe_error(e)}"
            logger.warning("⚠️ %s", error)
            return self._point_event(ev, error=error)

        self._ground_fraction = (fx, fy)
        logger.info(f"🎯 Grounded {self._subgoal!r} -> ({x}, {y})", extra={"color": "yellow"})
        return self._point_event(ev, x=x, y=y, confidence=confidence)

    # -- oneflow argument resolution -----------------------------------------

    async def _resolve_swipe(self, ev: GroundedDecisionEvent) -> GroundedPointEvent:
        """Resolve SWIPE to start/end points via the swipe oneflow.

        The workflow grounds the points on the screenshot; an LLM failure or an
        unusable answer is a failed step, never a blind gesture.
        """
        if not self._screenshot:
            return self._point_event(ev, error="SWIPE needs a screenshot.")
        llm = self._oneflow_llm()
        if llm is None:
            return self._point_event(
                ev, error="SWIPE needs the app_opener LLM for the swipe workflow."
            )
        try:
            agent = SwipeAgent(
                llm=llm, timeout=_ONEFLOW_TIMEOUT_S, stream=self._streaming()
            )
            result = await agent.run(
                goal=self.shared_state.instruction or "",
                subgoal=self._subgoal,
                image=self._screenshot,
            )
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001
            error = f"Swipe resolution failed: {describe_error(e)}"
            logger.warning("⚠️ %s", error)
            return self._point_event(ev, error=error)

        if not isinstance(result, dict) or "error" in result:
            detail = result.get("error") if isinstance(result, dict) else result
            return self._point_event(ev, error=f"Swipe resolution failed: {detail}")
        self._oneflow_usage = result.get("usage")
        try:
            start = (float(result["start"][0]), float(result["start"][1]))
            end = (float(result["end"][0]), float(result["end"][1]))
        except (KeyError, TypeError, ValueError, IndexError):
            return self._point_event(
                ev, error="Swipe resolution returned unusable coordinates."
            )
        return self._point_event(ev, swipe_start=start, swipe_end=end)

    async def _resolve_text(self, ev: GroundedDecisionEvent) -> GroundedPointEvent:
        """Resolve TYPE_TEXT to the exact value via the text-input oneflow."""
        llm = self._oneflow_llm()
        if llm is None:
            return self._point_event(
                ev, error="TYPE_TEXT needs the app_opener LLM for the text workflow."
            )
        try:
            agent = TextInputAgent(
                llm=llm, timeout=_ONEFLOW_TIMEOUT_S, stream=self._streaming()
            )
            result = await agent.run(
                goal=self.shared_state.instruction or "",
                subgoal=self._subgoal,
                focused_text=self.shared_state.focused_text or "",
                image=self._screenshot,
            )
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001
            error = f"Text resolution failed: {describe_error(e)}"
            logger.warning("⚠️ %s", error)
            return self._point_event(ev, error=error)

        self._oneflow_usage = result.get("usage") if isinstance(result, dict) else None
        text = str((result or {}).get("text") if isinstance(result, dict) else result or "").strip()
        if not text:
            return self._point_event(ev, error="TYPE_TEXT without a suitable text value.")
        text = self._complete_file_extension(text)
        logger.info(f"⌨️ Text resolved: {text!r}", extra={"color": "blue"})
        return self._point_event(ev, text_value=text)

    def _encoded_frame(self) -> List[str]:
        """The step's screenshot as a one-element base64 list, encoded once and
        reused by the decide and ground requests."""
        if not self._screenshot:
            return []
        if self._screenshot_b64 is None:
            encoded = encode_images(
                [self._screenshot], self.systemone.max_image_pixels
            )
            self._screenshot_b64 = encoded[0] if encoded else ""
        return [self._screenshot_b64] if self._screenshot_b64 else []

    def _oneflow_llm(self):
        """The LLM the argument-resolving oneflows run on.

        The ``app_opener`` profile is the framework's already-loaded
        general-purpose small LLM, which is exactly the slot the open_app
        fallback (AppStarter) uses.
        """
        return getattr(self.action_ctx, "app_opener_llm", None)

    def _streaming(self) -> bool:
        return bool(getattr(self.action_ctx, "streaming", False))

    def _screen_size(self) -> Tuple[int, int]:
        """The screen in pixels the labels are rendered in: the UIState's, else
        the shared state's."""
        ui = getattr(self.action_ctx, "ui", None)
        width = int(getattr(ui, "screen_width", 0) or self.shared_state.width or 0)
        height = int(getattr(ui, "screen_height", 0) or self.shared_state.height or 0)
        return width, height

    async def _execute(self, ev: GroundedPointEvent) -> GroundedActionResultEvent:
        if ev.error:
            self._thought = self._thought_line(ev, error=ev.error)
            self.shared_state.last_thought = self._thought
            return GroundedActionResultEvent(
                action={"action": ev.operation.lower() or "unknown"},
                success=False,
                error=ev.error,
                summary=ev.error,
            )

        logger.info(f"⚡ Executing {ev.operation}", extra={"color": "green"})
        try:
            action_dict, result = await self._execute_operation(ev)
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001
            try:
                action_dict = self._action_dict_for(ev)
            except Exception:  # noqa: BLE001
                action_dict = {"action": ev.operation.lower() or "unknown"}
            result = ActionResult(
                success=False,
                summary=f"Failed to execute {ev.operation}: {describe_error(e)}",
            )

        await asyncio.sleep(self.agent_config.after_sleep_action)

        # Record the executed decision whether it succeeded or not: the
        # operation model must see its failed attempts, like the reference's
        # history, instead of repeating them.
        self._remember(ev)

        self._thought = (
            self._thought_line(ev)
            if result.success
            else self._thought_line(ev, error=result.summary)
        )
        self.shared_state.last_thought = self._thought
        logger.debug(
            f"{'✅' if result.success else '❌'} Execution complete: {result.summary}"
        )

        return GroundedActionResultEvent(
            action=action_dict,
            success=result.success,
            error="" if result.success else result.summary,
            summary=result.summary,
        )

    # ========================================================================
    # Device state
    # ========================================================================

    async def _current_screenshot(self) -> Optional[bytes]:
        """A fresh raw, unannotated frame for this step.

        The Manager's frame was captured before its (possibly slow) LLM call,
        so it can be tens of seconds stale by dispatch time; take a fresh one
        and fall back to the Manager's only when the capture fails.
        """
        try:
            shot = await self.action_ctx.driver.screenshot()
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ Could not capture a screenshot for grounding: {e}")
            shot = None
        if shot:
            return bytes(shot)
        shared = self.shared_state.screenshot
        return bytes(shared) if shared else None

    async def _app_inventory(self) -> List[str]:
        """The device's app labels, fetched once per episode (best effort).

        Launchable apps only: an open_app target the model picks must be an app
        the driver can actually start. ``get_apps`` on the android-world
        backend lists every package (providers, viewers, overlays included), so
        the launcher query wins when the driver offers it. System apps stay:
        benchmark tasks routinely open Settings, Clock or Contacts, which
        ``include_system=False`` would hide.
        """
        if self._apps is not None:
            return self._apps
        labels: List[str] = []
        fetched = False
        try:
            items = await self._fetch_app_items()
            fetched = True
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                package = str(item.get("package") or "").strip()
                label = str(item.get("label") or package).strip()
                if not label or label in labels:
                    continue
                lowered = label.lower()
                if "auto_generated_rro" in lowered or ".overlay." in lowered:
                    continue
                labels.append(label)
                if package:
                    self._app_packages[label] = package
        except DeviceDisconnectedError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"⚠️ Could not list installed apps (OPEN_APP stays offered): {e}"
            )
        if fetched:
            # Cache only a successful read: a transient failure must not pin an
            # empty inventory (and drop OPEN_APP) for the rest of the episode.
            self._apps = labels
        return labels

    async def _fetch_app_items(self) -> List[Dict[str, str]]:
        """App rows (``{package, label}``): launchable apps when the driver can
        tell them apart, its full app list otherwise."""
        launchable = getattr(self.action_ctx.driver, "list_launchable_packages", None)
        if callable(launchable):
            try:
                packages = await launchable()
                if packages:
                    return [
                        {"package": str(package), "label": str(package)}
                        for package in packages
                    ]
            except DeviceDisconnectedError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "⚠️ Launcher-app query failed (%s); using the full package list",
                    describe_error(e)[:120],
                )
        try:
            return await self.action_ctx.driver.get_apps(include_system=True)
        except TypeError:  # drivers without the keyword
            return await self.action_ctx.driver.get_apps()

    # ========================================================================
    # Decisions
    # ========================================================================

    def _parse_decision(
        self, response: Dict[str, Any], request: jev_prompt.Request
    ) -> GroundedDecisionEvent:
        answers = response.get("answers") or {}
        operation_answer = answers.get("operation") or {}
        operation = str(operation_answer.get("choice") or "").strip().upper()
        confidence = operation_answer.get("confidence")
        confidence = confidence if isinstance(confidence, (int, float)) else None

        if not operation:
            return self._decision_event("The operation model returned no operation.")
        if operation not in _EXECUTABLE_OPERATIONS:
            return self._decision_event(
                f"Operation {operation} has no execution path.", operation, confidence
            )

        app_target: Optional[str] = None

        # Only OPEN_APP carries its argument in the answer: SWIPE and
        # TYPE_TEXT resolve theirs in the ground step (scroll / text oneflows).
        if operation == "OPEN_APP":
            choice = choice_of(answers.get("app_target"))
            app_target = request.criteria_of("app_target").get(choice) if choice else None
            if not app_target:
                return self._decision_event(
                    "OPEN_APP without an offered app target.", operation, confidence
                )

        logger.info(
            f"🧭 Decided {operation}" + (f" ({app_target})" if app_target else ""),
            extra={"color": "yellow"},
        )
        return GroundedDecisionEvent(
            operation=operation,
            app_target=app_target,
            confidence=confidence,
        )

    def _complete_file_extension(self, value: str) -> str:
        """Extend a chosen value to the task's exact file-name token.

        The step instruction can drop a file extension the task names verbatim
        ("2023_06_24_training" vs "2023_06_24_training.m4a"), and target apps
        commonly append their own format suffix -- typing the task's exact
        name is what lands the expected file on disk. Only a strict
        ``<value>.<ext>`` token that appears in the task instruction
        qualifies; nothing is invented and non-file values are untouched.
        """
        if not value:
            return value
        instruction = self.shared_state.instruction or ""
        pattern = re.compile(
            r"(?<![\w])" + re.escape(value) + r"\.[A-Za-z0-9]{1,8}(?![\w])"
        )
        match = pattern.search(instruction)
        return match.group(0) if match else value

    def _grounding_point(
        self, response: Dict[str, Any]
    ) -> Tuple[float, float, Optional[float]]:
        """The normalized ground point of a gui-actor answer.

        ``point`` is the server's decode of the pointer distribution; when it
        is missing, the answer's ``choice`` patch index is decoded against the
        patch grid instead (``y = idx // n_width``, patch centers).
        """
        answer = (response.get("answers") or {}).get(_GROUNDING_QUESTION_ID) or {}
        confidence = answer.get("confidence")
        confidence = confidence if isinstance(confidence, (int, float)) else None
        point = answer.get("point")
        if isinstance(point, (list, tuple)) and len(point) == 2:
            fx, fy = float(point[0]), float(point[1])
        else:
            raw_choice = answer.get("choice")
            index = -1 if raw_choice is None or raw_choice == "" else int(str(raw_choice))
            n_width = int(answer.get("n_width") or 0)
            n_height = int(answer.get("n_height") or 0)
            if index < 0 or n_width <= 0 or n_height <= 0:
                raise ValueError("no point and no patch grid in the answer")
            fx = ((index % n_width) + 0.5) / n_width
            fy = ((index // n_width) + 0.5) / n_height
        return min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0), confidence

    def _fraction_to_action_space(self, fx: float, fy: float) -> Tuple[int, int]:
        """Map a 0..1 screenshot fraction into the coordinate space actions take.

        The action layer validates and converts points the same way it does for
        an LLM: normalized 0..1000 when the provider says so, otherwise the
        model-facing space the provider declared (``model_screenshot_width`` /
        ``model_screenshot_height`` -- which cover both conventions: the tree
        provider's display space, and the screenshot-only provider's resized
        frame whose ``screen_width`` is already model-side).
        """
        ui = getattr(self.action_ctx, "ui", None)
        if getattr(ui, "use_normalized", False):
            return (int(round(fx * NORMALIZED_MAX)), int(round(fy * NORMALIZED_MAX)))

        model_w = getattr(ui, "model_screenshot_width", None)
        model_h = getattr(ui, "model_screenshot_height", None)
        if not model_w or not model_h:
            # No declared contract: the tree provider's screen is native and
            # the model space is native / coordinate scale.
            width = int(getattr(ui, "screen_width", 0) or self.shared_state.width or 0)
            height = int(getattr(ui, "screen_height", 0) or self.shared_state.height or 0)
            scale_x = float(getattr(ui, "coordinate_scale_x", 1.0) or 1.0)
            scale_y = float(getattr(ui, "coordinate_scale_y", 1.0) or 1.0)
            if width <= 0 or height <= 0 or scale_x <= 0 or scale_y <= 0:
                raise ValueError("the device did not report a screen size")
            model_w = width / scale_x
            model_h = height / scale_y
        model_w = max(1, int(round(float(model_w))))
        model_h = max(1, int(round(float(model_h))))
        # Stay strictly inside the space: the action layer rejects a point on
        # the far edge (``px >= width``), and a patched centre never sits there.
        return (
            min(int(round(fx * model_w)), model_w - 1),
            min(int(round(fy * model_h)), model_h - 1),
        )

    # ========================================================================
    # Execution
    # ========================================================================

    def _action_dict_for(self, ev: GroundedPointEvent) -> Dict[str, Any]:
        """The registry-style action dict this operation produces.

        Also used on the failure path, so the Manager/history still see which
        action was attempted.
        """
        operation = ev.operation
        if operation in ("TAP", "LONG_PRESS"):
            return {
                "action": "click_at" if operation == "TAP" else "long_press_at",
                "x": int(ev.x or 0),
                "y": int(ev.y or 0),
            }
        if operation == "SWIPE":
            start, end = self._swipe_points_for(ev)
            return {
                "action": "swipe",
                "coordinate": list(start),
                "coordinate2": list(end),
                "duration": _SCROLL_DURATION_S,
            }
        if operation in ("BACK", "HOME", "ENTER"):
            return {"action": "system_button", "button": operation.lower()}
        if operation == "OPEN_APP":
            target = ev.app_target or ""
            package = self._app_packages.get(target, "")
            action: Dict[str, Any] = {"action": "open_app", "text": target}
            if package:
                action["app_id"] = package
            return action
        if operation == "TYPE_TEXT":
            return {"action": "type_text", "text": ev.text_value or "", "clear": True}
        return {"action": operation.lower() or "unknown"}

    async def _execute_operation(
        self, ev: GroundedPointEvent
    ) -> Tuple[Dict[str, Any], ActionResult]:
        """Run one resolved operation through the action layer."""
        ctx = self.action_ctx
        operation = ev.operation
        action = self._action_dict_for(ev)

        if operation == "TAP":
            return action, await click_at(action["x"], action["y"], ctx=ctx)
        if operation == "LONG_PRESS":
            return action, await long_press_at(action["x"], action["y"], ctx=ctx)
        if operation == "SWIPE":
            return action, await swipe(
                action["coordinate"], action["coordinate2"], action["duration"], ctx=ctx
            )
        if operation in ("BACK", "HOME", "ENTER"):
            return action, await system_button(action["button"], ctx=ctx)
        if operation == "OPEN_APP":
            target = ev.app_target or ""
            # The target is a label from the inventory we showed the model;
            # launch its exact package when known (the AppStarter name
            # resolution is only the fallback for label-only inventories).
            if action.get("app_id"):
                return action, await open_bundle_id(app_id=action["app_id"], ctx=ctx)
            return action, await open_app(target, ctx=ctx)
        if operation == "TYPE_TEXT":
            return action, await type_text(ev.text_value or "", None, True, ctx=ctx)
        raise ValueError(f"Unsupported operation: {operation}")

    def _swipe_points_for(
        self, ev: GroundedPointEvent
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """The swipe endpoints (action space) for one SWIPE.

        The swipe workflow always resolves both points; a SWIPE without them
        never reaches here (it fails earlier), so a missing pair is a hard
        error rather than a guessed gesture.
        """
        if ev.swipe_start is None or ev.swipe_end is None:
            raise ValueError("SWIPE without resolved start/end points")
        return (
            self._fraction_to_action_space(*ev.swipe_start),
            self._fraction_to_action_space(*ev.swipe_end),
        )

    # ========================================================================
    # History / thought bookkeeping
    # ========================================================================

    def _absorb_pending(self) -> None:
        """Append the previous step's history entry.

        Screenshot-only reads no tree, so ``screenChanged`` stays None (it
        cannot be judged without a fingerprint) and the entry is queued as
        soon as it is recorded.
        """
        if self._pending is None:
            return
        self._history.append(self._pending)
        self._history = self._history[-5:]
        self._pending = None

    def _remember(self, ev: GroundedPointEvent) -> None:
        """Record one executed decision for the next request's recentActions.

        Labels are rendered without a tree: a point operation names its
        coordinate in device pixels (from the UIState's screen size), a swipe
        its start/end.
        """
        width, height = self._screen_size()
        tap_point = None
        if (
            ev.operation in _COORDINATE_OPERATIONS
            and self._ground_fraction is not None
            and width > 0
            and height > 0
        ):
            fx, fy = self._ground_fraction
            tap_point = (fx * width, fy * height)
        swipe_points = None
        if (
            ev.operation == jev_prompt.SWIPE_OPERATION
            and ev.swipe_start is not None
            and ev.swipe_end is not None
            and width > 0
            and height > 0
        ):
            swipe_points = (
                (ev.swipe_start[0] * width, ev.swipe_start[1] * height),
                (ev.swipe_end[0] * width, ev.swipe_end[1] * height),
            )
        label = jev_prompt.operation_label(
            ev.operation,
            tap_point=tap_point,
            app_name=ev.app_target,
            text=ev.text_value,
            swipe=swipe_points,
        )
        entry = jev_prompt.HistoryEntry(
            operation=ev.operation,
            label=label,
            text=ev.text_value if ev.operation == "TYPE_TEXT" else None,
        )
        # Absorb the previous entry now -- screenChanged is never known without
        # a tree, so nothing has to wait for the next observation.
        self._absorb_pending()
        self._pending = entry

    def _thought_line(self, ev: GroundedPointEvent, error: str = "") -> str:
        if error:
            return f"{ev.operation or 'Execution'} failed: {error}"
        detail = ""
        if ev.confidence is not None:
            detail = f" (confidence {ev.confidence:.2f})"
        if ev.operation in _COORDINATE_OPERATIONS and ev.x is not None:
            detail += f" grounded at ({ev.x}, {ev.y})"
        elif ev.operation == jev_prompt.SWIPE_OPERATION and ev.swipe_start is not None:
            detail += f" {ev.swipe_start} -> {ev.swipe_end}"
        elif ev.app_target:
            detail += f" ({ev.app_target})"
        elif ev.text_value:
            detail += f" ({ev.text_value!r})"
        return f"{ev.operation}{detail}"

    # ========================================================================
    # Event helpers
    # ========================================================================

    def _decision_event(
        self,
        error: str = "",
        operation: str = "",
        confidence: Optional[float] = None,
    ) -> GroundedDecisionEvent:
        return GroundedDecisionEvent(
            operation=operation,
            confidence=confidence if isinstance(confidence, (int, float)) else None,
            error=error,
        )

    def _point_event(
        self,
        decision: GroundedDecisionEvent,
        *,
        x: Optional[int] = None,
        y: Optional[int] = None,
        text_value: Optional[str] = None,
        swipe_start: Optional[Tuple[float, float]] = None,
        swipe_end: Optional[Tuple[float, float]] = None,
        confidence: Optional[float] = None,
        error: str = "",
    ) -> GroundedPointEvent:
        return GroundedPointEvent(
            operation=decision.operation,
            app_target=decision.app_target,
            text_value=text_value,
            x=x,
            y=y,
            swipe_start=swipe_start,
            swipe_end=swipe_end,
            confidence=confidence if confidence is not None else decision.confidence,
            error=error,
        )
