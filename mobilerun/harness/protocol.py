"""Wire protocol for the mobilerun evaluation harness (protocol v1).

The harness splits an evaluation into two services that only talk through the
models in this module:

- the **environment service** accepts an :class:`Action` and returns an
  :class:`Observation` (``POST /v1/sessions/{sid}/step``),
- the **agent service** accepts an :class:`Observation` and returns an
  :class:`Action` (``POST /v1/policies/{name}/act``).

Phase one runs built-in agents in *episode* mode: the agent drives the
environment through the same ``observe``/``step`` pair, so the environment
half of the contract is exercised end to end from day one, and the
state->action half can be added later without changing it.

Two invariants keep the split cheap:

- an :class:`Observation`'s ``ui_tree`` / ``phone_state`` / ``screen`` fields
  are byte-compatible with what :class:`mobilerun.tools.driver.android_world.
  AndroidWorldHttpDriver` already returns, so the agent-side state pipeline
  (``AndroidStateProvider``) consumes them unchanged;
- an :class:`Action` is a 1:1 mapping onto a ``DeviceDriver`` verb, so the
  environment side owns no new device I/O.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bumped only on breaking changes; both services echo it in /v1/health so a
# mismatched pair fails loudly instead of mis-parsing actions.
PROTOCOL_VERSION = "1.0"

# The coordinate space every coordinate-bearing action is expressed in. The
# observation declares which one it means; an action may override it per call.
CoordinateSpace = Literal["screenshot_pixels", "normalized_1000"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorCode(str, Enum):
    """Machine-readable error codes carried by every failed response."""

    BAD_REQUEST = "bad_request"
    UNKNOWN_SESSION = "unknown_session"
    UNKNOWN_TASK = "unknown_task"
    UNKNOWN_EPISODE = "unknown_episode"
    UNKNOWN_BACKEND = "unknown_backend"
    UNKNOWN_AGENT = "unknown_agent"
    SESSION_STATE = "session_state"
    STALE_OBSERVATION = "stale_observation"
    UNSUPPORTED_ACTION = "unsupported_action"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    CAPACITY = "capacity"
    PROTOCOL_VERSION = "protocol_version"
    UNAUTHORIZED = "unauthorized"
    NOT_IMPLEMENTED = "not_implemented"
    INTERNAL = "internal"


class ErrorBody(BaseModel):
    """Body of every non-2xx response."""

    code: ErrorCode
    message: str
    detail: Dict[str, Any] = Field(default_factory=dict)


class ProtocolError(Exception):
    """Raised by services to produce a structured ``ErrorBody`` response."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail or {}

    def body(self) -> ErrorBody:
        return ErrorBody(code=self.code, message=self.message, detail=self.detail)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class ScreenInfo(BaseModel):
    """Geometry of the current frame, in native screenshot pixels."""

    native_width: int
    native_height: int
    # Size the input layer expects, when it differs from the screenshot (iOS
    # points vs physical pixels). ``None`` means "same as the screenshot".
    input_width: Optional[int] = None
    input_height: Optional[int] = None
    density: Optional[float] = None

    @property
    def action_width(self) -> int:
        return self.input_width or self.native_width

    @property
    def action_height(self) -> int:
        return self.input_height or self.native_height


class Observation(BaseModel):
    """One device snapshot: the "state" half of the protocol.

    ``screenshot_png_b64`` and ``ui_tree`` mirror
    ``DeviceDriver.screenshot()`` / ``DeviceDriver.get_ui_tree()`` exactly, so
    an agent that already speaks the driver contract needs no adaptation.
    """

    model_config = ConfigDict(extra="allow")

    seq: int = 0
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    screenshot_png_b64: Optional[str] = None
    # Portal-shaped accessibility root: {"boundsInScreen": ..., "children": [...]}
    ui_tree: Optional[Dict[str, Any]] = None
    phone_state: Dict[str, Any] = Field(default_factory=dict)
    device_context: Dict[str, Any] = Field(default_factory=dict)
    screen: ScreenInfo
    coordinate_space: CoordinateSpace = "screenshot_pixels"
    # "<package>/<Activity>"; lets tree-less consumers still name the app.
    foreground: Optional[str] = None

    def screenshot_bytes(self) -> Optional[bytes]:
        if not self.screenshot_png_b64:
            return None
        return base64.b64decode(self.screenshot_png_b64)

    def state_dict(self) -> Dict[str, Any]:
        """The combined dict ``AndroidStateProvider`` consumes, verbatim."""
        return {
            "a11y_tree": self.ui_tree
            or {
                "boundsInScreen": {
                    "left": 0,
                    "top": 0,
                    "right": self.screen.native_width,
                    "bottom": self.screen.native_height,
                },
                "className": "",
                "text": "",
                "contentDescription": "",
                "resourceId": "",
                "packageName": "",
                "children": [],
            },
            "phone_state": dict(self.phone_state),
            "device_context": {
                **dict(self.device_context),
                "screen_bounds": {
                    "width": self.screen.native_width,
                    "height": self.screen.native_height,
                },
            },
        }

    @classmethod
    def build(
        cls,
        *,
        seq: int,
        screenshot: Optional[bytes],
        state: Optional[Dict[str, Any]],
        screen: ScreenInfo,
        coordinate_space: CoordinateSpace = "screenshot_pixels",
    ) -> "Observation":
        """Assemble an observation from a driver's two raw reads."""
        state = state or {}
        phone_state = dict(state.get("phone_state") or {})
        return cls(
            seq=seq,
            screenshot_png_b64=(
                base64.b64encode(screenshot).decode("ascii") if screenshot else None
            ),
            ui_tree=state.get("a11y_tree"),
            phone_state=phone_state,
            device_context=dict(state.get("device_context") or {}),
            screen=screen,
            coordinate_space=coordinate_space,
            foreground=str(phone_state.get("currentApp") or "") or None,
        )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ActionOp = Literal[
    "tap",
    "long_press",
    "swipe",
    "drag",
    "type",
    "key",
    "open_app",
    "stop_app",
    "wait",
    "complete",
]

# Arguments every op requires, and the JSON type each one must have. Keeping
# the table here (rather than in either service) means a malformed action is
# rejected at the protocol boundary with a precise message instead of failing
# deep inside a device call.
ACTION_ARGS: Dict[str, Dict[str, str]] = {
    "tap": {"x": "number", "y": "number"},
    "long_press": {"x": "number", "y": "number"},
    "swipe": {
        "x1": "number",
        "y1": "number",
        "x2": "number",
        "y2": "number",
    },
    "drag": {
        "x1": "number",
        "y1": "number",
        "x2": "number",
        "y2": "number",
    },
    "type": {"text": "string"},
    # ``key`` takes either a named button or a raw key code (exactly one).
    "key": {},
    "open_app": {"app": "string"},
    "stop_app": {"app": "string"},
    "wait": {"duration_s": "number"},
    "complete": {"success": "boolean", "message": "string"},
}

_OPTIONAL_ARGS: Dict[str, Dict[str, str]] = {
    "swipe": {"duration_ms": "number"},
    "drag": {"duration_s": "number"},
    "type": {"clear": "boolean"},
    "key": {"button": "string", "keycode": "number"},
    "open_app": {"activity": "string"},
}

_JSON_TYPES = {
    "string": str,
    "number": (int, float),
    "boolean": bool,
}


class Action(BaseModel):
    """One device operation: the "action" half of the protocol."""

    model_config = ConfigDict(extra="forbid")

    op: ActionOp
    args: Dict[str, Any] = Field(default_factory=dict)
    # Overrides ``Observation.coordinate_space`` for coordinate-bearing ops.
    coordinate_space: Optional[CoordinateSpace] = None

    @model_validator(mode="after")
    def _check_args(self) -> "Action":
        required = ACTION_ARGS[self.op]
        optional = _OPTIONAL_ARGS.get(self.op, {})
        merged = {**required, **optional}
        for name, kind in merged.items():
            if name not in self.args:
                if name in required:
                    raise ValueError(f"action '{self.op}' requires '{name}'")
                continue
            value = self.args[name]
            if value is None:
                continue
            expected = _JSON_TYPES[kind]
            # bool is an int subclass: never let True stand in for 1.
            if isinstance(value, bool) and kind != "boolean":
                raise ValueError(f"action '{self.op}.{name}' must be {kind}")
            if not isinstance(value, expected):
                raise ValueError(f"action '{self.op}.{name}' must be {kind}")
        unknown = set(self.args) - set(merged)
        if unknown:
            raise ValueError(
                f"action '{self.op}' does not take {sorted(unknown)}; "
                f"accepted: {sorted(merged)}"
            )
        if self.op == "key" and not (
            self.args.get("button") or self.args.get("keycode") is not None
        ):
            raise ValueError("action 'key' requires 'button' or 'keycode'")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.op == "complete"

    @property
    def touches_device(self) -> bool:
        return self.op not in ("wait", "complete")


# ---------------------------------------------------------------------------
# Tasks and capabilities
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    """One evaluable task, normalised across benchmark backends."""

    model_config = ConfigDict(extra="allow")

    id: str
    backend: str
    instruction: str = ""
    max_steps: int = 50
    complexity: Optional[float] = None
    start_on_home_screen: bool = True
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Capabilities(BaseModel):
    """What an environment backend (or an agent policy) can actually do."""

    # Protocol-level action vocabulary (``Action.op`` values).
    ops: List[str] = Field(default_factory=list)
    # Raw ``DeviceDriver`` verbs. The agent side rebuilds its capability set
    # from these so ``ToolRegistry.disable_unsupported`` prunes exactly the
    # tools this device cannot run.
    verbs: List[str] = Field(default_factory=list)
    buttons: List[str] = Field(default_factory=list)
    platform: str = "Android"
    has_ui_tree: bool = True
    has_screenshot: bool = True
    supports_stabilize: bool = False
    supports_answer: bool = False
    coordinate_space: CoordinateSpace = "screenshot_pixels"


class BackendInfo(BaseModel):
    id: str
    kind: str
    url: Optional[str] = None
    state: Literal["ready", "busy", "down"] = "ready"
    capabilities: Optional[Capabilities] = None


class HealthResponse(BaseModel):
    ok: bool
    protocol_version: str = PROTOCOL_VERSION
    backends: List[BackendInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sessions (environment side)
# ---------------------------------------------------------------------------


class SessionState(str, Enum):
    RUNNING = "running"
    FINISHING = "finishing"
    CLOSED = "closed"
    ERROR = "error"


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Optional[str] = None
    task_id: str
    task_idx: int = 0
    seed: Optional[int] = None
    # Passed through to the backend; e.g. AndroidWorld's suite re-init knobs.
    params: Dict[str, Any] = Field(default_factory=dict)


class SessionInfo(BaseModel):
    session_id: str
    backend: str
    state: SessionState
    task: TaskSpec
    capabilities: Capabilities
    screen: ScreenInfo
    created_at: str
    seq: int = 0
    steps: int = 0


class CreateSessionResponse(SessionInfo):
    observation: Observation


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    # Client-side monotonic counter. Replaying an already-applied index returns
    # the cached result instead of re-executing the action -- an at-least-once
    # HTTP retry must never become a double tap.
    step_index: int
    # Optional stale-frame guard: the seq of the observation the action was
    # decided on. Enforced only when supplied, because one observation may
    # legitimately drive several actions.
    expect_seq: Optional[int] = None
    # Ask for the post-action frame in this response. The agent-side driver
    # leaves it off: it captures lazily when the agent reads state next.
    include_observation: bool = False


class StepResponse(BaseModel):
    step_index: int
    seq: int
    ok: bool
    summary: str = ""
    error: Optional[str] = None
    observation: Optional[Observation] = None


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class FinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    declared_done: bool
    answer: Optional[str] = None
    # Seconds to let the UI settle before scoring.
    settle_s: Optional[float] = None


class FinishResponse(BaseModel):
    """Scoring result. ``gated_score`` is derived server-side, never trusted."""

    score: float = 0.0
    gated_score: float = 0.0
    declared_done: bool = False
    reason: str = ""
    eval_reason: str = ""


class DeviceDateResponse(BaseModel):
    date: str = ""


class DeviceAppsResponse(BaseModel):
    apps: List[Dict[str, str]] = Field(default_factory=list)
    packages: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Episodes (agent side, phase one: the agent owns the loop)
# ---------------------------------------------------------------------------


class EnvironmentRef(BaseModel):
    """Points the agent at one open environment session."""

    base_url: str
    session_id: str


class EpisodeLimits(BaseModel):
    max_steps: Optional[int] = None
    timeout_s: float = 1000.0


class EpisodeEvidenceConfig(BaseModel):
    enabled: bool = True
    dir: Optional[str] = None


class AgentOverrides(BaseModel):
    """Per-episode agent configuration overrides.

    Only fields the CLI can also set are exposed; anything else belongs in the
    config file the service was started with.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = None
    reasoning: Optional[bool] = None
    vision_only: Optional[bool] = None
    max_steps: Optional[int] = None


class EpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    environment: EnvironmentRef
    # Which registered agent should run this episode. Consumed by the
    # dispatcher; a worker ignores it (it is always itself).
    agent_id: Optional[str] = None
    overrides: AgentOverrides = Field(default_factory=AgentOverrides)
    limits: EpisodeLimits = Field(default_factory=EpisodeLimits)
    evidence: EpisodeEvidenceConfig = Field(default_factory=EpisodeEvidenceConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodeState(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class EpisodeResult(BaseModel):
    success: bool = False
    # True only when the agent itself declared the episode over -- the same
    # gate the benchmark runners apply before trusting a score.
    declared_done: bool = False
    answer: Optional[str] = None
    steps: int = 0
    reason: str = ""
    duration_s: float = 0.0
    usage: Optional[TokenUsage] = None
    evidence_dir: Optional[str] = None


class EpisodeInfo(BaseModel):
    episode_id: str
    state: EpisodeState
    result: Optional[EpisodeResult] = None
    error: Optional[str] = None
    created_at: str
    goal: str = ""
    #: Steps decided so far. Live while the episode runs, final afterwards.
    steps: int = 0
    # Set by the dispatcher: which registered agent is running this episode.
    agent_id: Optional[str] = None


class AgentSlotInfo(BaseModel):
    """One registered worker behind the agent dispatcher."""

    id: str
    name: str
    url: str
    state: Literal["ready", "busy", "down"] = "ready"
    # What the worker calls itself (``config.agent.name``), as opposed to the
    # label the operator registered it under.
    implementation: Optional[str] = None
    episodes_running: int = 0
    max_concurrent_episodes: int = 1
    name_mismatch: bool = False


class AgentsHealth(BaseModel):
    """Dispatcher health: one row per registered worker."""

    ok: bool
    protocol_version: str = PROTOCOL_VERSION
    mode: Literal["dispatcher"] = "dispatcher"
    agents: List[AgentSlotInfo] = Field(default_factory=list)


class EpisodeEvent(BaseModel):
    """One serialised workflow event on the SSE stream."""

    seq: int
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    type: str
    step: Optional[int] = None
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policies (agent side, phase two: the harness owns the loop)
# ---------------------------------------------------------------------------


class PolicyCapabilities(BaseModel):
    name: str
    ops: List[str] = Field(default_factory=list)
    needs_screenshot: bool = True
    needs_ui_tree: bool = False
    coordinate_space: CoordinateSpace = "screenshot_pixels"


class CreatePolicySessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    overrides: AgentOverrides = Field(default_factory=AgentOverrides)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreatePolicySessionResponse(BaseModel):
    policy_session_id: str
    capabilities: PolicyCapabilities


class ActRequest(BaseModel):
    """A single ``state -> action`` decision.

    ``step`` is monotonic per session and makes the call idempotent: replaying
    the same ``step`` returns the cached decision instead of advancing the
    agent's internal context twice.
    """

    model_config = ConfigDict(extra="forbid")

    policy_session_id: str
    step: int
    observation: Observation
    feedback: Optional[Dict[str, Any]] = None


class ActResponse(BaseModel):
    step: int
    action: Action
    done: bool = False
    thought: Optional[str] = None
    answer: Optional[str] = None
    usage: Optional[TokenUsage] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    """Short, log-friendly identifier for sessions and episodes."""
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:12]}"
