"""Evaluation harness: environment and agent as two cooperating services.

The wire contract lives in :mod:`mobilerun.harness.protocol` and is shared by
both halves. Nothing here changes how agents work: the agent service injects
a :class:`~mobilerun.harness.drivers.RemoteEnvDriver` and the existing
``MobileAgent`` runs unmodified.

Layout::

    protocol.py        pydantic wire models (Observation / Action / ...)
    actions.py         Action -> DeviceDriver mapping + coordinate conversion
    client.py          EnvClient / AgentClient (httpx)
    drivers.py         RemoteEnvDriver(DeviceDriver)
    env_service/       FastAPI app, backends, session registry, scoring
    agent_service/     FastAPI app, episode runtime, event projection
    runner.py          the thin orchestrator (``mobilerun eval``)

The FastAPI factories are deliberately not re-exported here: importing them
pulls in the server stack, which a client-only consumer does not need.
"""

from mobilerun.harness.actions import execute_action, supported_ops
from mobilerun.harness.client import AgentClient, EnvClient, ServiceError
from mobilerun.harness.drivers import RemoteEnvDriver
from mobilerun.harness.protocol import (
    PROTOCOL_VERSION,
    Action,
    ActionOp,
    Capabilities,
    EpisodeInfo,
    EpisodeRequest,
    EpisodeResult,
    EpisodeState,
    FinishResponse,
    Observation,
    ScreenInfo,
    SessionInfo,
    SessionState,
    StepRequest,
    StepResponse,
    TaskSpec,
)

__all__ = [
    "PROTOCOL_VERSION",
    "Action",
    "ActionOp",
    "AgentClient",
    "Capabilities",
    "EnvClient",
    "EpisodeInfo",
    "EpisodeRequest",
    "EpisodeResult",
    "EpisodeState",
    "FinishResponse",
    "Observation",
    "RemoteEnvDriver",
    "ScreenInfo",
    "ServiceError",
    "SessionInfo",
    "SessionState",
    "StepRequest",
    "StepResponse",
    "TaskSpec",
    "execute_action",
    "supported_ops",
]
