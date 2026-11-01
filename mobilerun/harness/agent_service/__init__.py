"""Agent service: one worker per implementation, plus a dispatcher in front.

* ``create_agent_app(config)`` — the **worker**: runs episodes in this process
  with ``MobileAgent``. It is a library factory, not a CLI command; the
  deployment serving it decides the host and port.
* ``create_agent_dispatcher(entries)`` — the **dispatcher**: routes each
  episode to one registered worker by ``agent_id``
  (``mobilerun serve agent id=name@url``).
"""

from mobilerun.harness.agent_service.episode import (
    EpisodeManager,
    EpisodeRuntime,
    declared_done,
)
from mobilerun.harness.agent_service.registry import (
    AgentRegistry,
    AgentSlot,
    composite_id,
    parse_entry,
)

__all__ = [
    "AgentRegistry",
    "AgentSlot",
    "EpisodeManager",
    "EpisodeRuntime",
    "composite_id",
    "declared_done",
    "parse_entry",
]


def create_agent_app(*args, **kwargs):
    """Lazy re-export of the worker app factory (imports the server stack)."""
    from mobilerun.harness.agent_service.app import create_agent_app as _create

    return _create(*args, **kwargs)


def create_agent_dispatcher(*args, **kwargs):
    """Lazy re-export of the dispatcher factory (imports the server stack)."""
    from mobilerun.harness.agent_service.dispatcher import (
        create_agent_dispatcher as _create,
    )

    return _create(*args, **kwargs)
