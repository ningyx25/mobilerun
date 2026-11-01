"""Environment service: devices, tasks, sessions and scoring."""

from mobilerun.harness.env_service.backend import (
    AndroidWorldBackend,
    DriverBackend,
    EnvBackend,
    MobileWorldBackend,
    create_backend,
)
from mobilerun.harness.env_service.scoring import finish_episode
from mobilerun.harness.env_service.session import Session, SessionRegistry

__all__ = [
    "AndroidWorldBackend",
    "DriverBackend",
    "EnvBackend",
    "MobileWorldBackend",
    "Session",
    "SessionRegistry",
    "create_backend",
    "finish_episode",
]


def create_env_app(*args, **kwargs):
    """Lazy re-export of the FastAPI factory (imports the server stack)."""
    from mobilerun.harness.env_service.app import create_env_app as _create

    return _create(*args, **kwargs)
