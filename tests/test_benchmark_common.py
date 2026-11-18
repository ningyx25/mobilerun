"""Shared benchmark helpers: error classification for abnormal endings."""

from types import SimpleNamespace

import pytest

from mobilerun.benchmark.common import (
    _infra_error_reason,
    _unrecognised_failure_reason,
)


def _run(*, success=None, reason="", error_kind=None):
    kwargs = {"reason": reason}
    if success is not None:
        kwargs["success"] = success
    if error_kind is not None:
        kwargs["error_kind"] = error_kind
    return SimpleNamespace(**kwargs)


@pytest.mark.parametrize(
    "reason",
    [
        "timed out after 600s",
        "Timed out after 30s",
        "WorkflowTimeoutError: Operation timed out after 1000 seconds",
    ],
)
def test_timeout_reason_on_failure_is_an_error(reason):
    assert _infra_error_reason(_run(success=False, reason=reason)) is not None


@pytest.mark.parametrize("reason", ["timed out after 600s", "WorkflowTimeoutError: x"])
def test_timeout_reason_on_success_is_not_an_error(reason):
    assert _infra_error_reason(_run(success=True, reason=reason)) is None


def test_timeout_reason_without_success_attribute_is_not_an_error():
    assert _infra_error_reason(_run(reason="timed out after 600s")) is None


@pytest.mark.parametrize(
    "reason",
    [
        "max_steps (50) reached",
        "Reached maximum steps (50)",
        "could not find the settings screen",
    ],
)
def test_normal_failures_are_not_errors(reason):
    assert _infra_error_reason(_run(success=False, reason=reason)) is None


def test_error_prefixed_provider_failure_is_still_an_error():
    reason = "Error: Error code: 429 - {'error': {'code': 'model_cooldown'}}"
    assert _infra_error_reason(_run(success=False, reason=reason)) is not None


def test_error_kind_still_wins():
    result = _run(success=False, reason="whatever", error_kind="infra")
    assert _infra_error_reason(result) == "InfraError: whatever"


def test_unrecognised_failure_is_default_denied():
    failed = _run(success=False, reason="watchdog aborted the episode")
    max_steps = _run(success=False, reason="max_steps (5) reached")
    declared = _run(success=False, reason="could not find the screen")
    succeeded = _run(success=True, reason="whatever")

    assert _unrecognised_failure_reason(failed, declared_done=False) is not None
    assert _unrecognised_failure_reason(max_steps, declared_done=False) is None
    assert _unrecognised_failure_reason(declared, declared_done=True) is None
    assert _unrecognised_failure_reason(succeeded, declared_done=False) is None
