"""Shared pytest fixtures for the test suite."""

import logging

import pytest


@pytest.fixture(autouse=True)
def restore_mobilerun_logger():
    """Undo the global logging changes the benchmark's CLI logging setup makes.

    ``_setup_benchmark_logging`` reconfigures the ``mobilerun`` logger
    (handlers, level, propagate) for a CLI run. Leaking that into later tests
    silences their log assertions -- and restoring the level by *direct
    assignment* is not enough: ``Logger.isEnabledFor`` memoizes its verdict per
    level in ``Logger._cache``, so a logger left at ERROR keeps answering
    "disabled" for WARNING even after ``.level`` is put back. ``setLevel``
    clears that cache; assigning ``.level`` does not.
    """
    log = logging.getLogger("mobilerun")
    saved_handlers = list(log.handlers)
    saved_level = log.level
    saved_propagate = log.propagate
    yield log
    log.handlers = saved_handlers
    log.setLevel(saved_level)  # also clears the per-level isEnabledFor cache
    log.propagate = saved_propagate
