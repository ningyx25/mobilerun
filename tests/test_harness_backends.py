"""Backend adapters: the task catalogue must report real per-task budgets."""

from __future__ import annotations

import asyncio

import httpx
import pytest

import mobilerun.harness.env_service.backend as backend_module
from mobilerun.benchmark import android_world as aw
from mobilerun.harness.protocol import ErrorCode, ProtocolError


class _FakeDriver:
    def __init__(self, url: str) -> None:
        self.url = url


class _FakeTaskClient:
    def __init__(self, url: str) -> None:
        self.url = url

    async def task_list(self):
        return ["Easy", "Hard", "Broken"]

    async def get_complexity(self, name: str, task_idx: int) -> float:
        if name == "Broken":
            raise RuntimeError("complexity endpoint failed")
        return {"Easy": 1.2, "Hard": 2.8}[name]

    async def close(self) -> None:
        return None


def test_android_world_catalogue_uses_each_task_complexity(monkeypatch):
    monkeypatch.setattr(backend_module, "AndroidWorldHttpDriver", _FakeDriver)
    monkeypatch.setattr(aw, "AndroidWorldTaskClient", _FakeTaskClient)

    backend = backend_module.AndroidWorldBackend("aw0", "http://aw.test")
    tasks = asyncio.run(backend.list_tasks())

    # AndroidWorld's budget is int(10 * complexity) per task, not a flat 50.
    # A task whose complexity cannot be read falls back to the old placeholder.
    assert [(t.id, t.complexity, t.max_steps) for t in tasks] == [
        ("Easy", 1.2, 12),
        ("Hard", 2.8, 28),
        ("Broken", None, 50),
    ]


class _DisconnectingDriver:
    """A driver whose device server drops the connection."""

    supported = {"tap"}

    async def screenshot(self):
        raise httpx.RemoteProtocolError(
            "Server disconnected without sending a response"
        )


class _ObserveBackend(backend_module.DriverBackend):
    async def list_tasks(self):
        return []

    async def prepare(self, task_id, **kwargs):
        raise NotImplementedError

    async def evaluate(self):
        return 0.0, ""


def test_observe_maps_a_transport_error_to_a_retryable_503():
    backend = _ObserveBackend("aw0", _DisconnectingDriver())
    with pytest.raises(ProtocolError) as caught:
        asyncio.run(backend.observe(seq=1))
    assert caught.value.status_code == 503
    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE
