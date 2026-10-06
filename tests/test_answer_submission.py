"""Tests for the P0 answer-submission fix (information-retrieval scoring) and
the degraded-input fallbacks around it.
"""

import asyncio
from types import SimpleNamespace

import httpx

from mobilerun.benchmark import android_world as bench


def run(coro):
    return asyncio.run(coro)


# The autouse ``restore_mobilerun_logger`` fixture lives in tests/conftest.py.


# ---------------------------------------------------------------------------
# AndroidWorldTaskClient.execute_action
# ---------------------------------------------------------------------------


def test_execute_action_posts_body_to_execute_action():
    recorded = []

    def handler(request):
        recorded.append(request)
        return httpx.Response(200, json={"status": "success"})

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )
    run(client.execute_action({"action_type": "answer", "text": "True"}))

    assert len(recorded) == 1
    request = recorded[0]
    assert request.method == "POST"
    assert request.url.path == "/execute_action"
    assert b'"action_type":"answer"' in request.content.replace(b" ", b"")
    assert b"True" in request.content


def test_execute_action_is_not_retried_on_5xx(monkeypatch):
    """A non-idempotent action may already have run; 5xx must not retry."""
    monkeypatch.setattr(bench, "_RETRY_BACKOFF_S", 0.0)
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        return httpx.Response(500, json={"detail": "boom"})

    client = bench.AndroidWorldTaskClient(
        "http://localhost:5000", transport=httpx.MockTransport(handler)
    )

    try:
        run(client.execute_action({"action_type": "answer", "text": "x"}))
    except RuntimeError:
        pass
    assert len(attempts) == 1


# ---------------------------------------------------------------------------
# _submit_episode_answer
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, raises=False):
        self.actions = []
        self.raises = raises

    async def execute_action(self, action):
        self.actions.append(action)
        if self.raises:
            raise RuntimeError("no route")


def _agent_with_answer(answer):
    return SimpleNamespace(shared_state=SimpleNamespace(answer=answer))


def test_submit_episode_answer_prefers_shared_state_answer():
    client = _FakeClient()
    run(
        bench._submit_episode_answer(
            client, _agent_with_answer("75"), reason="the answer is 75"
        )
    )
    assert client.actions == [{"action_type": "answer", "text": "75"}]


def test_submit_episode_answer_falls_back_to_reason():
    client = _FakeClient()
    run(bench._submit_episode_answer(client, _agent_with_answer(""), reason="True"))
    assert client.actions == [{"action_type": "answer", "text": "True"}]


def test_submit_episode_answer_skips_when_empty():
    client = _FakeClient()
    run(bench._submit_episode_answer(client, _agent_with_answer("  "), reason=""))
    assert client.actions == []


def test_submit_episode_answer_is_best_effort():
    client = _FakeClient(raises=True)
    # Must not raise.
    run(bench._submit_episode_answer(client, _agent_with_answer("True"), reason=""))
    assert client.actions  # it did try


# ---------------------------------------------------------------------------
# system_button normalization (P1)
# ---------------------------------------------------------------------------


class _RecordingDriver:
    def __init__(self):
        self.buttons = []

    async def press_button(self, button):
        # Mirrors AndroidWorldHttpDriver.press_button: strip + lower, then
        # validate against the supported set.
        normalized = str(button or "").strip().lower()
        if normalized not in ("back", "home", "enter"):
            raise ValueError(f"Button '{button}' not supported.")
        self.buttons.append(normalized)


def _ctx(driver):
    return SimpleNamespace(
        driver=driver, ui=None, shared_state=None, macro_recorder=None
    )


def test_system_button_strips_surrounding_whitespace():
    from mobilerun.agent.utils.actions import system_button

    driver = _RecordingDriver()
    result = run(system_button("\nback\n", ctx=_ctx(driver)))

    assert result.success is True
    assert driver.buttons == ["back"]


def test_system_button_normalizes_upper_and_spaced():
    from mobilerun.agent.utils.actions import system_button

    driver = _RecordingDriver()
    assert run(system_button("  Back  ", ctx=_ctx(driver))).success is True
    assert driver.buttons == ["back"]
