"""Tests for the grounded executor's argument-resolving oneflows.

``TextInputAgent`` supplies the exact text for a TYPE_TEXT step and
``SwipeAgent`` supplies the swipe start/end points for a SWIPE step --
replacing the operation model's former ``text_value`` / ``scroll_direct``
questions. Both ground their answer on the current screenshot.
"""

import asyncio
import base64
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ImageBlock, TextBlock

from mobilerun.agent.oneflows.swipe_agent import SwipeAgent
from mobilerun.agent.oneflows.text_input_agent import TextInputAgent


def run_oneflow(agent, **kwargs):
    """Run one oneflow inside a fresh event loop (``Workflow.run`` needs a
    running loop when it is called)."""

    async def _run():
        return await agent.run(**kwargs)

    return asyncio.run(_run())


def fake_completion(text):
    """A response-like object: acomplete_text reads `.text` off it and pulls
    usage best-effort (a plain object llm reports none)."""
    async def _complete(llm, prompt, **kwargs):
        return SimpleNamespace(text=text)

    return _complete


def fake_chat_completion(text, captured):
    async def _chat(llm, messages, **kwargs):
        captured.extend(messages)
        return SimpleNamespace(text=text)

    return _chat


# ---------------------------------------------------------------------------
# TextInputAgent
# ---------------------------------------------------------------------------


def test_text_parse_handles_json_bare_and_garbage():
    assert TextInputAgent._parse_text('{"text": "Welsh cakes"}') == "Welsh cakes"
    assert TextInputAgent._parse_text('here: {"text": "a b"}\ntrailing') == "a b"
    assert TextInputAgent._parse_text('"Welsh cakes"') == "Welsh cakes"
    assert TextInputAgent._parse_text("") == ""
    assert TextInputAgent._parse_text("line one\nline two") == ""
    # A sentence of prose is refused: it must never reach the device.
    assert (
        TextInputAgent._parse_text(
            'I would type "Welsh cakes" into the search field.'
        )
        == ""
    )


def test_text_agent_returns_the_llm_value(monkeypatch):
    monkeypatch.setattr(
        "mobilerun.agent.oneflows.common.acomplete_with_retries",
        fake_completion('{"text": "2023_06_24_training.m4a"}'),
    )
    agent = TextInputAgent(llm=object(), timeout=10)
    result = run_oneflow(
        agent,
        goal='Save it as "2023_06_24_training.m4a".',
        subgoal="Type the name into the field",
        focused_text="Record-1",
    )
    assert result["text"] == "2023_06_24_training.m4a"


def test_text_agent_attaches_the_screenshot(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "mobilerun.agent.oneflows.common.acall_with_retries",
        fake_chat_completion('{"text": "Welsh cakes"}', captured),
    )
    agent = TextInputAgent(llm=object(), timeout=10)
    result = run_oneflow(
        agent, goal="g", subgoal="s", focused_text="", image=b"\x89PNG-raw"
    )
    assert result["text"] == "Welsh cakes"
    blocks = captured[0].blocks
    assert any(isinstance(b, TextBlock) for b in blocks)
    # ImageBlock stores the bytes base64-encoded.
    expected = base64.b64encode(b"\x89PNG-raw")
    assert any(isinstance(b, ImageBlock) and b.image == expected for b in blocks)


# ---------------------------------------------------------------------------
# SwipeAgent
# ---------------------------------------------------------------------------


def test_swipe_parse_points_validates_and_clamps():
    parsed = SwipeAgent._parse_points('{"start": [500, 800], "end": [500, 200]}')
    assert parsed == ((500.0, 800.0), (500.0, 200.0))

    # Out-of-range points clamp to the normalized 0-1000 space.
    parsed = SwipeAgent._parse_points('{"start": [2000, 3000], "end": [-100, -100]}')
    assert parsed == ((1000.0, 1000.0), (0.0, 0.0))

    # Garbage, missing keys and a tap-like (degenerate) swipe are refused.
    assert SwipeAgent._parse_points("not json") is None
    assert SwipeAgent._parse_points('{"start": [1, 2]}') is None
    assert SwipeAgent._parse_points('{"start": [500, 500], "end": [501, 501]}') is None


def test_swipe_agent_attaches_the_screenshot_and_returns_fractions(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "mobilerun.agent.oneflows.common.acall_with_retries",
        fake_chat_completion('{"start": [500, 800], "end": [500, 200]}', captured),
    )
    agent = SwipeAgent(llm=object(), timeout=10)
    result = run_oneflow(
        agent, goal="Find WiFi", subgoal="Scroll down", image=b"\x89PNG-raw"
    )
    # Normalized 0-1000 points come back as 0..1 screen fractions.
    assert result["start"] == pytest.approx([0.5, 0.8])
    assert result["end"] == pytest.approx([0.5, 0.2])
    blocks = captured[0].blocks
    expected = base64.b64encode(b"\x89PNG-raw")
    assert any(isinstance(b, ImageBlock) and b.image == expected for b in blocks)
    prompt_text = next(b.text for b in blocks if isinstance(b, TextBlock))
    assert "normalized 0-1000" in prompt_text


def test_swipe_agent_reports_usage_key(monkeypatch):
    # The result always carries a "usage" key (None when the provider reports
    # no usage), so the executor can forward it unconditionally.
    monkeypatch.setattr(
        "mobilerun.agent.oneflows.common.acall_with_retries",
        fake_chat_completion('{"start": [500, 800], "end": [500, 200]}', []),
    )
    agent = SwipeAgent(llm=object(), timeout=10)
    result = run_oneflow(agent, goal="g", subgoal="s", image=b"\x89PNG-raw")
    assert "usage" in result


def test_swipe_agent_without_a_screenshot_errors(monkeypatch):
    # Nothing to ground the points on.
    agent = SwipeAgent(llm=object(), timeout=10)
    result = run_oneflow(agent, goal="g", subgoal="s", image=None)
    assert "no screenshot" in result["error"]


def test_swipe_agent_unusable_answer_is_a_failure(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "mobilerun.agent.oneflows.common.acall_with_retries",
        fake_chat_completion("garbage", captured),
    )
    agent = SwipeAgent(llm=object(), timeout=10)
    result = run_oneflow(agent, goal="g", subgoal="s", image=b"\x89PNG-raw")
    # No blind fallback: an unusable answer is reported, not guessed.
    assert "error" in result
