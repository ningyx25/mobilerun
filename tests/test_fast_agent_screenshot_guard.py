"""Regression tests: an undecodable screenshot must not abort a FastAgent run.

A truncated PNG used to raise ``OSError: image file is truncated`` inside the
message-building block (outside any try/except), bubble up to MobileAgent's
catch-all, and end the episode with ``score=0`` / ``finished=False``. The guard
drops that frame and runs the step without vision instead.
"""

import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock
from PIL import Image

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent


def _png_bytes(width=64, height=64):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(7, 8, 9)).save(buf, format="PNG")
    return buf.getvalue()


def _truncated_png():
    data = _png_bytes()
    return data[: len(data) // 2]


CLICK_RESPONSE = """I will tap the target.
<function_calls>
<invoke name="click">
<parameter name="index">12</parameter>
</invoke>
</function_calls>"""

COMPLETE_RESPONSE = """The task is complete.
<function_calls>
<invoke name="complete">
<parameter name="success">true</parameter>
<parameter name="message">Done</parameter>
</invoke>
</function_calls>"""


class _SequencedResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def __call__(self, _llm, messages, *, stream):
        self.requests.append(list(messages))
        if not self.responses:
            raise AssertionError("FastAgent made an unexpected extra LLM request")
        content = self.responses.pop(0)
        return ChatResponse(message=ChatMessage(role="assistant", content=content))


class _PromptResolver:
    def get_prompt(self, name):
        if name == "fast_agent_system":
            return "Use the documented XML tools: {{ tool_descriptions }}"
        if name == "fast_agent_user":
            return "Goal: {{ goal }}"
        return None


class _StateProvider:
    # Triggers the resize path in message building (the block that used to
    # raise on truncated bytes).
    requires_coordinate_tools = True

    async def get_state(self):
        return SimpleNamespace(
            formatted_text="Settings home with Network & internet visible",
            focused_text="",
            elements=[],
            phone_state={
                "packageName": "com.android.settings",
                "currentApp": "com.android.settings.Settings",
            },
        )


class _FramesDriver:
    """Returns a scripted screenshot per capture; the last frame repeats."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.captures = 0

    async def screenshot(self):
        frame = self.frames[min(self.captures, len(self.frames) - 1)]
        self.captures += 1
        return frame


class _Registry:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.executed = []
        self.tools = {"click": object(), "complete": object()}

    def get_tool_descriptions_xml(self):
        return '<tool name="click"/><tool name="complete"/>'

    def get_param_types(self):
        return {"index": "number", "success": "boolean"}

    async def execute(self, name, parameters, _action_ctx, workflow_ctx=None):
        self.executed.append((name, parameters))
        if name == "complete":
            await self.shared_state.complete(
                parameters.get("success", False),
                message=parameters.get("message", ""),
            )
        return ActionResult(success=True, summary=f"{name} executed")


def _run_sequence(frames, responses):
    async def _run():
        shared_state = MobileAgentState()
        registry = _Registry(shared_state)
        responder = _SequencedResponder(responses)
        driver = _FramesDriver(frames)
        config = SimpleNamespace(
            fast_agent=SimpleNamespace(vision=True, parallel_tools=False),
            max_steps=20,
            streaming=False,
            after_sleep_action=0,
        )
        action_ctx = SimpleNamespace(
            driver=driver,
            credential_manager=None,
            ui=None,
        )
        agent = FastAgent(
            llm=SimpleNamespace(),
            agent_config=config,
            registry=registry,
            action_ctx=action_ctx,
            state_provider=_StateProvider(),
            shared_state=shared_state,
            prompt_resolver=_PromptResolver(),
        )

        with (
            patch(
                "mobilerun.agent.fast_agent.fast_agent.acall_with_retries",
                new=responder,
            ),
            patch(
                "mobilerun.agent.fast_agent.fast_agent.get_usage_from_response",
                return_value=None,
            ),
            patch(
                "mobilerun.agent.fast_agent.fast_agent.record_langfuse_screenshot",
            ),
        ):
            result = await agent.run(input="Exercise the screenshot guard")

        return result, responder, registry, driver

    return asyncio.run(_run())


def _image_blocks(messages):
    return [
        block
        for message in messages
        for block in (message.blocks or [])
        if isinstance(block, ImageBlock)
    ]


def test_step_survives_a_truncated_frame_and_drops_the_image(capsys):
    result, responder, registry, driver = _run_sequence(
        [_truncated_png()], [COMPLETE_RESPONSE]
    )

    assert result["success"] is True
    assert [name for name, _params in registry.executed] == ["complete"]
    assert driver.captures == 1
    assert len(responder.requests) == 1
    # The step still reached the LLM, just without an image.
    assert _image_blocks(responder.requests[0]) == []
    # The repo logger writes to stdout (it does not propagate to caplog).
    assert "Dropping undecodable screenshot" in capsys.readouterr().out


def test_next_step_attaches_the_recovered_frame():
    result, responder, registry, _driver = _run_sequence(
        [_truncated_png(), _png_bytes()], [CLICK_RESPONSE, COMPLETE_RESPONSE]
    )

    assert result["success"] is True
    assert [name for name, _params in registry.executed] == ["click", "complete"]
    assert len(responder.requests) == 2
    assert _image_blocks(responder.requests[0]) == []

    blocks = _image_blocks(responder.requests[1])
    assert len(blocks) == 1
    # ImageBlock stores the bytes base64-encoded; the frame must decode.
    Image.open(io.BytesIO(base64.b64decode(blocks[0].image))).verify()
