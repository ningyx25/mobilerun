"""Shared helpers for the single-purpose oneflows.

The operation-argument oneflows (``text_input_agent`` / ``swipe_agent``) share
one LLM-call shape -- a plain completion, or a multimodal chat when a
screenshot is attached -- and one "answer is a JSON object we can parse out of
the text" contract.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock

from mobilerun.agent.usage import UsageResult, get_usage_from_response
from mobilerun.agent.utils.inference import acall_with_retries, acomplete_with_retries


async def acomplete_text(
    llm: Any,
    prompt: str,
    *,
    image: Optional[bytes] = None,
    stream: bool = False,
) -> Tuple[str, Optional[UsageResult]]:
    """The prompt's answer text and its token usage: a multimodal chat when a
    screenshot is attached, a plain completion otherwise.

    ``UsageResult`` is None when the provider reports no usage (best effort,
    like the executor's own calls); the oneflows surface it on their events so
    the benchmark's usage totals include them.
    """
    if not image:
        response = await acomplete_with_retries(llm, prompt, stream=stream)
    else:
        messages = [
            ChatMessage(
                role="user",
                blocks=[TextBlock(text=prompt), ImageBlock(image=image)],
            )
        ]
        response = await acall_with_retries(llm, messages, stream=stream)
    # CompletionResponse carries `.text`; ChatResponse carries the answer in
    # its message (and str() would prepend the role).
    message = getattr(response, "message", None)
    if message is not None and getattr(message, "content", None):
        text = str(message.content).strip()
    else:
        text = str(getattr(response, "text", None) or "").strip()
    try:
        usage = get_usage_from_response(llm.class_name(), response)
    except Exception:  # noqa: BLE001 - usage is best-effort
        usage = None
    return text, usage


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """The first ``{...}`` object in ``text`` as a dict, or None when there is
    none or it is not an object."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None
