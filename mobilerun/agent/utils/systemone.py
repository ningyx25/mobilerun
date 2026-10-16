"""Thin client for the TypeSafe ``/v1/systemone`` decision-model protocol.

The local decision stack (``kev.serve`` behind ``scripts/kev_gateway.py``, one
gateway address routing by the request's ``model`` field) answers
``POST /v1/systemone`` with typed decisions: the request carries a ``state``,
typed ``questions`` (``noul`` / ``choice`` / ``score``) and optional base64
``images``; the response carries one answer per question, a ``usage`` block and
``latency_ms``. This is a single-prefill decision API, not a chat completion,
so it does not fit the llama-index LLM plumbing -- this module is the whole
transport.

Reference callers: ``kev/scripts/systemone_api_example.py`` (field mapping) and
``dohnuts/src/dohnuts/mobile_jev_prompt.py`` (request construction).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import weakref
from typing import Any, Dict, List, Optional, Sequence

import httpx

logger = logging.getLogger("mobilerun")

# One client (connection pool) per running event loop, reused across requests:
# a fresh AsyncClient per call paid a full TCP handshake every step. The client
# is bound to its loop (and timeout), so a new loop -- each `asyncio.run`, or a
# test -- gets its own; the WeakKeyDictionary drops it when the loop is gone.
_clients: "weakref.WeakKeyDictionary[Any, Dict[float, httpx.AsyncClient]]" = (
    weakref.WeakKeyDictionary()
)


def _client(timeout: float) -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    per_loop = _clients.setdefault(loop, {})
    client = per_loop.get(timeout)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=timeout)
        per_loop[timeout] = client
    return client


class SystemOneError(RuntimeError):
    """A ``/v1/systemone`` request failed (transport or an HTTP error status)."""


async def systemone_post(
    base_url: str,
    body: Dict[str, Any],
    *,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """POST one request to ``{base_url}/v1/systemone`` and return the JSON body.

    An HTTP error status raises :class:`SystemOneError` carrying the server's
    ``detail`` (a 422 explains the refusal); a transport failure raises the
    same type with the underlying message. The caller decides whether a failed
    decision aborts the step or becomes feedback.
    """
    url = f"{base_url.rstrip('/')}/v1/systemone"
    try:
        response = await _client(timeout).post(
            url, json=body, headers={"content-type": "application/json"}
        )
    except httpx.HTTPError as exc:
        raise SystemOneError(f"{url} unreachable: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text
        try:
            payload = response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail", detail)
        except ValueError:
            pass
        raise SystemOneError(f"HTTP {response.status_code}: {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise SystemOneError(f"{url} returned a non-JSON body: {exc}") from exc


def encode_images(
    frames: Sequence[bytes], max_pixels: int | Sequence[int] = 0
) -> List[str]:
    """Base64-encode raw PNG/JPEG bytes for a request's ``images`` list.

    The wire format takes raw base64 (a ``data:`` URI also parses, but plain
    base64 is what the reference client sends). ``None``/empty frames are
    dropped so a missing screenshot never becomes an empty string on the wire.

    ``max_pixels`` (one cap or one per frame, 0 = no cap) downscales a frame
    whose pixel count exceeds it -- aspect ratio kept, LANCZOS -- the same
    client-side cap the training/eval path applies. The serving path does no
    resizing, and the server's sequence budget cannot hold several
    full-resolution frames.
    """
    caps = list(max_pixels) if isinstance(max_pixels, (list, tuple)) else [max_pixels]
    if len(caps) == 1:
        caps = caps * len(frames)
    elif len(caps) < len(frames):
        caps = caps + [caps[-1] if caps else 0] * (len(frames) - len(caps))
    elif len(caps) > len(frames):
        caps = caps[: len(frames)]
    out = []
    for frame, cap in zip(frames, caps, strict=True):
        if not frame:
            continue
        frame = _downscale_png(bytes(frame), int(cap or 0))
        out.append(base64.b64encode(frame).decode("ascii"))
    return out


def _downscale_png(data: bytes, max_pixels: int) -> bytes:
    """Downscale PNG/JPEG bytes to ``max_pixels`` pixels (0 = passthrough)."""
    if max_pixels <= 0:
        return data
    try:
        import io

        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hard dependency elsewhere
        return data
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height <= max_pixels:
            return data
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        resized = image.convert("RGB").resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:  # noqa: BLE001 - an undecodable frame is the caller's problem
        return data


def choice_of(answer: Optional[Dict[str, Any]]) -> Optional[str]:
    """The ``choice`` of a ``choice`` answer, or None when unusable."""
    if not isinstance(answer, dict):
        return None
    choice = answer.get("choice")
    return choice if isinstance(choice, str) and choice else None
