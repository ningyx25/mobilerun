"""Short, log-safe descriptions of provider errors."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit

_MAX_ERROR_CHARS = 2000
_MAX_TITLE_CHARS = 100
_TITLE_SCAN_CHARS = 20_000
_HTML_TAG = re.compile(r"<!doctype\s+html|<html[\s>]", re.IGNORECASE)
_PAGE_TAG = re.compile(r"<(?:head|body|title|script|/html)[\s>]", re.IGNORECASE)
_HTML_TITLE = re.compile(r"<title[^>]{0,200}>([^<]{0,500})</title>", re.IGNORECASE)
# Whitespace, escapes, XML prologs and comments before <!DOCTYPE html>/<html>.
_PAGE_PREAMBLE = re.compile(
    r"(?:\s|\ufeff|\\[nrt]|<\?xml[^>]{0,200}\?>|<!--.{0,500}?-->)*$", re.DOTALL
)
_ESCAPES = re.compile(r"\\[nrt]|\ufeff")
_TRUNCATION_MARKER = re.compile(r"\.\.\. \[\d+ more characters\]$")


def http_status_code(error: BaseException) -> int | None:
    def read_attribute(value: object, name: str) -> object | None:
        try:
            return getattr(value, name, None)
        except Exception:
            return None

    def parse_status(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped.isascii() or not stripped.isdecimal():
                return None
            parsed = int(stripped)
        else:
            return None
        return parsed if 100 <= parsed <= 599 else None

    response = read_attribute(error, "response")
    candidates = (
        read_attribute(error, "status_code"),
        read_attribute(response, "status_code"),
        read_attribute(error, "code"),
        read_attribute(error, "status"),
        read_attribute(response, "status"),
    )
    for status_code in candidates:
        parsed = parse_status(status_code)
        if parsed is not None:
            return parsed
    return None


def _request_url(error: BaseException) -> str | None:
    """Return scheme, host and path only: no credentials or query string."""
    try:
        response = getattr(error, "response", None)
        request = getattr(error, "request", None) or getattr(response, "request", None)
        url = getattr(request, "url", None) or getattr(response, "url", None)
        if not url:
            return None
        parts = urlsplit(str(url))
        host = parts.hostname
        if not parts.scheme or not host:
            return None
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        return f"{parts.scheme}://{host}{parts.path}"
    except Exception:
        return None


def _clean_prefix(prefix: str) -> str:
    # Keep the SDK's status text and drop the JSON wrapper around the body.
    prefix = _ESCAPES.sub(" ", prefix).split("{", 1)[0]
    return prefix.strip().rstrip(" -:.,'\"[").strip()


def _summarize_html_page(
    text: str,
    *,
    max_prefix: int,
    allow_prefix: bool = True,
    status: int | None = None,
    url: str | None = None,
) -> str | None:
    """Summarize ``text`` if it is an HTML page, optionally after a short prefix."""
    match = _HTML_TAG.search(text, 0, max_prefix + 1000)
    if match is None:
        return None
    raw_prefix = text[: match.start()]
    preamble = _PAGE_PREAMBLE.search(raw_prefix)
    head = raw_prefix[: preamble.start()] if preamble else raw_prefix
    if head.strip() and (not allow_prefix or len(head) > max_prefix):
        return None
    # Earlier tags or code fences mean the HTML is quoted, not the response.
    if "<" in head or "`" in head:
        return None
    if not _PAGE_TAG.search(text, match.end(), match.end() + _TITLE_SCAN_CHARS):
        return None
    prefix = _clean_prefix(head)
    parts = [prefix] if prefix else []
    page = "HTML error page"
    if status and str(status) not in prefix:
        page = f"HTTP {status} {page}"
    if prefix and "{" not in head and head.rstrip(" '\"").endswith(":"):
        parts[-1] = f"{prefix}:"
    parts.append(page)
    title = _HTML_TITLE.search(text, 0, _TITLE_SCAN_CHARS)
    if title:
        title_text = " ".join(title.group(1).split())[:_MAX_TITLE_CHARS]
        if title_text:
            parts.append(f"'{title_text}'")
    if url:
        parts.append(f"from {url}")
    return " ".join(parts)


def describe_error(error: BaseException, max_chars: int = _MAX_ERROR_CHARS) -> str:
    """Return ``str(error)`` with HTML error pages summarized and length capped."""
    # Timeouts such as httpx.ConnectTimeout often have no message.
    text = str(error) or type(error).__name__
    status = http_status_code(error)
    # Only HTTP errors wrap a response body in an SDK prefix.
    summary = _summarize_html_page(
        text,
        max_prefix=80,
        allow_prefix=status is not None,
        status=status,
        url=_request_url(error),
    )
    if summary is not None:
        text = summary
    if len(text) > max_chars and not _TRUNCATION_MARKER.search(text):
        omitted = len(text) - max_chars
        text = f"{text[:max_chars]}... [{omitted} more characters]"
    return text


# Transient transport/provider failures: the episode made no cognitive
# progress and a retry can succeed, so benchmarks record them as infra
# errors (retried on --resume) instead of agent failures (score 0).
_INFRA_ERROR_MARKERS = (
    "ratelimit",
    "rate limit",
    "model_cooldown",
    "apitimeout",
    "apiconnection",
    "apirequest",
    "connection error",
    "connection reset",
    "connection refused",
    "remoteprotocolerror",
    "server disconnected",
    "request timed out",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "overloaded",
)


def classify_llm_failure(error: BaseException) -> str | None:
    """Return ``"infra"`` for transient LLM/provider failures, else ``None``.

    Covers rate limits (429, cooldowns), transport timeouts and connection
    errors, and retryable HTTP statuses (408/429/5xx). Everything else —
    model protocol errors, tool failures, agent-level exceptions — is the
    agent's own failure and returns ``None``.
    """
    if isinstance(error, TimeoutError):  # incl. asyncio.TimeoutError
        return "infra"
    status = http_status_code(error)
    if status is not None and (status in (408, 429) or status >= 500):
        return "infra"
    name = type(error).__name__.lower()
    if any(marker in name for marker in ("ratelimit", "timeout", "connection")):
        return "infra"
    text = f"{name}: {describe_error(error, max_chars=400)}".lower()
    if any(marker in text for marker in _INFRA_ERROR_MARKERS):
        return "infra"
    return None


class HtmlErrorPageFilter(logging.Filter):
    """Shorten HTML error pages that third-party retry logs include verbatim."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        summary = _summarize_html_page(message, max_prefix=300)
        if summary is not None:
            record.msg = summary
            record.args = None
        return True
