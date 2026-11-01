"""Agent dispatcher: one public agent service in front of several workers.

The dispatcher speaks exactly the same ``/v1/episodes`` protocol as a worker,
so ``mobilerun eval`` cannot tell the difference. What it adds is selection:

- ``POST /v1/episodes`` accepts an optional ``agent_id``; without one it picks
  the least-loaded reachable worker.
- Every episode it starts is returned under a composite id
  ``"<agent_id>:<upstream_id>"``. Polling, the SSE stream and cancellation are
  routed back by that prefix, so the dispatcher keeps no per-episode table and
  can restart without orphaning anything.

Configuration is the ``{agent_id}={agent_name}@{server_url}`` triple, parsed by
``registry.parse_entry``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mobilerun.harness.agent_service.registry import (
    AgentRegistry,
    AgentSlot,
    composite_id,
)
from mobilerun.harness.client import ServiceError
from mobilerun.harness.protocol import (
    PROTOCOL_VERSION,
    AgentsHealth,
    EpisodeInfo,
    EpisodeRequest,
    ErrorBody,
    ErrorCode,
    ProtocolError,
)

logger = logging.getLogger("mobilerun")

SERVICE_TOKEN_ENV = "MOBILERUN_SERVICE_TOKEN"

# Finished episodes whose slot counter was already released. Bounded so a very
# long-running dispatcher cannot grow without limit; an evicted id can only be
# decremented twice if the same episode is polled again after 4096 newer ones.
_FINISHED_CACHE = 4096


def create_agent_dispatcher(
    entries: Sequence[str],
    *,
    token: Optional[str] = None,
    probe_interval: float = 5.0,
    probe_timeout: float = 2.0,
    max_per_agent: int = 4,
    client_factory=None,
) -> FastAPI:
    """Build the dispatcher around ``{agent_id}={agent_name}@{server_url}``."""
    token = token if token is not None else os.environ.get(SERVICE_TOKEN_ENV)
    registry = AgentRegistry(
        entries,
        token=token,
        probe_interval=probe_interval,
        probe_timeout=probe_timeout,
        max_per_agent=max_per_agent,
        client_factory=client_factory,
    )
    finished: "OrderedDict[str, None]" = OrderedDict()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await registry.probe(force=True)
        try:
            yield
        finally:
            await registry.close()

    app = FastAPI(
        title="mobilerun agent dispatcher",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    app.state.registry = registry

    @app.exception_handler(ProtocolError)
    async def _protocol_error(_: Request, exc: ProtocolError) -> JSONResponse:
        headers = {}
        if exc.code in (ErrorCode.CAPACITY, ErrorCode.BACKEND_UNAVAILABLE):
            headers["Retry-After"] = str(exc.detail.get("retry_after_s", 5))
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.body().model_dump(mode="json"),
            headers=headers,
        )

    @app.exception_handler(_UpstreamError)
    async def _upstream_error_handler(
        _: Request, exc: "_UpstreamError"
    ) -> JSONResponse:
        # A worker's own error is passed through verbatim: its code string may
        # be newer than this dispatcher's enum, and rewriting it would hide
        # which side actually failed.
        return JSONResponse(status_code=exc.status_code, content=exc.body)

    if token:

        @app.middleware("http")
        async def _auth(request: Request, call_next):
            if request.url.path in ("/v1/health", "/docs", "/openapi.json"):
                return await call_next(request)
            if request.headers.get("authorization", "") != f"Bearer {token}":
                return JSONResponse(
                    status_code=401,
                    content=ErrorBody(
                        code=ErrorCode.UNAUTHORIZED,
                        message="Missing or invalid bearer token",
                    ).model_dump(mode="json"),
                )
            return await call_next(request)

    # -- service -----------------------------------------------------------

    @app.get("/v1/health", response_model=AgentsHealth)
    async def health() -> AgentsHealth:
        await registry.probe()
        return registry.health()

    # -- episodes ----------------------------------------------------------

    @app.post("/v1/episodes", response_model=EpisodeInfo, status_code=202)
    async def start_episode(request: EpisodeRequest) -> EpisodeInfo:
        await registry.probe()
        slot = registry.select(request.agent_id)
        try:
            info = await slot.client.start_episode(request)
        except ServiceError as exc:
            raise _UpstreamError(slot.id, exc) from exc
        except ConnectionError as exc:
            slot.state = "down"
            raise _unreachable(slot, exc) from exc

        registry.note_started(slot)
        return _with_composite(slot, info)

    @app.get("/v1/episodes/{episode_id}", response_model=EpisodeInfo)
    async def get_episode(episode_id: str) -> EpisodeInfo:
        slot, upstream_id = registry.split(episode_id)
        try:
            info = await slot.client.get_episode(upstream_id)
        except ServiceError as exc:
            raise _UpstreamError(slot.id, exc) from exc
        except ConnectionError as exc:
            slot.state = "down"
            raise _unreachable(slot, exc) from exc
        _release_if_finished(slot, episode_id, info)
        return _with_composite(slot, info)

    @app.get("/v1/episodes/{episode_id}/events")
    async def episode_events(episode_id: str) -> StreamingResponse:
        slot, upstream_id = registry.split(episode_id)

        async def _stream() -> AsyncIterator[str]:
            async for frame in slot.client.stream_events(upstream_id):
                yield f"data: {_json(frame)}\n\n"

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.post("/v1/episodes/{episode_id}/cancel")
    async def cancel_episode(episode_id: str) -> dict:
        slot, upstream_id = registry.split(episode_id)
        try:
            await slot.client.cancel_episode(upstream_id)
        except ServiceError as exc:
            raise _UpstreamError(slot.id, exc) from exc
        except ConnectionError as exc:
            slot.state = "down"
            raise _unreachable(slot, exc) from exc
        _release(slot, episode_id)
        return {"ok": True, "agent_id": slot.id, "episode_id": upstream_id}

    # -- policies (phase two; the contract is frozen, nothing to route yet) --

    @app.post("/v1/policies/{agent_id}/{rest:path}")
    async def policies(agent_id: str, rest: str):
        registry.require(agent_id)
        return JSONResponse(
            status_code=501,
            content=ErrorBody(
                code=ErrorCode.NOT_IMPLEMENTED,
                message=(
                    "Policy (single-step state -> action) mode is planned for "
                    "phase two; use /v1/episodes for now."
                ),
            ).model_dump(mode="json"),
        )

    # -- helpers -----------------------------------------------------------

    def _with_composite(slot: AgentSlot, info: EpisodeInfo) -> EpisodeInfo:
        return info.model_copy(
            update={
                "episode_id": composite_id(slot, info.episode_id),
                "agent_id": slot.id,
            }
        )

    def _release_if_finished(
        slot: AgentSlot, composite: str, info: EpisodeInfo
    ) -> None:
        if info.state.value != "running":
            _release(slot, composite)

    def _release(slot: AgentSlot, composite: str) -> None:
        if composite in finished:
            return
        finished[composite] = None
        while len(finished) > _FINISHED_CACHE:
            finished.popitem(last=False)
        registry.note_finished(slot)

    return app


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _unreachable(slot: AgentSlot, exc: Exception) -> ProtocolError:
    return ProtocolError(
        503,
        ErrorCode.BACKEND_UNAVAILABLE,
        f"Agent '{slot.id}' at {slot.url} is unreachable: {exc}",
        {"retry_after_s": 5, "agent_id": slot.id, "url": slot.url},
    )


class _UpstreamError(Exception):
    """A worker's error response, preserved verbatim."""

    def __init__(self, agent_id: str, exc: ServiceError) -> None:
        super().__init__(f"{agent_id}: {exc}")
        self.status_code = exc.status_code
        self.body = {
            "code": exc.code,
            "message": exc.message,
            "detail": {**exc.detail, "agent_id": agent_id},
        }


__all__ = ["create_agent_dispatcher"]
