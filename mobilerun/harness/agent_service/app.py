"""HTTP surface of the agent service.

Phase one exposes *episodes*: the agent is given a goal plus an open
environment session and runs its own loop against it. The event stream is a
plain SSE endpoint so an orchestrator can watch a run without log scraping.

Phase two's ``/v1/policies/*`` routes -- the strict ``state -> action``
surface -- are declared here but answer ``501``: the wire contract is fixed
now, and implementing it only needs the single-step refactor inside the
agents, not a protocol change.

This is the *worker* app: a deployment serves it (and a dispatcher may
register it by URL). Build it in tests with ``create_agent_app(config)`` and
drive it through ``httpx.ASGITransport``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.harness.agent_service.episode import EpisodeManager
from mobilerun.harness.protocol import (
    PROTOCOL_VERSION,
    ActRequest,
    CreatePolicySessionRequest,
    EpisodeInfo,
    EpisodeRequest,
    ErrorBody,
    ErrorCode,
    ProtocolError,
)

logger = logging.getLogger("mobilerun")

SERVICE_TOKEN_ENV = "MOBILERUN_SERVICE_TOKEN"


def create_agent_app(
    config: MobileConfig,
    *,
    llms: Any = None,
    token: Optional[str] = None,
    evidence_root: Optional[Path] = None,
    name: Optional[str] = None,
) -> FastAPI:
    """Build the agent service around a loaded :class:`MobileConfig`.

    ``name`` is the label this worker is known by (``--name``); it defaults to
    the configured implementation name and is what a dispatcher registers.
    """
    token = token if token is not None else os.environ.get(SERVICE_TOKEN_ENV)
    label = name or config.agent.name
    manager = EpisodeManager(
        config, llms=llms, token=token, evidence_root=evidence_root
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(
        title="mobilerun agent service",
        version=PROTOCOL_VERSION,
        lifespan=lifespan,
    )
    app.state.manager = manager

    @app.exception_handler(ProtocolError)
    async def _protocol_error(_: Request, exc: ProtocolError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.body().model_dump(mode="json"),
        )

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

    @app.get("/v1/health")
    async def health() -> dict:
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            # ``name`` is the registered label, ``agent`` the implementation.
            "name": label,
            "agent": config.agent.name,
            "state": "ready",
            "reasoning": bool(config.agent.reasoning),
            "executor_kind": config.agent.executor.kind,
            "vision_only": bool(config.agent.vision_only),
            "episodes": len(manager._episodes),  # noqa: SLF001 - status only
        }

    # -- episodes ----------------------------------------------------------

    @app.post("/v1/episodes", response_model=EpisodeInfo, status_code=202)
    async def start_episode(request: EpisodeRequest) -> EpisodeInfo:
        return manager.start_episode(request)

    @app.get("/v1/episodes/{episode_id}", response_model=EpisodeInfo)
    async def get_episode(episode_id: str) -> EpisodeInfo:
        runtime = manager.get(episode_id)
        info = runtime.info()
        manager.prune()
        return info

    @app.get("/v1/episodes/{episode_id}/events")
    async def episode_events(episode_id: str) -> StreamingResponse:
        runtime = manager.get(episode_id)

        async def _stream() -> AsyncIterator[str]:
            queue = runtime.subscribe()
            try:
                while True:
                    event = await queue.get()
                    if event is None:  # episode finished
                        break
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                runtime.unsubscribe(queue)

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    @app.post("/v1/episodes/{episode_id}/cancel")
    async def cancel_episode(episode_id: str) -> dict:
        await manager.cancel(episode_id)
        return {"ok": True, "state": manager.get(episode_id).state.value}

    # -- policies (phase two; contract fixed, implementation pending) ------

    @app.post("/v1/policies/{policy}/sessions")
    async def create_policy_session(policy: str, request: CreatePolicySessionRequest):
        return _not_implemented(
            "Policy (single-step state -> action) mode is planned for phase two; "
            "use /v1/episodes for now."
        )

    @app.post("/v1/policies/{policy}/act")
    async def act(policy: str, request: ActRequest):
        return _not_implemented(
            "Policy (single-step state -> action) mode is planned for phase two; "
            "use /v1/episodes for now."
        )

    @app.delete("/v1/policies/{policy}/sessions/{policy_session_id}")
    async def delete_policy_session(policy: str, policy_session_id: str):
        return _not_implemented(
            "Policy (single-step state -> action) mode is planned for phase two."
        )

    return app


def _not_implemented(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content=ErrorBody(code=ErrorCode.NOT_IMPLEMENTED, message=message).model_dump(
            mode="json"
        ),
    )
