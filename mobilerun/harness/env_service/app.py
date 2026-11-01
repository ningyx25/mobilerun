"""HTTP surface of the environment service.

One process can front several devices (one backend each); a session leases one
backend exclusively for its lifetime. The service owns device state, task
setup and scoring -- the caller only ever sends actions and reads states.

Start it with ``mobilerun serve env``; build it in tests with
``create_env_app(backends)`` and drive it through ``httpx.ASGITransport``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mobilerun.harness.env_service.backend import EnvBackend
from mobilerun.harness.env_service.session import SessionRegistry
from mobilerun.harness.protocol import (
    AnswerRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    DeviceAppsResponse,
    DeviceDateResponse,
    ErrorBody,
    ErrorCode,
    FinishRequest,
    FinishResponse,
    HealthResponse,
    Observation,
    ProtocolError,
    SessionInfo,
    StepRequest,
    StepResponse,
    TaskSpec,
)

logger = logging.getLogger("mobilerun")

SERVICE_TOKEN_ENV = "MOBILERUN_SERVICE_TOKEN"


def create_env_app(
    backends: Sequence[EnvBackend],
    *,
    token: Optional[str] = None,
    session_ttl_s: Optional[float] = None,
) -> FastAPI:
    """Build the environment service around already-constructed backends."""
    token = token if token is not None else os.environ.get(SERVICE_TOKEN_ENV)
    registry = SessionRegistry(
        backends,
        session_ttl_s=session_ttl_s or 2 * 60 * 60.0,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await registry.start()
        try:
            yield
        finally:
            await registry.stop()

    app = FastAPI(
        title="mobilerun environment service",
        version="1.0",
        lifespan=lifespan,
    )
    app.state.registry = registry

    @app.exception_handler(ProtocolError)
    async def _protocol_error(_: Request, exc: ProtocolError) -> JSONResponse:
        body: ErrorBody = exc.body()
        headers = {}
        if exc.code is ErrorCode.CAPACITY:
            headers["Retry-After"] = str(exc.detail.get("retry_after_s", 5))
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(mode="json"),
            headers=headers,
        )

    if token:

        @app.middleware("http")
        async def _auth(request: Request, call_next):
            if request.url.path in ("/v1/health", "/docs", "/openapi.json"):
                return await call_next(request)
            header = request.headers.get("authorization", "")
            expected = f"Bearer {token}"
            if header != expected:
                return JSONResponse(
                    status_code=401,
                    content=ErrorBody(
                        code=ErrorCode.UNAUTHORIZED,
                        message="Missing or invalid bearer token",
                    ).model_dump(mode="json"),
                )
            return await call_next(request)

    # -- service -----------------------------------------------------------

    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return registry.health()

    @app.get("/v1/tasks", response_model=List[TaskSpec])
    async def tasks(backend: Optional[str] = None) -> List[TaskSpec]:
        targets = (
            [registry.require_backend(backend)]
            if backend
            else list(registry.backends().values())
        )
        collected: List[TaskSpec] = []
        for target in targets:
            collected.extend(await target.list_tasks())
        return collected

    @app.post("/v1/backends/{backend_id}/reinitialize")
    async def reinitialize(
        backend_id: str,
        seed: Optional[int] = None,
        n_task_combinations: Optional[int] = None,
    ) -> dict:
        backend = registry.require_backend(backend_id)
        await backend.reinitialize(seed=seed, n_task_combinations=n_task_combinations)
        return {"ok": True}

    # -- sessions ----------------------------------------------------------

    @app.post("/v1/sessions", response_model=CreateSessionResponse)
    async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
        return await registry.create(request)

    @app.get("/v1/sessions/{session_id}", response_model=SessionInfo)
    async def get_session(session_id: str) -> SessionInfo:
        return registry.get(session_id).info()

    @app.post("/v1/sessions/{session_id}/observe", response_model=Observation)
    async def observe(session_id: str, fresh: bool = True) -> Observation:
        return await registry.get(session_id).observe(fresh=fresh)

    @app.post("/v1/sessions/{session_id}/step", response_model=StepResponse)
    async def step(session_id: str, request: StepRequest) -> StepResponse:
        return await registry.get(session_id).step(request)

    @app.post("/v1/sessions/{session_id}/answer")
    async def answer(session_id: str, request: AnswerRequest) -> dict:
        await registry.get(session_id).submit_answer(request.text)
        return {"ok": True}

    @app.post("/v1/sessions/{session_id}/finish", response_model=FinishResponse)
    async def finish(session_id: str, request: FinishRequest) -> FinishResponse:
        return await registry.get(session_id).finish(request)

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        await registry.delete(session_id)
        return {"ok": True}

    @app.get("/v1/sessions/{session_id}/device/date", response_model=DeviceDateResponse)
    async def device_date(session_id: str) -> DeviceDateResponse:
        session = registry.get(session_id)
        return DeviceDateResponse(date=await session.backend.get_date())

    @app.get("/v1/sessions/{session_id}/device/apps", response_model=DeviceAppsResponse)
    async def device_apps(
        session_id: str, include_system: bool = True
    ) -> DeviceAppsResponse:
        session = registry.get(session_id)
        return await session.backend.get_apps(include_system=include_system)

    return app
