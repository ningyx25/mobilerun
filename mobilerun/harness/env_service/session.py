"""Session lifecycle: one open episode on one backend.

The environment service is stateful on purpose. AndroidWorld's contract is
``initialize -> score -> tear_down`` over live device state, so a "stateless
step function" cannot exist for it; what can be stateless is the *wire* format
(the observation/action pair), while the session object holds the device
lease, the step counter and the cached step results that make retries safe.

State machine::

    RUNNING --finish--> CLOSED
       |  \\--backend error--> ERROR
       \\--TTL----> CLOSED (scored as not-done)
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from mobilerun.harness.env_service.backend import EnvBackend
from mobilerun.harness.env_service.scoring import finish_episode
from mobilerun.harness.protocol import (
    BackendInfo,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorCode,
    FinishRequest,
    FinishResponse,
    HealthResponse,
    Observation,
    ProtocolError,
    SessionInfo,
    SessionState,
    StepRequest,
    StepResponse,
    TaskSpec,
    new_id,
)

logger = logging.getLogger("mobilerun")

# How many step results to keep for replay protection. An episode budget is
# tens of steps; 128 is comfortably above any single-session replay window.
_MAX_CACHED_STEPS = 128

_DEFAULT_SESSION_TTL_S = 2 * 60 * 60.0


@dataclass
class Session:
    """One open episode bound to one backend."""

    id: str
    backend: EnvBackend
    task: TaskSpec
    observation: Observation
    state: SessionState = SessionState.RUNNING
    steps: int = 0
    # Set when the agent sent a terminal ``complete`` action; the harness may
    # still call /finish without repeating it.
    declared_done: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    finish_result: Optional[FinishResponse] = None
    torn_down: bool = False
    _step_results: "OrderedDict[int, StepResponse]" = field(default_factory=OrderedDict)

    # -- observation -------------------------------------------------------

    async def observe(self, *, fresh: bool = True) -> Observation:
        async with self.lock:
            self._require_running()
            if not fresh:
                return self.observation
            self.observation = await self.backend.observe(seq=self.observation.seq + 1)
            return self.observation

    # -- actions -----------------------------------------------------------

    async def step(self, request: StepRequest) -> StepResponse:
        async with self.lock:
            cached = self._step_results.get(request.step_index)
            if cached is not None:
                # A retry of an already-applied step: never touch the device.
                return cached

            self._require_running()
            if (
                request.expect_seq is not None
                and request.expect_seq != self.observation.seq
            ):
                raise ProtocolError(
                    409,
                    ErrorCode.STALE_OBSERVATION,
                    "Action was decided on a stale observation",
                    {
                        "expected_seq": self.observation.seq,
                        "got_seq": request.expect_seq,
                        "observation": self.observation.model_dump(mode="json"),
                    },
                )
            if request.step_index != self.steps + 1:
                raise ProtocolError(
                    409,
                    ErrorCode.SESSION_STATE,
                    f"step_index out of order: expected {self.steps + 1}, "
                    f"got {request.step_index}",
                )

            action = request.action
            ok = True
            summary = ""
            error: Optional[str] = None
            # A rejected action is feedback for the agent, not a failed
            # episode: capture the message and keep the session running.
            try:
                summary = await self.backend.execute(
                    action, screen=self.observation.screen
                )
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                ok = False
                error = f"{type(exc).__name__}: {exc}"
                summary = f"Failed to execute {action.op}: {error}"
                logger.warning(
                    "Action %s failed on session %s: %s", action.op, self.id, error
                )

            self.steps = request.step_index
            if action.is_terminal:
                self.declared_done = bool(action.args.get("success", True))

            response = StepResponse(
                step_index=request.step_index,
                seq=self.observation.seq,
                ok=ok,
                summary=summary,
                error=error,
            )
            if request.include_observation and not action.is_terminal:
                self.observation = await self.backend.observe(
                    seq=self.observation.seq + 1
                )
                response.observation = self.observation

            self._remember(response)
            return response

    async def submit_answer(self, text: str) -> None:
        async with self.lock:
            self._require_running()
            await self.backend.submit_answer(text)

    # -- scoring -----------------------------------------------------------

    async def finish(self, request: FinishRequest) -> FinishResponse:
        async with self.lock:
            if self.finish_result is not None:
                return self.finish_result
            self._require_running()
            self.state = SessionState.FINISHING
            try:
                result = await finish_episode(
                    self.backend,
                    declared_done=bool(request.declared_done or self.declared_done),
                    answer=request.answer,
                    settle_s=request.settle_s,
                )
            except Exception as exc:
                self.state = SessionState.ERROR
                raise ProtocolError(
                    502,
                    ErrorCode.BACKEND_UNAVAILABLE,
                    f"Scoring failed: {type(exc).__name__}: {exc}",
                ) from exc
            self.finish_result = result
            self.state = SessionState.CLOSED
            return result

    # -- teardown ----------------------------------------------------------

    async def close(self) -> None:
        """Tear the backend task down. Idempotent.

        An episode that was never finished is *not* scored here: the score
        would be meaningless without a declared-done gate, and teardown is a
        cleanup step, not an evaluation one.
        """
        async with self.lock:
            if self.torn_down:
                return
            try:
                await self.backend.teardown(self.task)
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning("teardown for session %s failed: %s", self.id, exc)
            self.torn_down = True
            self.state = SessionState.CLOSED

    # -- helpers -----------------------------------------------------------

    def info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.id,
            backend=self.backend.id,
            state=self.state,
            task=self.task,
            capabilities=self.backend.capabilities(),
            screen=self.observation.screen,
            created_at=self.created_at,
            seq=self.observation.seq,
            steps=self.steps,
        )

    def _remember(self, response: StepResponse) -> None:
        self._step_results[response.step_index] = response
        while len(self._step_results) > _MAX_CACHED_STEPS:
            self._step_results.popitem(last=False)

    def _require_running(self) -> None:
        if self.state is not SessionState.RUNNING:
            raise ProtocolError(
                409,
                ErrorCode.SESSION_STATE,
                f"Session {self.id} is {self.state.value}, expected running",
            )


class SessionRegistry:
    """Owns the backends and hands each open session an exclusive lease."""

    def __init__(
        self,
        backends: Iterable[EnvBackend],
        *,
        session_ttl_s: float = _DEFAULT_SESSION_TTL_S,
    ) -> None:
        self._backends: Dict[str, EnvBackend] = {b.id: b for b in backends}
        self._leases: Dict[str, asyncio.Lock] = {
            backend_id: asyncio.Lock() for backend_id in self._backends
        }
        self._sessions: Dict[str, Session] = {}
        self._session_ttl_s = session_ttl_s
        self._sweeper: Optional[asyncio.Task] = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        for backend in self._backends.values():
            await backend.connect()
        self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
        for session in list(self._sessions.values()):
            try:
                await session.close()
            except Exception as exc:  # noqa: BLE001 - shutdown path
                logger.warning(
                    "Closing session %s at shutdown failed: %s", session.id, exc
                )
        self._sessions.clear()
        for backend in self._backends.values():
            try:
                await backend.close()
            except Exception as exc:  # noqa: BLE001 - shutdown path
                logger.warning("Closing backend %s failed: %s", backend.id, exc)

    # -- introspection -----------------------------------------------------

    def health(self) -> HealthResponse:
        return HealthResponse(ok=True, backends=self.backend_infos())

    def backend_infos(self) -> List[BackendInfo]:
        infos: List[BackendInfo] = []
        for backend_id, backend in self._backends.items():
            busy = self._leases[backend_id].locked()
            infos.append(
                BackendInfo(
                    id=backend_id,
                    kind=backend.kind,
                    url=getattr(backend, "url", None),
                    state="busy" if busy else "ready",
                    capabilities=backend.capabilities(),
                )
            )
        return infos

    def require_backend(self, backend_id: str) -> EnvBackend:
        backend = self._backends.get(backend_id)
        if backend is None:
            raise ProtocolError(
                404,
                ErrorCode.UNKNOWN_BACKEND,
                f"Unknown backend '{backend_id}'",
                {"available": sorted(self._backends)},
            )
        return backend

    def backends(self) -> Dict[str, EnvBackend]:
        return dict(self._backends)

    def sessions(self) -> List[Session]:
        return list(self._sessions.values())

    # -- session lifecycle -------------------------------------------------

    async def create(self, request: CreateSessionRequest) -> CreateSessionResponse:
        backend_id = request.backend or self._pick_backend()
        backend = self.require_backend(backend_id)
        lease = self._leases[backend_id]
        if lease.locked():
            raise ProtocolError(
                503,
                ErrorCode.CAPACITY,
                f"Backend '{backend_id}' is busy",
                {"retry_after_s": 5},
            )
        # asyncio.Lock has no owner; the lease is released by delete()/TTL.
        await lease.acquire()
        try:
            task = await backend.prepare(
                request.task_id,
                task_idx=request.task_idx,
                seed=request.seed,
                params=request.params,
            )
            observation = await backend.observe(seq=1)
        except ProtocolError:
            lease.release()
            raise
        except Exception as exc:
            lease.release()
            if isinstance(exc, (ConnectionError, OSError)):
                raise ProtocolError(
                    503,
                    ErrorCode.BACKEND_UNAVAILABLE,
                    f"Backend '{backend_id}' is unreachable: {exc}",
                ) from exc
            raise

        session = Session(
            id=new_id("s"), backend=backend, task=task, observation=observation
        )
        self._sessions[session.id] = session
        logger.info("session %s opened on %s for %s", session.id, backend_id, task.id)
        return CreateSessionResponse(
            **session.info().model_dump(),
            observation=observation,
        )

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise ProtocolError(
                404,
                ErrorCode.UNKNOWN_SESSION,
                f"Unknown session '{session_id}'",
            )
        return session

    async def delete(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            # DELETE is idempotent: a client that lost the response must be
            # able to retry without tripping over itself.
            return
        try:
            await session.close()
        finally:
            self._release(session)
            logger.info("session %s closed", session_id)

    async def expire_stale(self, now: Optional[float] = None) -> int:
        expired = 0
        for session in list(self._sessions.values()):
            age = _age_seconds(session.created_at, now)
            if age < self._session_ttl_s:
                continue
            logger.warning(
                "session %s exceeded its TTL (%.0fs); tearing down", session.id, age
            )
            await self.delete(session.id)
            expired += 1
        return expired

    # -- internals ---------------------------------------------------------

    def _release(self, session: Session) -> None:
        lease = self._leases.get(session.backend.id)
        if lease is not None and lease.locked():
            lease.release()

    def _pick_backend(self) -> str:
        for backend_id, lease in self._leases.items():
            if not lease.locked():
                return backend_id
        raise ProtocolError(
            503,
            ErrorCode.CAPACITY,
            "All environment backends are busy",
            {"retry_after_s": 5},
        )

    async def _sweep_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(60.0)
                try:
                    await self.expire_stale()
                except Exception as exc:  # noqa: BLE001 - sweeper must survive
                    logger.warning("Session TTL sweep failed: %s", exc)
        except asyncio.CancelledError:
            raise


def _age_seconds(created_at: str, now: Optional[float] = None) -> float:
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    reference = datetime.now(timezone.utc)
    return (reference - created).total_seconds()
