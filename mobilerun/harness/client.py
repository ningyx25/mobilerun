"""HTTP clients for the two harness services.

Both the harness runner and the agent-side :class:`RemoteEnvDriver` speak to
the environment service through :class:`EnvClient`, so there is exactly one
implementation of the wire contract on the client side too.

Retry policy: connection errors and 5xx/429 are retried (the environment's
``step`` endpoint is idempotent by ``step_index``, and every other call is
either a read or itself idempotent), while 4xx contract errors surface
immediately with their structured ``error`` body.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx

from mobilerun.harness.protocol import (
    AnswerRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    DeviceAppsResponse,
    DeviceDateResponse,
    EpisodeInfo,
    EpisodeRequest,
    ErrorBody,
    FinishRequest,
    FinishResponse,
    HealthResponse,
    Observation,
    SessionInfo,
    StepRequest,
    StepResponse,
    TaskSpec,
)

logger = logging.getLogger("mobilerun")

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_S = 1.0
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_DEFAULT_TIMEOUT = 30.0
# Environment calls that wait on the device (state reads, scoring) take longer
# than a plain JSON round trip.
_DEVICE_TIMEOUT = 180.0
_SCORE_TIMEOUT = 300.0


class ServiceError(RuntimeError):
    """A structured error returned by one of the harness services."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"{status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.detail = detail or {}

    @property
    def retryable(self) -> bool:
        return self.status_code in _RETRY_STATUSES


class _BaseServiceClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries
        headers = {"accept": "application/json"}
        if token:
            headers["authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers=headers,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "_BaseServiceClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # -- plumbing ----------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
        retry: bool = True,
    ) -> Any:
        attempts = self.retries if retry else 1
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    timeout=timeout or self.timeout,
                )
            except httpx.TransportError as exc:
                last_error = exc
                if attempt >= attempts:
                    raise ConnectionError(
                        f"{method} {self.base_url}{path} failed: {exc}"
                    ) from exc
                await asyncio.sleep(DEFAULT_BACKOFF_S * attempt)
                continue

            if response.status_code < 400:
                return response.json() if response.content else {}

            if response.status_code in _RETRY_STATUSES and attempt < attempts:
                retry_after = response.headers.get("retry-after")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else DEFAULT_BACKOFF_S * attempt
                )
                logger.debug(
                    "%s %s got %d; retry %d/%d in %.1fs",
                    method,
                    path,
                    response.status_code,
                    attempt,
                    self.retries - 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

            raise _service_error(response)

        raise ConnectionError(  # pragma: no cover - loop always returns or raises
            f"{method} {self.base_url}{path} failed: {last_error}"
        )


def _service_error(response: httpx.Response) -> ServiceError:
    try:
        body = ErrorBody.model_validate(response.json())
        return ServiceError(
            response.status_code, body.code.value, body.message, body.detail
        )
    except Exception:  # noqa: BLE001 - fall back to the raw body
        return ServiceError(
            response.status_code,
            "http_error",
            response.text[:400] or response.reason_phrase,
        )


class EnvClient(_BaseServiceClient):
    """Client for the environment service (``/v1/sessions/*``)."""

    # -- health / tasks ----------------------------------------------------

    async def health(self) -> HealthResponse:
        return HealthResponse.model_validate(await self._request("GET", "/v1/health"))

    async def list_tasks(self, backend: Optional[str] = None) -> List[TaskSpec]:
        params = {"backend": backend} if backend else None
        data = await self._request("GET", "/v1/tasks", params=params)
        return [TaskSpec.model_validate(item) for item in data]

    async def reinitialize(
        self,
        backend: str,
        *,
        seed: Optional[int] = None,
        n_task_combinations: Optional[int] = None,
    ) -> None:
        params: Dict[str, Any] = {}
        if seed is not None:
            params["seed"] = seed
        if n_task_combinations is not None:
            params["n_task_combinations"] = n_task_combinations
        await self._request(
            "POST",
            f"/v1/backends/{backend}/reinitialize",
            params=params or None,
            timeout=_SCORE_TIMEOUT,
        )

    # -- sessions ----------------------------------------------------------

    async def create_session(
        self,
        task_id: str,
        *,
        backend: Optional[str] = None,
        task_idx: int = 0,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = _DEVICE_TIMEOUT * 3,
    ) -> CreateSessionResponse:
        request = CreateSessionRequest(
            backend=backend,
            task_id=task_id,
            task_idx=task_idx,
            seed=seed,
            params=params or {},
        )
        return CreateSessionResponse.model_validate(
            await self._request(
                "POST",
                "/v1/sessions",
                json_body=request.model_dump(mode="json"),
                timeout=timeout,
            )
        )

    async def get_session(self, session_id: str) -> SessionInfo:
        return SessionInfo.model_validate(
            await self._request("GET", f"/v1/sessions/{session_id}")
        )

    async def observe(self, session_id: str, *, fresh: bool = True) -> Observation:
        return Observation.model_validate(
            await self._request(
                "POST",
                f"/v1/sessions/{session_id}/observe",
                params={"fresh": fresh},
                timeout=_DEVICE_TIMEOUT,
            )
        )

    async def step(
        self,
        session_id: str,
        request: StepRequest,
        *,
        timeout: float = _DEVICE_TIMEOUT,
    ) -> StepResponse:
        return StepResponse.model_validate(
            await self._request(
                "POST",
                f"/v1/sessions/{session_id}/step",
                json_body=request.model_dump(mode="json"),
                timeout=timeout,
            )
        )

    async def answer(self, session_id: str, text: str) -> None:
        await self._request(
            "POST",
            f"/v1/sessions/{session_id}/answer",
            json_body=AnswerRequest(text=text).model_dump(mode="json"),
            timeout=_DEVICE_TIMEOUT,
        )

    async def finish(
        self,
        session_id: str,
        *,
        declared_done: bool,
        answer: Optional[str] = None,
        settle_s: Optional[float] = None,
    ) -> FinishResponse:
        request = FinishRequest(
            declared_done=declared_done, answer=answer, settle_s=settle_s
        )
        return FinishResponse.model_validate(
            await self._request(
                "POST",
                f"/v1/sessions/{session_id}/finish",
                json_body=request.model_dump(mode="json"),
                timeout=_SCORE_TIMEOUT,
            )
        )

    async def delete_session(self, session_id: str) -> None:
        await self._request(
            "DELETE", f"/v1/sessions/{session_id}", timeout=_SCORE_TIMEOUT
        )

    async def device_date(self, session_id: str) -> str:
        data = await self._request("GET", f"/v1/sessions/{session_id}/device/date")
        return DeviceDateResponse.model_validate(data).date

    async def device_apps(
        self, session_id: str, *, include_system: bool = True
    ) -> DeviceAppsResponse:
        data = await self._request(
            "GET",
            f"/v1/sessions/{session_id}/device/apps",
            params={"include_system": include_system},
            timeout=_DEVICE_TIMEOUT,
        )
        return DeviceAppsResponse.model_validate(data)


class AgentClient(_BaseServiceClient):
    """Client for the agent service (``/v1/episodes/*``)."""

    async def health(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return await self._request("GET", "/v1/health", timeout=timeout, retry=False)

    async def start_episode(self, request: EpisodeRequest) -> EpisodeInfo:
        return EpisodeInfo.model_validate(
            await self._request(
                "POST",
                "/v1/episodes",
                json_body=request.model_dump(mode="json"),
                timeout=_DEVICE_TIMEOUT,
                # Creating an episode is not idempotent: retrying a 5xx would
                # start a second one. The caller retries at task granularity.
                retry=False,
            )
        )

    async def get_episode(self, episode_id: str) -> EpisodeInfo:
        return EpisodeInfo.model_validate(
            await self._request("GET", f"/v1/episodes/{episode_id}")
        )

    async def cancel_episode(self, episode_id: str) -> None:
        await self._request("POST", f"/v1/episodes/{episode_id}/cancel")

    async def wait_episode(
        self,
        episode_id: str,
        *,
        timeout_s: float,
        poll_s: float = 1.0,
        on_update: Optional[Callable[[EpisodeInfo], None]] = None,
    ) -> EpisodeInfo:
        """Poll until the episode leaves the running state (or times out)."""
        deadline = time.monotonic() + timeout_s
        while True:
            info = await self.get_episode(episode_id)
            if on_update is not None:
                try:
                    on_update(info)
                except Exception:  # noqa: BLE001 - progress must not break polling
                    logger.debug("wait_episode progress callback failed")
            if info.state.value != "running":
                return info
            if time.monotonic() >= deadline:
                return info
            await asyncio.sleep(poll_s)

    async def stream_events(self, episode_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Yield the episode's SSE frames as dicts (best-effort)."""
        async with self._client.stream(
            "GET", f"/v1/episodes/{episode_id}/events", timeout=None
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise _service_error(response)
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON SSE frame: %r", payload)
