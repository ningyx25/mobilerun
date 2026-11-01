"""Episode runtime for the agent service (phase one: the agent owns the loop).

The episode is the unit the built-in agents already implement end to end
(``MobileAgent.run()``), so nothing in ``mobilerun/agent`` changes: the only
swap is the driver (``RemoteEnvDriver``) that turns device I/O into
environment-service RPCs.

What this module adds on top of the bare workflow:

- a serialized event stream (bounded replay buffer + live subscribers) for
  remote observers,
- the ``declared_done`` gate that decides whether the environment's score is
  trustworthy,
- the per-episode evidence bundle (same layout the benchmark runners write).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import ManagerPlanEvent
from mobilerun.agent.utils.llm_loader import load_agent_llms
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.benchmark.evidence import EpisodeEvidence, gather_ui_tree
from mobilerun.config_manager.config_manager import MobileConfig
from mobilerun.harness.agent_service.events import to_episode_event
from mobilerun.harness.drivers import RemoteEnvDriver
from mobilerun.harness.protocol import (
    EpisodeEvent,
    EpisodeInfo,
    EpisodeRequest,
    EpisodeResult,
    EpisodeState,
    ErrorCode,
    ProtocolError,
    TokenUsage,
    new_id,
)

logger = logging.getLogger("mobilerun")

# Replay buffer per episode. A 50-step episode emits a few hundred events; this
# keeps a late SSE subscriber from re-reading a whole run while staying bounded.
_MAX_BUFFERED_EVENTS = 2000


def declared_done(
    agent: MobileAgent, *, saw_manager_answer: bool, reasoning: bool
) -> bool:
    """Whether the agent itself declared the episode over.

    Mirrors ``benchmark/android_world.py:1163-1167``: reasoning mode ends with
    the manager's final answer, direct mode with the ``complete`` tool having
    set ``shared_state.success``. The environment's score is only trusted when
    this is true, so the two paths must agree.
    """
    if reasoning:
        return saw_manager_answer
    return agent.shared_state.success is not None


class EpisodeRuntime:
    """One running episode and everything observable about it."""

    def __init__(
        self,
        request: EpisodeRequest,
        config: MobileConfig,
        llms: Any,
        *,
        token: Optional[str] = None,
        evidence_root: Optional[Path] = None,
    ) -> None:
        self.id = new_id("ep")
        self.request = request
        self.config = config
        self.llms = llms
        self.token = token
        self.evidence_root = evidence_root
        self.state = EpisodeState.RUNNING
        self.result: Optional[EpisodeResult] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()

        self._task: Optional[asyncio.Task] = None
        self._events: List[EpisodeEvent] = []
        self._seq = 0
        self._subscribers: List[asyncio.Queue] = []
        self._closed = False

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"episode:{self.id}")

    def info(self) -> EpisodeInfo:
        return EpisodeInfo(
            episode_id=self.id,
            state=self.state,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            goal=self.request.goal,
        )

    async def cancel(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        if self.state is EpisodeState.RUNNING:
            self.state = EpisodeState.CANCELLED

    def subscribe(self) -> asyncio.Queue:
        """Live event subscription. Buffered events are replayed first."""
        queue: asyncio.Queue = asyncio.Queue()
        for event in self._events:
            queue.put_nowait(event)
        if self._closed:
            # The episode already finished: replay the buffer and end the
            # stream immediately instead of leaving the subscriber waiting.
            queue.put_nowait(None)
        else:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def wait(self) -> EpisodeInfo:
        if self._task is not None:
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError:
                raise
        return self.info()

    # -- the episode -------------------------------------------------------

    async def _run(self) -> None:
        started = time.monotonic()
        driver: Optional[RemoteEnvDriver] = None
        agent: Optional[MobileAgent] = None
        evidence: Optional[EpisodeEvidence] = None
        ui_tree_before: Optional[Dict[str, Any]] = None
        ui_tree_after: Optional[Dict[str, Any]] = None
        saw_manager_answer = False
        # The terminal state is published only after the finally block below,
        # so "state != running" means the evidence bundle is already on disk.
        final_state = EpisodeState.RUNNING

        try:
            driver = RemoteEnvDriver(
                self.request.environment.base_url,
                self.request.environment.session_id,
                token=self.token,
            )
            await driver.connect()
            evidence = self._build_evidence()

            config = self._episode_config()
            llms = self._episode_llms(request_config=config)

            agent = MobileAgent(
                goal=self.request.goal,
                config=config,
                llms=llms,
                driver=driver,
                timeout=self.request.limits.timeout_s,
                runtype="benchmark",
            )
            if evidence is not None:
                ui_tree_before = await gather_ui_tree(driver)

            handler = agent.run()
            async for event in handler.stream_events():
                self._publish(event)
                if isinstance(event, ManagerPlanEvent) and (event.answer or "").strip():
                    saw_manager_answer = True
                if evidence is not None:
                    evidence.record_event(event)
            run_result = await handler

            is_done = declared_done(
                agent,
                saw_manager_answer=saw_manager_answer,
                reasoning=bool(config.agent.reasoning),
            )
            answer = str(getattr(agent.shared_state, "answer", "") or "").strip()
            self.result = EpisodeResult(
                success=bool(run_result.success),
                declared_done=is_done,
                answer=answer or None,
                steps=int(run_result.steps),
                reason=str(run_result.reason or ""),
                duration_s=time.monotonic() - started,
                usage=(
                    TokenUsage(**self._usage_totals(evidence))
                    if evidence is not None
                    else None
                ),
            )
            final_state = (
                EpisodeState.SUCCEEDED if run_result.success else EpisodeState.FAILED
            )
        except asyncio.CancelledError:
            final_state = EpisodeState.CANCELLED
            raise
        except Exception as exc:  # noqa: BLE001 - reported through the episode
            self.error = f"{type(exc).__name__}: {exc}"
            final_state = EpisodeState.ERROR
            logger.exception("Episode %s failed", self.id)
        finally:
            if evidence is not None:
                if ui_tree_after is None and driver is not None:
                    ui_tree_after = await gather_ui_tree(driver)
                self._finalize_evidence(
                    evidence, agent, ui_tree_before, ui_tree_after, final_state
                )
            if driver is not None:
                # The session itself belongs to the harness; only the
                # transport is ours to close.
                await driver.close()
            # Publish the terminal state last: observers that see a finished
            # episode can rely on the evidence bundle already being written.
            self.state = final_state
            self._close_subscribers()

    # -- config / llms -----------------------------------------------------

    def _episode_config(self) -> MobileConfig:
        config = copy.deepcopy(self.config)
        overrides = self.request.overrides
        if overrides.reasoning is not None:
            config.agent.reasoning = overrides.reasoning
        if overrides.vision_only is not None:
            config.agent.vision_only = overrides.vision_only
        if overrides.max_steps is not None:
            config.agent.max_steps = overrides.max_steps
        if self.request.limits.max_steps is not None:
            config.agent.max_steps = self.request.limits.max_steps
        # The driver is injected by this service, so the config must not try to
        # resolve one (a serial that looks like an env URL would auto-route).
        config.device.control_backend = None
        config.device.serial = None
        return config

    def _episode_llms(self, *, request_config: MobileConfig) -> Any:
        overrides = self.request.overrides
        if overrides.provider and overrides.model:
            kwargs: Dict[str, Any] = {}
            if overrides.temperature is not None:
                kwargs["temperature"] = overrides.temperature
            if overrides.base_url is not None:
                kwargs["base_url"] = overrides.base_url
            if overrides.api_base is not None:
                kwargs["api_base"] = overrides.api_base
            return load_llm(overrides.provider, model=overrides.model, **kwargs)
        if self.llms is not None:
            return self.llms
        return load_agent_llms(request_config)

    # -- evidence ----------------------------------------------------------

    def _build_evidence(self) -> Optional[EpisodeEvidence]:
        if not self.request.evidence.enabled:
            return None
        if self.request.evidence.dir:
            root = Path(self.request.evidence.dir)
        elif self.evidence_root is not None:
            root = self.evidence_root
        else:
            return None
        task_id = str(self.request.metadata.get("task_id") or self.id)
        task_idx = int(self.request.metadata.get("task_idx") or 0)
        try:
            return EpisodeEvidence(root / self.id, task_id, task_idx)
        except Exception as exc:  # noqa: BLE001 - evidence must never fail a run
            logger.warning("Could not create the episode evidence dir: %s", exc)
            return None

    def _finalize_evidence(
        self,
        evidence: EpisodeEvidence,
        agent: Optional[MobileAgent],
        ui_tree_before: Optional[Dict[str, Any]],
        ui_tree_after: Optional[Dict[str, Any]],
        final_state: EpisodeState = EpisodeState.RUNNING,
    ) -> None:
        try:
            # Publish the path before serialising the manifest, so the row the
            # runner reads and the bundle on disk agree.
            if self.result is not None:
                self.result.evidence_dir = str(evidence.dir)
            manifest: Dict[str, Any] = {
                "episode_id": self.id,
                "goal": self.request.goal,
                "environment": self.request.environment.model_dump(),
                "state": final_state.value,
                "error": self.error,
                "result": self.result.model_dump() if self.result else None,
                "request": self.request.model_dump(exclude={"environment"}),
            }
            if agent is not None:
                trajectory = getattr(agent, "trajectory", None)
                if trajectory is not None:
                    manifest["trajectory_dir"] = (
                        evidence.adopt_trajectory(trajectory.trajectory_folder) or ""
                    )
                shared_state = getattr(agent, "shared_state", None)
                if shared_state is not None:
                    manifest["conversation_path"] = evidence.write_conversation(
                        list(getattr(shared_state, "message_history", []) or [])
                    )
            manifest["ui_tree_paths"] = [
                name
                for name in (
                    evidence.write_ui_tree("ui_tree_before", ui_tree_before),
                    evidence.write_ui_tree("ui_tree_after", ui_tree_after),
                )
                if name
            ]
            manifest["timeline_path"] = evidence.write_timeline()
            manifest["usage_totals"] = dict(evidence.usage_totals)
            manifest["evidence_dir"] = str(evidence.dir)
            evidence.write_manifest(manifest)
            if self.result is not None:
                self.result.evidence_dir = str(evidence.dir)
        except Exception as exc:  # noqa: BLE001 - evidence must never fail a run
            logger.warning("Episode evidence finalize failed: %s", exc)

    @staticmethod
    def _usage_totals(evidence: Optional[EpisodeEvidence]) -> Dict[str, int]:
        totals = (evidence.usage_totals if evidence is not None else {}) or {}
        request_tokens = int(totals.get("request_tokens", 0) or 0)
        response_tokens = int(totals.get("response_tokens", 0) or 0)
        return {
            "prompt_tokens": request_tokens,
            "completion_tokens": response_tokens,
            "total_tokens": int(totals.get("total_tokens", 0) or 0),
        }

    # -- event stream ------------------------------------------------------

    def _publish(self, event: Any) -> None:
        self._seq += 1
        try:
            projected = to_episode_event(self._seq, event)
        except Exception:  # noqa: BLE001 - telemetry must never break an episode
            projected = None
        if projected is None:
            return
        self._events.append(projected)
        if len(self._events) > _MAX_BUFFERED_EVENTS:
            del self._events[: len(self._events) - _MAX_BUFFERED_EVENTS]
        for queue in list(self._subscribers):
            queue.put_nowait(projected)

    def _close_subscribers(self) -> None:
        self._closed = True
        for queue in list(self._subscribers):
            queue.put_nowait(None)
        self._subscribers.clear()


class EpisodeManager:
    """Owns the episode registry and the service-wide configuration."""

    def __init__(
        self,
        config: MobileConfig,
        *,
        llms: Any = None,
        token: Optional[str] = None,
        evidence_root: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.llms = llms
        self.token = token
        self.evidence_root = evidence_root
        self._episodes: Dict[str, EpisodeRuntime] = {}

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        # Load the shared LLMs once; per-episode overrides load their own.
        if self.llms is None:
            self.llms = load_agent_llms(self.config)

    async def stop(self) -> None:
        for runtime in list(self._episodes.values()):
            await runtime.cancel()
        self._episodes.clear()

    # -- episodes ----------------------------------------------------------

    def start_episode(self, request: EpisodeRequest) -> EpisodeInfo:
        runtime = EpisodeRuntime(
            request,
            self.config,
            self.llms,
            token=self.token,
            evidence_root=self.evidence_root,
        )
        self._episodes[runtime.id] = runtime
        runtime.start()
        logger.info("episode %s started: %s", runtime.id, request.goal)
        return runtime.info()

    def get(self, episode_id: str) -> EpisodeRuntime:
        runtime = self._episodes.get(episode_id)
        if runtime is None:
            raise ProtocolError(
                404,
                ErrorCode.UNKNOWN_EPISODE,
                f"Unknown episode '{episode_id}'",
            )
        return runtime

    async def cancel(self, episode_id: str) -> None:
        await self.get(episode_id).cancel()

    def prune(self, keep: int = 200) -> None:
        """Drop finished episodes once the registry grows past *keep*."""
        if len(self._episodes) <= keep:
            return
        finished = [
            episode_id
            for episode_id, runtime in self._episodes.items()
            if runtime.state is not EpisodeState.RUNNING
        ]
        for episode_id in finished[: len(self._episodes) - keep]:
            self._episodes.pop(episode_id, None)
