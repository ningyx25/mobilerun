"""AndroidWorld benchmark runner.

Drives an AndroidWorld docker environment server (``android_server.py``) over
its task API and runs the mobilerun agent on each of the 116 ``android_world``
tasks, mirroring the canonical episode flow of the reference docker eval
runner (``android_world/run_on_docker.py``)::

    initialize -> goal -> complexity -> start_on_home_screen -> reset
    -> agent episode (budget _allocate_step_budget(complexity) steps) -> score -> tear_down

Scores are computed server-side from the live device state at ``/task/score``
time and are recorded both raw and "gated" the AndroidWorld way (0 unless the
agent declared the episode finished via the ``complete`` tool). ``/close`` is
deliberately never called: it kills the environment without stopping the
emulator, leaving the container unrecoverable.

``task_metadata.json`` is a copy of the android_world repo's task metadata
(same task names as the server suite); it supplies difficulty/tags for
selection and reporting. Keep it in sync when android_world changes its task
list.

Shared result/task/evidence plumbing lives in ``mobilerun.benchmark.common``;
this module keeps only AndroidWorld-specific client and episode logic.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
from rich.table import Table

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import ManagerPlanEvent
from mobilerun.agent.external import list_agents
from mobilerun.agent.utils.llm_loader import load_agent_llms
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.benchmark.common import (  # backward-compatible re-exports
    TaskResult as TaskResult,
    _FILE_LEVEL_META_KEYS as _FILE_LEVEL_META_KEYS,
    _RESULTS_LOCK_TIMEOUT_S as _RESULTS_LOCK_TIMEOUT_S,
    _RETRY_ATTEMPTS as _RETRY_ATTEMPTS,
    _RETRY_BACKOFF_S as _RETRY_BACKOFF_S,
    _RETRY_STATUSES as _RETRY_STATUSES,
    _SCORE_TIMEOUT as _SCORE_TIMEOUT,
    _aggregate_times as _aggregate_times,
    _build_summary as _build_summary,
    _finalize_episode_evidence as _finalize_episode_evidence,
    _infra_error_reason as _infra_error_reason,
    _load_results as _load_results,
    _load_run_history as _load_run_history,
    _load_run_meta as _load_run_meta,
    _merge_runs as _merge_runs,
    _new_run_id as _new_run_id,
    _parse_task_index_range as _parse_task_index_range,
    _print_summary as _print_summary,
    _read_results_payload as _read_results_payload,
    _run_record as _run_record,
    _runs_from_payload as _runs_from_payload,
    _sanitize_name as _sanitize_name,
    _select_tasks as _select_tasks,
    _setup_benchmark_logging as _setup_benchmark_logging,
    _write_json as _write_json,
    _write_results as _write_results,
    console as console,
    logger as logger,
)
from mobilerun.benchmark.evidence import EpisodeEvidence, gather_ui_tree
from mobilerun.harness.env_service.backend import _allocate_step_budget
from mobilerun.config_manager import ConfigLoader
from mobilerun.tools.driver.android_world import (
    ANDROID_WORLD_CONNECTION,
    ANDROID_WORLD_DEFAULT_URL,
    AndroidWorldHttpDriver,
    validate_android_world_url,
)

_METADATA_PATH = Path(__file__).with_name("task_metadata.json")

# Timeouts mirror android_world's env/interface.py client: task setup may
# install apps / download data, reset may rebuild the whole controller.
_DEFAULT_TIMEOUT = 60.0
_INITIALIZE_TIMEOUT = 600.0
_RESET_TIMEOUT = 300.0
_REINITIALIZE_TIMEOUT = 300.0
# /state with wait_to_stabilize runs the server's internal stability loop
# (up to ~6s); the reference client budgets 120s for it.
_STABILIZE_TIMEOUT = 120.0
# Second-chance score read for a completed-but-scored-0 episode, after the
# final screen has had another beat to settle.
_SCORE_RECHECK_DELAY_S = 1.0

_BENCHMARK_NAME = "androidworld"


def load_task_metadata() -> Dict[str, Dict[str, Any]]:
    """Load the bundled task metadata keyed by task name."""
    with open(_METADATA_PATH, encoding="utf-8") as handle:
        entries = json.load(handle)
    return {entry["task_name"]: entry for entry in entries}


class AndroidWorldTaskClient:
    """HTTP client for the environment server's /suite/* and /task/* API."""

    def __init__(
        self,
        base_url: str = ANDROID_WORLD_DEFAULT_URL,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    async def close(self) -> None:
        await self._client.aclose()

    # -- plumbing ----------------------------------------------------------

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except Exception:
            return ""
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("reason") or body.get("message")
            if detail:
                return str(detail)
        return ""

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Any = None,
        retry_statuses: Optional[frozenset] = _RETRY_STATUSES,
        timeout: Optional[float] = None,
    ) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    timeout=timeout or self.timeout,
                )
            except httpx.TransportError as exc:
                last_error = exc
                if attempt < _RETRY_ATTEMPTS:
                    logger.debug(
                        "%s %s failed (%s); retry %d/%d in %.0fs",
                        method,
                        path,
                        exc,
                        attempt,
                        _RETRY_ATTEMPTS - 1,
                        _RETRY_BACKOFF_S,
                    )
                    await asyncio.sleep(_RETRY_BACKOFF_S)
                    continue
                raise ConnectionError(
                    f"{method} {self.base_url}{path} failed: {exc}"
                ) from exc

            if (
                attempt < _RETRY_ATTEMPTS
                and retry_statuses
                and response.status_code in retry_statuses
            ):
                logger.debug(
                    "%s %s got %d; retry %d/%d in %.0fs",
                    method,
                    path,
                    response.status_code,
                    attempt,
                    _RETRY_ATTEMPTS - 1,
                    _RETRY_BACKOFF_S,
                )
                await asyncio.sleep(_RETRY_BACKOFF_S)
                continue

            if response.status_code >= 400:
                detail = self._error_detail(response)
                raise RuntimeError(
                    f"{method} {path} failed with {response.status_code}"
                    + (f": {detail}" if detail else "")
                )
            return response.json() if response.content else {}
        raise ConnectionError(  # pragma: no cover - unreachable
            f"{method} {path} failed: {last_error}"
        )

    # -- health ------------------------------------------------------------

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=10.0)
        except httpx.TransportError:
            return False
        return response.status_code == 200

    async def wait_healthy(self, timeout: float, poll_interval: float = 5.0) -> bool:
        """Poll /health until healthy or *timeout* seconds elapse.

        Cold containers take 5-10 minutes to boot the emulator; /health is a
        deep probe (env + QEMU + adb) so this covers the whole boot.
        """
        deadline = time.monotonic() + timeout
        while True:
            if await self.health():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_interval)

    # -- suite -------------------------------------------------------------

    async def task_list(self) -> List[str]:
        data = await self._request("GET", "/suite/task_list", params={"max_index": -1})
        return [str(name) for name in (data.get("task_list") or [])]

    async def task_length(self, task_type: str) -> int:
        data = await self._request(
            "GET", "/suite/task_length", params={"task_type": task_type}
        )
        return int(data.get("length", 0))

    async def reinitialize_suite(
        self,
        *,
        n_task_combinations: Optional[int] = None,
        seed: Optional[int] = None,
        task_family: str = "android_world",
        use_identical_params: bool = False,
    ) -> None:
        params: Dict[str, Any] = {
            "task_family": task_family,
            "use_identical_params": use_identical_params,
        }
        if n_task_combinations is not None:
            params["n_task_combinations"] = n_task_combinations
        if seed is not None:
            params["seed"] = seed
        await self._request(
            "POST", "/suite/reinitialize", params=params, timeout=_REINITIALIZE_TIMEOUT
        )

    # -- task lifecycle ----------------------------------------------------

    async def initialize_task(self, task_type: str, task_idx: int) -> None:
        await self._request(
            "POST",
            "/task/initialize",
            params={"task_type": task_type, "task_idx": task_idx},
            timeout=_INITIALIZE_TIMEOUT,
        )

    async def get_goal(self, task_type: str, task_idx: int) -> str:
        data = await self._request(
            "GET", "/task/goal", params={"task_type": task_type, "task_idx": task_idx}
        )
        return str(data.get("goal", ""))

    async def get_complexity(self, task_type: str, task_idx: int) -> float:
        data = await self._request(
            "POST",
            "/task/complexity",
            params={"task_type": task_type, "task_idx": task_idx},
        )
        return float(data.get("complexity", 1.0))

    async def start_on_home_screen(self, task_type: str, task_idx: int) -> bool:
        data = await self._request(
            "POST",
            "/task/start_on_home_screen",
            params={"task_type": task_type, "task_idx": task_idx},
        )
        return bool(data.get("start_on_home_screen"))

    async def stabilize(self) -> None:
        """Ask the server to wait for a stable screen.

        The server scores from the live accessibility tree, so the screen
        must be settled at /task/score time. The official docker runner gets
        this for free: its agent reads every observation with
        ``wait_to_stabilize=True`` (``transition_pause=None`` ->
        ``get_post_transition_state``), so the final frame of an episode is
        already stable. This call reproduces that guarantee for a
        client-side episode before scoring.
        """
        await self._request(
            "POST",
            "/state",
            params={"wait_to_stabilize": True},
            timeout=_STABILIZE_TIMEOUT,
        )

    async def execute_action(self, action: dict) -> None:
        """Execute a JSON action server-side (``/execute_action``).

        HTTP-status retries are disabled: a non-idempotent action may already
        have run server-side, so only connection-level errors retry.
        """
        await self._request(
            "POST",
            "/execute_action",
            json=action,
            retry_statuses=frozenset(),
            timeout=_SCORE_TIMEOUT,
        )

    async def get_score(self, task_type: str, task_idx: int) -> float:
        data = await self._request(
            "GET",
            "/task/score",
            params={"task_type": task_type, "task_idx": task_idx},
            timeout=_SCORE_TIMEOUT,
        )
        return float(data.get("score", 0.0))

    async def tear_down(self, task_type: str, task_idx: int) -> None:
        await self._request(
            "POST",
            "/task/tear_down",
            params={"task_type": task_type, "task_idx": task_idx},
            timeout=_SCORE_TIMEOUT,
        )

    async def reset(self, go_home: bool) -> None:
        await self._request(
            "POST", "/reset", params={"go_home": go_home}, timeout=_RESET_TIMEOUT
        )


async def _submit_episode_answer(
    client: "AndroidWorldTaskClient", agent: MobileAgent, reason: str
) -> None:
    """Submit the agent's answer into the server's interaction_cache.

    Answer-type (information retrieval) tasks are graded purely from the text
    submitted through the ``answer`` action, so an episode that never submits
    one always scores 0 regardless of correctness. Prefer the agent's own
    finish answer (``shared_state.answer``) and fall back to the episode
    reason. Best-effort: a submission failure must not fail the episode.
    """
    answer = getattr(agent.shared_state, "answer", "") or reason or ""
    answer = str(answer).strip()
    if not answer:
        return
    try:
        await client.execute_action({"action_type": "answer", "text": answer})
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.warning("Submitting the episode answer failed: %s", exc)


# Provenance fields that, when changed, make resumed records incomparable.
_RESUME_PROVENANCE_FIELDS = (
    "server_url",
    "seed",
    "n_task_combinations",
    "provider",
    "model",
    "executor_kind",
    "task_idx",
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_androidworld_benchmark(
    server_url: Optional[str] = None,
    config_path: Optional[str] = None,
    tasks: Optional[str] = None,
    task_index_range: Optional[str] = None,
    difficulty: Sequence[str] = (),
    task_idx: int = 0,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    n_task_combinations: Optional[int] = None,
    output: Optional[str] = None,
    resume: bool = False,
    health_timeout: float = 900.0,
    task_timeout: float = 1000.0,
    save_trajectory: Optional[str] = None,
    capture_evidence: bool = True,
    telemetry: Optional[bool] = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Optional[float] = None,
    debug: Optional[bool] = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run the AndroidWorld task suite with the mobilerun agent.

    Returns True when the run completed (individual task failures included),
    False on fatal errors (unhealthy server, bad selection, interrupt).
    """
    server_url = (server_url or ANDROID_WORLD_DEFAULT_URL).rstrip("/")
    try:
        # Validate before the (potentially minutes-long) health wait so a bad
        # URL fails immediately with a clear message instead of a timeout.
        server_url = validate_android_world_url(server_url)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/]")
        return False

    if (provider is None) != (model is None):
        console.print(
            "[red]Error: --provider and --model must be provided together.[/]"
        )
        return False

    if task_index_range:
        try:
            _parse_task_index_range(task_index_range)  # format check, pre-wait
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/]")
            return False

    config = ConfigLoader.load(config_path)
    # This benchmark always drives an AndroidWorld environment server.
    config.device.control_backend = ANDROID_WORLD_CONNECTION
    config.device.serial = server_url
    if config.agent.name in list_agents():
        # External agents take MobileAgent's early-exit path, which drives adb
        # directly and would bypass the injected AndroidWorld driver.
        console.print(
            f"[red]Error: agent '{config.agent.name}' is an external agent; "
            "runbench requires a built-in mobilerun agent.[/]"
        )
        return False
    if save_trajectory is not None:
        config.logging.save_trajectory = save_trajectory
    if capture_evidence and config.logging.save_trajectory == "none":
        # Evidence needs the agent's raw events; turn trajectory capture on.
        config.logging.save_trajectory = "step"
    if telemetry is not None:
        config.telemetry.enabled = telemetry

    debug_mode = debug if debug is not None else config.logging.debug
    _setup_benchmark_logging(debug_mode, verbose)

    if output is None:
        output = (
            f"runs/{_BENCHMARK_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    output_path = Path(output)
    # One directory per episode under <output>/evidences/ holds every artifact.
    evidence_root = output_path.parent / f"{output_path.stem}_evidence"

    client = AndroidWorldTaskClient(server_url)
    driver: Optional[AndroidWorldHttpDriver] = None
    fatal_error = False
    run_id = _new_run_id()
    run_meta: Dict[str, Any] = {
        "benchmark": _BENCHMARK_NAME,
        "run_id": run_id,
        "server_url": server_url,
        "agent": config.agent.name,
        "reasoning": config.agent.reasoning,
        "executor_kind": config.agent.executor.kind,
        "provider": provider,
        "model": model,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "updated_at": None,
        "task_idx": task_idx,
        "seed": seed,
        "n_task_combinations": n_task_combinations,
        # Which slice of the suite this invocation was asked to cover; without
        # these the file cannot say how a sharded run was produced. (Named
        # ``tasks_filter`` because top-level ``tasks`` holds the task rows.)
        "task_index_range": task_index_range,
        "tasks_filter": tasks,
        "difficulty": list(difficulty),
        "resumed": resume,
        "results_path": str(output_path),
        "capture_evidence": capture_evidence,
        "evidence_root": str(evidence_root) if capture_evidence else None,
    }
    # The results file accumulates across sharded / resumed invocations; the
    # history of those invocations is merged back in on every write.
    run_history = _load_run_history(output_path) if output_path.exists() else []

    try:
        console.print(
            f"⏳ Waiting for AndroidWorld server at [cyan]{server_url}[/] ..."
        )
        if not await client.wait_healthy(health_timeout):
            console.print(
                f"[red]Server at {server_url} did not become healthy within "
                f"{health_timeout:.0f}s.[/]"
            )
            return False
        console.print("✅ Server healthy")

        if seed is not None or n_task_combinations is not None:
            console.print(
                f"♻️  Reinitializing suite (seed={seed}, "
                f"n_task_combinations={n_task_combinations}) ..."
            )
            await client.reinitialize_suite(
                n_task_combinations=n_task_combinations, seed=seed
            )

        available = await client.task_list()
        if not available:
            console.print("[red]The server returned an empty task list.[/]")
            return False

        metadata = load_task_metadata()
        try:
            selected = _select_tasks(
                available,
                tasks=tasks,
                task_index_range=task_index_range,
                difficulty=difficulty,
                metadata=metadata,
            )
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/]")
            return False

        run_meta["selected_count"] = len(selected)
        # Run-level provenance embedded in each episode's evidence manifest so
        # results and evidence can be cross-checked in both directions.
        run_context = {
            key: run_meta.get(key)
            for key in (
                "run_id",
                "benchmark",
                "server_url",
                "agent",
                "provider",
                "model",
                "seed",
                "task_idx",
                "task_index_range",
                "tasks_filter",
                "difficulty",
                "started_at",
            )
        }

        if task_idx < 0 or task_idx >= await client.task_length(selected[0]):
            console.print(
                f"[red]Error: --task-idx {task_idx} is out of range for "
                f"'{selected[0]}'.[/]"
            )
            return False

        if dry_run:
            table = Table(title=f"Tasks to run ({len(selected)})")
            table.add_column("#", style="dim")
            table.add_column("Task", style="cyan")
            table.add_column("Difficulty", style="green")
            table.add_column("Optimal steps", style="white")
            for index, name in enumerate(selected, start=1):
                entry = metadata.get(name, {})
                table.add_row(
                    str(index),
                    name,
                    entry.get("difficulty", ""),
                    str(entry.get("optimal_steps", "")),
                )
            console.print(table)
            console.print(f"Results would be written to [cyan]{output_path}[/]")
            return True

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # -- LLMs: load once, share across every episode --------------------
        if provider and model:
            llm_kwargs: Dict[str, Any] = {}
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            if base_url is not None:
                llm_kwargs["base_url"] = base_url
            if api_base is not None:
                llm_kwargs["api_base"] = api_base
            llms: Any = load_llm(provider, model=model, **llm_kwargs)
        else:
            llm_kwargs = {
                key: value
                for key, value in (
                    ("temperature", temperature),
                    ("base_url", base_url),
                    ("api_base", api_base),
                )
                if value is not None
            }
            llms = load_agent_llms(config, **llm_kwargs)

        # One connected driver shared by every episode: MobileAgent never
        # connects or closes an injected driver, and per-episode drivers would
        # leak one httpx client per task.
        driver = AndroidWorldHttpDriver(url=server_url)
        await driver.connect()

        results: Dict[str, TaskResult] = {}
        if resume and output_path.exists():
            results = _load_results(output_path)
            previous_meta = _load_run_meta(output_path)
            changed = []
            for field in _RESUME_PROVENANCE_FIELDS:
                if previous_meta.get(field) == run_meta.get(field):
                    continue
                if (
                    field in ("seed", "n_task_combinations")
                    and run_meta.get(field) is None
                ):
                    # No reinit requested: the container's suite is unchanged,
                    # so the recorded rows stay comparable.
                    continue
                changed.append(field)
            if changed:
                console.print(
                    "[yellow]⚠️ Resuming with different "
                    f"{', '.join(changed)} than the recorded run; previously "
                    "recorded tasks may not be comparable.[/]"
                )
            console.print(
                f"⏩ Resuming: {len(results)} recorded task(s) in {output_path}"
            )

        total = len(selected)
        console.print(f"🏁 Running {total} task(s) on {server_url}")
        for index, task_type in enumerate(selected, start=1):
            key = f"{task_type}#{task_idx}"
            recorded = results.get(key)
            if resume and recorded is not None and recorded.error is None:
                console.print(
                    f"[dim][{index}/{total}] {task_type} — already recorded, skipped[/]"
                )
                continue

            result = TaskResult(
                task_type=task_type,
                task_idx=task_idx,
                difficulty=metadata.get(task_type, {}).get("difficulty", ""),
                timestamp=datetime.now().isoformat(timespec="seconds"),
                run_id=run_id,
                server_url=server_url,
            )
            task_started = time.monotonic()
            initialized = False
            evidence: Optional[EpisodeEvidence] = None
            agent: Optional[MobileAgent] = None
            ui_tree_before: Optional[Dict[str, Any]] = None
            ui_tree_after: Optional[Dict[str, Any]] = None
            if capture_evidence:
                evidence = EpisodeEvidence(
                    evidence_root / f"{_sanitize_name(task_type)}_{task_idx}",
                    task_type,
                    task_idx,
                )
            try:
                await client.initialize_task(task_type, task_idx)
                initialized = True
                result.goal = await client.get_goal(task_type, task_idx)
                complexity = await client.get_complexity(task_type, task_idx)
                start_on_home = await client.start_on_home_screen(task_type, task_idx)
                result.complexity = complexity
                result.start_on_home_screen = start_on_home
                # Fresh episode: reset with go_home mirroring the task's
                # start_on_home_screen, exactly like the reference runner's
                # agent.reset(start_on_home_screen).
                await client.reset(go_home=start_on_home)

                max_steps = steps if steps is not None else _allocate_step_budget(complexity)
                result.max_steps = max_steps
                task_config = copy.deepcopy(config)
                task_config.agent.max_steps = max_steps

                agent = MobileAgent(
                    goal=result.goal,
                    config=task_config,
                    llms=llms,
                    driver=driver,
                    timeout=task_timeout,
                    runtype="benchmark",
                )
                if evidence is not None:
                    # The real accessibility tree (empty under vision_only
                    # providers), read straight from the driver. Polls briefly
                    # because the tree is empty mid-transition after reset.
                    ui_tree_before = await gather_ui_tree(driver)
                handler = agent.run()
                manager_answered = False
                # Forward events to the CLI handler: agents that report each
                # step only as events (Jev) would otherwise be silent here —
                # the loop below feeds evidence, never the console. Imported
                # lazily: mobilerun.cli imports this module at package level.
                from mobilerun.cli.event_handler import EventHandler

                event_handler = EventHandler()
                async for event in handler.stream_events():
                    event_handler.handle(event)
                    # Reasoning mode: the manager's final answer is the
                    # episode-completion signal (AndroidWorld's `done`).
                    if (
                        isinstance(event, ManagerPlanEvent)
                        and (event.answer or "").strip()
                    ):
                        manager_answered = True
                    if evidence is not None:
                        evidence.record_event(event)
                run_result = await handler

                result.success = bool(run_result.success)
                result.steps = run_result.steps
                result.reason = run_result.reason
                # Transient provider failures become retriable errors so
                # --resume re-runs the episode instead of keeping the 0.
                infra_reason = _infra_error_reason(run_result)
                if infra_reason:
                    result.error = infra_reason
                # AndroidWorld gates the score on the agent declaring the
                # episode finished. Direct mode: complete() ran, which sets
                # shared_state.success and survives the run (FastAgent only
                # clears the transient `finished` flag). Reasoning mode: the
                # manager finalized with an answer.
                if task_config.agent.reasoning:
                    declared_done = manager_answered
                else:
                    declared_done = agent.shared_state.success is not None
                result.finished = declared_done

                # Answer-type tasks (information retrieval) are graded only
                # from the text the agent submits via the `answer` action into
                # the server's interaction_cache -- the score never looks at
                # the screen. Submit it after the episode but before scoring
                # (and before any further device action) so those tasks are
                # gradable; UI-scored tasks ignore this cache, so it is safe
                # for them too.
                if declared_done:
                    await _submit_episode_answer(client, agent, run_result.reason)

                # Score before tear_down: the server evaluates live device
                # state and requires the task to still be initialized. Settle
                # the screen first (the official runner's episode ends on a
                # stabilized frame) so the tree the server reads reflects the
                # agent's final action instead of a mid-animation snapshot.
                # Best-effort: a failed settle must not cost a good episode; a
                # score read that follows can itself surface a real outage.
                try:
                    await client.stabilize()
                except Exception as exc:  # noqa: BLE001 - best effort
                    logger.warning("stabilize before scoring failed: %s", exc)
                result.score = await client.get_score(task_type, task_idx)
                # Second chance: a declared-done episode that scored 0 may have
                # been judged on a frame that was still settling; give the UI
                # another beat and keep the better reading.
                if declared_done and not result.score:
                    await asyncio.sleep(_SCORE_RECHECK_DELAY_S)
                    try:
                        await client.stabilize()
                    except Exception as exc:  # noqa: BLE001 - best effort
                        logger.warning("stabilize before re-scoring failed: %s", exc)
                    result.score = max(
                        result.score, await client.get_score(task_type, task_idx)
                    )
                result.gated_score = result.score if declared_done else 0.0
            except Exception as exc:  # noqa: BLE001 - record and continue
                result.error = f"{type(exc).__name__}: {exc}"
                # Infra failures (server or device unreachable) make every
                # remaining episode fail the same way: probe the deep /health
                # endpoint (env + QEMU + adb) and wait for recovery. Plain
                # ConnectionError from the task client and httpx transport
                # errors from the driver both land here, as do wrapped
                # state-fetch failures that a health probe can still detect.
                if (
                    isinstance(exc, (ConnectionError, httpx.TransportError))
                    or not await client.health()
                ):
                    console.print(
                        "[yellow]Connection failure; waiting for the server to recover ...[/]"
                    )
                    if not await client.wait_healthy(health_timeout):
                        result.duration_s = time.monotonic() - task_started
                        results[key] = result
                        # Aborted: finished_at stays unset so the file's
                        # aggregate end time is not moved by a failed run.
                        _write_results(output_path, run_meta, results, run_history)
                        console.print(
                            "[red]Server did not recover; aborting the run.[/]"
                        )
                        fatal_error = True
                        break
            finally:
                result.duration_s = time.monotonic() - task_started
                if evidence is not None and ui_tree_after is None:
                    # Capture the final device state right before tear_down, so
                    # every outcome (success, error, timeout) gets a snapshot.
                    ui_tree_after = await gather_ui_tree(driver)
                if initialized:
                    try:
                        await client.tear_down(task_type, task_idx)
                    except Exception as exc:  # noqa: BLE001 - best effort
                        console.print(
                            f"[yellow]⚠️ tear_down for {task_type} failed: {exc}[/]"
                        )
                if evidence is not None:
                    _finalize_episode_evidence(
                        evidence,
                        result,
                        agent,
                        ui_tree_before,
                        ui_tree_after,
                        run_context,
                    )

            results[key] = result
            # Incremental write: no finished_at here -- only a truly finished
            # invocation sets it, so a mid-run write cannot claim the file is
            # done (updated_at records the write itself).
            _write_results(output_path, run_meta, results, run_history)

            status = "green" if (result.gated_score or 0.0) > 0.5 else "red"
            if result.error is not None:
                console.print(
                    f"[red][{index}/{total}] {task_type} ({result.difficulty}) "
                    f"ERROR: {result.error[:160]} ({result.duration_s:.1f}s)[/]"
                )
            else:
                console.print(
                    f"[{status}][{index}/{total}] {task_type} ({result.difficulty}) "
                    f"score={result.score:.2f} gated={result.gated_score:.2f} "
                    f"steps={result.steps} finished={result.finished} "
                    f"({result.duration_s:.1f}s)[/]"
                )

        # Only a completed invocation stamps finished_at; the file's end time
        # is the latest finished run across every invocation in runs[].
        run_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_results(output_path, run_meta, results, run_history)
        console.print(f"💾 Results written to [cyan]{output_path}[/]")
        if capture_evidence:
            console.print(f"🗂️  Evidence written to [cyan]{evidence_root}[/]")

        ordered = [
            results[f"{name}#{task_idx}"]
            for name in selected
            if f"{name}#{task_idx}" in results
        ]
        _print_summary(ordered, "AndroidWorld Benchmark Summary")
        return not fatal_error
    except asyncio.CancelledError:
        # Ctrl-C: per-task results were already written incrementally.
        console.print(
            "\n[yellow]⏹️ Benchmark interrupted; partial results were kept.[/]"
        )
        raise
    finally:
        if driver is not None:
            await driver.close()
        await client.close()
