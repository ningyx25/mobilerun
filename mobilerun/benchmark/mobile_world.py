"""MobileWorld benchmark runner.

Drives a MobileWorld environment server (``src/mobile_world/core/server.py``)
over its task API and runs the mobilerun agent on its tasks, mirroring the
per-episode flow of the reference runner (``src/mobile_world/core/runner.py``)::

    initialize -> goal -> agent episode -> submit answer -> evaluate -> tear_down

Differences from the AndroidWorld server worth knowing:

- Tasks are named, single-instance (no suite / task index / complexity), so
  the step budget is a flat ``--steps`` (default 50, matching the reference
  runner's ``--max_round 50``).
- ``/task/init`` performs all device setup server-side (snapshot, time sync,
  home), so no client-side reset is needed.
- ``/task/eval`` is a GET carrying a JSON body.
- Answer-style tasks grade only from the value submitted via the ``/step``
  ``answer`` action, so the runner must submit it before scoring.

By default only pure-GUI tasks run: tasks tagged ``agent-mcp`` or
``agent-user-interaction`` are excluded unless ``--enable-mcp`` /
``--enable-user-interaction`` is passed (they need subsystems the mobilerun
agent does not provide yet).
"""

from __future__ import annotations

import asyncio
import copy
import difflib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from rich.console import Console
from rich.table import Table

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import ManagerPlanEvent
from mobilerun.agent.external import list_agents
from mobilerun.agent.utils.llm_loader import load_agent_llms
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.benchmark.android_world import (
    _RETRY_ATTEMPTS,
    _RETRY_BACKOFF_S,
    _RETRY_STATUSES,
    _SCORE_TIMEOUT,
    TaskResult,
    _finalize_episode_evidence,
    _load_results,
    _load_run_history,
    _load_run_meta,
    _new_run_id,
    _parse_task_index_range,
    _print_summary,
    _sanitize_name,
    _setup_benchmark_logging,
    _write_results,
)
from mobilerun.benchmark.evidence import EpisodeEvidence, gather_ui_tree
from mobilerun.config_manager import ConfigLoader
from mobilerun.tools.driver.mobile_world import (
    MOBILE_WORLD_CONNECTION,
    MOBILE_WORLD_DEFAULT_DEVICE,
    MOBILE_WORLD_DEFAULT_URL,
    MobileWorldHttpDriver,
    validate_mobile_world_url,
)

logger = logging.getLogger("mobilerun")
console = Console()

_BENCHMARK_NAME = "mobileworld"

# No per-task complexity exists on this server (``/task/complexity`` 500s), so
# the step budget is flat; 50 matches the reference runner's --max_round.
_DEFAULT_MAX_STEPS = 50
_DEFAULT_TIMEOUT = 60.0
_INITIALIZE_TIMEOUT = 600.0

_MCP_TAG = "agent-mcp"
_USER_INTERACTION_TAG = "agent-user-interaction"

# Provenance fields that, when changed, make resumed records incomparable.
_MOBILE_WORLD_RESUME_FIELDS = (
    "server_url",
    "provider",
    "model",
    "tasks_filter",
    "enable_mcp",
    "enable_user_interaction",
)


class MobileWorldTaskClient:
    """HTTP client for the MobileWorld server's task API."""

    def __init__(
        self,
        base_url: str = MOBILE_WORLD_DEFAULT_URL,
        device: str = MOBILE_WORLD_DEFAULT_DEVICE,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.device = device or MOBILE_WORLD_DEFAULT_DEVICE
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
            return response.text[:200]
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("reason") or body.get("message")
            if detail:
                return str(detail)
        return str(body)

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
                raise RuntimeError(
                    f"{method} {path} failed with {response.status_code}"
                    f": {self._error_detail(response)}"
                )
            return response.json() if response.content else {}
        raise ConnectionError(  # pragma: no cover - unreachable
            f"{method} {path} failed after {_RETRY_ATTEMPTS} attempts"
        )

    # -- health ------------------------------------------------------------

    async def health(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=10.0)
        except httpx.TransportError:
            return False
        if response.status_code != 200:
            return False
        try:
            return bool(response.json().get("ok"))
        except Exception:
            return False

    async def wait_healthy(self, timeout: float, poll_interval: float = 5.0) -> bool:
        """Poll /health until healthy or *timeout* seconds elapse."""
        deadline = time.monotonic() + timeout
        while True:
            if await self.health():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(poll_interval)

    # -- tasks -------------------------------------------------------------

    async def task_list(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/task/list")
        return [entry for entry in (data or []) if isinstance(entry, dict)]

    async def get_goal(self, task_name: str) -> str:
        data = await self._request("GET", "/task/goal", params={"task_name": task_name})
        return str(data) if not isinstance(data, dict) else str(data.get("goal", ""))

    async def initialize(self, task_name: str) -> None:
        # Non-idempotent device setup: only connection-level errors retry.
        await self._request(
            "POST",
            "/task/init",
            json={"task_name": task_name, "req_device": self.device},
            retry_statuses=frozenset(),
            timeout=_INITIALIZE_TIMEOUT,
        )

    async def evaluate(self, task_name: str) -> Tuple[float, str]:
        """Run the task's success check; returns (score, reason)."""
        # The server binds TaskOperationRequest from the body of this GET.
        data = await self._request(
            "GET",
            "/task/eval",
            json={"task_name": task_name, "req_device": self.device},
            timeout=_SCORE_TIMEOUT,
        )
        body = data if isinstance(data, dict) else {}
        score = float(body.get("score", 0.0))
        reason = str(body.get("reason") or f"No reason provided for {task_name}")
        return score, reason

    async def tear_down(self, task_name: str) -> None:
        await self._request(
            "POST",
            "/task/tear_down",
            json={"task_name": task_name, "req_device": self.device},
            retry_statuses=frozenset(),
            timeout=_SCORE_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


def _is_gui_task(
    entry: Dict[str, Any], *, enable_mcp: bool, enable_user_interaction: bool
) -> bool:
    """True when the task needs only GUI interaction (no MCP / user agent)."""
    tags = set(entry.get("tags") or [])
    if not enable_mcp and _MCP_TAG in tags:
        return False
    if not enable_user_interaction and _USER_INTERACTION_TAG in tags:
        return False
    return True


def _select_mobile_world_tasks(
    entries: Sequence[Dict[str, Any]],
    *,
    tasks: Optional[str] = None,
    task_index_range: Optional[str] = None,
    difficulty: Sequence[str] = (),
    enable_mcp: bool = False,
    enable_user_interaction: bool = False,
) -> List[str]:
    """Filter the server task list down to the tasks to run.

    By default only pure-GUI tasks survive; ``enable_mcp`` /
    ``enable_user_interaction`` admit the corresponding tag groups.
    """
    all_names = [str(entry.get("name", "")) for entry in entries if entry.get("name")]
    base: List[str] = []
    for entry in entries:
        name = str(entry.get("name", ""))
        if name and _is_gui_task(
            entry,
            enable_mcp=enable_mcp,
            enable_user_interaction=enable_user_interaction,
        ):
            base.append(name)

    selected = list(base)
    if task_index_range:
        start, end = _parse_task_index_range(task_index_range, len(base))
        selected = base[start - 1 : end]

    if tasks:
        requested = [name.strip() for name in tasks.split(",") if name.strip()]
        base_set = set(base)
        all_set = set(all_names)
        unknown = [name for name in requested if name not in all_set]
        if unknown:
            hints = []
            for name in unknown:
                matches = difflib.get_close_matches(name, all_names, n=3, cutoff=0.5)
                if matches:
                    hints.append(f"'{name}' (did you mean: {', '.join(matches)}?)")
                else:
                    hints.append(f"'{name}'")
            raise ValueError(f"Unknown task(s): {', '.join(hints)}")
        gated = [name for name in requested if name not in base_set]
        if gated:
            raise ValueError(
                f"Task(s) {', '.join(gated)} are tagged '{_MCP_TAG}' or "
                f"'{_USER_INTERACTION_TAG}'; pass --enable-mcp / "
                "--enable-user-interaction to run them."
            )
        wanted = set(requested)
        selected = [name for name in selected if name in wanted]

    if difficulty:
        logger.warning(
            "MobileWorld tasks have no difficulty field; ignoring --difficulty."
        )

    if not selected:
        raise ValueError("Task selection is empty.")
    return selected


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_mobileworld_benchmark(
    server_url: Optional[str] = None,
    config_path: Optional[str] = None,
    tasks: Optional[str] = None,
    task_index_range: Optional[str] = None,
    difficulty: Sequence[str] = (),
    steps: Optional[int] = None,
    output: Optional[str] = None,
    resume: bool = False,
    health_timeout: float = 900.0,
    task_timeout: float = 1000.0,
    enable_mcp: bool = False,
    enable_user_interaction: bool = False,
    device_id: Optional[str] = None,
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
    """Run the MobileWorld task suite with the mobilerun agent.

    Returns True when the run completed (individual task failures included),
    False on fatal errors (unhealthy server, bad selection, interrupt).
    """
    server_url = (server_url or MOBILE_WORLD_DEFAULT_URL).rstrip("/")
    try:
        server_url = validate_mobile_world_url(server_url)
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

    device = device_id or MOBILE_WORLD_DEFAULT_DEVICE
    config = ConfigLoader.load(config_path)
    # This benchmark always drives a MobileWorld environment server.
    config.device.control_backend = MOBILE_WORLD_CONNECTION
    config.device.serial = server_url
    if config.agent.name in list_agents():
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
    evidence_root = output_path.parent / f"{output_path.stem}_evidence"

    client = MobileWorldTaskClient(server_url, device)
    driver: Optional[MobileWorldHttpDriver] = None
    fatal_error = False
    run_id = _new_run_id()
    run_meta: Dict[str, Any] = {
        "benchmark": _BENCHMARK_NAME,
        "run_id": run_id,
        "server_url": server_url,
        "device_id": device,
        "agent": config.agent.name,
        "reasoning": config.agent.reasoning,
        "provider": provider,
        "model": model,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "updated_at": None,
        "task_index_range": task_index_range,
        "tasks_filter": tasks,
        "difficulty": list(difficulty),
        "enable_mcp": enable_mcp,
        "enable_user_interaction": enable_user_interaction,
        "resumed": resume,
        "results_path": str(output_path),
        "capture_evidence": capture_evidence,
        "evidence_root": str(evidence_root) if capture_evidence else None,
    }
    run_history = _load_run_history(output_path) if output_path.exists() else []

    try:
        console.print(f"⏳ Waiting for MobileWorld server at [cyan]{server_url}[/] ...")
        if not await client.wait_healthy(health_timeout):
            console.print(
                f"[red]Server at {server_url} did not become healthy within "
                f"{health_timeout:.0f}s.[/]"
            )
            return False
        console.print("✅ Server healthy")

        entries = await client.task_list()
        if not entries:
            console.print("[red]The server returned an empty task list.[/]")
            return False

        try:
            selected = _select_mobile_world_tasks(
                entries,
                tasks=tasks,
                task_index_range=task_index_range,
                difficulty=difficulty,
                enable_mcp=enable_mcp,
                enable_user_interaction=enable_user_interaction,
            )
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/]")
            return False

        by_name = {str(entry.get("name")): entry for entry in entries}
        run_meta["selected_count"] = len(selected)
        run_context = {
            key: run_meta.get(key)
            for key in (
                "run_id",
                "benchmark",
                "server_url",
                "device_id",
                "agent",
                "provider",
                "model",
                "task_index_range",
                "tasks_filter",
                "enable_mcp",
                "enable_user_interaction",
                "started_at",
            )
        }

        if dry_run:
            table = Table(title=f"Tasks to run ({len(selected)})")
            table.add_column("#", style="dim")
            table.add_column("Task", style="cyan")
            table.add_column("Tags", style="green")
            table.add_column("Apps", style="white")
            for index, name in enumerate(selected, start=1):
                entry = by_name.get(name, {})
                table.add_row(
                    str(index),
                    name,
                    ", ".join(entry.get("tags") or []),
                    ", ".join(entry.get("apps") or []),
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

        # One connected driver shared by every episode (MobileAgent never
        # connects or closes an injected driver).
        driver = MobileWorldHttpDriver(url=server_url, device=device)
        await driver.connect()

        results: Dict[str, TaskResult] = {}
        if resume and output_path.exists():
            results = _load_results(output_path)
            previous_meta = _load_run_meta(output_path)
            changed = [
                field
                for field in _MOBILE_WORLD_RESUME_FIELDS
                if previous_meta.get(field) != run_meta.get(field)
            ]
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
            key = f"{task_type}#0"
            recorded = results.get(key)
            if resume and recorded is not None and recorded.error is None:
                console.print(
                    f"[dim][{index}/{total}] {task_type} — already recorded, skipped[/]"
                )
                continue

            result = TaskResult(
                task_type=task_type,
                task_idx=0,
                start_on_home_screen=True,
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
                    evidence_root / f"{_sanitize_name(task_type)}_0", task_type, 0
                )
            try:
                await client.initialize(task_type)
                initialized = True
                result.goal = await client.get_goal(task_type)

                max_steps = steps if steps is not None else _DEFAULT_MAX_STEPS
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
                # Same completion gate as the reference runner: score only
                # counts when the agent declared the episode finished.
                if task_config.agent.reasoning:
                    declared_done = manager_answered
                else:
                    declared_done = agent.shared_state.success is not None
                result.finished = declared_done

                # Answer-style tasks grade purely from the submitted answer, so
                # it must reach the server before /task/eval. Best-effort.
                if declared_done:
                    answer = (
                        getattr(agent.shared_state, "answer", "") or run_result.reason
                    )
                    answer = str(answer).strip()
                    if answer:
                        try:
                            await driver.submit_answer(answer)
                        except Exception as exc:  # noqa: BLE001 - best effort
                            logger.warning(
                                "Submitting the episode answer failed: %s", exc
                            )

                result.score, result.eval_reason = await client.evaluate(task_type)
                result.gated_score = result.score if declared_done else 0.0
            except Exception as exc:  # noqa: BLE001 - record and continue
                result.error = f"{type(exc).__name__}: {exc}"
                # Infra failures make every remaining episode fail the same
                # way: probe /health and wait for recovery.
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
                        _write_results(output_path, run_meta, results, run_history)
                        console.print(
                            "[red]Server did not recover; aborting the run.[/]"
                        )
                        fatal_error = True
                        break
            finally:
                result.duration_s = time.monotonic() - task_started
                if evidence is not None and ui_tree_after is None:
                    ui_tree_after = await gather_ui_tree(driver)
                if initialized:
                    try:
                        await client.tear_down(task_type)
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
            _write_results(output_path, run_meta, results, run_history)

            status = "green" if (result.gated_score or 0.0) > 0.5 else "red"
            if result.error is not None:
                console.print(
                    f"[red][{index}/{total}] {task_type} "
                    f"ERROR: {result.error[:160]} ({result.duration_s:.1f}s)[/]"
                )
            else:
                console.print(
                    f"[{status}][{index}/{total}] {task_type} "
                    f"score={result.score:.2f} gated={result.gated_score:.2f} "
                    f"steps={result.steps} finished={result.finished} "
                    f"({result.duration_s:.1f}s)[/]"
                )

        run_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_results(output_path, run_meta, results, run_history)
        console.print(f"💾 Results written to [cyan]{output_path}[/]")
        if capture_evidence:
            console.print(f"🗂️  Evidence written to [cyan]{evidence_root}[/]")

        ordered = [results[f"{name}#0"] for name in selected if f"{name}#0" in results]
        _print_summary(ordered)
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
