"""The harness: the thin orchestrator between the two services.

It owns no domain knowledge. For each selected task it

1. opens an environment session (task setup + first observation),
2. asks the agent service to run an episode against that session,
3. asks the environment to finish and score it,
4. closes the session and records the row.

Results are written with the benchmark runner's own machinery
(``TaskResult`` / ``_write_results`` / ``_load_run_history``), so a harness run
and a ``mobilerun runbench`` run land in files that are directly comparable --
and resume works the same way in both.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from mobilerun.benchmark.android_world import (  # shared result/provenance plumbing
    TaskResult,
    _load_results,
    _load_run_history,
    _load_run_meta,
    _new_run_id,
    _parse_task_index_range,
    _print_summary,
    _setup_benchmark_logging,
    _write_results,
)
from mobilerun.harness.client import AgentClient, EnvClient, ServiceError
from mobilerun.harness.protocol import (
    PROTOCOL_VERSION,
    AgentOverrides,
    EnvironmentRef,
    EpisodeEvidenceConfig,
    EpisodeLimits,
    EpisodeRequest,
    EpisodeState,
    TaskSpec,
)

logger = logging.getLogger("mobilerun")
console = Console()

_BENCHMARK_NAME = "harness"
_CAPACITY_RETRY_S = 5.0
_HEARTBEAT_S = 60.0


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


def _select_tasks(
    tasks: Sequence[TaskSpec],
    *,
    backend: Optional[str] = None,
    names: Optional[str] = None,
    task_index_range: Optional[str] = None,
    difficulty: Sequence[str] = (),
) -> List[TaskSpec]:
    """Filter the environment's task catalogue down to what to run."""
    selected = [task for task in tasks if backend is None or task.backend == backend]

    if backend is None:
        # ``/v1/tasks`` returns one copy per backend, each tagged with its own
        # id. Without a pinned backend the copies are the same suite: keep one
        # and let the environment lease whichever backend is free per session.
        seen = set()
        deduped = []
        for task in selected:
            if task.id in seen:
                continue
            seen.add(task.id)
            deduped.append(task)
        selected = deduped

    if task_index_range:
        start, end = _parse_task_index_range(task_index_range, len(selected))
        selected = selected[start - 1 : end]

    if names:
        requested = [name.strip() for name in names.split(",") if name.strip()]
        by_id = {task.id: task for task in selected}
        unknown = [name for name in requested if name not in by_id]
        if unknown:
            raise ValueError(
                f"Unknown task(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(by_id)[:10])}…"
            )
        selected = [by_id[name] for name in requested]

    if difficulty:
        wanted = set(difficulty)
        selected = [
            task
            for task in selected
            if str(task.metadata.get("difficulty", "")) in wanted
        ]

    if not selected:
        raise ValueError("Task selection is empty.")
    return selected


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------


def _task_key(task: TaskSpec) -> str:
    """Stable result key, matching ``TaskResult.key`` (``"<name>#<idx>"``)."""
    name, _, idx_text = task.id.partition("#")
    return f"{name}#{int(idx_text or 0)}"


async def _run_one(
    task: TaskSpec,
    *,
    env: EnvClient,
    agent: AgentClient,
    agent_env_url: Optional[str],
    overrides: AgentOverrides,
    steps: Optional[int],
    task_timeout: float,
    seed: Optional[int],
    evidence: bool,
    evidence_root: Optional[Path],
    backend: Optional[str],
    agent_id: Optional[str] = None,
) -> TaskResult:
    """Run one task end to end and return its result row."""
    task_type, _, idx_text = task.id.partition("#")
    task_idx = int(idx_text or 0)
    result = TaskResult(
        task_type=task_type,
        task_idx=task_idx,
        difficulty=str(task.metadata.get("difficulty", "")),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        server_url=str(env.base_url),
        agent_id=agent_id or "",
    )
    started = time.monotonic()
    session_id: Optional[str] = None
    episode_id: Optional[str] = None
    try:
        session = await env.create_session(
            task.id.partition("#")[0],
            # ``None`` lets the environment lease any free backend. Falling
            # back to ``task.backend`` here would pin every task to the first
            # backend in the catalogue and serialise parallel runs.
            backend=backend,
            task_idx=task_idx,
            seed=seed,
        )
        session_id = session.session_id
        result.goal = session.task.instruction
        result.complexity = session.task.complexity
        result.start_on_home_screen = session.task.start_on_home_screen
        result.max_steps = steps or session.task.max_steps

        episode = await agent.start_episode(
            EpisodeRequest(
                goal=session.task.instruction,
                environment=EnvironmentRef(
                    base_url=agent_env_url or env.base_url, session_id=session_id
                ),
                agent_id=agent_id,
                overrides=overrides,
                limits=EpisodeLimits(
                    max_steps=steps or session.task.max_steps,
                    timeout_s=task_timeout,
                ),
                evidence=EpisodeEvidenceConfig(
                    enabled=evidence, dir=str(evidence_root) if evidence_root else None
                ),
                metadata={"task_id": task_type, "task_idx": task_idx},
            )
        )
        episode_id = episode.episode_id

        final = await agent.wait_episode(
            episode_id,
            timeout_s=task_timeout + _AGENT_GRACE_S,
            poll_s=1.0,
        )
        if final.state is EpisodeState.RUNNING:
            # The agent is wedged; cancel and record it as an error instead of
            # hanging the whole suite.
            await agent.cancel_episode(episode_id)
            final = await agent.get_episode(episode_id)
            result.error = f"agent episode timed out after {task_timeout:.0f}s"
        if final.state is EpisodeState.ERROR:
            result.error = final.error or "agent episode failed"

        # The dispatcher echoes the slot it actually used; a direct worker
        # leaves it empty and the requested id stands.
        result.agent_id = final.agent_id or result.agent_id
        run = final.result
        if run is not None:
            result.success = run.success
            result.finished = run.declared_done
            result.steps = run.steps
            result.reason = run.reason
            result.evidence_dir = run.evidence_dir or ""
            if run.usage is not None:
                # Same key names the benchmark runner writes, so results files
                # from both paths aggregate identically.
                result.usage_totals = {
                    "request_tokens": run.usage.prompt_tokens,
                    "response_tokens": run.usage.completion_tokens,
                    "total_tokens": run.usage.total_tokens,
                }

        finished = await env.finish(
            session_id,
            declared_done=bool(run.declared_done) if run is not None else False,
            answer=run.answer if run is not None else None,
        )
        result.score = finished.score
        result.gated_score = finished.gated_score
        result.eval_reason = finished.eval_reason
    except ServiceError as exc:
        result.error = f"{exc.code}: {exc.message}"
    except Exception as exc:  # noqa: BLE001 - record and continue
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.duration_s = time.monotonic() - started
        if session_id is not None:
            try:
                await env.delete_session(session_id)
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning("Deleting session %s failed: %s", session_id, exc)
    return result


# Extra time the agent gets over the episode timeout before the harness gives
# up on it: the workflow enforces its own deadline, so this only guards against
# a wedged event loop.
_AGENT_GRACE_S = 60.0


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


async def run_harness_suite(
    env_url: str,
    agent_url: str,
    *,
    backend: Optional[str] = None,
    agent_id: Optional[str] = None,
    tasks: Optional[str] = None,
    task_index_range: Optional[str] = None,
    difficulty: Sequence[str] = (),
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    n_task_combinations: Optional[int] = None,
    output: Optional[str] = None,
    resume: bool = False,
    health_timeout: float = 900.0,
    task_timeout: float = 1000.0,
    capture_evidence: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: Optional[float] = None,
    parallel: int = 1,
    verbose: bool = False,
    debug: Optional[bool] = None,
    dry_run: bool = False,
    token: Optional[str] = None,
    agent_env_url_override: Optional[str] = None,
) -> bool:
    """Run a task suite across the agent and environment services."""
    env_url = env_url.rstrip("/")
    agent_url = agent_url.rstrip("/")
    _setup_benchmark_logging(bool(debug or False), verbose)

    if output is None:
        output = (
            f"runs/{_BENCHMARK_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    output_path = Path(output)
    evidence_root = output_path.parent / f"{output_path.stem}_evidence"

    env = EnvClient(env_url, token=token)
    agent = AgentClient(agent_url, token=token)
    run_id = _new_run_id()
    run_meta: Dict[str, Any] = {
        "benchmark": _BENCHMARK_NAME,
        "run_id": run_id,
        "env_url": env_url,
        "agent_url": agent_url,
        "agent_env_url": agent_env_url_override or env_url,
        "protocol_version": PROTOCOL_VERSION,
        "backend": backend,
        "agent_id": agent_id,
        "provider": provider,
        "model": model,
        "executor_kind": None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "updated_at": None,
        "task_index_range": task_index_range,
        "tasks_filter": tasks,
        "difficulty": list(difficulty),
        "seed": seed,
        "n_task_combinations": n_task_combinations,
        "resumed": resume,
        "parallel": parallel,
        "capture_evidence": capture_evidence,
        "results_path": str(output_path),
        "evidence_root": str(evidence_root) if capture_evidence else None,
    }

    try:
        if not await _wait_healthy(env, agent, health_timeout):
            return False

        try:
            health = await agent.health()
            run_meta["agent"] = health.get("agent")
            run_meta["executor_kind"] = health.get("executor_kind")
            run_meta["reasoning"] = health.get("reasoning")
            run_meta["vision_only"] = health.get("vision_only")
            # A dispatcher reports one row per worker; a worker reports itself.
            if health.get("agents"):
                run_meta["agents"] = health["agents"]
        except Exception as exc:  # noqa: BLE001 - provenance is best effort
            logger.debug("Could not read the agent's health block: %s", exc)

        catalogue = await env.list_tasks(backend=backend)
        if not catalogue:
            console.print("[red]The environment returned an empty task list.[/]")
            return False
        try:
            selected = _select_tasks(
                catalogue,
                backend=backend,
                names=tasks,
                task_index_range=task_index_range,
                difficulty=difficulty,
            )
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/]")
            return False

        run_meta["selected_count"] = len(selected)

        if dry_run:
            table = Table(title=f"Tasks to run ({len(selected)})")
            table.add_column("#", style="dim")
            table.add_column("Task", style="cyan")
            table.add_column("Backend", style="green")
            table.add_column("Steps", style="white")
            for index, task in enumerate(selected, start=1):
                table.add_row(
                    str(index), task.id, backend or "any free", str(task.max_steps)
                )
            console.print(table)
            console.print(f"Results would be written to [cyan]{output_path}[/]")
            return True

        if seed is not None or n_task_combinations is not None:
            if backend:
                targets = [backend]
            else:
                # A suite seed has to be applied to every device, otherwise
                # tasks run on the other backends use a different instance set.
                targets = [info.id for info in (await env.health()).backends]
            for target in targets:
                await env.reinitialize(
                    target,
                    seed=seed,
                    n_task_combinations=n_task_combinations,
                )

        overrides = AgentOverrides(
            provider=provider,
            model=model,
            base_url=base_url,
            api_base=api_base,
            temperature=temperature,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results: Dict[str, TaskResult] = {}
        run_history = _load_run_history(output_path) if output_path.exists() else []
        if resume and output_path.exists():
            results = _load_results(output_path)
            previous_meta = _load_run_meta(output_path)
            changed = [
                field
                for field in ("env_url", "agent_url", "provider", "model", "backend")
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

        pending: List[TaskSpec] = []
        for task in selected:
            recorded = results.get(_task_key(task))
            if resume and recorded is not None and recorded.error is None:
                console.print(f"[dim]{task.id} — already recorded, skipped[/]")
                continue
            pending.append(task)

        console.print(
            f"🏁 Running {len(pending)} task(s) "
            f"(env={env_url}, agent={agent_url}, parallel={parallel})"
        )

        write_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, parallel))
        completed = 0
        running: Dict[str, tuple[str, float]] = {}

        async def _worker(task: TaskSpec) -> None:
            nonlocal completed
            async with semaphore:
                key = _task_key(task)
                running[key] = (task.id, time.monotonic())
                console.print(
                    f"[dim]▶ {task.id} " f"(budget {steps or task.max_steps} steps)[/]"
                )
                try:
                    result = await _run_with_capacity_retry(
                        task,
                        env=env,
                        agent=agent,
                        agent_env_url=agent_env_url_override,
                        overrides=overrides,
                        steps=steps,
                        task_timeout=task_timeout,
                        seed=seed,
                        evidence=capture_evidence,
                        evidence_root=evidence_root,
                        backend=None if parallel > 1 else backend,
                        agent_id=agent_id,
                    )
                finally:
                    running.pop(key, None)
            key = result.key
            results[key] = result
            async with write_lock:
                completed += 1
                _write_results(output_path, run_meta, results, run_history)
                _print_result_line(completed, len(pending), result)

        async def _heartbeat() -> None:
            """A long episode is otherwise silent; report what is in flight."""
            while True:
                await asyncio.sleep(_HEARTBEAT_S)
                if not running:
                    continue
                now = time.monotonic()
                parts = [
                    f"{task_id} {_format_elapsed(now - started_at)}"
                    for task_id, started_at in running.values()
                ]
                console.print(
                    f"[dim]… {completed}/{len(pending)} done | "
                    f"running: {', '.join(parts)}[/]"
                )

        heartbeat = asyncio.create_task(_heartbeat())
        try:
            await asyncio.gather(*(_worker(task) for task in pending))
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

        run_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_results(output_path, run_meta, results, run_history)
        console.print(f"💾 Results written to [cyan]{output_path}[/]")
        if capture_evidence:
            console.print(f"🗂️  Evidence written to [cyan]{evidence_root}[/]")
        ordered = [
            results[_task_key(task)] for task in selected if _task_key(task) in results
        ]
        _print_summary(ordered)
        return True
    except asyncio.CancelledError:
        console.print("\n[yellow]⏹️ Harness interrupted; partial results were kept.[/]")
        raise
    finally:
        await env.close()
        await agent.close()


async def _run_with_capacity_retry(
    task: TaskSpec, *, env: EnvClient, **kwargs: Any
) -> TaskResult:
    """Retry a task while the environment reports no free backend."""
    deadline = time.monotonic() + float(kwargs.get("task_timeout", 1000.0))
    while True:
        result = await _run_one(task, env=env, **kwargs)
        if result.error is None or "capacity" not in result.error.lower():
            return result
        if time.monotonic() >= deadline:
            return result
        await asyncio.sleep(_CAPACITY_RETRY_S)


async def _wait_healthy(env: EnvClient, agent: AgentClient, timeout: float) -> bool:
    console.print(
        f"⏳ Waiting for services (env={env.base_url}, agent={agent.base_url}) ..."
    )
    deadline = time.monotonic() + timeout
    while True:
        env_ok = agent_ok = False
        try:
            env_ok = bool((await env.health()).ok)
        except Exception:  # noqa: BLE001 - reported by the loop
            env_ok = False
        try:
            agent_ok = bool((await agent.health()).get("ok"))
        except Exception:  # noqa: BLE001 - reported by the loop
            agent_ok = False
        if env_ok and agent_ok:
            console.print("✅ Services healthy")
            return True
        if time.monotonic() >= deadline:
            console.print(
                f"[red]Services did not become healthy within {timeout:.0f}s "
                f"(env={env_ok}, agent={agent_ok}).[/]"
            )
            return False
        await asyncio.sleep(5.0)


def _format_elapsed(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{rest:02d}s"


def _print_result_line(index: int, total: int, result: TaskResult) -> None:
    status = "green" if (result.gated_score or 0.0) > 0.5 else "red"
    if result.error is not None:
        console.print(
            f"[red][{index}/{total}] {result.task_type} "
            f"ERROR: {result.error[:160]} ({result.duration_s:.1f}s)[/]"
        )
        return
    console.print(
        f"[{status}][{index}/{total}] {result.task_type} "
        f"score={result.score:.2f} gated={result.gated_score:.2f} "
        f"steps={result.steps} finished={result.finished} "
        f"({result.duration_s:.1f}s)[/]"
    )
