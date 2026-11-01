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
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich.console import Console
from rich.markup import escape
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
    EpisodeResult,
    EpisodeState,
    TaskSpec,
)

logger = logging.getLogger("mobilerun")
console = Console()

_BENCHMARK_NAME = "harness"
_CAPACITY_RETRY_S = 5.0


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
    run_id: str = "",
    verbose: bool = False,
    debug: bool = False,
) -> TaskResult:
    """Run one task end to end and return its result row."""
    task_type, _, idx_text = task.id.partition("#")
    task_idx = int(idx_text or 0)
    result = TaskResult(
        task_type=task_type,
        task_idx=task_idx,
        difficulty=str(task.metadata.get("difficulty", "")),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        run_id=run_id,
        server_url=str(env.base_url),
        agent_id=agent_id or "",
    )
    started = time.monotonic()
    session_id: Optional[str] = None
    episode_id: Optional[str] = None
    progress_task: Optional[asyncio.Task] = None
    try:
        session = await env.create_session(
            task.id.partition("#")[0],
            # ``None`` lets the environment lease any free backend. Falling
            # back to ``task.backend`` here would pin every task to the first
            # backend in the catalogue and serialise parallel runs.
            backend=backend,
            task_idx=task_idx,
            # The suite seed is applied once before the loop. Passing it per
            # session makes AndroidWorldBackend.prepare reinitialize the whole
            # suite for every task.
            seed=None,
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
        if debug:
            # Live agent events, rendered like `runbench --debug`.
            progress_task = asyncio.create_task(
                _stream_episode_progress(agent, episode_id, task_type)
            )

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
        if (verbose or debug) and run is not None:
            # Same info-level line runbench emits when the workflow finalizes.
            _print_goal_line(task_type, run)
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

        if result.error is None:
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
        if progress_task is not None:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
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
    evidence_dir: Optional[str] = None,
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
    evidence_root = (
        Path(evidence_dir)
        if evidence_dir
        else output_path.parent / f"{output_path.stem}_evidence"
    )

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

        catalogue = await _list_tasks(env, backend)
        if catalogue is None:
            return False
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
            try:
                for target in targets:
                    await env.reinitialize(
                        target,
                        seed=seed,
                        n_task_combinations=n_task_combinations,
                    )
            except (ServiceError, ConnectionError) as exc:
                console.print(f"[red]Could not reinitialize the suite: {exc}[/]")
                return False

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
                for field in (
                    "env_url",
                    "agent_url",
                    "agent_id",
                    "provider",
                    "model",
                    "backend",
                    "seed",
                    "n_task_combinations",
                )
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
        total = len(selected)
        for index, task in enumerate(selected, start=1):
            recorded = results.get(_task_key(task))
            if resume and recorded is not None and recorded.error is None:
                console.print(
                    f"[dim][{index}/{total}] {task.id} — already recorded, skipped[/]"
                )
                continue
            pending.append(task)

        console.print(
            f"🏁 Running {len(pending)} task(s) "
            f"(env={env_url}, agent={agent_url}, parallel={parallel})"
        )

        write_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(max(1, parallel))
        completed = 0

        async def _worker(task: TaskSpec) -> None:
            nonlocal completed
            async with semaphore:
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
                    backend=backend,
                    agent_id=agent_id,
                    run_id=run_meta.get("run_id") or "",
                    verbose=verbose,
                    debug=bool(debug),
                )
            key = result.key
            _write_episode_manifest(result, run_meta)
            results[key] = result
            async with write_lock:
                completed += 1
                _write_results(output_path, run_meta, results, run_history)
                _print_result_line(completed, len(pending), result)

        await asyncio.gather(*(_worker(task) for task in pending))

        run_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_results(output_path, run_meta, results, run_history)
        console.print(f"💾 Results written to [cyan]{output_path}[/]")
        if capture_evidence:
            console.print(f"🗂️  Evidence written to [cyan]{evidence_root}[/]")
        ordered = [
            results[_task_key(task)] for task in selected if _task_key(task) in results
        ]
        _print_summary(ordered)
        # A run where every task failed is not a successful invocation.
        ran = [row for row in results.values() if row.run_id == run_meta["run_id"]]
        return not (ran and all(row.error is not None for row in ran))
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


async def _list_tasks(
    env: EnvClient, backend: Optional[str]
) -> Optional[List[TaskSpec]]:
    try:
        return await env.list_tasks(backend=backend)
    except (ServiceError, ConnectionError) as exc:
        console.print(f"[red]Could not read the task list: {exc}[/]")
        return None


async def _wait_healthy(env: EnvClient, agent: AgentClient, timeout: float) -> bool:
    console.print(
        f"⏳ Waiting for services (env={env.base_url}, agent={agent.base_url}) ..."
    )
    deadline = time.monotonic() + timeout
    while True:
        env_ok = agent_ok = False
        try:
            env_health = await env.health()
            # ``ok`` alone is unconditional; a service whose every backend is
            # down cannot run anything.
            env_ok = bool(env_health.ok) and any(
                info.state != "down" for info in env_health.backends
            )
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


def _write_episode_manifest(result: TaskResult, run_meta: Dict[str, Any]) -> None:
    """Write the reference-format ``episode.json`` into the episode's bundle.

    The agent service writes the conversation, timeline, trees and trajectory,
    but only the runner knows the score, so the manifest is completed here.
    Best effort: a remote agent's filesystem may not be reachable, and that
    must not fail the run.
    """
    if not result.evidence_dir:
        return
    bundle = Path(result.evidence_dir)
    if not bundle.is_dir():
        return

    # The timeline sums every model call; the episode result only carries the
    # usage attached to decisions. Prefer the timeline, like the reference.
    usage_totals = result.usage_totals or None
    timeline_exists = (bundle / "timeline.json").exists()
    if timeline_exists:
        try:
            timeline = json.loads((bundle / "timeline.json").read_text())
            usage_totals = timeline.get("usage_totals") or usage_totals
            if result.usage_totals is not None:
                # runbench rows carry the request count too; the summary sums it.
                result.usage_totals["requests"] = int(
                    (usage_totals or {}).get("requests", 0)
                )
        except Exception:  # noqa: BLE001 - evidence is best effort
            pass

    payload = {
        "task_type": result.task_type,
        "task_idx": result.task_idx,
        "difficulty": result.difficulty,
        "goal": result.goal,
        "score": result.score,
        "gated_score": result.gated_score,
        "success": result.success,
        "finished": result.finished,
        "steps": result.steps,
        "reason": result.reason,
        "eval_reason": result.eval_reason,
        "duration_s": result.duration_s,
        "error": result.error,
        "timestamp": result.timestamp,
        "run_id": run_meta.get("run_id"),
        "server_url": result.server_url,
        "complexity": result.complexity,
        "start_on_home_screen": result.start_on_home_screen,
        "max_steps": result.max_steps,
        "evidence_dir": str(bundle),
        "trajectory_dir": ("trajectory" if (bundle / "trajectory").is_dir() else None),
        "usage_totals": usage_totals,
        "conversation_path": (
            "conversation.json" if (bundle / "conversation.json").exists() else None
        ),
        "timeline_path": "timeline.json" if timeline_exists else None,
        "ui_tree_paths": [
            name
            for name in ("ui_tree_before.json", "ui_tree_after.json")
            if (bundle / name).exists()
        ],
        "run": run_meta,
    }
    try:
        target = bundle / "episode.json"
        temp = target.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        temp.replace(target)
    except Exception as exc:  # noqa: BLE001 - evidence is best effort
        logger.warning(
            "Writing the episode manifest for %s failed: %s", result.task_type, exc
        )


def _preview(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def _print_goal_line(task_type: str, run: EpisodeResult) -> None:
    """The info-level finalize line runbench prints per task."""
    ok = bool(run.success)
    console.print(
        f"[dim]{task_type}[/] [{'green' if ok else 'red'}]"
        f"{'🎉 Goal achieved' if ok else '❌ Goal failed'}: {escape(run.reason)}[/]"
    )


async def _stream_episode_progress(
    agent: AgentClient, episode_id: str, label: str
) -> None:
    try:
        async for event in agent.stream_events(episode_id):
            _render_episode_event(label, event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - progress is best effort
        logger.debug("Progress stream for %s ended: %s", episode_id, exc)


def _render_episode_event(label: str, event: Dict[str, Any]) -> None:
    """Render one SSE frame with runbench's event vocabulary.

    Two agent services can sit behind ``--agent``: agent-harness emits the
    harness names (``llm_response``/``action_result``/``agent_end``); the
    mobilerun worker forwards workflow class names. Both are rendered here.
    """
    kind = event.get("type")
    data = event.get("data") or {}
    prefix = f"[dim]{label}[/] "

    if kind in ("llm_response", "FastAgentResponseEvent"):
        thought = str(data.get("thought") or "").strip()
        if thought:
            console.print(
                f"{prefix}[cyan]🧠 Thinking: {escape(_preview(thought, 150))}[/]"
            )
        code = data.get("code")
        if code:
            console.print(f"{prefix}[yellow]💻 Executing action code[/]")
            console.print(f"{prefix}[blue]{escape(str(code))}[/]")
    elif kind == "ManagerPlanEvent":
        thought = str(data.get("thought") or "").strip()
        if thought:
            console.print(
                f"{prefix}[cyan]🧠 Thinking: {escape(_preview(thought, 150))}[/]"
            )
        subgoal = str(data.get("current_subgoal") or "").strip()
        if subgoal:
            console.print(
                f"{prefix}[yellow]📋 Next step: {escape(_preview(subgoal, 150))}[/]"
            )
        answer = str(data.get("answer") or "").strip()
        if answer:
            console.print(
                f"{prefix}[green]💬 Answer: {escape(_preview(answer, 200))}[/]"
            )
    elif kind in (
        "action_result",
        "ExecutorActionResultEvent",
        "GroundedActionResultEvent",
    ):
        success = data.get("ok", data.get("success"))
        summary = str(data.get("summary") or "")
        if success:
            console.print(
                f"{prefix}[green]⚡ Action result: {escape(_preview(summary, 100))}[/]"
            )
        else:
            error = str(data.get("error") or summary)
            console.print(
                f"{prefix}[red]❌ Action error: {escape(_preview(error, 100))}[/]"
            )
    elif kind == "FastAgentOutputEvent":
        output = str(data.get("output") or "")
        if "Error" in output or "Exception" in output:
            console.print(
                f"{prefix}[red]❌ Action error: {escape(_preview(output, 100))}[/]"
            )
        else:
            console.print(
                f"{prefix}[green]⚡ Action result: {escape(_preview(output, 100))}[/]"
            )
    elif kind in ("agent_end", "FastAgentEndEvent"):
        ok = bool(data.get("success"))
        console.print(
            f"{prefix}[{'green' if ok else 'red'}]"
            f"■ {'done' if ok else 'failed'}: {escape(str(data.get('reason') or ''))}[/]"
        )


def _print_result_line(index: int, total: int, result: TaskResult) -> None:
    status = "green" if (result.gated_score or 0.0) > 0.5 else "red"
    if result.error is not None:
        console.print(
            f"[red][{index}/{total}] {result.task_type} ({result.difficulty}) "
            f"ERROR: {result.error[:160]} ({result.duration_s:.1f}s)[/]"
        )
        return
    console.print(
        f"[{status}][{index}/{total}] {result.task_type} ({result.difficulty}) "
        f"score={result.score:.2f} gated={result.gated_score:.2f} "
        f"steps={result.steps} finished={result.finished} "
        f"({result.duration_s:.1f}s)[/]"
    )
