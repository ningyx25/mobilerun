"""MobileGym benchmark runner.

Runs MobileGym's task suite with the mobilerun agent ``in process``::

    TaskRegistry/split -> Controller.setup (grounded) -> MobileAgent episode
    -> final state snapshot -> Evaluator (official JSON judge) -> TaskResult

Unlike AndroidWorld / MobileWorld, MobileGym has no environment server: the
simulator is a React SPA served over HTTPS (``https://localhost:4180``) and all
control flows through Playwright ``page.evaluate()`` of ``window.__SIM__`` and
friends. ``bench_env`` (from the sibling ``mobilegym`` repository) is therefore
imported lazily and must be reachable on ``sys.path`` -- set::

    PYTHONPATH=/path/to/mobilegym

Grounded mode is used throughout: ``Controller.setup`` injects the AnswerSheet
app state and ``Evaluator`` judges from the resulting JSON state, no VLM and no
HTTP score endpoint involved. The simulator exposes no accessibility tree, so
this benchmark is vision-only.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import ManagerPlanEvent
from mobilerun.agent.external import list_agents
from mobilerun.agent.utils.llm_loader import load_agent_llms
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.benchmark.android_world import (
    _infra_error_reason,
    TaskResult,
    _finalize_episode_evidence,
    _load_results,
    _load_run_history,
    _load_run_meta,
    _new_run_id,
    _parse_task_index_range,
    _print_summary,
    _sanitize_name,
    _select_tasks,
    _setup_benchmark_logging,
    _write_results,
)
from mobilerun.benchmark.evidence import EpisodeEvidence, gather_ui_tree
from mobilerun.config_manager import ConfigLoader
from mobilerun.tools.driver.mobile_gym import (
    MOBILE_GYM_CONNECTION,
    MOBILE_GYM_DEFAULT_URL,
    MobileGymBrowserDriver,
    validate_mobile_gym_url,
)

logger = logging.getLogger("mobilerun")
console = Console()

_BENCHMARK_NAME = "mobilegym"
_DEFAULT_MAX_STEPS = 30
_DEFAULT_SIM_URL = MOBILE_GYM_DEFAULT_URL

# bench_env classifies difficulty as L1..L4; runbench's --difficulty uses the
# easy/medium/hard vocabulary shared with the other two benchmarks.
_DIFFICULTY_MAP = {"L1": "easy", "L2": "medium", "L3": "hard", "L4": "hard"}

# Provenance fields that, when changed, make resumed records incomparable.
_MOBILEGYM_RESUME_FIELDS = (
    "server_url",
    "split",
    "provider",
    "model",
    "executor_kind",
)


def _import_bench_env() -> SimpleNamespace:
    """Import the optional ``bench_env`` package with an actionable error."""
    try:
        import bench_env  # noqa: F401
        from bench_env.env.mobile_gym import MobileGymEnv
        from bench_env.runner.base import Controller, Evaluator, ExecutionResult
        from bench_env.splits import base_task_id, resolve_split
        from bench_env.task.registry import TaskRegistry
    except ImportError as exc:
        raise ImportError(
            "MobileGym benchmark requires the 'bench_env' package from the "
            "mobilegym repository. Add it to sys.path, e.g. "
            "export PYTHONPATH=/home/swaydy/docker-proj/test-some-idea/mobilegym"
        ) from exc
    return SimpleNamespace(
        MobileGymEnv=MobileGymEnv,
        Controller=Controller,
        Evaluator=Evaluator,
        ExecutionResult=ExecutionResult,
        base_task_id=base_task_id,
        resolve_split=resolve_split,
        TaskRegistry=TaskRegistry,
    )


async def _wait_for_sim(page: Any, timeout: float = 60.0) -> None:
    """Poll ``window.__SIM__.getState`` until the simulator is initialized."""
    deadline = time.monotonic() + max(1.0, timeout)
    while True:
        try:
            ready = await page.evaluate(
                "() => Boolean(window.__SIM__ && "
                "typeof window.__SIM__.getState === 'function' && "
                "window.__SIM__.getState())"
            )
            if ready:
                return
        except Exception:  # noqa: BLE001 - page still loading
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"MobileGym simulator did not become ready within {timeout:.0f}s"
            )
        await asyncio.sleep(0.5)


def _load_task_list(
    be: SimpleNamespace,
    *,
    split: Optional[str],
    tasks: Optional[str],
    task_index_range: Optional[str],
    difficulty: Sequence[str],
) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
    """Discover MobileGym tasks, apply the split, then the runbench filters."""
    discovered = be.TaskRegistry().discover_all()
    if split:
        split_ids = be.resolve_split(split)
        discovered = [t for t in discovered if be.base_task_id(t.id) in split_ids]
    if not discovered:
        raise ValueError(
            "No MobileGym tasks found (check --split and PYTHONPATH/bench_env)."
        )

    metadata: Dict[str, Dict[str, Any]] = {
        task.id: {
            "difficulty": _DIFFICULTY_MAP.get(str(getattr(task, "difficulty", "")), ""),
            "max_steps": int(getattr(task, "max_steps", None) or _DEFAULT_MAX_STEPS),
            "apps": ",".join(getattr(task, "apps", []) or []),
        }
        for task in discovered
    }
    selected = _select_tasks(
        list(metadata),
        tasks=tasks,
        task_index_range=task_index_range,
        difficulty=difficulty,
        metadata=metadata,
    )
    by_id = {task.id: task for task in discovered}
    return [by_id[name] for name in selected], metadata


async def run_mobilegym_benchmark(
    sim_url: Optional[str] = None,
    config_path: Optional[str] = None,
    tasks: Optional[str] = None,
    task_index_range: Optional[str] = None,
    difficulty: Sequence[str] = (),
    split: str = "test",
    steps: Optional[int] = None,
    output: Optional[str] = None,
    resume: bool = False,
    headless: bool = True,
    startup_timeout: float = 120.0,
    task_timeout: float = 600.0,
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
    """Run the MobileGym suite with the mobilerun agent.

    Returns True when the run completed (individual task failures included),
    False on fatal errors (missing bench_env, unhealthy simulator, bad
    selection, interrupt).
    """
    try:
        sim_url = validate_mobile_gym_url(sim_url or _DEFAULT_SIM_URL)
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
    # This benchmark always drives the in-process MobileGym simulator.
    config.device.control_backend = MOBILE_GYM_CONNECTION
    config.device.serial = sim_url
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

    try:
        be = _import_bench_env()
    except ImportError as exc:
        console.print(f"[red]Error: {exc}[/]")
        return False

    try:
        selected_tasks, metadata = _load_task_list(
            be,
            split=split,
            tasks=tasks,
            task_index_range=task_index_range,
            difficulty=difficulty,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Error: {exc}[/]")
        return False

    run_id = _new_run_id()
    run_meta: Dict[str, Any] = {
        "benchmark": _BENCHMARK_NAME,
        "run_id": run_id,
        "server_url": sim_url,
        "split": split,
        "headless": headless,
        "agent": config.agent.name,
        "reasoning": config.agent.reasoning,
        "executor_kind": config.agent.executor.kind,
        "provider": provider,
        "model": model,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "updated_at": None,
        "task_index_range": task_index_range,
        "tasks_filter": tasks,
        "difficulty": list(difficulty),
        "resumed": resume,
        "results_path": str(output_path),
        "capture_evidence": capture_evidence,
        "evidence_root": str(evidence_root) if capture_evidence else None,
        "selected_count": len(selected_tasks),
    }
    run_history = _load_run_history(output_path) if output_path.exists() else []
    run_context = {
        key: run_meta.get(key)
        for key in (
            "run_id",
            "benchmark",
            "server_url",
            "split",
            "agent",
            "provider",
            "model",
            "task_index_range",
            "tasks_filter",
            "difficulty",
            "started_at",
        )
    }

    if dry_run:
        table = Table(title=f"Tasks to run ({len(selected_tasks)})")
        table.add_column("#", style="dim")
        table.add_column("Task", style="cyan")
        table.add_column("Difficulty", style="green")
        table.add_column("Apps", style="white")
        table.add_column("Max steps", style="white")
        for index, task in enumerate(selected_tasks, start=1):
            entry = metadata[task.id]
            table.add_row(
                str(index),
                task.id,
                entry["difficulty"],
                entry["apps"],
                str(entry["max_steps"]),
            )
        console.print(table)
        console.print(f"Results would be written to [cyan]{output_path}[/]")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- LLMs: load once, share across every episode ------------------------
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

    env: Any = None
    driver: Optional[MobileGymBrowserDriver] = None
    try:
        console.print(f"⏳ Starting MobileGym simulator at [cyan]{sim_url}[/] ...")
        env = be.MobileGymEnv(
            url=sim_url,
            headless=headless,
            coord_space="physical",
            delay_after_action=0.0,
            viewport_size=(360, 800),
            physical_size=(1080, 2400),
            device_scale_factor=3,
            verbose=bool(verbose),
        )
        await env.start()
        await env.wait_ready(timeout_ms=int(startup_timeout * 1000))
        await _wait_for_sim(env.page, startup_timeout)
        console.print("✅ Simulator ready")

        # One env/driver shared by every episode; MobileAgent never connects or
        # closes an injected driver, and task.setup() resets the simulator.
        driver = MobileGymBrowserDriver(env=env)
        await driver.connect()

        results: Dict[str, TaskResult] = {}
        if resume and output_path.exists():
            results = _load_results(output_path)
            previous_meta = _load_run_meta(output_path)
            changed = [
                field
                for field in _MOBILEGYM_RESUME_FIELDS
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

        total = len(selected_tasks)
        console.print(f"🏁 Running {total} task(s) on {sim_url}")
        for index, task in enumerate(selected_tasks, start=1):
            key = f"{task.id}#0"
            recorded = results.get(key)
            if resume and recorded is not None and recorded.error is None:
                console.print(
                    f"[dim][{index}/{total}] {task.id} — already recorded, skipped[/]"
                )
                continue

            result = TaskResult(
                task_type=task.id,
                task_idx=0,
                difficulty=metadata[task.id]["difficulty"],
                timestamp=datetime.now().isoformat(timespec="seconds"),
                run_id=run_id,
                server_url=sim_url,
            )
            task_started = time.monotonic()
            # Stale answers must not leak into this episode's judge input.
            driver._answer = ""
            initialized = False
            evidence: Optional[EpisodeEvidence] = None
            agent: Optional[MobileAgent] = None
            ui_tree_before: Optional[Dict[str, Any]] = None
            ui_tree_after: Optional[Dict[str, Any]] = None
            if capture_evidence:
                evidence = EpisodeEvidence(
                    evidence_root / f"{_sanitize_name(task.id)}_0", task.id, 0
                )
            try:
                # Grounded setup: task.setup() resets/warms/open apps/samples
                # and Controller injects apps.answer_sheet into the state.
                init_obs, _params = await be.Controller.setup(
                    env, task, eval_mode="grounded"
                )
                initialized = True
                result.goal = task.description

                max_steps = (
                    steps if steps is not None else metadata[task.id]["max_steps"]
                )
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
                    # MobileGym has no a11y tree; one attempt is enough.
                    ui_tree_before = await gather_ui_tree(driver, tries=1)
                handler = agent.run()
                manager_answered = False
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
                # Transient provider failures become retriable errors so
                # --resume re-runs the episode instead of keeping the 0.
                infra_reason = _infra_error_reason(run_result)
                if infra_reason:
                    result.error = infra_reason
                # Same completion gate as the other two benchmarks: the score
                # only counts when the agent declared the episode finished.
                if task_config.agent.reasoning:
                    declared_done = manager_answered
                else:
                    declared_done = agent.shared_state.success is not None
                result.finished = declared_done

                answer = str(
                    getattr(agent.shared_state, "answer", "") or run_result.reason or ""
                ).strip()
                if answer:
                    await driver.submit_answer(answer)

                # step() returns observations with state={}; the judge needs a
                # full final state, so re-read it explicitly before scoring.
                state = await env.get_state(
                    required_apps=list(getattr(task, "apps", []) or [])
                )
                last_obs = await env.get_observation()
                last_obs = replace(last_obs, state=state)

                exec_result = be.ExecutionResult(
                    steps=int(result.steps or 0),
                    trace=[],
                    runtime_s=time.monotonic() - task_started,
                    finished=declared_done,
                    truncated=False,
                    agent_answer=driver._answer or getattr(env, "agent_answer", None),
                )
                judge = await be.Evaluator(
                    judge_mode="state", eval_mode="grounded"
                ).evaluate(task, init_obs, last_obs, exec_result)
                result.score = 1.0 if judge.passed else 0.0
                result.eval_reason = json.dumps(
                    {
                        "success": judge.success,
                        "clean": judge.clean,
                        "progress": judge.progress,
                        "issues": judge.issues,
                        "warnings": judge.warnings,
                    },
                    ensure_ascii=False,
                    default=str,
                )[:2000]
                if judge.judge_error:
                    result.error = f"JudgeError: {judge.judge_error}"
                result.gated_score = result.score if declared_done else 0.0
            except Exception as exc:  # noqa: BLE001 - record and continue
                result.error = f"{type(exc).__name__}: {exc}"
            finally:
                result.duration_s = time.monotonic() - task_started
                if evidence is not None and ui_tree_after is None:
                    ui_tree_after = await gather_ui_tree(driver, tries=1)
                if initialized:
                    try:
                        task.teardown(env)
                    except Exception as exc:  # noqa: BLE001 - best effort
                        console.print(
                            f"[yellow]⚠️ teardown for {task.id} failed: {exc}[/]"
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
                    f"[red][{index}/{total}] {task.id} ({result.difficulty}) "
                    f"ERROR: {result.error[:160]} ({result.duration_s:.1f}s)[/]"
                )
            else:
                console.print(
                    f"[{status}][{index}/{total}] {task.id} ({result.difficulty}) "
                    f"score={result.score:.2f} gated={result.gated_score:.2f} "
                    f"steps={result.steps} finished={result.finished} "
                    f"({result.duration_s:.1f}s)[/]"
                )

        run_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_results(output_path, run_meta, results, run_history)
        console.print(f"💾 Results written to [cyan]{output_path}[/]")
        if capture_evidence:
            console.print(f"🗂️  Evidence written to [cyan]{evidence_root}[/]")

        ordered = [results[f"{task.id}#0"] for task in selected_tasks]
        _print_summary(ordered, "MobileGym Benchmark Summary")
        return True
    except asyncio.CancelledError:
        console.print(
            "\n[yellow]⏹️ Benchmark interrupted; partial results were kept.[/]"
        )
        raise
    finally:
        if driver is not None:
            await driver.close()
        if env is not None:
            await env.close()
