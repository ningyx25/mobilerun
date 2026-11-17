"""Shared plumbing for the benchmark runners.

Canonical home of the result model, results-file IO, task selection,
episode-evidence finalization, benchmark logging and the summary table.
``android_world``, ``mobile_world`` and ``mobile_gym`` are sibling runners:
they import from here, never from each other.

Dependency direction: this module must not import any benchmark runner or
``mobilerun.harness``. The runners pull in the harness/agent stack, so
importing it here would invert the dependency and recreate the import cycle
that ``harness.env_service.scoring`` already works around.
"""

from __future__ import annotations

import difflib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from filelock import FileLock
from rich.console import Console
from rich.table import Table

from mobilerun.log_handlers import CLILogHandler, configure_logging

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.benchmark.evidence import EpisodeEvidence

logger = logging.getLogger("mobilerun")
console = Console()

# Request timeout for score/evaluation calls, shared by the benchmark clients.
_SCORE_TIMEOUT = 120.0

# Lock timeout for merge-writing the results file.
_RESULTS_LOCK_TIMEOUT_S = 30.0

# Retry policy for idempotent HTTP calls.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 5.0
_RETRY_STATUSES = frozenset({500, 502, 503, 504})


@dataclass
class TaskResult:
    """Recorded outcome of one task episode."""

    task_type: str
    task_idx: int = 0
    difficulty: str = ""
    goal: str = ""
    # Server-side score from live device state, and the gated score (0 unless
    # the agent declared the episode finished).
    score: Optional[float] = None
    gated_score: Optional[float] = None
    success: Optional[bool] = None  # agent-declared success (ResultEvent)
    finished: Optional[bool] = None  # agent declared completion (complete tool)
    steps: Optional[int] = None
    reason: str = ""
    # Server-side explanation of the score (mobileworld's /task/eval returns
    # one; androidworld's /task/score does not).
    eval_reason: str = ""
    duration_s: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = ""
    # -- provenance: which invocation produced this row (a results file can
    # accumulate rows from several sharded / resumed runs) --
    run_id: str = ""
    server_url: str = ""
    # Which registered agent ran the episode (dispatcher mode only).
    agent_id: str = ""
    # Task-level attempts made by the invocation that produced this row
    # (1 = no retry). Capacity polling inside the harness runner is not counted.
    attempts: int = 1
    # -- evidence (all optional; populated when evidence_dir is enabled) --
    complexity: Optional[float] = None
    start_on_home_screen: Optional[bool] = None
    max_steps: Optional[int] = None
    evidence_dir: str = ""
    trajectory_dir: str = ""
    usage_totals: Optional[Dict[str, int]] = None
    conversation_path: Optional[str] = None
    timeline_path: Optional[str] = None
    ui_tree_paths: List[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.task_type}#{self.task_idx}"


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------


def _parse_task_index_range(value: str, total: Optional[int] = None) -> tuple[int, int]:
    """Parse a 1-based inclusive START-END range over the task list.

    With *total* set, the end bound is validated against the suite size;
    format-only validation (*total* None) can run before the server is up.
    """
    parts = value.strip().split("-")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(f"Invalid --task-index-range '{value}'; expected START-END.")
    start, end = int(parts[0]), int(parts[1])
    if start < 1 or end < start:
        raise ValueError(
            f"Invalid --task-index-range '{value}'; expected 1 <= START <= END."
        )
    if total is not None and end > total:
        raise ValueError(
            f"--task-index-range '{value}' exceeds the {total} tasks in the suite."
        )
    return start, end


def _select_tasks(
    available: Sequence[str],
    *,
    tasks: Optional[str] = None,
    task_index_range: Optional[str] = None,
    difficulty: Sequence[str] = (),
    metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """Filter a benchmark task list by range, names and difficulty."""
    selected = list(available)

    if task_index_range:
        start, end = _parse_task_index_range(task_index_range, len(available))
        selected = available[start - 1 : end]

    if tasks:
        requested = [name.strip() for name in tasks.split(",") if name.strip()]
        unknown = [name for name in requested if name not in available]
        if unknown:
            hints = []
            for name in unknown:
                matches = difflib.get_close_matches(
                    name, list(available), n=3, cutoff=0.5
                )
                if matches:
                    hints.append(f"'{name}' (did you mean: {', '.join(matches)}?)")
                else:
                    hints.append(f"'{name}'")
            raise ValueError(f"Unknown task(s): {', '.join(hints)}")
        wanted = set(requested)
        selected = [name for name in selected if name in wanted]

    if difficulty:
        wanted = set(difficulty)
        metadata = metadata or {}
        selected = [
            name
            for name in selected
            if metadata.get(name, {}).get("difficulty") in wanted
        ]

    if not selected:
        raise ValueError("Task selection is empty.")
    return selected


# ---------------------------------------------------------------------------
# Results IO
# ---------------------------------------------------------------------------


def _read_results_payload(path: Path) -> Dict[str, Any]:
    """Raw payload of an existing results file, or {} if unreadable/absent."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _build_summary(results: Dict[str, TaskResult]) -> Dict[str, Any]:
    """Aggregate metrics over every recorded task."""
    tasks = list(results.values())
    scored = [r for r in tasks if r.score is not None]
    scored_finished = [r for r in scored if r.finished]
    usage: Dict[str, int] = {}
    for result in tasks:
        for key, value in (result.usage_totals or {}).items():
            usage[key] = usage.get(key, 0) + int(value or 0)
    return {
        "tasks": len(tasks),
        "scored": len(scored),
        "errors": sum(1 for r in tasks if r.error is not None),
        "mean_score": (sum(r.score for r in scored) / len(scored)) if scored else 0.0,
        "mean_gated_score": (
            (sum(r.gated_score or 0.0 for r in scored) / len(scored)) if scored else 0.0
        ),
        "completion_rate": (len(scored_finished) / len(scored)) if scored else 0.0,
        # Sum of episode durations: unlike wall clock it is unaffected by
        # --resume, so it describes the whole file (matches the reference
        # runner's total_runtime_s).
        "total_runtime_s": sum(r.duration_s or 0.0 for r in tasks),
        "total_tokens": usage.get("total_tokens", 0),
        "total_requests": usage.get("requests", 0),
    }


# Keys that describe the *file* rather than a single invocation; excluded from
# the per-run record in ``runs[]``.
_FILE_LEVEL_META_KEYS = frozenset({"runs", "summary", "tasks"})


def _run_record(
    run_meta: Dict[str, Any], results: Dict[str, TaskResult]
) -> Dict[str, Any]:
    """One ``runs[]`` entry: what this invocation was and what it produced."""
    record = {k: v for k, v in run_meta.items() if k not in _FILE_LEVEL_META_KEYS}
    run_id = record.get("run_id")
    record["ran_count"] = sum(1 for r in results.values() if r.run_id == run_id)
    return record


def _runs_from_payload(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Invocation history recorded in a results payload.

    Files written before ``runs[]`` existed have only the top-level metadata
    of whichever invocation wrote last; synthesize one legacy entry from it so
    that origin is not lost when the file is next merged.
    """
    runs = data.get("runs")
    if isinstance(runs, list) and runs:
        return [r for r in runs if isinstance(r, dict)]
    if not data:
        return []
    legacy = {
        k: v
        for k, v in data.items()
        if k not in _FILE_LEVEL_META_KEYS and k != "updated_at"
    }
    legacy.setdefault("run_id", "")
    legacy["legacy"] = True
    return [legacy]


def _load_run_history(path: Path) -> List[Dict[str, Any]]:
    """Invocation history of an existing results file ([] when absent)."""
    return _runs_from_payload(_read_results_payload(path))


def _merge_runs(
    prior_runs: List[Dict[str, Any]], current: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Keep every prior invocation exactly once, current last and up to date.

    Earlier entries win over later duplicates so a prior run's own final write
    is not clobbered by a stale snapshot; the current run is always replaced by
    the caller's (newest) copy.
    """
    run_id = current.get("run_id")
    merged: List[Dict[str, Any]] = []
    seen: set = set()
    for record in prior_runs:
        record_id = record.get("run_id")
        if run_id is not None and record_id == run_id:
            continue  # re-appended below with the latest state
        if record_id and record_id in seen:
            continue
        if record_id:
            seen.add(record_id)
        merged.append(record)
    merged.append(current)
    return merged


def _aggregate_times(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """File-level start/finish across every invocation in ``runs``."""
    starts = [r.get("started_at") for r in runs if r.get("started_at")]
    finishes = [r.get("finished_at") for r in runs if r.get("finished_at")]
    return {
        "started_at": min(starts) if starts else None,
        # Only truly finished runs count, so an incremental write mid-run
        # cannot move the file's end time forward.
        "finished_at": max(finishes) if finishes else None,
    }


def _write_results(
    path: Path,
    run_meta: Dict[str, Any],
    results: Dict[str, TaskResult],
    runs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Merge-write the results file under a lock.

    The file accumulates: tasks on disk that this invocation did not produce
    are preserved (so sharded / concurrent runs cannot clobber each other),
    and the invocation history in ``runs[]`` keeps every run that contributed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=_RESULTS_LOCK_TIMEOUT_S):
        disk = _read_results_payload(path)
        merged_results: Dict[str, TaskResult] = {}
        names = {f.name for f in fields(TaskResult)}
        for entry in disk.get("tasks", []) or []:
            if isinstance(entry, dict):
                row = TaskResult(**{k: v for k, v in entry.items() if k in names})
                merged_results[row.key] = row
        merged_results.update(results)  # this invocation's rows win

        # Disk history first so a prior invocation's own final write wins over
        # our (possibly stale) snapshot; the current run is appended last.
        merged_runs = _merge_runs(
            [*_runs_from_payload(disk), *(runs or [])],
            _run_record(run_meta, merged_results),
        )

        payload = dict(run_meta)
        payload.update(_aggregate_times(merged_runs))
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        payload["runs"] = merged_runs
        payload["summary"] = _build_summary(merged_results)
        payload["tasks"] = [
            asdict(result)
            for result in sorted(
                merged_results.values(), key=lambda r: (r.task_type, r.task_idx)
            )
        ]
        _write_json(path, payload)


def _load_results(path: Path) -> Dict[str, TaskResult]:
    data = _read_results_payload(path)
    names = {field.name for field in fields(TaskResult)}
    results: Dict[str, TaskResult] = {}
    for entry in data.get("tasks", []):
        result = TaskResult(**{k: v for k, v in entry.items() if k in names})
        results[result.key] = result
    return results


def _load_run_meta(path: Path) -> Dict[str, Any]:
    """Run-level metadata from an existing results file (for resume checks)."""
    return {
        k: v
        for k, v in _read_results_payload(path).items()
        if k not in _FILE_LEVEL_META_KEYS
    }


def _sanitize_name(value: str) -> str:
    """Filesystem-safe component for an evidence directory name."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _infra_error_reason(run_result: Any) -> str | None:
    """Infra-failure reason from an episode result, if the episode died on one.

    A transient provider failure (429 cooldown, request timeout, disconnect)
    is absorbed by the agent into ``reason="Error: ..."`` with no exception,
    so without this check ``--resume`` would treat the poisoned 0-score
    record as final and never retry it. The explicit ``error_kind`` set by
    MobileAgent wins; the reason-text scan is the fallback for older paths.
    """
    kind = getattr(run_result, "error_kind", None)
    if kind:
        reason = str(getattr(run_result, "reason", "") or "")
        return f"InfraError: {reason[:300]}"
    reason = str(getattr(run_result, "reason", "") or "")
    if reason.startswith("Error:"):
        text = reason.lower()
        markers = (
            "error code: 429",
            "rate limit",
            "model_cooldown",
            "timed out",
            "timeout",
            "connection",
            "temporarily unavailable",
            "overloaded",
        )
        if any(marker in text for marker in markers):
            return f"InfraError: {reason[:300]}"
    return None


def _new_run_id() -> str:
    """Identifier for one invocation of the benchmark runner."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def _finalize_episode_evidence(
    evidence: EpisodeEvidence,
    result: TaskResult,
    agent: Optional[MobileAgent],
    ui_tree_before: Optional[Dict[str, Any]],
    ui_tree_after: Optional[Dict[str, Any]],
    run_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Write every remaining artifact for a finished episode (best-effort)."""
    try:
        if agent is not None:
            trajectory = getattr(agent, "trajectory", None)
            if trajectory is not None:
                result.trajectory_dir = (
                    evidence.adopt_trajectory(trajectory.trajectory_folder) or ""
                )
            shared_state = getattr(agent, "shared_state", None)
            if shared_state is not None:
                result.conversation_path = evidence.write_conversation(
                    list(getattr(shared_state, "message_history", []) or [])
                )
        result.ui_tree_paths = [
            name
            for name in (
                evidence.write_ui_tree("ui_tree_before", ui_tree_before),
                evidence.write_ui_tree("ui_tree_after", ui_tree_after),
            )
            if name
        ]
        result.timeline_path = evidence.write_timeline()
        result.usage_totals = dict(evidence.usage_totals)
        result.evidence_dir = str(evidence.dir)
        # The run block links this episode back to the invocation (and thus
        # the shard / server / model) that produced it.
        manifest = asdict(result)
        if run_context is not None:
            manifest["run"] = run_context
        evidence.write_manifest(manifest)
    except Exception as exc:  # noqa: BLE001 - evidence must never fail a run
        logger.warning("Episode evidence finalize failed: %s", exc)


# ---------------------------------------------------------------------------
# Logging / summary
# ---------------------------------------------------------------------------


def _setup_benchmark_logging(debug: bool, verbose: bool) -> None:
    """CLI logging for the benchmark: quiet unless --verbose/--debug."""
    handler = CLILogHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s %(message)s", "%H:%M:%S")
        if debug
        else logging.Formatter("%(message)s")
    )
    configure_logging(debug=debug, handler=handler)
    if not verbose and not debug:
        # A full suite of per-step agent logs (and retry warnings) would drown
        # the progress output; errors still surface.
        logging.getLogger("mobilerun").setLevel(logging.ERROR)


def _print_summary(results: List[TaskResult], title: str) -> None:
    completed = [result for result in results if result.error is None]
    scored = [result for result in completed if result.score is not None]
    errors = [result for result in results if result.error is not None]

    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    table.add_row("Tasks", str(len(results)))
    table.add_row("Completed", f"{len(completed)}")
    table.add_row("Errors", str(len(errors)))
    if scored:
        table.add_row("Mean score (server)", f"{_mean([r.score for r in scored]):.4f}")
        table.add_row(
            "Mean score (gated)", f"{_mean([r.gated_score or 0.0 for r in scored]):.4f}"
        )
        finished = [r for r in scored if r.finished]
        table.add_row(
            "Completion rate",
            f"{len(finished) / len(scored):.1%} ({len(finished)}/{len(scored)})",
        )
    # Sum of episode durations rather than this invocation's wall clock, so
    # the number also holds for sharded / resumed runs.
    total_runtime_s = sum(result.duration_s or 0.0 for result in results)
    table.add_row("Total runtime", f"{total_runtime_s:.1f}s")
    console.print(table)

    by_difficulty: Dict[str, List[TaskResult]] = {}
    for result in scored:
        by_difficulty.setdefault(result.difficulty or "unknown", []).append(result)
    if by_difficulty:
        diff_table = Table(title="Mean score (gated) by difficulty")
        diff_table.add_column("Difficulty", style="cyan")
        diff_table.add_column("Tasks", style="white")
        diff_table.add_column("Mean gated", style="green")
        diff_table.add_column("Mean raw", style="yellow")
        for name in ("easy", "medium", "hard", "unknown"):
            entries = by_difficulty.get(name)
            if not entries:
                continue
            diff_table.add_row(
                name,
                str(len(entries)),
                f"{_mean([r.gated_score or 0.0 for r in entries]):.4f}",
                f"{_mean([r.score or 0.0 for r in entries]):.4f}",
            )
        console.print(diff_table)

    if errors:
        console.print("[red]Failed tasks:[/]")
        for result in errors:
            console.print(f"  [red]{result.task_type}[/]: {result.error}")
