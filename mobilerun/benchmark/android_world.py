"""AndroidWorld benchmark runner.

Drives an AndroidWorld docker environment server (``android_server.py``) over
its task API and runs the mobilerun agent on each of the 116 ``android_world``
tasks, mirroring the canonical episode flow of the reference docker eval
runner (``android_world/run_on_docker.py``)::

    initialize -> goal -> complexity -> start_on_home_screen -> reset
    -> agent episode (budget int(10 * complexity) steps) -> score -> tear_down

Scores are computed server-side from the live device state at ``/task/score``
time and are recorded both raw and "gated" the AndroidWorld way (0 unless the
agent declared the episode finished via the ``complete`` tool). ``/close`` is
deliberately never called: it kills the environment without stopping the
emulator, leaving the container unrecoverable.

``task_metadata.json`` is a copy of the android_world repo's task metadata
(same task names as the server suite); it supplies difficulty/tags for
selection and reporting. Keep it in sync when android_world changes its task
list.
"""

from __future__ import annotations

import asyncio
import copy
import difflib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx
from filelock import FileLock
from rich.console import Console
from rich.table import Table

from mobilerun.agent.droid.droid_agent import MobileAgent
from mobilerun.agent.droid.events import ManagerPlanEvent
from mobilerun.agent.external import list_agents
from mobilerun.agent.utils.llm_loader import load_agent_llms
from mobilerun.agent.utils.llm_picker import load_llm
from mobilerun.benchmark.evidence import EpisodeEvidence, gather_ui_tree
from mobilerun.config_manager import ConfigLoader
from mobilerun.log_handlers import CLILogHandler, configure_logging
from mobilerun.tools.driver.android_world import (
    ANDROID_WORLD_CONNECTION,
    ANDROID_WORLD_DEFAULT_URL,
    AndroidWorldHttpDriver,
    validate_android_world_url,
)

logger = logging.getLogger("mobilerun")
console = Console()

_METADATA_PATH = Path(__file__).with_name("task_metadata.json")

# Timeouts mirror android_world's env/interface.py client: task setup may
# install apps / download data, reset may rebuild the whole controller.
_DEFAULT_TIMEOUT = 60.0
_INITIALIZE_TIMEOUT = 600.0
_SCORE_TIMEOUT = 120.0
_RESET_TIMEOUT = 300.0
_REINITIALIZE_TIMEOUT = 300.0
# /state with wait_to_stabilize runs the server's internal stability loop
# (up to ~6s); the reference client budgets 120s for it.
_STABILIZE_TIMEOUT = 120.0
# Second-chance score read for a completed-but-scored-0 episode, after the
# final screen has had another beat to settle.
_SCORE_RECHECK_DELAY_S = 1.0

# Results files accumulate across sharded / resumed / concurrent invocations,
# so every write is a locked read-merge-write.
_RESULTS_LOCK_TIMEOUT_S = 30.0

# Idempotent calls retry on transient server flakiness (same policy as the
# reference client); connection errors retry for every call.
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = 5.0
_RETRY_STATUSES = frozenset({500, 502, 503, 504})

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


@dataclass
class TaskResult:
    """Recorded outcome of one task episode."""

    task_type: str
    task_idx: int = 0
    difficulty: str = ""
    goal: str = ""
    # Server-side score from live device state, and the AndroidWorld-style
    # gated score (0 unless the agent declared the episode finished).
    score: Optional[float] = None
    gated_score: Optional[float] = None
    success: Optional[bool] = None  # agent-declared success (ResultEvent)
    finished: Optional[bool] = None  # agent declared completion (complete tool)
    steps: Optional[int] = None
    reason: str = ""
    duration_s: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = ""
    # -- provenance: which invocation produced this row (a results file can
    # accumulate rows from several sharded / resumed runs) --
    run_id: str = ""
    server_url: str = ""
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
    """Filter the server task list (mirrors run_on_docker._select_tasks)."""
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
        if k not in ("tasks", "runs", "summary")
    }


def _sanitize_name(value: str) -> str:
    """Filesystem-safe component for an evidence directory name."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


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


# Provenance fields that, when changed, make resumed records incomparable.
_RESUME_PROVENANCE_FIELDS = (
    "server_url",
    "seed",
    "n_task_combinations",
    "provider",
    "model",
    "task_idx",
)


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
        # 116 episodes of per-step agent logs (and retry warnings) would drown
        # the progress output; errors still surface.
        logging.getLogger("mobilerun").setLevel(logging.ERROR)


def _print_summary(results: List[TaskResult]) -> None:
    completed = [result for result in results if result.error is None]
    scored = [result for result in completed if result.score is not None]
    errors = [result for result in results if result.error is not None]

    table = Table(title="AndroidWorld Benchmark Summary")
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

                max_steps = steps if steps is not None else int(10 * complexity)
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
                async for event in handler.stream_events():
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
