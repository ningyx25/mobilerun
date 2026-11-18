"""MobileGym backend: the in-process simulator plus its official state judge.

MobileGym has no environment server. ``bench_env.MobileGymEnv`` drives the SPA
at the simulator URL through Playwright in this process, so the backend owns
one browser per backend id (one session at a time, enforced by the registry's
per-backend lease). The simulator is vision-only: ``get_ui_tree`` returns an
empty portal stub, so this backend removes that verb from its own driver
instance and never advertises an accessibility tree.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from mobilerun.harness.env_service.backend import DriverBackend
from mobilerun.harness.protocol import (
    Capabilities,
    ErrorCode,
    Observation,
    ProtocolError,
    ScreenInfo,
    TaskSpec,
)
from mobilerun.tools.driver.mobile_gym import (
    MOBILE_GYM_DEFAULT_MAX_STEPS,
    MobileGymBrowserDriver,
    discover_tasks,
    load_bench_env,
    resolve_app_identifier,
    validate_mobile_gym_url,
    wait_for_sim,
)

logger = logging.getLogger("mobilerun")

_DEFAULT_HEADLESS = True
_DEFAULT_STARTUP_TIMEOUT_S = 120.0
_QUERY_OPTIONS = frozenset({"split", "headless", "startup_timeout", "browser_path"})
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class MobileGymOptions:
    url: str
    split: Optional[str] = "test"
    headless: bool = _DEFAULT_HEADLESS
    startup_timeout: float = _DEFAULT_STARTUP_TIMEOUT_S
    browser_path: Optional[str] = None


def _parse_query(query: str) -> Dict[str, str]:
    """Parse ``a=b&c=d`` without form-decoding ``+`` (split names use it)."""
    parsed: Dict[str, str] = {}
    for chunk in query.split("&"):
        if not chunk:
            continue
        key, sep, value = chunk.partition("=")
        key = unquote(key)
        value = unquote(value) if sep else ""
        if key in parsed:
            raise ValueError(f"MobileGym option '{key}' was given more than once")
        parsed[key] = value
    return parsed


def parse_mobile_gym_backend_url(raw: str) -> MobileGymOptions:
    """Split ``<simulator-url>?option=value`` into validated options."""
    parts = urlsplit(str(raw or "").strip())
    if parts.fragment:
        raise ValueError("MobileGym backend URL must not contain a fragment")
    query = _parse_query(parts.query)
    unknown = sorted(set(query) - _QUERY_OPTIONS)
    if unknown:
        raise ValueError(
            f"Unknown MobileGym backend option(s) {unknown}; "
            f"supported: {sorted(_QUERY_OPTIONS)}"
        )

    url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    if not url:
        raise ValueError("MobileGym backend URL must not be empty")
    url = validate_mobile_gym_url(url)

    split: Optional[str] = "test"
    if "split" in query:
        split = query["split"].strip()
        if not split:
            raise ValueError("MobileGym option 'split' must not be empty")

    headless = _DEFAULT_HEADLESS
    if "headless" in query:
        value = query["headless"].strip().lower()
        if value in _TRUE:
            headless = True
        elif value in _FALSE:
            headless = False
        else:
            raise ValueError(
                f"MobileGym option 'headless' must be a boolean, got '{value}'"
            )

    startup_timeout = _DEFAULT_STARTUP_TIMEOUT_S
    if "startup_timeout" in query:
        try:
            startup_timeout = float(query["startup_timeout"])
        except ValueError as exc:
            raise ValueError(
                "MobileGym option 'startup_timeout' must be a number of seconds"
            ) from exc
        if startup_timeout <= 0:
            raise ValueError("MobileGym option 'startup_timeout' must be > 0")

    browser_path: Optional[str] = None
    if "browser_path" in query:
        browser_path = query["browser_path"].strip()
        if not browser_path:
            raise ValueError("MobileGym option 'browser_path' must not be empty")
        if not Path(browser_path).is_file():
            raise ValueError(
                f"MobileGym option 'browser_path' does not exist: {browser_path}"
            )

    return MobileGymOptions(url, split, headless, startup_timeout, browser_path)


class MobileGymBackend(DriverBackend):
    """One MobileGym simulator browser, exposed as an ``EnvBackend``."""

    kind = "mobile-gym"

    def __init__(self, backend_id: str, url: str) -> None:
        options = parse_mobile_gym_backend_url(url)
        be = load_bench_env()  # ImportError -> CLI reports the PYTHONPATH hint
        self._be = be
        self.url = options.url
        self.split = options.split
        self.headless = options.headless
        self.startup_timeout = options.startup_timeout

        try:
            self._split_ids = be.resolve_split(self.split) if self.split else None
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(
                f"Invalid MobileGym split '{self.split}' for backend "
                f"'{backend_id}': {exc}"
            ) from exc

        self._registry = be.TaskRegistry()
        env = be.MobileGymEnv(
            url=self.url,
            headless=self.headless,
            coord_space="physical",
            delay_after_action=0.0,
            viewport_size=(360, 800),
            physical_size=(1080, 2400),
            device_scale_factor=3,
            verbose=False,
        )
        if options.browser_path:
            self._apply_browser_path(env, options.browser_path)
        driver = MobileGymBrowserDriver(env=env)
        # Vision-only: never fetch the empty portal stub. This keeps observe()
        # and capabilities() consistent without touching the shared driver.
        driver.supported = set(driver.supported) - {"get_ui_tree"}
        super().__init__(backend_id, driver)

        self._env = env
        self._connected = False
        self._catalogue: Optional[Dict[str, Dict[str, Any]]] = None
        self._active_task_obj: Any = None
        self._init_obs: Any = None
        self._declared_done = False
        self._answer = ""
        self._steps = 0
        self._episode_started = 0.0

    @staticmethod
    def _apply_browser_path(env: Any, browser_path: str) -> None:
        """Point bench_env's Playwright launch at a local Chromium binary.

        Escape hatch for hosts where the installed Playwright cannot fetch its
        bundled browser (e.g. an OS the current Playwright release dropped).
        """
        base = type(env).get_launch_args

        def launch_args(headless, proxy=None, browser_type="chromium"):
            args = base(headless, proxy, browser_type)
            args["executable_path"] = browser_path
            return args

        env.get_launch_args = launch_args

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            # Fail fast on a bad split/catalogue before launching a browser.
            self._task_metadata()
            # Idempotent; also discards a dead page/context after a crash.
            await self._env.close()
            await self._env.start()
            await self._env.wait_ready(timeout_ms=int(self.startup_timeout * 1000))
            await wait_for_sim(self._env.page, self.startup_timeout)
            await self.driver.connect()
        except Exception as exc:  # noqa: BLE001 - wrap for the lifespan
            hint = (
                " (run `playwright install chromium`, or add "
                "`?browser_path=/path/to/chrome` to the backend URL)"
                if "executable" in str(exc).lower()
                else ""
            )
            try:
                await self._env.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            raise ConnectionError(
                f"MobileGym simulator at {self.url} did not start for backend "
                f"'{self.id}': {type(exc).__name__}: {exc}{hint}"
            ) from exc
        self._connected = True

    async def close(self) -> None:
        self._connected = False
        try:
            await self.driver.close()
        finally:
            await self._env.close()

    def _mark_env_failed(self, exc: BaseException) -> None:
        if self._connected:
            logger.warning(
                "MobileGym backend '%s' lost its simulator: %s", self.id, exc
            )
        self._connected = False

    async def _ensure_started(self) -> None:
        if not self._connected:
            await self.connect()

    # -- tasks -------------------------------------------------------------

    def _task_metadata(self) -> Dict[str, Dict[str, Any]]:
        if self._catalogue is None:
            _, metadata = discover_tasks(
                self._be, split_ids=self._split_ids, registry=self._registry
            )
            self._catalogue = metadata
        return self._catalogue

    async def list_tasks(self) -> List[TaskSpec]:
        return [
            TaskSpec(
                id=task_id,
                backend=self.id,
                # The concrete instruction is produced by prepare() after the
                # grounded answer-sheet hint is appended.
                instruction="",
                max_steps=int(entry.get("max_steps") or MOBILE_GYM_DEFAULT_MAX_STEPS),
                start_on_home_screen=False,
                metadata=dict(entry),
            )
            for task_id, entry in self._task_metadata().items()
        ]

    async def prepare(
        self,
        task_id: str,
        *,
        task_idx: int = 0,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TaskSpec:
        params = params or {}
        meta = self._task_metadata()
        if task_id not in meta:
            raise ProtocolError(
                404,
                ErrorCode.UNKNOWN_TASK,
                f"Unknown MobileGym task '{task_id}'",
                {"available": sorted(meta)[:20]},
            )
        await self._ensure_started()

        # Every prepare uses a fresh instance: Controller.setup mutates
        # task.task_name (grounded hint) and sampling mutates task.params.
        create_kwargs = {"_seed": int(seed)} if seed is not None else {}
        try:
            task = self._registry.create_task(task_id, **create_kwargs)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(
                404,
                ErrorCode.UNKNOWN_TASK,
                f"Cannot create MobileGym task '{task_id}': {exc}",
            ) from exc

        self._active_task_obj = None
        self._init_obs = None
        self._active = None
        self._declared_done = False
        self._answer = ""
        self._steps = 0
        self.driver._answer = ""
        started = time.monotonic()

        try:
            init_obs, _sampled = await self._be.Controller.setup(
                self._env, task, eval_mode="grounded"
            )
        except Exception as exc:  # noqa: BLE001 - setup is device I/O
            self._mark_env_failed(exc)
            raise ProtocolError(
                503,
                ErrorCode.BACKEND_UNAVAILABLE,
                f"MobileGym setup failed for '{task_id}': "
                f"{type(exc).__name__}: {exc}",
                {"retry_after_s": 5},
            ) from exc

        self._active_task_obj = task
        self._init_obs = init_obs
        self._episode_started = started
        max_steps = int(
            params.get("max_steps")
            or getattr(task, "max_steps", None)
            or meta[task_id].get("max_steps")
            or MOBILE_GYM_DEFAULT_MAX_STEPS
        )
        spec = TaskSpec(
            id=f"{task_id}#0",
            backend=self.id,
            instruction=str(task.description),
            max_steps=max_steps,
            start_on_home_screen=False,
            metadata={
                **meta[task_id],
                "task_name": task_id,
                "task_idx": 0,
            },
        )
        self._active = spec
        return spec

    # -- episode I/O -------------------------------------------------------

    async def observe(self, *, seq: int) -> Observation:
        try:
            return await super().observe(seq=seq)
        except ProtocolError:
            raise
        except Exception as exc:  # noqa: BLE001 - simulator/Playwright failure
            self._mark_env_failed(exc)
            raise ProtocolError(
                503,
                ErrorCode.BACKEND_UNAVAILABLE,
                f"MobileGym simulator for '{self.id}' failed: "
                f"{type(exc).__name__}: {exc}",
                {"retry_after_s": 2},
            ) from exc

    async def execute(self, action, *, screen: ScreenInfo) -> str:
        if action.op == "open_app":
            app = str(action.args.get("app", "")).strip()
            translated = resolve_app_identifier(self._env, app)
            if translated != app:
                action = action.model_copy(
                    update={"args": {**action.args, "app": translated}}
                )
        summary = await super().execute(action, screen=screen)
        if action.touches_device:
            self._steps += 1
        return summary

    # -- scoring -----------------------------------------------------------

    async def submit_answer(self, text: str) -> None:
        self._answer = str(text)
        try:
            await self.driver.submit_answer(self._answer)
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("Submitting the MobileGym answer failed: %s", exc)

    def set_finish_context(
        self, *, declared_done: bool, answer: Optional[str] = None
    ) -> None:
        self._declared_done = bool(declared_done)
        if answer:
            self._answer = str(answer)
            self.driver._answer = self._answer

    async def evaluate(self) -> tuple[float, str]:
        task = self._require_active_task()
        try:
            state = await self._env.get_state(
                required_apps=list(getattr(task, "apps", []) or []) or None
            )
            last_obs = await self._env.get_observation()
            last_obs = replace(last_obs, state=state)
        except Exception as exc:  # noqa: BLE001 - final device read
            self._mark_env_failed(exc)
            raise ProtocolError(
                503,
                ErrorCode.BACKEND_UNAVAILABLE,
                f"MobileGym final observation failed: " f"{type(exc).__name__}: {exc}",
                {"retry_after_s": 2},
            ) from exc

        exec_result = self._be.ExecutionResult(
            steps=int(self._steps),
            trace=[],
            runtime_s=max(0.0, time.monotonic() - self._episode_started),
            finished=bool(self._declared_done),
            truncated=False,
            agent_answer=(
                self._answer
                or getattr(self.driver, "_answer", "")
                or getattr(self._env, "agent_answer", None)
            )
            or None,
        )
        judge = await self._be.Evaluator(
            judge_mode="state", eval_mode="grounded"
        ).evaluate(task, self._init_obs, last_obs, exec_result)
        payload = {
            "success": getattr(judge, "success", False),
            "clean": getattr(judge, "clean", True),
            "progress": getattr(judge, "progress", 0.0),
            "issues": getattr(judge, "issues", []),
            "warnings": getattr(judge, "warnings", []),
        }
        reason = json.dumps(payload, ensure_ascii=False, default=str)[:2000]
        if getattr(judge, "judge_error", None):
            # A judge failure means the score is meaningless. Report it as a
            # scoring failure (502) instead of a scoreable 0: the runner then
            # records an `error` and `--max-retry` / `--resume` retry the task.
            logger.warning(
                "MobileGym judge error for %s: %s", task.id, judge.judge_error
            )
            raise ProtocolError(
                502,
                ErrorCode.BACKEND_UNAVAILABLE,
                f"JudgeError: {judge.judge_error}",
            )
        return (1.0 if judge.passed else 0.0), reason

    async def teardown(self, task: TaskSpec) -> None:
        active, self._active_task_obj = self._active_task_obj, None
        if active is not None:
            try:
                active.teardown(self._env)  # sync by contract
            except Exception as exc:  # noqa: BLE001 - best effort
                logger.warning("MobileGym teardown for %s failed: %s", task.id, exc)
        self._init_obs = None
        self._active = None
        self._declared_done = False
        self._answer = ""
        self._steps = 0
        self.driver._answer = ""

    def _require_active_task(self) -> Any:
        if self._active_task_obj is None or self._init_obs is None:
            raise ProtocolError(
                409,
                ErrorCode.SESSION_STATE,
                "No task is initialized on this backend",
            )
        return self._active_task_obj

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        caps = super().capabilities()
        return caps.model_copy(
            update={"has_ui_tree": False, "supports_stabilize": False}
        )
