"""Environment backends: the device/task/scoring side of an evaluation.

A backend owns exactly one device (one emulator, one container) and exposes a
uniform surface to the environment service:

- task catalogue (``list_tasks``),
- episode setup (``prepare`` — AndroidWorld's initialize/goal/reset dance),
- observation (``observe``), action execution (``execute``),
- scoring (``submit_answer`` / ``get_score`` / ``stabilize``),
- teardown (``teardown``).

The two real backends are thin adapters over clients that already exist and
are already battle-tested by ``mobilerun runbench``:
:class:`AndroidWorldTaskClient` + :class:`AndroidWorldHttpDriver`, and
:class:`MobileWorldTaskClient` + :class:`MobileWorldHttpDriver`. No device
I/O is reimplemented here.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from mobilerun.harness.actions import screen_from_state, supported_ops
from mobilerun.harness.protocol import (
    Capabilities,
    DeviceAppsResponse,
    ErrorCode,
    Observation,
    ProtocolError,
    ScreenInfo,
    TaskSpec,
)
from mobilerun.tools.driver.android_world import AndroidWorldHttpDriver
from mobilerun.tools.driver.mobile_world import MobileWorldHttpDriver
from mobilerun.tools.helpers.images import image_dimensions

logger = logging.getLogger("mobilerun")


class EnvBackend(ABC):
    """One device + one task family, addressable by the environment service."""

    kind: str = "base"

    def __init__(self, backend_id: str) -> None:
        self.id = backend_id
        self._active: Optional[TaskSpec] = None

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:  # noqa: B027 - optional hook
        """Prepare the backend for use (health probe, warm-up)."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release transport resources owned by the backend."""

    # -- tasks -------------------------------------------------------------

    @abstractmethod
    async def list_tasks(self) -> List[TaskSpec]:
        """Every task this backend can run."""

    @abstractmethod
    async def prepare(
        self,
        task_id: str,
        *,
        task_idx: int = 0,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TaskSpec:
        """Bring the device into the task's initial state and return its spec."""

    async def reinitialize(  # noqa: B027 - optional hook
        self, *, seed: Optional[int] = None, n_task_combinations: Optional[int] = None
    ) -> None:
        """Reset the suite itself (seed/instance selection). No-op by default."""

    # -- episode I/O -------------------------------------------------------

    @abstractmethod
    async def observe(self, *, seq: int) -> Observation:
        """Capture the current device state without changing it."""

    @abstractmethod
    async def execute(self, action, *, screen: ScreenInfo) -> str:
        """Apply one protocol action; return a human-readable summary."""

    # -- scoring -----------------------------------------------------------

    async def submit_answer(self, text: str) -> None:  # noqa: B027 - optional hook
        """Forward an answer for answer-graded tasks (best effort)."""

    @abstractmethod
    async def evaluate(self) -> tuple[float, str]:
        """Read the task's current score (and the environment's own reason)."""

    async def stabilize(self) -> None:  # noqa: B027 - optional hook
        """Ask the environment to wait for a settled screen before scoring."""

    async def teardown(self, task: TaskSpec) -> None:  # noqa: B027 - optional hook
        """Undo the task's setup. Best effort."""

    # -- device queries ----------------------------------------------------

    async def get_date(self) -> str:
        return ""

    async def get_apps(self, include_system: bool = True) -> DeviceAppsResponse:
        return DeviceAppsResponse()

    # -- capabilities ------------------------------------------------------

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """What this backend can execute, derived from its driver."""


class DriverBackend(EnvBackend):
    """Shared plumbing for backends that speak ``DeviceDriver``.

    ``observe`` and ``execute`` are identical for every driver-based backend;
    only the task/scoring half differs, and that is what subclasses add.
    """

    def __init__(self, backend_id: str, driver: Any) -> None:
        super().__init__(backend_id)
        self.driver = driver
        self._input_size: Optional[tuple[int, int]] = None

    # -- episode I/O -------------------------------------------------------

    async def observe(self, *, seq: int) -> Observation:
        try:
            screenshot = await self.driver.screenshot()
            width, height = image_dimensions(screenshot)
            state: Optional[Dict[str, Any]] = None
            if "get_ui_tree" in getattr(self.driver, "supported", set()):
                state = await self.driver.get_ui_tree()
            input_size = await self._driver_input_size(width, height)
        except httpx.TransportError as exc:
            # The device server dropped the connection. Surface it as a
            # retryable service error instead of a 500 traceback.
            raise ProtocolError(
                503,
                ErrorCode.BACKEND_UNAVAILABLE,
                f"Device server for '{self.id}' disconnected: {exc}",
                {"retry_after_s": 2},
            ) from exc
        screen = screen_from_state(
            state,
            native_width=width,
            native_height=height,
            input_size=input_size,
        )
        return Observation.build(
            seq=seq,
            screenshot=screenshot,
            state=state,
            screen=screen,
        )

    async def execute(self, action, *, screen: ScreenInfo) -> str:
        from mobilerun.harness.actions import execute_action

        return await execute_action(self.driver, action, screen=screen)

    async def _driver_input_size(
        self, width: int, height: int
    ) -> Optional[tuple[int, int]]:
        """The driver's input space, when it differs from the screenshot."""
        input_coordinate_size = getattr(self.driver, "input_coordinate_size", None)
        if input_coordinate_size is None:
            return None
        try:
            size = await input_coordinate_size(width, height)
        except Exception as exc:  # noqa: BLE001 - a probe, never fatal
            logger.debug("input_coordinate_size probe failed: %s", exc)
            return None
        if not size or size == (width, height):
            return None
        return int(size[0]), int(size[1])

    # -- device queries ----------------------------------------------------

    async def get_date(self) -> str:
        try:
            return await self.driver.get_date()
        except Exception as exc:  # noqa: BLE001 - informational only
            logger.debug("get_date failed: %s", exc)
            return ""

    async def get_apps(self, include_system: bool = True) -> DeviceAppsResponse:
        try:
            apps = await self.driver.get_apps(include_system=include_system)
        except Exception as exc:  # noqa: BLE001 - informational only
            logger.debug("get_apps failed: %s", exc)
            apps = []
        try:
            packages = await self.driver.list_packages(include_system=False)
        except Exception as exc:  # noqa: BLE001 - informational only
            logger.debug("list_packages failed: %s", exc)
            packages = []
        return DeviceAppsResponse(apps=list(apps or []), packages=list(packages or []))

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        verbs = set(getattr(self.driver, "supported", set()) or set())
        return Capabilities(
            ops=supported_ops(self.driver),
            verbs=sorted(verbs),
            buttons=sorted(set(getattr(self.driver, "supported_buttons", set()) or ())),
            platform=str(getattr(self.driver, "platform", "") or "Android"),
            has_ui_tree="get_ui_tree" in verbs,
            has_screenshot="screenshot" in verbs,
            supports_stabilize=True,
            supports_answer=True,
        )


class AndroidWorldBackend(DriverBackend):
    """Adapter over the AndroidWorld docker server (``:5000*``)."""

    kind = "android-world"

    def __init__(self, backend_id: str, url: str) -> None:
        driver = AndroidWorldHttpDriver(url=url)
        super().__init__(backend_id, driver)
        # Imported lazily to keep the benchmark module out of the import graph
        # of the harness protocol (it pulls MobileAgent in via its package).
        from mobilerun.benchmark.android_world import (
            AndroidWorldTaskClient,
            load_task_metadata,
        )

        self.url = url
        self.client = AndroidWorldTaskClient(url)
        self._metadata = load_task_metadata()

    async def connect(self) -> None:
        await self.driver.connect()

    async def close(self) -> None:
        await self.driver.close()
        await self.client.close()

    # -- tasks -------------------------------------------------------------

    async def list_tasks(self) -> List[TaskSpec]:
        names = await self.client.task_list()
        # AndroidWorld's catalogue endpoint carries no budget; the reference
        # runner uses ``int(10 * task.complexity)`` per instance. Ask the
        # server so the catalogue shows the real per-task budget instead of a
        # flat placeholder.
        semaphore = asyncio.Semaphore(8)

        async def complexity(name: str) -> Optional[float]:
            async with semaphore:
                try:
                    return await self.client.get_complexity(name, 0)
                except Exception as exc:  # noqa: BLE001 - catalogue is best effort
                    logger.debug("Could not read complexity for %s: %s", name, exc)
                    return None

        complexities = await asyncio.gather(*(complexity(name) for name in names))
        tasks: List[TaskSpec] = []
        for name, value in zip(names, complexities, strict=True):
            meta = self._metadata.get(name, {})
            tasks.append(
                TaskSpec(
                    id=name,
                    backend=self.id,
                    instruction="",
                    # Nominal budget for instance 0; prepare() recomputes the
                    # exact value for the instance it initializes.
                    max_steps=int(10 * value) if value is not None else 50,
                    complexity=value,
                    metadata=dict(meta),
                )
            )
        return tasks

    async def prepare(
        self,
        task_id: str,
        *,
        task_idx: int = 0,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TaskSpec:
        params = params or {}
        if seed is not None or params.get("n_task_combinations") is not None:
            await self.reinitialize(
                seed=seed,
                n_task_combinations=params.get("n_task_combinations"),
            )

        await self.client.initialize_task(task_id, task_idx)
        goal = await self.client.get_goal(task_id, task_idx)
        complexity = await self.client.get_complexity(task_id, task_idx)
        start_on_home = await self.client.start_on_home_screen(task_id, task_idx)
        # Fresh episode: reset with go_home mirroring the task's
        # start_on_home_screen, exactly like the reference runner's
        # agent.reset(start_on_home_screen).
        await self.client.reset(go_home=start_on_home)

        meta = dict(self._metadata.get(task_id, {}))
        meta.update({"task_type": task_id, "task_idx": task_idx})
        task = TaskSpec(
            id=f"{task_id}#{task_idx}",
            backend=self.id,
            instruction=goal,
            max_steps=int(params.get("max_steps") or int(10 * complexity)),
            complexity=complexity,
            start_on_home_screen=start_on_home,
            metadata=meta,
        )
        self._active = task
        return task

    async def reinitialize(
        self, *, seed: Optional[int] = None, n_task_combinations: Optional[int] = None
    ) -> None:
        await self.client.reinitialize_suite(
            n_task_combinations=n_task_combinations, seed=seed
        )

    # -- scoring -----------------------------------------------------------

    async def submit_answer(self, text: str) -> None:
        # AndroidWorld grades answer-type tasks from interaction_cache, which
        # the server fills through its own `answer` action.
        try:
            await self.client.execute_action({"action_type": "answer", "text": text})
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("Submitting the episode answer failed: %s", exc)

    async def evaluate(self) -> tuple[float, str]:
        task = self._require_active()
        score = await self.client.get_score(
            task.metadata["task_type"], task.metadata["task_idx"]
        )
        return float(score), ""

    async def stabilize(self) -> None:
        await self.client.stabilize()

    async def teardown(self, task: TaskSpec) -> None:
        task_type = task.metadata.get("task_type")
        if not task_type:
            return
        try:
            await self.client.tear_down(
                task_type, int(task.metadata.get("task_idx", 0))
            )
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("tear_down for %s failed: %s", task_type, exc)
        finally:
            self._active = None

    def _require_active(self) -> TaskSpec:
        if self._active is None:
            raise ProtocolError(
                409, ErrorCode.SESSION_STATE, "No task is initialized on this backend"
            )
        return self._active


class MobileWorldBackend(DriverBackend):
    """Adapter over the MobileWorld docker server (``:6800*``)."""

    kind = "mobile-world"

    def __init__(
        self, backend_id: str, url: str, *, device: Optional[str] = None
    ) -> None:
        from mobilerun.tools.driver.mobile_world import MOBILE_WORLD_DEFAULT_DEVICE

        device = device or MOBILE_WORLD_DEFAULT_DEVICE
        driver = MobileWorldHttpDriver(url=url, device=device)
        super().__init__(backend_id, driver)
        from mobilerun.benchmark.mobile_world import MobileWorldTaskClient

        self.url = url
        self.device = device
        self.client = MobileWorldTaskClient(url, device)

    async def connect(self) -> None:
        await self.driver.connect()

    async def close(self) -> None:
        await self.driver.close()
        await self.client.close()

    async def list_tasks(self) -> List[TaskSpec]:
        entries = await self.client.task_list()
        tasks: List[TaskSpec] = []
        for entry in entries:
            name = str(entry.get("name") or "")
            if not name:
                continue
            tasks.append(
                TaskSpec(
                    id=name,
                    backend=self.id,
                    instruction="",
                    max_steps=50,
                    tags=[str(tag) for tag in (entry.get("tags") or [])],
                    metadata={
                        "apps": list(entry.get("apps") or []),
                        "tags": list(entry.get("tags") or []),
                    },
                )
            )
        return tasks

    async def prepare(
        self,
        task_id: str,
        *,
        task_idx: int = 0,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> TaskSpec:
        params = params or {}
        # /task/init performs all device setup server-side (snapshot, time
        # sync, home); there is no client-side reset on this server.
        await self.client.initialize(task_id)
        goal = await self.client.get_goal(task_id)
        task = TaskSpec(
            id=f"{task_id}#0",
            backend=self.id,
            instruction=goal,
            max_steps=int(params.get("max_steps") or 50),
            metadata={"task_name": task_id},
        )
        self._active = task
        return task

    async def submit_answer(self, text: str) -> None:
        try:
            await self.driver.submit_answer(text)
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("Submitting the episode answer failed: %s", exc)

    async def evaluate(self) -> tuple[float, str]:
        task = self._require_active()
        score, reason = await self.client.evaluate(task.metadata["task_name"])
        return float(score), str(reason)

    async def teardown(self, task: TaskSpec) -> None:
        name = task.metadata.get("task_name")
        if not name:
            return
        try:
            await self.client.tear_down(name)
        except Exception as exc:  # noqa: BLE001 - best effort
            logger.warning("tear_down for %s failed: %s", name, exc)
        finally:
            self._active = None

    def capabilities(self) -> Capabilities:
        caps = super().capabilities()
        # The MobileWorld server exposes no stabilize endpoint; scoring reads
        # the live device state without a settle step.
        return caps.model_copy(update={"supports_stabilize": False})

    def _require_active(self) -> TaskSpec:
        if self._active is None:
            raise ProtocolError(
                409, ErrorCode.SESSION_STATE, "No task is initialized on this backend"
            )
        return self._active


_BACKEND_KINDS = {
    "android-world": AndroidWorldBackend,
    "mobile-world": MobileWorldBackend,
}


def create_backend(
    kind: str,
    backend_id: str,
    url: str,
    *,
    options: Optional[Dict[str, Any]] = None,
) -> EnvBackend:
    """Build a backend from its ``kind`` (used by the CLI and the tests)."""
    if kind == "mobile-world":
        return MobileWorldBackend(backend_id, url, device=(options or {}).get("device"))
    if kind == "android-world":
        return AndroidWorldBackend(backend_id, url)
    raise ValueError(
        f"Unknown backend kind '{kind}'; expected one of {sorted(_BACKEND_KINDS)}"
    )
