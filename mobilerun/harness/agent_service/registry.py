"""Registry of agent workers behind the dispatcher.

The dispatcher's job is the mirror image of the environment service's: instead
of leasing one device per session, it keeps a small table of **already running
agent workers** and routes each episode to one of them by id.

Nothing here knows what an agent implementation is. A slot is a URL, a label
and a client; selection is by id or by least-loaded, and every follow-up call
is routed back by the composite episode id (``"<agent_id>:<upstream_id>"``),
so the dispatcher keeps no per-episode state and can restart freely.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mobilerun.harness.client import AgentClient, ServiceError
from mobilerun.harness.protocol import (
    AgentsHealth,
    AgentSlotInfo,
    ErrorCode,
    ProtocolError,
)

logger = logging.getLogger("mobilerun")

# ``agent_id`` is a routing key and part of the composite episode id, so it may
# not contain the separator.
_ENTRY_RE = re.compile(
    r"^(?P<id>[A-Za-z0-9_.-]+)=(?P<name>[^@\s]+)@(?P<url>https?://\S+)$"
)
_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_DEFAULT_MAX_PER_AGENT = 4


def parse_entry(entry: str) -> Tuple[str, str, str]:
    """Parse ``{agent_id}={agent_name}@{server_url}``."""
    match = _ENTRY_RE.match(entry.strip())
    if not match:
        raise ValueError(
            f"Invalid agent entry '{entry}'; expected "
            "AGENT_ID=AGENT_NAME@http://host:port"
        )
    return match.group("id"), match.group("name"), match.group("url").rstrip("/")


@dataclass
class AgentSlot:
    """One registered worker."""

    id: str
    name: str
    url: str
    client: AgentClient
    # Episodes this dispatcher has started on the slot and not seen finish.
    running: int = 0
    max_concurrent: int = _DEFAULT_MAX_PER_AGENT
    state: str = "ready"  # ready | busy | down
    implementation: Optional[str] = None
    name_mismatch: bool = False
    probed: bool = False
    probed_at: float = 0.0

    def info(self) -> AgentSlotInfo:
        return AgentSlotInfo(
            id=self.id,
            name=self.name,
            url=self.url,
            state=self.state,  # type: ignore[arg-type]
            implementation=self.implementation,
            episodes_running=self.running,
            max_concurrent_episodes=self.max_concurrent,
            name_mismatch=self.name_mismatch,
        )


class AgentRegistry:
    """Owns the slots and the health probe cache."""

    def __init__(
        self,
        entries: Sequence[str],
        *,
        token: Optional[str] = None,
        probe_interval: float = 5.0,
        probe_timeout: float = 2.0,
        max_per_agent: int = _DEFAULT_MAX_PER_AGENT,
        client_factory=None,
    ) -> None:
        self._slots: Dict[str, AgentSlot] = {}
        self._probe_interval = probe_interval
        self._probe_timeout = probe_timeout
        self._max_per_agent = max(1, max_per_agent)
        self._client_factory = client_factory or (
            lambda url: AgentClient(url, token=token)
        )

        for entry in entries:
            agent_id, name, url = parse_entry(entry)
            if agent_id in self._slots:
                raise ValueError(f"Duplicate agent id '{agent_id}'")
            self._slots[agent_id] = AgentSlot(
                id=agent_id,
                name=name,
                url=url,
                client=self._client_factory(url),
                max_concurrent=self._max_per_agent,
            )

        urls = [slot.url for slot in self._slots.values()]
        for url in sorted(set(urls)):
            if urls.count(url) > 1:
                logger.warning(
                    "Agent URL %s is registered under several ids; selection "
                    "counters are tracked per id.",
                    url,
                )

    # -- introspection -----------------------------------------------------

    def slots(self) -> List[AgentSlot]:
        return list(self._slots.values())

    def require(self, agent_id: str) -> AgentSlot:
        slot = self._slots.get(agent_id)
        if slot is None:
            raise ProtocolError(
                404,
                ErrorCode.UNKNOWN_AGENT,
                f"Unknown agent '{agent_id}'",
                {"available": sorted(self._slots)},
            )
        return slot

    async def close(self) -> None:
        for slot in self._slots.values():
            try:
                await slot.client.close()
            except Exception as exc:  # noqa: BLE001 - shutdown path
                logger.debug("Closing agent client %s failed: %s", slot.id, exc)

    # -- health ------------------------------------------------------------

    async def probe(self, *, force: bool = False) -> None:
        """Refresh every slot's state (cached for ``probe_interval``)."""
        now = time.monotonic()
        for slot in self._slots.values():
            if (
                not force
                and slot.probed
                and now - slot.probed_at < self._probe_interval
            ):
                continue
            await self._probe_slot(slot, now)

    async def _probe_slot(self, slot: AgentSlot, now: float) -> None:
        slot.probed = True
        slot.probed_at = now
        try:
            health = await slot.client.health(timeout=self._probe_timeout)
        except ServiceError as exc:
            # A reachable worker that refuses /v1/health is still unusable.
            slot.state = "down"
            logger.debug(
                "Agent %s health returned %s: %s", slot.id, exc.status_code, exc
            )
            return
        except Exception as exc:  # noqa: BLE001 - a probe never fails the dispatcher
            slot.state = "down"
            logger.debug("Agent %s is unreachable: %s", slot.id, exc)
            return

        slot.implementation = (
            str(health.get("agent") or health.get("implementation") or "") or None
        )
        reported_name = str(health.get("name") or slot.implementation or "")
        slot.name_mismatch = bool(
            reported_name and slot.name.lower() != reported_name.lower()
        )
        if slot.name_mismatch:
            logger.info(
                "Agent %s registered as '%s' but reports '%s'",
                slot.id,
                slot.name,
                reported_name,
            )

        draining = str(health.get("state") or "") == "draining"
        if not health.get("ok") or draining:
            slot.state = "busy"
        elif slot.running >= slot.max_concurrent:
            slot.state = "busy"
        else:
            slot.state = "ready"

    def health(self) -> AgentsHealth:
        return AgentsHealth(
            ok=any(slot.state != "down" for slot in self._slots.values()),
            agents=[slot.info() for slot in self._slots.values()],
        )

    # -- selection ---------------------------------------------------------

    def select(self, agent_id: Optional[str] = None) -> AgentSlot:
        """Pick a slot for a new episode (call after :meth:`probe`)."""
        if agent_id:
            slot = self.require(agent_id)
            if slot.state == "down":
                raise ProtocolError(
                    503,
                    ErrorCode.BACKEND_UNAVAILABLE,
                    f"Agent '{agent_id}' is unreachable",
                    {"retry_after_s": 5, "url": slot.url},
                )
            return slot

        ready = [slot for slot in self._slots.values() if slot.state == "ready"]
        if not ready:
            raise ProtocolError(
                503,
                ErrorCode.CAPACITY,
                "No agent is available",
                {"retry_after_s": 5, "agents": sorted(self._slots)},
            )
        # Least-loaded first; ties broken by id so selection is deterministic.
        return min(ready, key=lambda slot: (slot.running, slot.id))

    # -- routing -----------------------------------------------------------

    def split(self, episode_id: str) -> Tuple[AgentSlot, str]:
        """Split a composite id into (slot, upstream episode id)."""
        agent_id, separator, upstream_id = episode_id.partition(":")
        if not separator or not upstream_id:
            raise ProtocolError(
                400,
                ErrorCode.BAD_REQUEST,
                f"Episode id '{episode_id}' is not a dispatcher id",
                {"expected": "<agent_id>:<episode_id>"},
            )
        if not _ID_RE.match(agent_id):
            raise ProtocolError(
                400,
                ErrorCode.BAD_REQUEST,
                f"Invalid agent id '{agent_id}' in '{episode_id}'",
            )
        return self.require(agent_id), upstream_id

    # -- counters ----------------------------------------------------------

    def note_started(self, slot: AgentSlot) -> None:
        slot.running += 1
        if slot.running >= slot.max_concurrent and slot.state == "ready":
            slot.state = "busy"

    def note_finished(self, slot: AgentSlot) -> None:
        slot.running = max(0, slot.running - 1)
        if slot.state == "busy" and slot.running < slot.max_concurrent:
            slot.state = "ready"


def composite_id(slot: AgentSlot, upstream_id: str) -> str:
    return f"{slot.id}:{upstream_id}"
