"""Episode scoring: the one and only implementation of the finish sequence.

The order below is not incidental. Each step fixes a real failure mode the
benchmark runners hit and paid for:

1. **answer before score** — AndroidWorld/MobileWorld grade answer-style tasks
   from the text submitted into the server's interaction cache, so an episode
   that never submits one always scores 0 no matter how right the answer was
   (``fix/benchmark-answer-submission``).
2. **stabilize before score** — the environment scores the *live* accessibility
   tree; a snapshot taken mid-animation scores the wrong frame
   (``fix/benchmark-score-race``).
3. **recheck a zero** — a declared-done episode that read 0 may have been
   judged on a still-settling frame; wait one beat, stabilize again, keep the
   better reading.
4. **gate on declared-done** — the raw score is only trusted when the agent
   itself declared the episode over.

Keeping this in the environment service means no future runner can get the
order wrong, and the derived ``gated_score`` cannot be forged by a client.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from mobilerun.harness.env_service.backend import EnvBackend
from mobilerun.harness.protocol import FinishResponse

logger = logging.getLogger("mobilerun")

# Shared with the benchmark runner so both paths wait the same amount.
try:  # pragma: no cover - import shape only
    from mobilerun.benchmark.android_world import _SCORE_RECHECK_DELAY_S
except Exception:  # pragma: no cover - benchmark module unavailable
    _SCORE_RECHECK_DELAY_S = 1.0


async def finish_episode(
    backend: EnvBackend,
    *,
    declared_done: bool,
    answer: Optional[str] = None,
    settle_s: Optional[float] = None,
) -> FinishResponse:
    """Score the open episode on *backend*, applying the invariants above."""
    # 1. Answer-type tasks are graded purely from what the server received.
    if declared_done and answer:
        await backend.submit_answer(answer)

    # 2 + 3. Settle, score, and re-check a zero once.
    score, eval_reason = await _score(backend, settle_s)
    if declared_done and not score:
        await asyncio.sleep(_SCORE_RECHECK_DELAY_S)
        rescored, rescore_reason = await _score(backend, settle_s)
        if rescored > score:
            score, eval_reason = rescored, rescore_reason or eval_reason

    # 4. The raw score is only trusted when the agent declared the episode over.
    gated = score if declared_done else 0.0
    return FinishResponse(
        score=float(score),
        gated_score=float(gated),
        declared_done=declared_done,
        reason="" if declared_done else "episode not declared finished by the agent",
        eval_reason=str(eval_reason or ""),
    )


async def _score(backend: EnvBackend, settle_s: Optional[float]) -> tuple[float, str]:
    """Settle the screen (best effort), then read the score."""
    if settle_s:
        await asyncio.sleep(settle_s)
    try:
        await backend.stabilize()
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.warning("stabilize before scoring failed: %s", exc)
    return await backend.evaluate()
