# Evaluation harness — protocol v1

Two services, one wire contract:

```
                 ┌─────────────────────────────┐
   orchestrator  │  mobilerun eval (runner)    │   no domain knowledge:
                 └──────┬───────────────┬──────┘   open session → run episode
                        │               │          → score → close
          ┌─────────────▼──────┐   ┌────▼──────────────────┐
          │ agent dispatcher   │   │ environment service   │
          │ `serve agent`      │   │ `serve env`           │
          │  ┌──────────────┐  │   │ backends:             │
          │  │ agent worker │  │   │  android-world :500*  │
          │  │ (library:    │  │   │  mobile-world  :680*  │
          │  │  create_     │  │   │                       │
          │  │  agent_app)  │  │   │                       │
          │  │ MobileAgent  │  │   │                       │
          │  │ +RemoteEnv   │──┼──▶│                       │
          │  └──────────────┘  │   │                       │
          └────────────────────┘   └───────────────────────┘
```

The CLI surface is three commands: `serve env`, `serve agent`, `eval`. A
**worker** (the process a dispatcher registers) is not a CLI command in this
project — it is the FastAPI app returned by
`mobilerun.harness.agent_service.create_agent_app`, mounted by whatever
deployment runs it.

* **The environment owns the device, the task and the score.** It is the only
  party that knows what a successful episode means.
* **The agent owns the prompts, tools, decisions and evidence.** It is the only
  party that sees the screen.
* **The runner owns neither.** It opens a session, hands the goal to the agent,
  asks the environment to score, records the row.

Nothing in `mobilerun/agent/` changed to make this work: `MobileAgent` accepts
an injected `DeviceDriver` and never connects or closes it
(`agent/droid/droid_agent.py:326`, `:865`), and every agent reaches the device
only through `DeviceDriver` verbs. The harness swaps that driver for
`RemoteEnvDriver`, which speaks the protocol below.

## The contract in one line

```
environment:  POST /v1/sessions/{sid}/step  { action }      -> { observation }
agent:        POST /v1/policies/{name}/act  { observation } -> { action }      (phase two)
```

## Models (`mobilerun/harness/protocol.py`)

`Observation` carries `screenshot_png_b64`, `ui_tree` (portal-shaped
accessibility root), `phone_state`, `device_context`, `screen` and a monotonic
`seq`. The `ui_tree` / `phone_state` / `device_context` triple is **the same
shape `AndroidWorldHttpDriver.get_ui_tree()` already returns**, so the agent's
existing state pipeline (`AndroidStateProvider`, tree filters, coordinate
contract, screenshot resizing) consumes it unchanged.

`Action` is one of

| op | args | device verb |
|---|---|---|
| `tap` | `x, y` | `tap` |
| `long_press` | `x, y` | `swipe(x, y, x, y, 1000)` |
| `swipe` | `x1, y1, x2, y2, duration_ms?` | `swipe` |
| `drag` | `x1, y1, x2, y2, duration_s?` | `drag` |
| `type` | `text, clear?` | `input_text` |
| `key` | `button` **or** `keycode` | `press_button` / `press_key_code` |
| `open_app` | `app, activity?` | `start_app` |
| `stop_app` | `app` | `stop_app` |
| `wait` | `duration_s` | `asyncio.sleep` (the environment owns the clock) |
| `complete` | `success, message` | terminal marker, no device I/O |

Coordinates live in `Observation.coordinate_space`
(`screenshot_pixels` by default, or `normalized_1000`); an action may override
it per call and the environment converts into its input space.

## Environment service

| method | path | notes |
|---|---|---|
| GET | `/v1/health` | protocol version + per-backend state/capabilities |
| GET | `/v1/tasks?backend=` | task catalogue |
| POST | `/v1/backends/{id}/reinitialize?seed=&n_task_combinations=` | AndroidWorld suite re-init |
| POST | `/v1/sessions` | `{backend?, task_id, task_idx?, seed?, params?}` → session + first observation |
| GET | `/v1/sessions/{sid}` | state machine state, step counter, capabilities |
| POST | `/v1/sessions/{sid}/observe?fresh=` | observation; never changes the device |
| POST | `/v1/sessions/{sid}/step` | `{action, step_index, expect_seq?, include_observation?}` |
| POST | `/v1/sessions/{sid}/answer` | answer-graded tasks |
| POST | `/v1/sessions/{sid}/finish` | `{declared_done, answer?, settle_s?}` → score |
| DELETE | `/v1/sessions/{sid}` | teardown, releases the backend |
| GET | `/v1/sessions/{sid}/device/date`, `/device/apps` | driver queries |

Semantics that matter:

* **One backend is one device.** A session leases it exclusively for its whole
  life; a second `POST /sessions` while it is leased returns `503` + `Retry-After`
  instead of queueing behind a slow episode.
* **`step` is idempotent by `step_index`.** A retried request returns the cached
  result; it never becomes a double tap. Out-of-order indices are a `409`.
* **`expect_seq` is an optional stale-frame guard.** Omitting it is legitimate:
  one observation may drive several actions.
* **A rejected action is feedback, not a failed episode.** Device errors come
  back as `{ok: false, error}` on a `200`, in line with the tool registry's rule.
* **`finish` owns the scoring order** (`env_service/scoring.py`, the only copy):
  submit answer → stabilize → score → recheck a zero once → gate on
  `declared_done`. `gated_score` is derived server-side and cannot be forged.
  The answer comes from the finish body (`answer`), which the runner passes
  down; a `complete` action's `message` is prose and is never submitted as one.
* **Sessions expire.** Abandoned sessions are torn down after `--session-ttl`
  (default 2 h) so a crashed client cannot hold a device forever.

## Agent service

Phase one — the agent runs its own loop against an environment session:

| method | path | notes |
|---|---|---|
| GET | `/v1/health` | agent name/mode/reasoning/executor kind |
| POST | `/v1/episodes` | 202 → `{episode_id}` |
| GET | `/v1/episodes/{eid}` | `{state, result?, error?}` |
| GET | `/v1/episodes/{eid}/events` | SSE; replays a bounded buffer, then streams live |
| POST | `/v1/episodes/{eid}/cancel` | |

`EpisodeRequest` = `{goal, environment: {base_url, session_id}, overrides,
limits, evidence, metadata}`. `EpisodeResult` carries `declared_done`, `answer`,
`steps`, `reason`, `duration_s`, `usage`, `evidence_dir`. The terminal state is
published only after the evidence bundle is written.

Phase two — the strict `state → action` surface. The routes and models are
already fixed (`PolicyCapabilities`, `CreatePolicySessionRequest`, `ActRequest`,
`ActResponse`); they answer `501` until the single-step refactor lands.

```jsonc
POST /v1/policies/jev/act
{
  "policy_session_id": "ps_7f3c",
  "step": 7,                       // idempotent key: replaying a step returns
  "observation": { ... },          // the cached decision, not a new one
  "feedback": {"last_action_ok": true, "last_summary": "Tapped at (512, 1400)"}
}
→ {"action": {"op": "tap", "args": {"x": 512, "y": 1400}}, "done": false}
```

Implementing it needs no protocol change: extract a re-entrant `step()` from
`FastAgent` (its chat history is already an instance attribute) and model the
manager/executor context explicitly. Suggested order: `jev` → `fast_agent` →
`manager+executor`.

## Running

```bash
# environment side
mobilerun serve env --backends aw0=http://localhost:5000,aw1=http://localhost:5001 --port 9000

# agent side: start the workers with your deployment, then register them
mobilerun serve agent fast0=fast-agent@http://localhost:9201 \
                    gr0=grounded@http://localhost:9202 --port 9100

mobilerun eval --env http://localhost:9000 --agent http://localhost:9100 \
    --backend aw0 --agent-id fast0 --task-index-range 1-30 \
    --output runs/harness_aw.json --resume
```

A worker is just the FastAPI app from `create_agent_app`; this project ships the
factory, not a CLI command for it:

```python
import uvicorn
from mobilerun.config_manager import ConfigLoader
from mobilerun.harness.agent_service import create_agent_app

app = create_agent_app(ConfigLoader.load("fast.yaml"), name="fast-agent")
uvicorn.run(app, host="0.0.0.0", port=9201)
```

`--parallel N` runs N episodes concurrently; each needs its own backend, so
leave `--backend` unset and let the environment pick a free one. Without a
pinned backend the per-backend copies of the catalogue are de-duplicated, so a
full run is the suite size, not suite size × devices; `--parallel` above the
number of backends only causes capacity retries. `--seed` and
`--n-task-combinations` apply to every backend in that mode.

Results are written with the benchmark runner's own plumbing (`TaskResult`,
`_write_results`, resume history), so a harness file is directly comparable to
a `mobilerun runbench` file. Provenance adds `env_url`, `agent_url`,
`agent_env_url`, `agent_id`, the dispatcher's `agents` table, and
`protocol_version`.

## Agent dispatcher

`mobilerun serve agent` is a **dispatcher**: it registers already-running
workers and picks one per episode. Each argument is
`AGENT_ID=AGENT_NAME@SERVER_URL`:

* `AGENT_ID` — routing key and the provenance field written into results;
* `AGENT_NAME` — a label (`fast-agent`, `jev`, …). The dispatcher only uses it
  for reporting and to spot a worker that calls itself something else
  (`name_mismatch` on `/v1/health`); behaviour never depends on it;
* `SERVER_URL` — a running worker (`create_agent_app` served over HTTP).

**Selection.** `POST /v1/episodes` accepts an optional `agent_id`. With one,
that worker is used (404 if unknown, 503 if it is down). Without one, the
dispatcher picks the least-loaded ready worker, and answers `503` +`Retry-After`
when none is available.

**Composite ids.** Every episode started through the dispatcher comes back as
`"<agent_id>:<upstream_id>"` (e.g. `fast0:ep_ab12`). Polling, the SSE stream and
cancellation are routed by that prefix, so the dispatcher keeps no per-episode
table and restarts cleanly. A bare id (no `:`) is a `400`; an unknown prefix is
`404 unknown_agent`.

**Health.** `GET /v1/health` returns one row per worker
(`ready` / `busy` / `down`), probed with a 2s timeout and cached for
`--probe-interval` seconds (default 5). The dispatcher never fails `/v1/health`.

The worker's own API is unchanged, so `mobilerun eval --agent http://worker:9201`
still works directly — the dispatcher is an optional layer, not a requirement.

## Extending

* **New environment backend** — implement `EnvBackend`
  (`env_service/backend.py`). `DriverBackend` already provides `observe`,
  `execute`, `get_date`, `get_apps` and capability derivation for anything that
  speaks `DeviceDriver`; a backend only adds task setup, scoring and teardown.
  `mobilerun/harness/testing.py::ScriptedBackend` is the minimal reference.
* **New agent** — either serve it as a worker and register it with `serve agent`
  (wrap its loop the way
  `episode.py` wraps `MobileAgent`), or implement `/v1/policies/*/act` for a
  single-step policy.
* **New action** — add it to `ACTION_ARGS` in `protocol.py`, map it in
  `actions.execute_action`, and it becomes available everywhere.

## Security

Both services bind `127.0.0.1` by default. Set `MOBILERUN_SERVICE_TOKEN` (or
`--token`) to require `Authorization: Bearer …`; the environment service can
drive a real device, so do not expose it unauthenticated.
