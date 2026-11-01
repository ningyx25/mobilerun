"""``mobilerun serve`` / ``mobilerun eval``: the harness command surface.

Kept out of ``cli/main.py`` so the CLI entry module stays a thin registry; the
import of ``fastapi`` / ``uvicorn`` happens inside the commands, so a plain
``mobilerun`` CLI install never pays for the server stack.
"""

from __future__ import annotations

import asyncio
import sys
from functools import wraps
from typing import List, Tuple

import click
from rich.console import Console

console = Console()


def _coro(f):
    """Run *f* (a coroutine function) inside a fresh event loop."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def _parse_backends(spec: str) -> List[Tuple[str, str, str]]:
    """Parse ``id=kind@url`` / ``id=url`` into ``(id, kind, url)`` triples."""
    from mobilerun.tools.driver.mobile_world import looks_like_mobile_world_url

    parsed: List[Tuple[str, str, str]] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise click.BadParameter(
                f"'{entry}' must be id=url or id=kind@url", param_hint="--backends"
            )
        backend_id, _, rest = entry.partition("=")
        backend_id = backend_id.strip()
        if "@" in rest:
            kind, _, url = rest.partition("@")
            kind = kind.strip()
        else:
            url = rest
            kind = (
                "mobile-world" if looks_like_mobile_world_url(url) else "android-world"
            )
        url = url.strip()
        if not backend_id or not url:
            raise click.BadParameter(
                f"'{entry}' must be id=url or id=kind@url", param_hint="--backends"
            )
        parsed.append((backend_id, kind, url))
    if not parsed:
        raise click.BadParameter(
            "at least one backend is required", param_hint="--backends"
        )
    return parsed


@click.group()
def serve() -> None:
    """Run one half of the evaluation harness as an HTTP service."""


@serve.command("env")
@click.option(
    "--backends",
    required=True,
    help="Comma-separated id=kind@url (or id=url) pairs, e.g. "
    "aw0=android-world@http://localhost:5000,aw1=http://localhost:5001",
)
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", default=9000, type=int, help="Bind port")
@click.option(
    "--session-ttl",
    default=7200.0,
    type=float,
    help="Seconds before an abandoned session is torn down",
)
@click.option(
    "--token",
    default=None,
    help="Bearer token clients must present (default: $MOBILERUN_SERVICE_TOKEN)",
)
def serve_env(
    backends: str, host: str, port: int, session_ttl: float, token: str | None
) -> None:
    """Serve environments: tasks, sessions, device actions and scoring."""
    from mobilerun.harness.env_service.app import create_env_app
    from mobilerun.harness.env_service.backend import create_backend

    instances = []
    parsed = _parse_backends(backends)
    for backend_id, kind, url in parsed:
        try:
            instances.append(create_backend(kind, backend_id, url))
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise click.ClickException(
            "The harness server needs fastapi + uvicorn: "
            "`pip install 'mobilerun[harness]'`."
        ) from exc

    app = create_env_app(instances, token=token, session_ttl_s=session_ttl)
    console.print(
        f"[green]Environment service[/] on http://{host}:{port} "
        f"backends=[{', '.join(f'{i}={k}@{u}' for i, k, u in parsed)}]"
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@serve.command("agent")
@click.argument(
    "entries",
    nargs=-1,
    required=True,
    metavar="[AGENT_ID=AGENT_NAME@SERVER_URL]...",
)
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", default=9100, type=int, help="Bind port")
@click.option(
    "--probe-interval",
    default=5.0,
    type=float,
    help="Seconds to cache worker health probes",
)
@click.option(
    "--max-per-agent",
    default=4,
    type=int,
    help="Episodes the dispatcher will route to one worker at a time",
)
@click.option(
    "--token",
    default=None,
    help="Bearer token clients must present (default: $MOBILERUN_SERVICE_TOKEN)",
)
def serve_agent(
    entries: Tuple[str, ...],
    host: str,
    port: int,
    probe_interval: float,
    max_per_agent: int,
    token: str | None,
) -> None:
    """Serve a dispatcher over running agent workers.

    Each entry registers one worker: AGENT_ID=AGENT_NAME@SERVER_URL, e.g.

        mobilerun serve agent fast0=fast-agent@http://localhost:9201 \
                               jev0=jev@http://localhost:9202
    """
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise click.ClickException(
            "The harness server needs fastapi + uvicorn: "
            "`pip install 'mobilerun[harness]'`."
        ) from exc

    from mobilerun.harness.agent_service.dispatcher import create_agent_dispatcher
    from mobilerun.harness.agent_service.registry import parse_entry

    parsed = []
    for entry in entries:
        try:
            parsed.append(parse_entry(entry))
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="ENTRIES") from exc

    app = create_agent_dispatcher(
        list(entries),
        token=token,
        probe_interval=probe_interval,
        max_per_agent=max_per_agent,
    )
    console.print(
        f"[green]Agent dispatcher[/] on http://{host}:{port} "
        f"agents=[{', '.join(f'{i}={n}@{u}' for i, n, u in parsed)}]"
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


@click.command("eval")
@click.option("--env", "env_url", required=True, help="Environment service base URL")
@click.option("--agent", "agent_url", required=True, help="Agent service base URL")
@click.option("--backend", default=None, help="Backend id to run on (default: any)")
@click.option(
    "--agent-id",
    default=None,
    help="Agent id to run on (dispatcher deployments; default: least loaded)",
)
@click.option("--tasks", default=None, help="Comma-separated task ids")
@click.option(
    "--task-index-range",
    default=None,
    help="1-based inclusive START-END slice of the task list",
)
@click.option(
    "--difficulty", multiple=True, help="Only tasks with this difficulty (repeatable)"
)
@click.option(
    "--steps", default=None, type=int, help="Override the per-task step budget"
)
@click.option("--seed", default=None, type=int, help="Suite seed (AndroidWorld)")
@click.option(
    "--n-task-combinations",
    default=None,
    type=int,
    help="AndroidWorld instance count per task (suite re-init)",
)
@click.option("--output", default=None, help="Results JSON path")
@click.option("--resume", is_flag=True, default=False, help="Skip recorded tasks")
@click.option("--health-timeout", default=900.0, type=float)
@click.option("--task-timeout", default=1000.0, type=float)
@click.option("--evidence/--no-evidence", default=True)
@click.option("--provider", default=None, help="LLM provider override")
@click.option("--model", default=None, help="LLM model override")
@click.option("--base-url", default=None, help="LLM base URL override")
@click.option("--api-base", default=None, help="LLM api_base override")
@click.option("--temperature", default=None, type=float)
@click.option(
    "--parallel",
    default=1,
    type=int,
    help="Concurrent episodes (one environment backend per episode)",
)
@click.option(
    "--agent-env-url",
    default=None,
    help="Environment URL as seen by the agent service (cross-machine setups)",
)
@click.option("--token", default=None, help="Bearer token for both services")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--debug/--no-debug", default=None)
@click.option("--dry-run", is_flag=True, default=False)
@_coro
async def eval_command(
    env_url: str,
    agent_url: str,
    backend: str | None,
    agent_id: str | None,
    tasks: str | None,
    task_index_range: str | None,
    difficulty: Tuple[str, ...],
    steps: int | None,
    seed: int | None,
    n_task_combinations: int | None,
    output: str | None,
    resume: bool,
    health_timeout: float,
    task_timeout: float,
    evidence: bool,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_base: str | None,
    temperature: float | None,
    parallel: int,
    agent_env_url: str | None,
    token: str | None,
    verbose: bool,
    debug: bool | None,
    dry_run: bool,
) -> None:
    """Drive a task suite across a running environment service and agent service."""
    from mobilerun.harness.runner import run_harness_suite

    ok = await run_harness_suite(
        env_url,
        agent_url,
        backend=backend,
        agent_id=agent_id,
        tasks=tasks,
        task_index_range=task_index_range,
        difficulty=difficulty,
        steps=steps,
        seed=seed,
        n_task_combinations=n_task_combinations,
        output=output,
        resume=resume,
        health_timeout=health_timeout,
        task_timeout=task_timeout,
        capture_evidence=evidence,
        provider=provider,
        model=model,
        base_url=base_url,
        api_base=api_base,
        temperature=temperature,
        parallel=parallel,
        verbose=verbose,
        debug=debug,
        dry_run=dry_run,
        token=token,
        agent_env_url_override=agent_env_url,
    )
    sys.exit(0 if ok else 1)


def register(cli: click.Group) -> None:
    """Attach the harness commands to the main CLI group."""
    cli.add_command(serve)
    cli.add_command(eval_command)
