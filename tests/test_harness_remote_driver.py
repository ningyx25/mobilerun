"""RemoteEnvDriver: the agent-side half of the wire contract."""

import asyncio

import httpx
import pytest

from mobilerun.harness.client import EnvClient
from mobilerun.harness.drivers import RemoteEnvDriver
from mobilerun.harness.env_service.app import create_env_app
from mobilerun.harness.testing import ScriptedBackend


class _Env:
    """In-process environment service reachable over ASGI, not a socket."""

    def __init__(self, backend: ScriptedBackend) -> None:
        self.backend = backend
        self.app = create_env_app([backend])
        self.transport = httpx.ASGITransport(app=self.app)

    async def __aenter__(self) -> EnvClient:
        await self.app.state.registry.start()
        self.client = EnvClient("http://env.test", transport=self.transport)
        return self.client

    async def __aexit__(self, *exc_info) -> None:
        await self.client.close()
        await self.app.state.registry.stop()


def _run(coro):
    return asyncio.run(coro)


def test_connect_advertises_the_backend_capabilities():
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        async with _Env(backend) as client:
            session = await client.create_session("Alpha")
            driver = RemoteEnvDriver(
                "http://env.test", session.session_id, client=client
            )
            await driver.connect()
            return driver, session

    driver, session = _run(scenario())
    assert driver.platform == "Android"
    assert {
        "tap",
        "swipe",
        "input_text",
        "press_button",
        "get_ui_tree",
    } <= driver.supported
    assert driver.supported_buttons == {"back", "home", "enter"}
    assert driver.capabilities["has_ui_tree"] is True
    assert session.session_id


def test_state_reads_share_one_capture_per_step():
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        async with _Env(backend) as client:
            session = await client.create_session("Alpha")
            driver = RemoteEnvDriver(
                "http://env.test", session.session_id, client=client
            )
            await driver.connect()
            after_connect = backend.observe_count

            tree = await driver.get_ui_tree()
            screenshot = await driver.screenshot()
            after_reads = backend.observe_count

            await driver.tap(10, 20)
            await driver.get_ui_tree()
            after_step = backend.observe_count
            return after_connect, after_reads, after_step, tree, screenshot

    after_connect, after_reads, after_step, tree, screenshot = _run(scenario())
    assert after_connect == 2  # session creation + connect warm-up
    assert after_reads == after_connect  # tree + screenshot share the frame
    assert after_step == after_reads + 1  # exactly one capture after the action
    assert tree["a11y_tree"]["children"][0]["text"] == "Continue"
    assert screenshot.startswith(b"\x89PNG")


def test_driver_verbs_map_to_protocol_actions():
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        async with _Env(backend) as client:
            session = await client.create_session("Alpha")
            driver = RemoteEnvDriver(
                "http://env.test", session.session_id, client=client
            )
            await driver.connect()
            await driver.swipe(1, 2, 3, 4, 500)
            await driver.drag(1, 2, 3, 4, 2.0)
            await driver.input_text("hello", clear=True)
            await driver.press_button("back")
            await driver.press_key_code(66)
            await driver.start_app("com.example", ".Main")
            await driver.stop_app("com.example")
            return backend.actions

    actions = _run(scenario())
    assert [action.op for action in actions] == [
        "swipe",
        "drag",
        "type",
        "key",
        "key",
        "open_app",
        "stop_app",
    ]
    assert actions[0].args["duration_ms"] == 500.0
    assert actions[2].args == {"text": "hello", "clear": True}
    assert actions[3].args == {"button": "back"}
    assert actions[4].args == {"keycode": 66}
    assert actions[5].args == {"app": "com.example", "activity": ".Main"}


def test_unsupported_button_is_rejected_locally():
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        async with _Env(backend) as client:
            session = await client.create_session("Alpha")
            driver = RemoteEnvDriver(
                "http://env.test", session.session_id, client=client
            )
            await driver.connect()
            with pytest.raises(ValueError):
                await driver.press_button("volume_up")
            return backend.actions

    assert _run(scenario()) == []


def test_failed_action_raises_so_the_tool_layer_reports_feedback():
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        async with _Env(backend) as client:
            session = await client.create_session("Alpha")
            driver = RemoteEnvDriver(
                "http://env.test", session.session_id, client=client
            )
            await driver.connect()
            with pytest.raises(RuntimeError) as excinfo:
                await driver.start_app("explode")
            return str(excinfo.value)

    assert "exploded" in _run(scenario())


def test_close_leaves_the_session_open_for_the_harness():
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        async with _Env(backend) as client:
            session = await client.create_session("Alpha")
            driver = RemoteEnvDriver(
                "http://env.test", session.session_id, client=client
            )
            await driver.connect()
            await driver.close()
            info = await client.get_session(session.session_id)
            # Read before the registry shuts down (which tears sessions down).
            return info.state.value, list(backend.teardowns)

    state, teardowns = _run(scenario())
    assert state == "running"
    assert teardowns == []


def test_device_queries_go_through_the_session():
    async def scenario():
        backend = ScriptedBackend(tasks=["Alpha"])
        async with _Env(backend) as client:
            session = await client.create_session("Alpha")
            driver = RemoteEnvDriver(
                "http://env.test", session.session_id, client=client
            )
            await driver.connect()
            return (
                await driver.get_date(),
                await driver.get_apps(),
                await driver.list_packages(),
            )

    date, apps, packages = _run(scenario())
    assert date == "2026-01-01T00:00:00"
    assert apps == [{"package": "com.example", "label": "Example"}]
    assert packages == ["com.example"]
