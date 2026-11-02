import asyncio
from pathlib import Path
from types import SimpleNamespace

from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent

SYSTEM_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "mobilerun"
    / "config"
    / "prompts"
    / "fast_agent"
    / "system.jinja2"
)


class _Registry:
    tools = {"click": object(), "complete": object()}

    def get_tool_descriptions_xml(self):
        return '<tool name="click"/>'

    def get_param_types(self):
        return {}


class _StateProvider:
    requires_coordinate_tools = False


class _PromptResolver:
    def get_prompt(self, name):
        return None


def _system_prompt(device_date: str) -> str:
    shared_state = MobileAgentState()
    shared_state.device_date = device_date
    agent = FastAgent(
        llm=SimpleNamespace(),
        agent_config=SimpleNamespace(
            fast_agent=SimpleNamespace(vision=False, parallel_tools=False),
            max_steps=5,
            get_fast_agent_system_prompt_path=lambda: str(SYSTEM_TEMPLATE),
        ),
        registry=_Registry(),
        action_ctx=SimpleNamespace(),
        state_provider=_StateProvider(),
        shared_state=shared_state,
        prompt_resolver=_PromptResolver(),
    )
    return asyncio.run(agent._build_system_prompt()).content


def test_system_prompt_carries_the_device_clock():
    text = _system_prompt("Sunday, October 15, 2023 12:00 PM")

    assert "Sunday, October 15, 2023 12:00 PM" in text
    assert "device_date" in text
    assert "never guess it" in text


def test_system_prompt_omits_the_clock_when_unknown():
    text = _system_prompt("")

    assert "device_date" not in text
    assert "never guess it" not in text
