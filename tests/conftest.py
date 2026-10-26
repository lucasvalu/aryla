"""Shared fixtures: test settings, fake models per role, tool policies, runtime."""

from __future__ import annotations

import pytest

from aryla.config import Settings
from aryla.llm import ALL_ROLES
from aryla.runtime import Runtime
from aryla.tools.policy import ToolPolicy
from tests.fakes.fake_chat_model import FakeChatModel

READ_TOOL_NAMES = ["list_directory", "read_text_file", "search_files"]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        APP_CONFIDENCE_THRESHOLD=0.6,
        APP_MAX_RECENT_MESSAGES=6,
        APP_KEEP_LAST_MESSAGES=2,
        APP_MAX_TOOL_ITERATIONS=3,
    )


@pytest.fixture
def fake_models() -> dict[str, FakeChatModel]:
    return {role: FakeChatModel() for role in ALL_ROLES}


@pytest.fixture
def runtime(settings: Settings, fake_models: dict[str, FakeChatModel]) -> Runtime:
    return Runtime(settings=settings, models=dict(fake_models))


@pytest.fixture
def read_only_policy() -> ToolPolicy:
    return ToolPolicy.from_config({"read": READ_TOOL_NAMES, "write": []}, server_name="filesystem")


@pytest.fixture
def policy_with_writes() -> ToolPolicy:
    return ToolPolicy.from_config(
        {"read": READ_TOOL_NAMES, "write": ["write_file"]}, server_name="filesystem"
    )
