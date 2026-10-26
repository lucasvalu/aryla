import pytest
from langchain_core.tools import tool

from aryla.tools.policy import ToolKind, ToolPolicy


@tool
def read_text_file(jql: str) -> str:
    """search"""
    return ""


@tool
def write_file(summary: str) -> str:
    """create"""
    return ""


@tool
def move_file(issue_key: str) -> str:
    """delete"""
    return ""


@pytest.fixture
def policy() -> ToolPolicy:
    return ToolPolicy.from_config(
        {"read": ["read_text_file"], "write": ["write_file"]}, server_name="filesystem"
    )


def test_classifies_read_write_and_blocked(policy: ToolPolicy):
    assert policy.classify("read_text_file") is ToolKind.READ
    assert policy.classify("write_file") is ToolKind.WRITE
    assert policy.classify("move_file") is ToolKind.BLOCKED
    assert policy.classify("anything_else") is ToolKind.BLOCKED


def test_strips_server_prefix_only_when_known(policy: ToolPolicy):
    assert policy.classify("filesystem_read_text_file") is ToolKind.READ
    assert policy.classify("filesystem_whatever") is ToolKind.BLOCKED


def test_filter_tools_drops_blocked(policy: ToolPolicy):
    allowed, blocked = policy.filter_tools([read_text_file, write_file, move_file])
    assert [t.name for t in allowed] == ["read_text_file", "write_file"]
    assert blocked == ["move_file"]


def test_read_and_write_cannot_overlap():
    with pytest.raises(ValueError):
        ToolPolicy.from_config({"read": ["a"], "write": ["a"]})


def test_deny_all_blocks_everything():
    policy = ToolPolicy.deny_all()
    assert policy.classify("read_text_file") is ToolKind.BLOCKED
    assert policy.filter_tools([read_text_file]) == ([], ["read_text_file"])


def test_shipped_config_has_no_write_tools(settings):
    config = settings.load_mcp_config()
    policy = ToolPolicy.from_config(config["policies"]["filesystem"], server_name="filesystem")
    assert policy.write == frozenset()
    for name in ("write_file", "edit_file", "create_directory", "move_file"):
        assert policy.classify(name) is ToolKind.BLOCKED
    assert policy.classify("read_text_file") is ToolKind.READ


def test_shipped_server_args_expand_the_workspace(settings, monkeypatch):
    from aryla.tools.mcp import load_server_config

    monkeypatch.setenv("CODING_WORKSPACE", "/tmp/repo")
    server = load_server_config(settings)
    assert server.connection["args"][-1] == "/tmp/repo"
    assert server.missing_env() == []

    monkeypatch.delenv("CODING_WORKSPACE")
    server = load_server_config(settings)
    assert server.connection["args"][-1] == ""
    assert server.missing_env() == ["CODING_WORKSPACE"]
