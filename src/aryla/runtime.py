"""What the graph needs at run time: settings, one model per role, tools and their policy."""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from aryla.config import Settings
from aryla.llm import load_models
from aryla.tools.mcp import load_server_config, open_server_tools
from aryla.tools.policy import ToolPolicy


@dataclass
class Runtime:
    settings: Settings
    models: dict[str, BaseChatModel]
    coding_tools: list[BaseTool] = field(default_factory=list)
    coding_policy: ToolPolicy = field(default_factory=ToolPolicy.deny_all)

    def model(self, role: str) -> BaseChatModel:
        if role not in self.models:
            raise KeyError(f"No model for role '{role}'; available: {sorted(self.models)}")
        return self.models[role]


async def open_runtime(
    settings: Settings, stack: AsyncExitStack, show_server_logs: bool = False
) -> Runtime:
    """Load the models and, when a workspace is configured, open the filesystem MCP server."""
    runtime = Runtime(settings=settings, models=load_models(settings))

    server = load_server_config(settings)
    missing = server.missing_env()
    if missing:
        print(
            f"[warning] coding agent disabled: set {', '.join(missing)} in .env to enable it. "
            "Continuing with the general agent only."
        )
        return runtime

    workspace = settings.coding_workspace
    if workspace is None or not workspace.is_dir():
        print(f"[warning] coding agent disabled: CODING_WORKSPACE is not a directory: {workspace}")
        return runtime

    server_stderr = sys.stderr if show_server_logs else None
    try:
        tools = await stack.enter_async_context(open_server_tools(server, server_stderr))
    except Exception as exc:
        print(f"[error] coding agent disabled, MCP server failed: {type(exc).__name__}: {exc}")
        return runtime

    runtime.coding_tools = tools
    runtime.coding_policy = server.policy
    return runtime
