"""MCP server configuration and tool loading through langchain.mcp."""

from __future__ import annotations

import os
import string
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TextIO

from langchain_core.tools import BaseTool

from aryla.config import Settings
from aryla.observability import flow
from aryla.tools.policy import ToolPolicy

DEFAULT_SERVER = "filesystem"


@dataclass
class ServerConfig:
    name: str
    policy: ToolPolicy
    connection: dict[str, Any]
    required_env: list[str] = field(default_factory=list)

    def missing_env(self) -> list[str]:
        return [name for name in self.required_env if not os.environ.get(name)]

    def as_mcp_config(self) -> dict[str, Any]:
        return {"mcpServers": {self.name: self.connection}}


def load_server_config(settings: Settings, server_name: str = DEFAULT_SERVER) -> ServerConfig:
    config = settings.load_mcp_config()
    servers = config.get("servers") or {}
    if server_name not in servers:
        raise KeyError(f"Server '{server_name}' not found in {settings.mcp_config_path}")
    server = servers[server_name]

    policy_name = server.get("policy")
    policies = config.get("policies") or {}
    if policy_name not in policies:
        raise KeyError(f"Policy '{policy_name}' for server '{server_name}' not found in mcp.yaml")

    return ServerConfig(
        name=server_name,
        policy=ToolPolicy.from_config(policies[policy_name], server_name=server_name),
        connection=expand_env(dict(server.get("connection") or {})),
        required_env=list(server.get("required_env") or []),
    )


@asynccontextmanager
async def open_server_tools(
    server: ServerConfig, server_stderr: TextIO | None = None
) -> AsyncIterator[list[BaseTool]]:
    """Open a configured server; a stdio server's stderr goes to `server_stderr` or is discarded."""
    flow.info("mcp: connecting to server '%s'", server.name)
    if "command" in server.connection:
        target = _stdio_client(server, server_stderr)
    else:
        target = server.as_mcp_config()
    async with open_tools(target, server.policy) as tools:
        yield tools


@asynccontextmanager
async def open_tools(target: Any, policy: ToolPolicy) -> AsyncIterator[list[BaseTool]]:
    """Open an MCPAdapter on any target it accepts (config dict, FastMCP instance, client, URL)
    and yield the tools allowed by the policy. Keep the context open while the tools are in use."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # langchain.mcp is beta and warns on import
        from langchain.mcp import MCPAdapter

    async with MCPAdapter(target) as adapter:
        all_tools = await adapter.list_tools()
        allowed, blocked = policy.filter_tools(all_tools)
        flow.info("mcp: %d tool(s) allowed: %s", len(allowed), policy.describe(allowed))
        if blocked:
            flow.info("mcp: %d tool(s) blocked by policy: %s", len(blocked), ", ".join(blocked))
        yield allowed


def expand_env(connection: dict[str, Any]) -> dict[str, Any]:
    """Expand `${VAR}` in command, args, url and env; env entries that end up empty are dropped."""
    for key in ("command", "url"):
        if isinstance(connection.get(key), str):
            connection[key] = _substitute(connection[key])

    if isinstance(connection.get("args"), list):
        connection["args"] = [_substitute(str(arg)) for arg in connection["args"]]

    env = connection.get("env")
    if isinstance(env, dict):
        expanded: dict[str, str] = {}
        for key, value in env.items():
            text = _substitute(str(value))
            if text:
                expanded[key] = text
        connection["env"] = expanded
    return connection


def _substitute(text: str) -> str:
    expanded = string.Template(text).safe_substitute(os.environ)
    if "${" in expanded:
        return ""
    return expanded


def _stdio_client(server: ServerConfig, server_stderr: TextIO | None) -> Any:
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    transport = StdioTransport(
        command=server.connection["command"],
        args=list(server.connection.get("args", [])),
        env=server.connection.get("env") or None,
        log_file=server_stderr or open(os.devnull, "w"),  # noqa: SIM115 - lives as long as the process
    )
    return Client(transport)
