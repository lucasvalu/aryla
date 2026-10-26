"""Tool allowlist: which MCP tools the agent may call, and which ones need confirmation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.tools import BaseTool


class ToolKind(StrEnum):
    READ = "read"
    WRITE = "write"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ToolPolicy:
    read: frozenset[str]
    write: frozenset[str]
    server_prefix: str | None = None

    @classmethod
    def from_config(
        cls, policy_config: dict[str, Any], server_name: str | None = None
    ) -> ToolPolicy:
        read = frozenset(policy_config.get("read") or [])
        write = frozenset(policy_config.get("write") or [])
        overlap = read & write
        if overlap:
            raise ValueError(f"Tools listed as both read and write: {sorted(overlap)}")
        return cls(read=read, write=write, server_prefix=server_name)

    @classmethod
    def deny_all(cls) -> ToolPolicy:
        return cls(read=frozenset(), write=frozenset())

    def normalize(self, tool_name: str) -> str:
        """Strip the `<server>_` prefix (multi-server setups) when the rest is a known tool."""
        if self.server_prefix:
            prefix = f"{self.server_prefix}_"
            stripped = tool_name[len(prefix) :]
            if tool_name.startswith(prefix) and stripped in (self.read | self.write):
                return stripped
        return tool_name

    def classify(self, tool_name: str) -> ToolKind:
        name = self.normalize(tool_name)
        if name in self.read:
            return ToolKind.READ
        if name in self.write:
            return ToolKind.WRITE
        return ToolKind.BLOCKED

    def is_write(self, tool_name: str) -> bool:
        return self.classify(tool_name) is ToolKind.WRITE

    def filter_tools(self, tools: Iterable[BaseTool]) -> tuple[list[BaseTool], list[str]]:
        allowed: list[BaseTool] = []
        blocked: list[str] = []
        for tool in tools:
            if self.classify(tool.name) is ToolKind.BLOCKED:
                blocked.append(tool.name)
            else:
                allowed.append(tool)
        return allowed, blocked

    def describe(self, tools: Sequence[BaseTool]) -> str:
        parts = [f"{tool.name} ({self.classify(tool.name).value})" for tool in tools]
        return ", ".join(parts) if parts else "(none)"
