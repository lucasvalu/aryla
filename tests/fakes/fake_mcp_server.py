"""In-process FastMCP server with fixed filesystem-like responses for integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastmcp import FastMCP

FILES: dict[str, str] = {
    "/repo/README.md": "# demo\n\nA small demo repository.\n",
    "/repo/main.py": "def main():\n    print('hello')\n",
    "/repo/utils.py": "def add(a, b):\n    return a + b\n",
}


@dataclass
class FakeFilesystemServer:
    mcp: FastMCP
    writes: list[dict[str, Any]] = field(default_factory=list)
    moves: list[tuple[str, str]] = field(default_factory=list)


def create_fake_filesystem_server() -> FakeFilesystemServer:
    server = FastMCP("filesystem")
    fake = FakeFilesystemServer(mcp=server)
    writes, moves = fake.writes, fake.moves

    @server.tool
    def list_directory(path: str) -> str:
        """List a directory."""
        return "\n".join(f"[FILE] {name.rsplit('/', 1)[-1]}" for name in FILES)

    @server.tool
    def read_text_file(path: str) -> str:
        """Read a text file."""
        if path not in FILES:
            raise ValueError(f"File not found: {path}")
        return FILES[path]

    @server.tool
    def search_files(path: str, pattern: str) -> str:
        """Search file names."""
        return "\n".join(name for name in FILES if pattern in name)

    @server.tool
    def write_file(path: str, content: str) -> str:
        """Write a file (write)."""
        writes.append({"path": path, "content": content})
        return f"Successfully wrote to {path}"

    @server.tool
    def move_file(source: str, destination: str) -> str:
        """Move a file (write; must be blocked by policy)."""
        moves.append((source, destination))
        return "moved"

    return fake
