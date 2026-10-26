"""Application settings loaded from the environment, .env and the YAML config files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MCP servers are child processes and inherit os.environ, so .env must be exported.
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    confidence_threshold: float = Field(default=0.6, alias="APP_CONFIDENCE_THRESHOLD", ge=0, le=1)
    max_recent_messages: int = Field(default=10, alias="APP_MAX_RECENT_MESSAGES", ge=2)
    keep_last_messages: int = Field(default=4, alias="APP_KEEP_LAST_MESSAGES", ge=1)
    max_tool_iterations: int = Field(default=6, alias="APP_MAX_TOOL_ITERATIONS", ge=1)
    checkpoint_db: Path = Field(default=Path(".data/checkpoints.sqlite"), alias="APP_CHECKPOINT_DB")
    coding_workspace: Path | None = Field(default=None, alias="CODING_WORKSPACE")

    models_config_path: Path = Field(default=PROJECT_ROOT / "config" / "models.yaml")
    mcp_config_path: Path = Field(default=PROJECT_ROOT / "config" / "mcp.yaml")

    def checkpoint_db_path(self) -> Path:
        path = self.checkpoint_db
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load_models_config(self) -> dict[str, Any]:
        return _read_yaml(self.models_config_path)

    def load_mcp_config(self) -> dict[str, Any]:
        return _read_yaml(self.mcp_config_path)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def get_settings() -> Settings:
    return Settings()
