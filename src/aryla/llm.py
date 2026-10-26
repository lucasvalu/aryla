"""Chat model loading; the only module that knows about model providers."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from aryla.config import Settings
from aryla.observability import flow

ROLE_ORCHESTRATOR = "orchestrator"
ROLE_CODING = "coding"
ROLE_GENERAL = "general"
ROLE_SUMMARIZER = "summarizer"
ALL_ROLES = (ROLE_ORCHESTRATOR, ROLE_CODING, ROLE_GENERAL, ROLE_SUMMARIZER)


class ModelProfile(BaseModel):
    provider: str
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    num_ctx: int = Field(default=8192, ge=2048)
    reasoning: bool | None = False


def load_models(settings: Settings) -> dict[str, BaseChatModel]:
    return {role: load_model(role, settings) for role in ALL_ROLES}


def load_model(role: str, settings: Settings) -> BaseChatModel:
    profile = resolve_profile(settings.load_models_config(), role)
    flow.info(
        "models: %s -> %s/%s (temperature %s, num_ctx %s, reasoning %s)",
        role,
        profile.provider,
        profile.model,
        profile.temperature,
        profile.num_ctx,
        profile.reasoning,
    )
    return build_chat_model(profile, settings)


def resolve_profile(models_config: dict[str, Any], role: str) -> ModelProfile:
    """Return the profile for a role, following string aliases; unknown roles use `default`."""
    profiles: dict[str, Any] = {"default": models_config.get("default", {})}
    profiles.update(models_config.get("roles", {}) or {})

    entry: Any = profiles.get(role, "default")
    visited: list[str] = [role]
    while isinstance(entry, str):
        if entry in visited:
            raise ValueError(f"Cyclic alias in models.yaml: {' -> '.join(visited + [entry])}")
        visited.append(entry)
        if entry not in profiles:
            raise ValueError(f"Profile '{entry}' (used by role '{role}') does not exist")
        entry = profiles[entry]

    if not entry:
        raise ValueError(f"No profile for role '{role}' and no 'default' profile")
    return ModelProfile.model_validate(entry)


def build_chat_model(profile: ModelProfile, settings: Settings) -> BaseChatModel:
    if profile.provider == "ollama":
        from langchain_ollama import ChatOllama

        # num_ctx has to be set here: a per-call `options=` replaces the whole options dict.
        return ChatOllama(
            model=profile.model,
            temperature=profile.temperature,
            num_ctx=profile.num_ctx,
            reasoning=profile.reasoning,
            base_url=settings.ollama_base_url,
        )

    from langchain.chat_models import init_chat_model

    return init_chat_model(
        profile.model,
        model_provider=profile.provider,
        temperature=profile.temperature,
    )
