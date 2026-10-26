import pytest
from langchain_ollama import ChatOllama

from aryla.llm import ModelProfile, build_chat_model, resolve_profile

CONFIG = {
    "default": {"provider": "ollama", "model": "qwen3:4b", "temperature": 0},
    "roles": {
        "orchestrator": {"provider": "ollama", "model": "qwen3:1.7b", "num_ctx": 16384},
        "coding": "default",
        "summarizer": "coding",
    },
}


def test_full_profile_is_validated():
    profile = resolve_profile(CONFIG, "orchestrator")
    assert profile.model == "qwen3:1.7b"
    assert profile.num_ctx == 16384
    assert profile.reasoning is False


def test_alias_and_chained_alias_resolve_to_default():
    assert resolve_profile(CONFIG, "coding").model == "qwen3:4b"
    assert resolve_profile(CONFIG, "summarizer").model == "qwen3:4b"


def test_unknown_role_falls_back_to_default():
    assert resolve_profile(CONFIG, "missing_role").model == "qwen3:4b"


def test_cyclic_alias_is_reported():
    config = {"default": CONFIG["default"], "roles": {"a": "b", "b": "a"}}
    with pytest.raises(ValueError, match="Cyclic"):
        resolve_profile(config, "a")


def test_missing_alias_target_is_reported():
    config = {"default": CONFIG["default"], "roles": {"a": "nope"}}
    with pytest.raises(ValueError, match="does not exist"):
        resolve_profile(config, "a")


def test_ollama_model_gets_context_window_and_reasoning(settings):
    profile = ModelProfile(provider="ollama", model="qwen3:4b", num_ctx=8192, reasoning=False)
    model = build_chat_model(profile, settings)
    assert isinstance(model, ChatOllama)
    assert model.num_ctx == 8192
    assert model.reasoning is False
    assert model.base_url == settings.ollama_base_url


def test_shipped_models_yaml_resolves_every_role(settings):
    config = settings.load_models_config()
    for role in ("orchestrator", "coding", "general", "summarizer"):
        assert resolve_profile(config, role).num_ctx >= 8192
