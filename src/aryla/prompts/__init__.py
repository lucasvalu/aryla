"""Prompt texts, one Markdown file per prompt, loaded by name."""

from importlib import resources


def load(name: str) -> str:
    return (resources.files(__package__) / f"{name}.md").read_text(encoding="utf-8")
