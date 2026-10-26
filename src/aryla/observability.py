"""Logging (an opt-in flow log of the main actions) and optional Langfuse tracing."""

from __future__ import annotations

import functools
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

APP_LOGGER_NAME = "aryla"
FLOW_LOGGER_NAME = "aryla.flow"


class FlowHandler(logging.StreamHandler):
    """Calls `before_emit` first so the CLI can close a half-printed line before a log line."""

    before_emit: Callable[[], None] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if self.before_emit:
            self.before_emit()
        super().emit(record)


flow = logging.getLogger(FLOW_LOGGER_NAME)
flow_handler = FlowHandler(sys.stderr)


def setup_logging(debug: bool = False, verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "mcp", "fastmcp"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if debug else logging.WARNING)

    flow_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    flow.handlers.clear()
    flow.addHandler(flow_handler)
    flow.propagate = False
    flow.setLevel(logging.INFO if verbose or debug else logging.WARNING)


def describe_content(text: str, limit: int = 300) -> str:
    """Return the text for logs only at DEBUG; otherwise just its length."""
    if not logging.getLogger(APP_LOGGER_NAME).isEnabledFor(logging.DEBUG):
        return f"<{len(text)} chars>"
    if len(text) <= limit:
        return repr(text)
    return repr(text[:limit] + "...")


def short(value: object, limit: int = 80) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# --- tracing ---------------------------------------------------------------

tracing_logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


@functools.cache
def get_callbacks() -> list[Any]:
    if not tracing_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        tracing_logger.warning(
            "LANGFUSE_* is set but langfuse is not installed; run `uv sync --extra tracing`"
        )
        return []
    tracing_logger.info(
        "Langfuse tracing enabled (%s)", os.environ.get("LANGFUSE_BASE_URL", "default host")
    )
    return [CallbackHandler()]


def trace_metadata(thread_id: str) -> dict[str, Any]:
    if tracing_enabled():
        return {"langfuse_session_id": thread_id}
    return {}


def shutdown_tracing() -> None:
    if not tracing_enabled():
        return
    try:
        from langfuse import get_client
    except ImportError:
        return
    get_client().shutdown()
