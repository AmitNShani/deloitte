"""Provider-neutral interface for the chat agent.

A provider owns its native message format: `history` is an opaque list it appends to, and it
reports progress (streamed text, tool calls, tool results) through `on_event`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from src.agent.tools import Tool


@dataclass
class AgentEvent:
    kind: str  # "text" | "tool_call" | "tool_result" | "notice" | "usage"
    data: Any


EventHandler = Callable[[AgentEvent], None]


class MissingCredentialsError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def run_turn(self, history: list, user_text: str, system: str, tools: list[Tool],
                 on_event: EventHandler) -> str:
        """Answer one user message, calling tools as needed. Appends to `history`; returns final text."""
        ...
