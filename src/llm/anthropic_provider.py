"""Claude provider: a streaming manual tool-use loop on the Anthropic Messages API."""
from __future__ import annotations

import json
import os

import anthropic

from src.agent.tools import Tool, run_tool
from src.llm.base import AgentEvent, EventHandler, MissingCredentialsError

DEFAULT_MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept server-side `fallbacks: "default"` on the Claude API.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
MAX_TOOL_ROUNDS = 12


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 effort: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        if not (self.client.api_key or self.client.auth_token or self.client.credentials):
            raise MissingCredentialsError(
                "No Anthropic API key. Enter one, or set ANTHROPIC_API_KEY (see .env.example).")
        self.model = model or DEFAULT_MODEL
        # Medium suits multi-step tool use; set explicitly since defaults differ by model.
        self.effort = effort or os.getenv("LLM_EFFORT", "medium")

    @staticmethod
    def _tool_params(tools: list[Tool]) -> list[dict]:
        # Tool inputs here are tiny ({"iata": "SFO"}), so they arrive whole and validated by the API;
        # only the answer text is streamed.
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools]

    def _request_options(self) -> dict:
        opts: dict = {}
        if "haiku" not in self.model:  # Haiku 4.5 rejects the effort parameter
            opts["output_config"] = {"effort": self.effort}
        if self.model in FALLBACK_MODELS:
            # On a safety decline, the API re-runs the request on Anthropic's recommended fallback model.
            opts["betas"] = [FALLBACK_BETA]
            opts["fallbacks"] = "default"
        return opts

    def _stream_once(self, history: list, system: str, tools: list[dict], on_event: EventHandler):
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=32000,
            # Breakpoint after tools + system: shared by every conversation. Top-level cache_control
            # adds a second breakpoint that caches the growing conversation within one chat.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=history,
            cache_control={"type": "ephemeral"},
            **self._request_options(),
        ) as stream:
            for event in stream:
                if event.type == "text":
                    on_event(AgentEvent("text", event.text))
            return stream.get_final_message()

    def run_turn(self, history: list, user_text: str, system: str, tools: list[Tool],
                 on_event: EventHandler) -> str:
        history.append({"role": "user", "content": user_text})
        tool_params = self._tool_params(tools)
        final_text: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            response = self._stream_once(history, system, tool_params, on_event)
            on_event(AgentEvent("usage", {
                "input": response.usage.input_tokens,
                "cache_read": response.usage.cache_read_input_tokens,
                "cache_write": response.usage.cache_creation_input_tokens,
                "output": response.usage.output_tokens,
                "model": response.model,
            }))
            history.append({"role": "assistant", "content": response.content})
            final_text = [b.text for b in response.content if b.type == "text"]
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason in ("refusal", "max_tokens") and tool_uses:
                # The turn stopped mid tool call. Every tool_use in history needs a tool_result, or
                # every later request in this conversation is rejected, so close them as not run.
                history.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                     "content": f"Not run: the response stopped early ({response.stop_reason})."}
                    for b in tool_uses]})
            if response.stop_reason == "refusal":
                on_event(AgentEvent("notice", "The model declined this request."))
                return "\n".join(final_text) or "I can't help with that request."
            if response.stop_reason == "pause_turn":
                continue
            if not tool_uses:
                return "\n".join(final_text)
            if response.stop_reason == "max_tokens":
                raise RuntimeError("response was cut off mid tool call; try a narrower question")

            results = []
            for block in tool_uses:
                on_event(AgentEvent("tool_call", {"id": block.id, "name": block.name, "input": block.input}))
                result, is_error = run_tool(block.name, block.input)
                on_event(AgentEvent("tool_result", {"id": block.id, "name": block.name,
                                                    "is_error": is_error, "result": result}))
                results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": is_error,
                                "content": json.dumps(result, default=str)})
            history.append({"role": "user", "content": results})

        on_event(AgentEvent("notice", f"Stopped after {MAX_TOOL_ROUNDS} tool rounds."))
        return "\n".join(final_text)
