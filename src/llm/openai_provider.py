"""OpenAI-compatible provider: a streaming tool-call loop on the Chat Completions API.

Chat Completions is the format OpenAI, Google Gemini (OpenAI-compatible endpoint), Ollama, vLLM
and many others accept, so one implementation covers all of them via `base_url`.
"""
from __future__ import annotations

import json

import openai

from src.agent.tools import Tool, run_tool
from src.llm.base import AgentEvent, EventHandler

MAX_TOOL_ROUNDS = 12


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def _tool_params(tools: list[Tool]) -> list[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.input_schema}} for t in tools]

    def _stream_once(self, history: list, tools: list[dict], on_event: EventHandler):
        """One model call. Returns (text, tool_calls, finish_reason)."""
        stream = self.client.chat.completions.create(model=self.model, messages=history, tools=tools,
                                                     stream=True)
        text: list[str] = []
        calls: dict = {}  # stream index -> {"id", "name", "arguments"}
        finish = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                text.append(delta.content)
                on_event(AgentEvent("text", delta.content))
            for tc in delta.tool_calls or []:
                # Some compatible servers send whole calls without an index; key by id then.
                key = tc.index if tc.index is not None else (tc.id or len(calls))
                slot = calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
            if choice.finish_reason:
                finish = choice.finish_reason
        tool_calls = []
        for i, c in enumerate(calls.values()):
            tool_calls.append({"id": c["id"] or f"call_{i}", "type": "function",
                               "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}})
        return "".join(text), tool_calls, finish

    def run_turn(self, history: list, user_text: str, system: str, tools: list[Tool],
                 on_event: EventHandler) -> str:
        if not history:
            history.append({"role": "system", "content": system})
        history.append({"role": "user", "content": user_text})
        tool_params = self._tool_params(tools)
        text = ""

        for _ in range(MAX_TOOL_ROUNDS):
            text, tool_calls, finish = self._stream_once(history, tool_params, on_event)
            message = {"role": "assistant", "content": text or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            history.append(message)

            if tool_calls and finish in ("content_filter", "length"):
                # The turn stopped mid tool call. Every tool call in history needs a tool message, or
                # every later request in this conversation is rejected, so close them as not run.
                history.extend({"role": "tool", "tool_call_id": call["id"],
                                "content": f"Not run: the response stopped early ({finish})."}
                               for call in tool_calls)
            if finish == "content_filter":
                on_event(AgentEvent("notice", "The provider's content filter stopped this response."))
                return text
            if not tool_calls:
                return text
            if finish == "length":
                raise RuntimeError("response was cut off mid tool call; try a narrower question")

            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = None
                on_event(AgentEvent("tool_call", {"id": call["id"], "name": name, "input": args}))
                if args is None:
                    result, is_error = {"error": "tool arguments were not valid JSON; retry the call"}, True
                else:
                    result, is_error = run_tool(name, args)
                on_event(AgentEvent("tool_result", {"id": call["id"], "name": name,
                                                    "is_error": is_error, "result": result}))
                history.append({"role": "tool", "tool_call_id": call["id"],
                                "content": json.dumps(result, default=str)})

        on_event(AgentEvent("notice", f"Stopped after {MAX_TOOL_ROUNDS} tool rounds."))
        return text
