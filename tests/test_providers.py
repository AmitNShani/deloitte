"""Provider plumbing tests with fake model responses (no network, no API key)."""
import json
from types import SimpleNamespace as NS

import pytest

from src.llm import factory, openai_provider
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import MissingCredentialsError
from src.llm.openai_provider import OpenAICompatibleProvider


def chunk(content=None, tool_calls=None, finish=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls), finish_reason=finish)])


def tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class FakeCompletions:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"])))  # snapshot
        return iter(self.scripted.pop(0))


def make(scripted):
    provider = OpenAICompatibleProvider(model="test-model", api_key="k", base_url="http://localhost:1")
    fake = FakeCompletions(scripted)
    provider.client = NS(chat=NS(completions=fake))
    return provider, fake


def test_streamed_tool_call_is_reassembled_and_answered(monkeypatch):
    calls = []
    monkeypatch.setattr(openai_provider, "run_tool",
                        lambda name, args: (calls.append((name, args)) or {"ok": True}, False))
    provider, fake = make([
        [chunk(content="Let me check. "),
         chunk(tool_calls=[tc(0, id="call_1", name="get_airport_profile", args='{"ia')]),
         chunk(tool_calls=[tc(0, args='ta": "SFO"}')], finish="tool_calls")],
        [chunk(content="SFO looks busy."), chunk(finish="stop")],
    ])
    history, events = [], []
    answer = provider.run_turn(history, "How is SFO?", "SYSTEM", [], events.append)

    assert answer == "SFO looks busy."
    assert calls == [("get_airport_profile", {"iata": "SFO"})]
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool", "assistant"]
    assert history[2]["tool_calls"][0]["function"]["arguments"] == '{"iata": "SFO"}'
    assert history[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
    assert fake.requests[1][-1]["role"] == "tool"  # second request carried the tool result
    assert [e.kind for e in events].count("tool_call") == 1


def test_invalid_tool_json_is_reported_to_model_not_raised(monkeypatch):
    monkeypatch.setattr(openai_provider, "run_tool", lambda *a: pytest.fail("must not run"))
    provider, _ = make([
        [chunk(tool_calls=[tc(0, id="c", name="get_airport_profile", args="{not json")], finish="tool_calls")],
        [chunk(content="Sorry.", finish="stop")],
    ])
    history = []
    provider.run_turn(history, "q", "S", [], lambda e: None)
    assert "not valid JSON" in history[3]["content"]


def test_calls_without_index_are_kept_separate(monkeypatch):
    seen = []
    monkeypatch.setattr(openai_provider, "run_tool", lambda name, args: (seen.append(args) or {}, False))
    provider, _ = make([
        [chunk(tool_calls=[tc(None, id="a", name="get_live_status", args='{"iata": "SFO"}'),
                           tc(None, id="b", name="get_live_status", args='{"iata": "LAX"}')],
               finish="tool_calls")],
        [chunk(content="done", finish="stop")],
    ])
    provider.run_turn([], "q", "S", [], lambda e: None)
    assert seen == [{"iata": "SFO"}, {"iata": "LAX"}]


def test_factory_requires_key_for_hosted_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError):
        factory.make_provider(factory.LLMSettings("openai", "gpt-5.5"))


def test_factory_allows_keyless_local_server(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    p = factory.make_provider(factory.LLMSettings("openai-compatible", "llama3.1"))
    assert str(p.client.base_url).startswith("http://localhost:11434")


def test_factory_uses_gemini_endpoint():
    p = factory.make_provider(factory.LLMSettings("gemini", "gemini-3.8-flash", api_key="k"))
    assert "generativelanguage.googleapis.com" in str(p.client.base_url)


# --- a turn that stops mid tool call must leave a valid history (code review finding) ---------

def test_openai_truncated_tool_call_closes_history(monkeypatch):
    monkeypatch.setattr(openai_provider, "run_tool", lambda *a: pytest.fail("must not run"))
    provider, _ = make([
        [chunk(tool_calls=[tc(0, id="c1", name="get_airport_profile", args='{"iata": "SF')], finish="length")],
    ])
    history = []
    with pytest.raises(RuntimeError, match="cut off"):
        provider.run_turn(history, "q", "S", [], lambda e: None)
    # Every tool call in history has its tool message, so the next turn won't be rejected.
    assert history[-1]["role"] == "tool" and history[-1]["tool_call_id"] == "c1"
    assert "stopped early" in history[-1]["content"]


def test_openai_content_filter_mid_tool_call_closes_history(monkeypatch):
    provider, _ = make([
        [chunk(tool_calls=[tc(0, id="c1", name="get_live_status", args="{}")], finish="content_filter")],
    ])
    history = []
    provider.run_turn(history, "q", "S", [], lambda e: None)
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool"]


def anthropic_response(stop_reason, blocks):
    usage = NS(input_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=1)
    return NS(content=blocks, stop_reason=stop_reason, usage=usage, model="claude-test")


TOOL_USE = NS(type="tool_use", id="tu1", name="get_airport_profile", input={"iata": "SFO"})


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_anthropic_stop_mid_tool_call_closes_history(monkeypatch, stop_reason):
    provider = AnthropicProvider(model="claude-test", api_key="test-key")
    monkeypatch.setattr(provider, "_stream_once",
                        lambda *a: anthropic_response(stop_reason, [NS(type="text", text="Checking."), TOOL_USE]))
    history = []
    if stop_reason == "max_tokens":
        with pytest.raises(RuntimeError, match="cut off"):
            provider.run_turn(history, "q", "S", [], lambda e: None)
    else:
        provider.run_turn(history, "q", "S", [], lambda e: None)
    closing = history[-1]
    assert closing["role"] == "user"
    assert [(r["tool_use_id"], r["is_error"]) for r in closing["content"]] == [("tu1", True)]
