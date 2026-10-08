"""Provider presets and construction. The UI and CLI both build providers through here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.llm.base import LLMProvider, MissingCredentialsError


@dataclass(frozen=True)
class Preset:
    label: str
    kind: str                          # "anthropic" | "openai" (OpenAI Chat Completions format)
    models: list[str] = field(default_factory=list)
    key_env: str | None = None      # env var holding the API key
    base_url: str | None = None
    key_required: bool = True


PRESETS = {
    "anthropic": Preset("Anthropic Claude", "anthropic",
                        ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5"], "ANTHROPIC_API_KEY"),
    "openai": Preset("OpenAI", "openai", ["gpt-5.5"], "OPENAI_API_KEY"),
    "gemini": Preset("Google Gemini", "openai", ["gemini-3.8-flash"], "GEMINI_API_KEY",
                     "https://generativelanguage.googleapis.com/v1beta/openai/"),
    "openai-compatible": Preset("Other OpenAI-compatible (Ollama, vLLM, ...)", "openai", [],
                                "LLM_API_KEY", "http://localhost:11434/v1", key_required=False),
}


@dataclass(frozen=True)
class LLMSettings:
    preset: str
    model: str
    api_key: str = ""     # empty = read from the preset's environment variable
    base_url: str = ""    # empty = preset default

    @classmethod
    def from_env(cls) -> LLMSettings:
        preset = os.getenv("LLM_PROVIDER", "anthropic")
        if preset not in PRESETS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(PRESETS)}")
        p = PRESETS[preset]
        return cls(preset, os.getenv("LLM_MODEL") or (p.models[0] if p.models else ""),
                   base_url=os.getenv("LLM_BASE_URL", ""))


def env_key(preset: str) -> str:
    p = PRESETS[preset]
    return os.getenv(p.key_env, "") if p.key_env else ""


def make_provider(s: LLMSettings) -> LLMProvider:
    p = PRESETS[s.preset]
    api_key = s.api_key or env_key(s.preset)
    if not s.model:
        raise ValueError("choose a model name")
    if p.kind == "anthropic":
        from src.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=s.model, api_key=api_key or None)
    if p.key_required and not api_key:
        raise MissingCredentialsError(f"No API key for {p.label}. Enter one, or set {p.key_env}.")
    from src.llm.openai_provider import OpenAICompatibleProvider
    return OpenAICompatibleProvider(model=s.model, api_key=api_key or "not-needed",
                                    base_url=s.base_url or p.base_url)
