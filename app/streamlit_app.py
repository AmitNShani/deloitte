"""Chat UI for the Airport Investment Intelligence Agent.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic
import openai
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

# app/ is on sys.path because Streamlit runs this file as a script, so its sibling module imports directly.
from chat_bar import chat_bar

from src import config
from src.agent.agent import AirportAgent
from src.data import repository
from src.llm.base import AgentEvent, MissingCredentialsError
from src.llm.factory import PRESETS, LLMSettings, env_key, make_provider
from src.voice.stt import TRANSCRIPTION_ERRORS, transcribe

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]

st.set_page_config(page_title="Airport Investment Agent", page_icon="✈️", layout="wide")


@st.cache_resource(show_spinner="Loading public aviation data (first run downloads ~400 MB from BTS)...")
def load_data():
    return repository.dataset()


def _remember(widget_key: str) -> str:
    """Keep a widget's value while it's hidden: Streamlit drops state of widgets not rendered in a run."""
    saved = st.session_state.setdefault("saved_inputs", {})
    if widget_key not in st.session_state and widget_key in saved:
        st.session_state[widget_key] = saved[widget_key]
    return widget_key


def _save(widget_key: str, value):
    st.session_state["saved_inputs"][widget_key] = value
    return value


def model_settings() -> LLMSettings:
    """Sidebar controls for provider, model, base URL and API key (key kept in session memory only)."""
    defaults = LLMSettings.from_env()
    ids = list(PRESETS)
    preset = st.selectbox("Provider", ids, index=ids.index(defaults.preset),
                          format_func=lambda k: PRESETS[k].label)
    p = PRESETS[preset]
    models = p.models or ([defaults.model] if defaults.model and preset == defaults.preset else [])
    # Per-provider widget keys: each provider remembers its own model and key, so a key pasted for
    # one provider is never sent to another.
    model = _save(f"model_{preset}", st.selectbox(
        "Model", models, index=0 if models else None, accept_new_options=True,
        placeholder="Type a model name, e.g. llama3.1", key=_remember(f"model_{preset}")))
    base_url = ""
    if preset == "openai-compatible":
        base_url = _save(f"url_{preset}", st.text_input(
            "Base URL", value=defaults.base_url or p.base_url, key=_remember(f"url_{preset}")))
    from_env = env_key(preset)
    hint = f"Using {p.key_env} from environment" if from_env else (
        "Optional for local servers" if not p.key_required else f"Paste your {p.label} API key")
    api_key = _save(f"key_{preset}", st.text_input(
        "API key", type="password", placeholder=hint, key=_remember(f"key_{preset}"),
        help="Kept only in this browser session's memory; never written to disk."))
    return LLMSettings(preset, model or "", api_key.strip(), base_url.strip())


def start_conversation(settings: LLMSettings):
    st.session_state.settings = settings
    st.session_state.turns = []  # [{"role", "text", "tools": [...]}] for display
    st.session_state.agent = None
    st.session_state.agent_error = None
    try:
        st.session_state.agent = AirportAgent(make_provider(settings))
    except (MissingCredentialsError, ValueError) as exc:
        st.session_state.agent_error = str(exc)


ds = load_data()

# --- sidebar ---------------------------------------------------------------------------------
with st.sidebar:
    st.header("✈️ Airport Investment Agent")
    st.caption("Screens US airports for renovation opportunities: where demand presses against "
               "capacity and is still growing.")

    with st.expander("Model settings", expanded=True):
        settings = model_settings()
    if st.session_state.get("settings") != settings:
        start_conversation(settings)  # histories are provider-specific, so a change starts fresh
    if st.button("New conversation", use_container_width=True):
        start_conversation(settings)
        st.rerun()

    st.subheader("Try asking")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True, disabled=st.session_state.agent is None):
            st.session_state.pending = q

    with st.expander("How scoring works"):
        st.markdown(
            f"**1. Gate** (must pass both): passenger growth > {config.GATE_MIN_PASSENGER_GROWTH_PCT:.0f}% "
            f"and load factor ≥ {config.GATE_MIN_LOAD_FACTOR_PCT:.0f}%.\n\n"
            "**2. Weighted rank** of percentiles vs the top "
            f"{config.REFERENCE_SET_SIZE} US airports:\n\n"
            + "\n".join(f"- {k.title()}: {w:.0%}" for k, w in config.WEIGHTS.items())
            + "\n\nLive FAA status is shown as context, never scored. See `docs/DESIGN.md`."
        )
    with st.expander("Data sources"):
        st.caption(f"Window: {ds.window_label} (trailing 12 months)")
        for s in ds.sources:
            st.markdown(f"**{s['source']}**  \nfetched {s['fetched_at']}"
                        + (" ⚠️ stale copy" if s["stale_copy"] else "")
                        + (f"  \n_{s['note']}_" if s["note"] else ""))
        st.markdown("**BTS T-100 Segment (TranStats FMG)**: route data, loaded on demand  \n"
                    "**FAA NAS Status**: live, on demand")


def render_tools(tools: list[dict]):
    if not tools:
        return
    with st.expander(f"🔧 {len(tools)} tool call{'s' if len(tools) != 1 else ''}"):
        for t in tools:
            icon = "❌" if t.get("is_error") else "✅"
            st.markdown(f"{icon} `{t['name']}({json.dumps(t['input'])})`")


# --- history ---------------------------------------------------------------------------------
st.title("Airport Investment Intelligence")
if st.session_state.agent is None:
    st.info(f"**Set up a model to start chatting.** {st.session_state.agent_error}  \n"
            "Choose a provider and paste an API key under *Model settings* in the sidebar.")


def user_text(turn: dict) -> str:
    return f"🎤 {turn['text']}" if turn.get("voice") else turn["text"]


for turn in st.session_state.turns:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant":
            render_tools(turn["tools"])
        st.markdown(user_text(turn) if turn["role"] == "user" else turn["text"])

# Chat bar pinned to the bottom of the page: one button that is a mic when the box is empty and
# send when there's text.
with st.bottom:
    if st.session_state.agent is not None:
        typed, recording = chat_bar()
    else:
        typed, recording = st.chat_input("Set up a model in the sidebar to start chatting", disabled=True), None
question, voice = typed, False
if recording is not None:
    try:
        with st.spinner("Transcribing (first use downloads a ~140 MB speech model)..."):
            question = transcribe(recording)
        voice = True
        if not question:
            st.warning("I couldn't hear a question in that recording. Try again, closer to the mic.")
    except TRANSCRIPTION_ERRORS as exc:  # shouldn't break the chat; typing still works
        st.error(f"Voice transcription failed: {exc}. You can type the question instead.")
question = question or st.session_state.pop("pending", None)

# --- new turn --------------------------------------------------------------------------------
if question and st.session_state.agent is not None:
    turn = {"role": "user", "text": question, "tools": [], "voice": voice}
    st.session_state.turns.append(turn)
    with st.chat_message("user"):
        st.markdown(user_text(turn))

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)
        placeholder = st.empty()
        buffer: list[str] = []
        tools: list[dict] = []

        def on_event(e: AgentEvent):
            if e.kind == "text":
                buffer.append(e.data)
                placeholder.markdown("".join(buffer) + "▌")
            elif e.kind == "tool_call":
                if buffer and not buffer[-1].endswith("\n\n"):
                    buffer.append("\n\n")
                tools.append({"name": e.data["name"], "input": e.data["input"]})
                status.update(label=f"Running {e.data['name']}...")
                status.write(f"`{e.data['name']}({json.dumps(e.data['input'])})`")
            elif e.kind == "tool_result":
                tools[-1]["is_error"] = e.data["is_error"]
                if e.data["is_error"]:
                    status.write(f"⚠️ {e.data['result'].get('error')}")
            elif e.kind == "notice":
                status.write(f"ℹ️ {e.data}")

        try:
            st.session_state.agent.ask(question, on_event)
            status.update(label=f"Done · {len(tools)} tool call{'s' if len(tools) != 1 else ''}",
                          state="complete")
        except (anthropic.AuthenticationError, openai.AuthenticationError):
            status.update(label="Not authenticated", state="error")
            buffer.append("**The API key was rejected.** Check it under *Model settings*.")
        except (anthropic.NotFoundError, openai.NotFoundError):
            status.update(label="Model not found", state="error")
            buffer.append(f"**Model `{st.session_state.settings.model}` was not found** by the provider. "
                          "Check the model name under *Model settings*.")
        except (anthropic.APIError, openai.APIError) as exc:
            status.update(label="API error", state="error")
            buffer.append(f"**The model request failed:** {exc}")
        except RuntimeError as exc:
            status.update(label="Stopped", state="error")
            buffer.append(f"**Stopped:** {exc}")

        text = "".join(buffer).strip()
        placeholder.markdown(text)
        st.session_state.turns.append({"role": "assistant", "text": text, "tools": tools})
