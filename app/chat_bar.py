"""Chat input bar with voice: a Streamlit custom component (inline HTML/CSS/JS, no build step).

One button next to the text box: a mic when the box is empty, send when there's text, stop while
recording. Voice recording plays a chime, shows a level meter and stops by itself when the speaker
pauses (see chat_bar.js).
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit as st

_DIR = Path(__file__).parent
_HTML = ('<div class="bar"><textarea rows="1" aria-label="Message"></textarea>'
         '<button type="button"></button><div class="meter"><div></div></div></div>'
         '<div class="status" aria-live="polite"></div>')

_component = st.components.v2.component(
    "chat_bar",
    html=_HTML,
    css=(_DIR / "chat_bar.css").read_text(),
    js=(_DIR / "chat_bar.js").read_text(),
)


def chat_bar(key: str = "chat_bar") -> tuple[str | None, io.BytesIO | None]:
    """Render the bar. Returns (typed text, WAV recording) on the run right after a submission."""
    result = _component(key=key, on_text_change=lambda: None, on_audio_change=lambda: None)
    text = getattr(result, "text", None)
    audio = getattr(result, "audio", None)
    return text or None, io.BytesIO(base64.b64decode(audio)) if audio else None
