"""Speech-to-text for voice questions, run locally with faster-whisper.

Local on purpose: it needs no API key, so voice works whichever LLM provider the user picked, and
the recording never leaves the machine. The model (~140 MB for base.en) downloads on first use.
"""
from __future__ import annotations

import wave
from functools import lru_cache
from typing import BinaryIO

import numpy as np

from src import config

_SAMPLE_RATE = 16_000  # what Whisper expects

# What transcription can raise: faster-whisper missing or broken (ImportError), model download or
# network failure (OSError), a malformed or truncated WAV (wave.Error, EOFError, ValueError) and
# inference failures in the model runtime (RuntimeError). The app shows these as a message; typing
# still works.
TRANSCRIPTION_ERRORS = (ImportError, OSError, wave.Error, EOFError, ValueError, RuntimeError)

# Whisper uses this as preceding context, which biases it toward the domain's vocabulary
# (airport codes and aviation terms that a general model would otherwise mis-hear).
VOCABULARY_PROMPT = (
    "A question about US airports, such as SFO, LAX, SNA, BOS, JFK, ANC, Anchorage, Santa Ana, "
    "New England, load factor, congestion, terminal expansion, long-haul flights, unmet demand."
)


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")


def _wav_samples(audio: BinaryIO) -> np.ndarray:
    """16-bit PCM WAV -> mono float32 samples at Whisper's 16 kHz.

    Decoded here rather than by faster-whisper's PyAV path, which breaks with av>=19 (it passes an
    argument av removed). Streamlit's recorder always produces WAV, so stdlib `wave` is enough.
    """
    with wave.open(audio, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got {8 * width}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != _SAMPLE_RATE:
        n = int(len(samples) * _SAMPLE_RATE / rate)
        samples = np.interp(np.linspace(0, len(samples), n, endpoint=False),
                            np.arange(len(samples)), samples).astype(np.float32)
    return samples


def transcribe(audio: BinaryIO) -> str:
    """Transcribe an English WAV recording. Returns an empty string if nothing was said."""
    segments, _ = _model().transcribe(_wav_samples(audio), language="en", vad_filter=True,
                                      initial_prompt=VOCABULARY_PROMPT)
    return " ".join(s.text.strip() for s in segments).strip()  # segments decode lazily here
