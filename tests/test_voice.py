"""WAV decoding for voice input (no model download, no network)."""
import io
import wave

import numpy as np
import pytest

from src.voice.stt import _wav_samples


def make_wav(samples: np.ndarray, rate: int, channels: int = 1, width: int = 2) -> io.BytesIO:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    buf.seek(0)
    return buf


def test_mono_16k_passes_through_scaled():
    out = _wav_samples(make_wav(np.array([0, 16384, -32768], dtype="<i2"), 16_000))
    assert out.dtype == np.float32
    assert out.tolist() == [0.0, 0.5, -1.0]


def test_stereo_is_averaged_to_mono():
    interleaved = np.array([1000, 3000, -2000, -4000], dtype="<i2")  # two stereo frames
    out = _wav_samples(make_wav(interleaved, 16_000, channels=2))
    assert out == pytest.approx(np.array([2000, -3000]) / 32768.0)


def test_other_rates_are_resampled_to_16k():
    one_second_44k = np.zeros(44_100, dtype="<i2")
    assert len(_wav_samples(make_wav(one_second_44k, 44_100))) == 16_000


def test_rejects_non_16_bit():
    with pytest.raises(ValueError, match="16-bit"):
        _wav_samples(make_wav(np.zeros(10, dtype="u1"), 16_000, width=1))


def test_missing_speech_package_is_a_handled_transcription_error(monkeypatch):
    # If faster-whisper isn't installed or is broken, the app must show a message, not a traceback.
    import sys

    from src.voice import stt

    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    stt._model.cache_clear()
    with pytest.raises(stt.TRANSCRIPTION_ERRORS):
        stt.transcribe(make_wav(np.zeros(1600, dtype="<i2"), 16_000))
    stt._model.cache_clear()
