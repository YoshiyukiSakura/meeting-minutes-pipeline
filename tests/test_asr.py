import sys
from types import SimpleNamespace

from meeting_minutes import asr


def _result():
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": " hello",
                "words": [{"word": " hello", "start": 0.0, "end": 1.0}],
            }
        ],
    }


def test_transcribe_requests_word_timestamps(monkeypatch, tmp_path):
    calls = []

    def transcribe(path, **kwargs):
        calls.append((path, kwargs))
        return _result()

    monkeypatch.setattr(asr, "_ensure_ffmpeg_on_path", lambda: {"status": "ok"})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))

    segments, status = asr.transcribe_audio(tmp_path / "meeting.wav", model="model-id", language="en")

    assert calls[0][1]["word_timestamps"] is True
    assert calls[0][1]["verbose"] is False
    assert segments[0]["words"][0]["word"] == " hello"
    assert status["segments_with_word_timestamps"] == 1


def test_transcribe_keeps_word_timestamps_when_verbose_is_unsupported(monkeypatch, tmp_path):
    calls = []

    def transcribe(path, **kwargs):
        calls.append(kwargs)
        if "verbose" in kwargs:
            raise TypeError("verbose is unsupported")
        return _result()

    monkeypatch.setattr(asr, "_ensure_ffmpeg_on_path", lambda: {"status": "ok"})
    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))

    asr.transcribe_audio(tmp_path / "meeting.wav", model="model-id", language="auto")

    assert len(calls) == 2
    assert calls[1]["word_timestamps"] is True
    assert "verbose" not in calls[1]
