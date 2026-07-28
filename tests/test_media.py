import subprocess
from pathlib import Path

import pytest

from meeting_minutes import media


def test_make_clip_prefers_bundled_ffmpeg(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "find_ffmpeg", lambda: "/tmp/ffmpeg")
    monkeypatch.setattr(media, "run_cmd", lambda args, **_kwargs: commands.append(args))

    destination = media.make_clip(Path("/tmp/input.mov"), tmp_path / "clip.mov", start=12.5, duration=34.25)

    assert destination == tmp_path / "clip.mov"
    assert commands == [
        [
            "/tmp/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            "12.500",
            "-i",
            "/tmp/input.mov",
            "-t",
            "34.250",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(tmp_path / "clip.mov"),
        ]
    ]


def test_make_clip_falls_back_to_avconvert(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "find_ffmpeg", lambda: None)
    monkeypatch.setattr(media, "run_cmd", lambda args, **_kwargs: commands.append(args))

    media.make_clip(Path("/tmp/input.mov"), tmp_path / "clip.mov", start=5, duration=10)

    assert commands[0][0] == "avconvert"


def test_extract_audio_uses_bundled_ffmpeg(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "find_ffmpeg", lambda: "/tmp/ffmpeg")
    monkeypatch.setattr(media, "run_cmd", lambda args, **_kwargs: commands.append(args))

    media.extract_audio(Path("/tmp/input.mov"), tmp_path / "audio.wav")

    assert commands[0][0] == "/tmp/ffmpeg"
    assert commands[0][-1] == str(tmp_path / "audio.wav")


def test_run_cmd_wraps_timeout(monkeypatch):
    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["slow-tool"], timeout=5)

    monkeypatch.setattr(media.subprocess, "run", raise_timeout)

    with pytest.raises(media.PipelineError, match="timed out after 5s: slow-tool"):
        media.run_cmd(["slow-tool"], timeout=5)
