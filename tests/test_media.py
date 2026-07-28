import subprocess
from types import SimpleNamespace
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


def test_ocr_frames_filters_malformed_records_without_rewriting_raw_manifest(monkeypatch, tmp_path):
    payloads: list[tuple[Path, list[dict]]] = []

    def fake_run(args, **_kwargs):
        payload_path = Path(args[-1])
        payloads.append((payload_path, media.json.loads(payload_path.read_text(encoding="utf-8"))))
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(media, "run_cmd", fake_run)
    frames = [
        {"time": 1.0, "actualTime": 0.99, "path": "/tmp/valid.jpg"},
        {"time": 2.0, "error": "AVFoundation Cannot Open"},
        {"time": "3.0", "actualTime": 3.0, "path": "/tmp/string-time.jpg"},
        {"time": True, "actualTime": 4.0, "path": "/tmp/bool-time.jpg"},
        {"time": 5.0, "actualTime": float("nan"), "path": "/tmp/nonfinite.jpg"},
    ]

    with pytest.warns(RuntimeWarning, match="AVFoundation Cannot Open"):
        assert media.ocr_frames(frames, tmp_path) == []

    payload_path, payload = payloads[0]
    assert payload == [{"time": 1.0, "actualTime": 0.99, "path": "/tmp/valid.jpg"}]
    assert not payload_path.exists()
    assert media.json.loads((tmp_path / "frames_for_ocr.json").read_text(encoding="utf-8"))[1]["error"] == "AVFoundation Cannot Open"
    diagnostics = media.ocr_manifest_diagnostics(tmp_path)
    assert diagnostics["status"] == "ok"
    assert diagnostics["accepted_records"] == 1
    assert diagnostics["dropped_record_count"] == 4
    assert diagnostics["dropped_records"][0] == {
        "index": 1,
        "reason": "invalid_actual_time",
        "time": 2.0,
        "error": "AVFoundation Cannot Open",
    }


def test_ocr_frames_fails_distinctly_when_no_valid_records(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "run_cmd", lambda args, **_kwargs: commands.append(args))

    with pytest.raises(media.PipelineError, match="all 1 frame records were malformed"):
        media.ocr_frames([{"time": 1.0, "error": "Cannot Open"}], tmp_path)

    assert commands == []
    diagnostics = media.ocr_manifest_diagnostics(tmp_path)
    assert diagnostics["status"] == "all_records_invalid"
    assert diagnostics["dropped_record_count"] == 1


def test_ocr_frames_fails_distinctly_when_no_frames_were_extracted(monkeypatch, tmp_path):
    commands: list[list[str]] = []
    monkeypatch.setattr(media, "run_cmd", lambda args, **_kwargs: commands.append(args))

    with pytest.raises(media.PipelineError, match="no frames extracted"):
        media.ocr_frames([], tmp_path)

    assert commands == []
    assert media.ocr_manifest_diagnostics(tmp_path)["status"] == "no_frames_extracted"
