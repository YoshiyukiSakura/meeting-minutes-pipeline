from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def _is_repetition_hallucination(text: str, start: float, end: float) -> bool:
    compact = "".join(text.split())
    if not compact:
        return True
    known_bad = ("点赞订阅转发", "明镜与点点栏目", "打赏支持")
    if any(phrase in compact for phrase in known_bad):
        return True
    if end <= start and len(compact) > 8:
        return True
    if len(compact) >= 60:
        counts = {char: compact.count(char) for char in set(compact)}
        if max(counts.values()) / len(compact) >= 0.55:
            return True
    return False


def _dedupe_short_repetitions(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    cleaned: list[dict[str, Any]] = []
    previous = None
    run = 0
    dropped = 0
    for segment in segments:
        normalized = str(segment["text"]).strip().lower()
        if normalized == previous:
            run += 1
        else:
            previous = normalized
            run = 1
        if run > 3 and len(normalized) <= 12:
            dropped += 1
            continue
        cleaned.append(segment)
    return cleaned, dropped


def _ensure_ffmpeg_on_path() -> dict[str, Any]:
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    local_bin = Path(__file__).resolve().parents[1] / ".local_bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    alias = local_bin / "ffmpeg"
    if not alias.exists():
        try:
            alias.symlink_to(ffmpeg)
        except OSError:
            shutil.copy2(ffmpeg, alias)
            alias.chmod(0o755)
    os.environ["PATH"] = f"{local_bin}{os.pathsep}{ffmpeg.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["FFMPEG_BINARY"] = str(alias)
    return {"status": "ok", "path": str(alias), "target": str(ffmpeg)}


def transcribe_audio(audio_path: Path, *, model: str, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ffmpeg_status = _ensure_ffmpeg_on_path()
    try:
        import mlx_whisper  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return [], {
            "engine": "mlx-whisper",
            "status": "unavailable",
            "ffmpeg": ffmpeg_status,
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        kwargs: dict[str, Any] = {
            "path_or_hf_repo": model,
            "verbose": False,
        }
        if language and language.lower() not in {"auto", "detect"}:
            kwargs["language"] = language
        result = mlx_whisper.transcribe(
            str(audio_path),
            **kwargs,
        )
    except TypeError:
        kwargs = {"path_or_hf_repo": model}
        if language and language.lower() not in {"auto", "detect"}:
            kwargs["language"] = language
        result = mlx_whisper.transcribe(
            str(audio_path),
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        return [], {
            "engine": "mlx-whisper",
            "model": model,
            "status": "failed",
            "ffmpeg": ffmpeg_status,
            "error": f"{type(exc).__name__}: {exc}",
        }

    segments: list[dict[str, Any]] = []
    dropped_hallucinations = 0
    for index, segment in enumerate(result.get("segments", []), start=1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if _is_repetition_hallucination(text, start, end):
            dropped_hallucinations += 1
            continue
        words = segment.get("words") or []
        segments.append(
            {
                "id": f"seg_{index:05d}",
                "start": start,
                "end": end,
                "text": text,
                "words": words,
                "speaker": "Speaker Unknown",
                "speaker_confidence": 0.0,
                "name": None,
                "name_source": None,
                "name_confidence": 0.0,
                "frame_refs": [],
            }
        )

    segments, dropped_short_repetitions = _dedupe_short_repetitions(segments)

    return segments, {
        "engine": "mlx-whisper",
        "model": model,
        "language": result.get("language", language),
        "status": "ok",
        "ffmpeg": ffmpeg_status,
        "segments": len(segments),
        "dropped_hallucinations": dropped_hallucinations,
        "dropped_short_repetitions": dropped_short_repetitions,
    }
