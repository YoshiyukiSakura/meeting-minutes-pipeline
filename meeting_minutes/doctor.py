from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sys
import urllib.error
import urllib.request
from typing import Any


def _check_executable(name: str, *, required: bool, note: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {
        "name": name,
        "kind": "executable",
        "required": required,
        "status": "ok" if path else "missing",
        "path": path,
        "note": note,
    }


def _check_module(name: str, *, required: bool, note: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    return {
        "name": name,
        "kind": "python_module",
        "required": required,
        "status": "ok" if spec else "missing",
        "note": note,
    }


def _check_env(name: str, *, required: bool, note: str) -> dict[str, Any]:
    value = os.environ.get(name)
    return {
        "name": name,
        "kind": "environment",
        "required": required,
        "status": "ok" if value else "missing",
        "note": note,
    }


def _check_ollama(timeout: float = 1.5) -> dict[str, Any]:
    req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            "name": "ollama",
            "kind": "local_service",
            "required": False,
            "status": "missing",
            "note": f"Optional local LLM summaries are unavailable: {type(exc).__name__}.",
        }
    models = [str(item.get("name")) for item in payload.get("models", []) if item.get("name")]
    return {
        "name": "ollama",
        "kind": "local_service",
        "required": False,
        "status": "ok",
        "models": models,
        "note": "Optional local LLM summaries are available.",
    }


def collect_environment_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = [
        {
            "name": "python",
            "kind": "runtime",
            "required": True,
            "status": "ok" if sys.version_info >= (3, 11) else "missing",
            "path": sys.executable,
            "version": platform.python_version(),
            "note": "Python >= 3.11 is required.",
        },
        _check_executable("swift", required=True, note="Required for macOS media probing, frame extraction, and OCR."),
        _check_executable("avconvert", required=False, note="Required only when creating source clips with --start/--duration."),
        _check_executable("afconvert", required=False, note="Audio extraction fallback when ffmpeg is not on PATH."),
        _check_executable("ffmpeg", required=False, note="Preferred audio/video utility; imageio-ffmpeg can provide a bundled binary."),
        _check_module("imageio_ffmpeg", required=True, note="Provides a bundled ffmpeg binary when system ffmpeg is unavailable."),
        _check_module("mlx_whisper", required=True, note="Primary local Whisper ASR backend on Apple Silicon."),
        _check_module("speechbrain", required=False, note="Recommended no-token ECAPA speaker clustering backend."),
        _check_module("torchaudio", required=False, note="Used by local diarization and voiceprint code paths."),
        _check_module("pyannote.audio", required=False, note="Optional diarization backend; requires HF_TOKEN and model access."),
        _check_module("PIL", required=True, note="Pillow is used for keyframe comparison and visual identity contact sheets."),
        _check_module("numpy", required=True, note="Used by diarization and visual scoring helpers."),
        _check_env("HF_TOKEN", required=False, note="Required only for pyannote diarization."),
        _check_ollama(),
    ]
    return checks


def render_doctor_report(checks: list[dict[str, Any]]) -> str:
    lines = ["# Meeting Minutes Pipeline Doctor", ""]
    for check in checks:
        marker = "OK" if check["status"] == "ok" else "MISSING"
        required = "required" if check.get("required") else "optional"
        detail = check.get("path") or check.get("version") or ""
        if check.get("models"):
            detail = ", ".join(check["models"][:5])
        suffix = f" - {detail}" if detail else ""
        lines.append(f"- [{marker}] {check['name']} ({required}){suffix}")
        if check.get("note"):
            lines.append(f"  {check['note']}")
    missing_required = [check["name"] for check in checks if check.get("required") and check["status"] != "ok"]
    lines += ["", "## Result"]
    if missing_required:
        lines.append("- Missing required checks: " + ", ".join(missing_required))
    else:
        lines.append("- Required checks passed.")
    return "\n".join(lines) + "\n"


def doctor_exit_code(checks: list[dict[str, Any]], *, strict: bool) -> int:
    if strict and any(check.get("required") and check["status"] != "ok" for check in checks):
        return 1
    return 0
