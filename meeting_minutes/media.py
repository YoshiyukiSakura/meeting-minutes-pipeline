from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

from .jsonio import write_json


ROOT = Path(__file__).resolve().parents[1]
SWIFT_HELPER = ROOT / "swift" / "macos_media.swift"
OCR_MANIFEST_DIAGNOSTICS = "ocr_manifest_diagnostics.json"


class PipelineError(RuntimeError):
    pass


def run_cmd(args: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"Command timed out after {timeout}s: {' '.join(args)}") from exc
    if proc.returncode != 0:
        raise PipelineError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def probe_media(input_path: Path) -> dict[str, Any]:
    proc = run_cmd(["swift", str(SWIFT_HELPER), "probe", str(input_path)], timeout=120)
    return json.loads(proc.stdout)


def find_ffmpeg() -> str | None:
    if ffmpeg := shutil.which("ffmpeg"):
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    return str(bundled) if bundled.is_file() else None


def make_clip(input_path: Path, output_path: Path, *, start: float, duration: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg := find_ffmpeg():
        run_cmd(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(input_path),
                "-t",
                f"{duration:.3f}",
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
                str(output_path),
            ],
            timeout=900,
        )
        return output_path
    run_cmd(
        [
            "avconvert",
            "--source",
            str(input_path),
            "--preset",
            "PresetPassthrough",
            "--output",
            str(output_path),
            "--start",
            f"{start:.3f}",
            "--duration",
            f"{duration:.3f}",
            "--replace",
        ],
        timeout=900,
    )
    return output_path


def extract_audio(input_path: Path, wav_path: Path) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        run_cmd(
            [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                str(wav_path),
            ],
            timeout=1800,
        )
        return wav_path

    run_cmd(
        [
            "afconvert",
            str(input_path),
            str(wav_path),
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
        ],
        timeout=1800,
    )
    return wav_path


def extract_frames(
    input_path: Path,
    times: list[float],
    out_dir: Path,
    *,
    max_width: int = 1280,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    times_path = out_dir / "requested_times.json"
    write_json(times_path, sorted({round(max(0.0, t), 3) for t in times}))
    proc = run_cmd(
        [
            "swift",
            str(SWIFT_HELPER),
            "frames",
            str(input_path),
            str(times_path),
            str(out_dir),
            str(max_width),
        ],
        timeout=1800,
    )
    manifest = json.loads(proc.stdout)
    write_json(out_dir / "frames_manifest.json", manifest)
    return manifest


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _dropped_ocr_record(index: int, record: Any, reason: str) -> dict[str, Any]:
    dropped: dict[str, Any] = {"index": index, "reason": reason}
    if not isinstance(record, dict):
        return dropped
    time_value = _finite_number(record.get("time"))
    if time_value is not None:
        dropped["time"] = time_value
    error = record.get("error")
    if isinstance(error, str) and error:
        dropped["error"] = error
    return dropped


def _normalized_ocr_manifest(frames_manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for index, record in enumerate(frames_manifest):
        if not isinstance(record, dict):
            dropped.append(_dropped_ocr_record(index, record, "record_not_object"))
            continue
        time_value = _finite_number(record.get("time"))
        if time_value is None:
            dropped.append(_dropped_ocr_record(index, record, "invalid_time"))
            continue
        actual_time = _finite_number(record.get("actualTime"))
        if actual_time is None:
            dropped.append(_dropped_ocr_record(index, record, "invalid_actual_time"))
            continue
        path = record.get("path")
        if not isinstance(path, str) or not path.strip():
            dropped.append(_dropped_ocr_record(index, record, "invalid_path"))
            continue
        normalized.append({"time": time_value, "actualTime": actual_time, "path": path})
    return normalized, dropped


def ocr_manifest_diagnostics(work_dir: Path) -> dict[str, Any]:
    path = work_dir / OCR_MANIFEST_DIAGNOSTICS
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def ocr_frames(frames_manifest: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "frames_for_ocr.json"
    write_json(manifest_path, frames_manifest)
    normalized, dropped = _normalized_ocr_manifest(frames_manifest)
    diagnostics_path = work_dir / OCR_MANIFEST_DIAGNOSTICS
    diagnostics = {
        "raw_manifest_path": str(manifest_path),
        "diagnostics_path": str(diagnostics_path),
        "total_records": len(frames_manifest),
        "accepted_records": len(normalized),
        "dropped_record_count": len(dropped),
        "dropped_records": dropped,
    }
    if not frames_manifest:
        diagnostics["status"] = "no_frames_extracted"
        write_json(diagnostics_path, diagnostics)
        raise PipelineError(f"OCR cannot run: no frames extracted; raw manifest: {manifest_path}; diagnostics: {diagnostics_path}")
    if not normalized:
        diagnostics["status"] = "all_records_invalid"
        write_json(diagnostics_path, diagnostics)
        raise PipelineError(
            f"OCR cannot run: all {len(frames_manifest)} frame records were malformed; "
            f"raw manifest: {manifest_path}; diagnostics: {diagnostics_path}"
        )

    diagnostics["status"] = "ok"
    write_json(diagnostics_path, diagnostics)
    if dropped:
        details = "; ".join(
            f"index={item['index']} time={item.get('time', 'unknown')} error={item.get('error', item['reason'])}"
            for item in dropped
        )
        warnings.warn(
            f"OCR skipped {len(dropped)} malformed frame record(s); raw manifest: {manifest_path}; "
            f"diagnostics: {diagnostics_path}; {details}",
            RuntimeWarning,
            stacklevel=2,
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".ocr_payload_",
        suffix=".json",
        dir=work_dir,
        delete=False,
    ) as handle:
        json.dump(normalized, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        payload_path = Path(handle.name)
    try:
        proc = run_cmd(["swift", str(SWIFT_HELPER), "ocr", str(payload_path)], timeout=1800)
        return json.loads(proc.stdout)
    finally:
        payload_path.unlink(missing_ok=True)


def ocr_regions(regions_manifest: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    """Run Apple Vision only on calibrated nameplate crops, not whole frames."""
    manifest_path = work_dir / "nameplate_regions_for_ocr.json"
    write_json(manifest_path, regions_manifest)
    proc = run_cmd(["swift", str(SWIFT_HELPER), "ocr-regions", str(manifest_path)], timeout=1800)
    return json.loads(proc.stdout)
