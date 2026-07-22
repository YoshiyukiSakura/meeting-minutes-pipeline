from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .jsonio import write_json


ROOT = Path(__file__).resolve().parents[1]
SWIFT_HELPER = ROOT / "swift" / "macos_media.swift"


class PipelineError(RuntimeError):
    pass


def run_cmd(args: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise PipelineError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def probe_media(input_path: Path) -> dict[str, Any]:
    proc = run_cmd(["swift", str(SWIFT_HELPER), "probe", str(input_path)], timeout=120)
    return json.loads(proc.stdout)


def make_clip(input_path: Path, output_path: Path, *, start: float, duration: float) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    ffmpeg = shutil.which("ffmpeg")
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


def ocr_frames(frames_manifest: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    manifest_path = work_dir / "frames_for_ocr.json"
    write_json(manifest_path, frames_manifest)
    proc = run_cmd(["swift", str(SWIFT_HELPER), "ocr", str(manifest_path)], timeout=1800)
    return json.loads(proc.stdout)


def ocr_regions(regions_manifest: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    """Run Apple Vision only on calibrated nameplate crops, not whole frames."""
    manifest_path = work_dir / "nameplate_regions_for_ocr.json"
    write_json(manifest_path, regions_manifest)
    proc = run_cmd(["swift", str(SWIFT_HELPER), "ocr-regions", str(manifest_path)], timeout=1800)
    return json.loads(proc.stdout)
