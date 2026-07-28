from __future__ import annotations

from pathlib import Path
from typing import Any


KEYWORDS = (
    "决定",
    "结论",
    "下一步",
    "风险",
    "截止",
    "负责人",
    "行动项",
    "todo",
    "action",
    "decision",
    "deadline",
    "owner",
    "risk",
    "hash",
    "double",
    "confirm",
    "confirmation",
    "deploy",
    "developer",
    "environment",
    "transaction",
)


def regular_times(duration: float, interval: float) -> list[float]:
    times: list[float] = []
    t = 0.0
    while t <= duration:
        times.append(round(t, 3))
        t += interval
    if duration and (not times or times[-1] < duration):
        times.append(round(duration, 3))
    return times


def keyword_times(segments: list[dict[str, Any]]) -> list[float]:
    times: list[float] = []
    for segment in segments:
        text = str(segment.get("text", "")).lower()
        if any(keyword.lower() in text for keyword in KEYWORDS):
            times.append(float(segment.get("start", 0.0)))
            times.append((float(segment.get("start", 0.0)) + float(segment.get("end", 0.0))) / 2)
    return times


def _image_diff(path_a: Path, path_b: Path) -> float:
    from PIL import Image, ImageChops, ImageStat  # type: ignore

    with Image.open(path_a) as img_a, Image.open(path_b) as img_b:
        a = img_a.convert("L").resize((32, 32))
        b = img_b.convert("L").resize((32, 32))
        diff = ImageChops.difference(a, b)
        stat = ImageStat.Stat(diff)
        return float(stat.mean[0]) / 255.0


def choose_keyframes(
    frames: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    scene_threshold: float = 0.18,
) -> list[dict[str, Any]]:
    keyword_starts = set(round(t, 0) for t in keyword_times(segments))
    selected: list[dict[str, Any]] = []
    previous_path: Path | None = None
    for index, frame in enumerate(frames):
        path_value = frame.get("path")
        if not path_value:
            continue
        path = Path(str(path_value))
        reasons: list[str] = []
        if index == 0 or previous_path is None:
            reasons.append("opening_frame")
        if round(float(frame.get("time", 0.0)), 0) in keyword_starts:
            reasons.append("keyword_nearby")
        if previous_path and path.exists():
            try:
                diff = _image_diff(previous_path, path)
                if diff >= scene_threshold:
                    reasons.append(f"scene_change:{diff:.2f}")
            except Exception:
                pass
        if reasons:
            selected.append({**frame, "reasons": reasons})
        previous_path = path if path.exists() else previous_path
    return selected
