from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from meeting_minutes.jsonio import read_json, write_json

Box = tuple[float, float, float, float]
BoxesByMode = dict[str, dict[str, Box]]


def load_boxes(path: Path) -> BoxesByMode:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    if payload and all(isinstance(value, list) for value in payload.values()):
        payload = {"default": payload}

    boxes_by_mode: BoxesByMode = {}
    for mode, boxes in payload.items():
        if not isinstance(boxes, dict):
            raise ValueError(f"Expected object of boxes for mode {mode!r}")
        parsed: dict[str, Box] = {}
        for label, raw_box in boxes.items():
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                raise ValueError(f"Expected four-number box for {mode}.{label}")
            box = tuple(float(value) for value in raw_box)
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Box for {mode}.{label} must be normalized x1,y1,x2,y2")
            parsed[str(label)] = box  # type: ignore[assignment]
        boxes_by_mode[str(mode)] = parsed
    return boxes_by_mode


def _pixel_box(image: Image.Image, box: Box) -> tuple[int, int, int, int]:
    width, height = image.size
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, math.floor(x1 * width))),
        max(0, min(height - 1, math.floor(y1 * height))),
        max(1, min(width, math.ceil(x2 * width))),
        max(1, min(height, math.ceil(y2 * height))),
    )


def score_highlight_border(
    image: Image.Image,
    box: Box,
    *,
    edge_pixels: int = 5,
    saturation_threshold: float = 0.25,
    value_threshold: float = 0.48,
) -> float:
    x1, y1, x2, y2 = _pixel_box(image, box)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32) / 255.0
    if crop.size == 0:
        return 0.0
    edge = max(1, min(edge_pixels, crop.shape[0] // 2 or 1, crop.shape[1] // 2 or 1))
    strips = np.concatenate(
        [
            crop[:edge, :, :].reshape(-1, 3),
            crop[-edge:, :, :].reshape(-1, 3),
            crop[:, :edge, :].reshape(-1, 3),
            crop[:, -edge:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    value = strips.max(axis=1)
    saturation = (value - strips.min(axis=1)) / np.maximum(value, 1e-6)
    highlighted = (saturation > saturation_threshold) & (value > value_threshold)
    return float(highlighted.mean())


def score_frame(
    image: Image.Image,
    boxes_by_mode: BoxesByMode,
    *,
    edge_pixels: int = 5,
    saturation_threshold: float = 0.25,
    value_threshold: float = 0.48,
) -> list[dict[str, Any]]:
    records = []
    for mode, boxes in boxes_by_mode.items():
        scores = {
            label: score_highlight_border(
                image,
                box,
                edge_pixels=edge_pixels,
                saturation_threshold=saturation_threshold,
                value_threshold=value_threshold,
            )
            for label, box in boxes.items()
        }
        best = max(scores, key=scores.get) if scores else ""
        records.append(
            {
                "mode": mode,
                "best": best,
                "best_score": scores[best] if best else 0.0,
                "scores": scores,
            }
        )
    return records


def score_keyframes(
    keyframes_path: Path,
    boxes_by_mode: BoxesByMode,
    *,
    edge_pixels: int = 5,
    saturation_threshold: float = 0.25,
    value_threshold: float = 0.48,
) -> list[dict[str, Any]]:
    keyframes = read_json(keyframes_path)
    records = []
    for frame in keyframes:
        frame_path = Path(str(frame["path"]))
        image = Image.open(frame_path).convert("RGB")
        for scored in score_frame(
            image,
            boxes_by_mode,
            edge_pixels=edge_pixels,
            saturation_threshold=saturation_threshold,
            value_threshold=value_threshold,
        ):
            records.append(
                {
                    "time": float(frame.get("time", frame.get("actualTime", 0.0))),
                    "actualTime": float(frame.get("actualTime", frame.get("time", 0.0))),
                    "path": str(frame_path),
                    "reasons": frame.get("reasons", []),
                    **scored,
                }
            )
    return records


def write_highlight_scores(
    keyframes_path: Path,
    boxes_json: Path,
    output_path: Path,
    *,
    edge_pixels: int = 5,
    saturation_threshold: float = 0.25,
    value_threshold: float = 0.48,
) -> list[dict[str, Any]]:
    records = score_keyframes(
        keyframes_path,
        load_boxes(boxes_json),
        edge_pixels=edge_pixels,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
    )
    write_json(output_path, records)
    return records
