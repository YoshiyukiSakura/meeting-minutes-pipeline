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
    colored_edge = (saturation > saturation_threshold) & (value > value_threshold)

    # A colored tile background can otherwise look like an active border. Compare
    # the outer pixels with the tile interior so uniform colored cards score low.
    inset = min(max(edge * 2, edge + 1), max(1, min(crop.shape[:2]) // 3))
    if crop.shape[0] <= inset * 2 or crop.shape[1] <= inset * 2:
        return float(colored_edge.mean())
    interior = crop[inset:-inset, inset:-inset, :]
    baseline = np.median(interior.reshape(-1, 3), axis=0)
    contrast = np.linalg.norm(strips - baseline, axis=1) / np.sqrt(3.0)
    highlighted = colored_edge & (contrast >= 0.18)
    return float(highlighted.mean())


def score_green_speaker_cue(image: Image.Image, box: Box) -> float:
    """Score a tightly profiled green active-speaker waveform cue.

    Slack Huddles renders this cue as a small bright-green waveform inside a
    white pill near the lower-left of a participant tile. The caller must pass
    the cue box itself rather than the whole tile; this is not a general green
    object detector.
    """
    x1, y1, x2, y2 = _pixel_box(image, box)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32) / 255.0
    if crop.size == 0:
        return 0.0
    red, green, blue = np.moveaxis(crop, -1, 0)
    waveform_green = (
        (green >= 0.34)
        & (green >= red * 1.28)
        & (green >= blue * 1.18)
        & ((green - red) >= 0.10)
        & ((green - blue) >= 0.06)
    )
    # A Slack active-speaker waveform sits inside a small white pill. Requiring
    # that background prevents a bright-green tile border from becoming a false
    # speaker cue when a screen share changes the participant layout.
    white_pill = (
        (red >= 0.80)
        & (green >= 0.80)
        & (blue >= 0.80)
        & ((np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)) <= 0.15)
    )
    if float(white_pill.mean()) < 0.03:
        return 0.0
    expected_pixels = max(1, int(round(crop.shape[0] * crop.shape[1] * 0.005)))
    return float(min(1.0, int(waveform_green.sum()) / expected_pixels))


def score_green_highlight_border(image: Image.Image, box: Box, *, edge_pixels: int = 5) -> float:
    """Score a rectangular green active-speaker border conservatively.

    A green avatar or green screen-share must not be treated as an active
    speaker. A valid border therefore needs continuous green coverage on at
    least three tile sides and a rim that is substantially greener than the
    tile interior.
    """
    x1, y1, x2, y2 = _pixel_box(image, box)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = np.asarray(image.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.float32) / 255.0
    if crop.size == 0:
        return 0.0
    edge = max(1, min(edge_pixels, crop.shape[0] // 2 or 1, crop.shape[1] // 2 or 1))
    red, green, blue = np.moveaxis(crop, -1, 0)
    highlight_green = (
        (green >= 0.34)
        & (green >= red * 1.28)
        & (green >= blue * 1.18)
        & ((green - red) >= 0.10)
        & ((green - blue) >= 0.06)
    )

    side_coverages = np.asarray(
        [
            highlight_green[:edge, :].any(axis=0).mean(),
            highlight_green[-edge:, :].any(axis=0).mean(),
            highlight_green[:, :edge].any(axis=1).mean(),
            highlight_green[:, -edge:].any(axis=1).mean(),
        ],
        dtype=np.float32,
    )
    # The third-highest side is the weakest side in a required 3-of-4 outline.
    # This rejects a green avatar or UI mark that only touches one or two edges.
    outline_score = float(np.sort(side_coverages)[-3])
    if outline_score < 0.55:
        return 0.0

    inset = min(max(edge * 2, edge + 1), max(1, min(crop.shape[:2]) // 3))
    if crop.shape[0] <= inset * 2 or crop.shape[1] <= inset * 2:
        return 0.0 if float(highlight_green.mean()) > 0.35 else outline_score
    interior_green = float(highlight_green[inset:-inset, inset:-inset].mean())
    if float(side_coverages.mean()) - interior_green < 0.25:
        return 0.0
    return outline_score


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
