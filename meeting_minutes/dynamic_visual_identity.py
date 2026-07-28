from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .visual_highlight import score_green_highlight_border
from .visual_identity import _resolve_candidate, has_trusted_identity


class DynamicVisualProfileError(ValueError):
    pass


_DEFAULTS: dict[str, Any] = {
    "samples_per_segment": 3,
    "min_active_score": 0.75,
    "minimum_tile_width": 0.06,
    "minimum_tile_height": 0.06,
    "horizontal_run_min_pixels": 60,
    "minimum_segment_vote_share": 0.67,
    "short_segment_seconds": 1.5,
    "time_offset_seconds": 0.0,
    "search_region": [0.20, 0.10, 0.98, 1.0],
    "nameplate_lower_fraction": 0.65,
    "strong_single_score": 0.85,
}


def _box(raw: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise DynamicVisualProfileError(f"{label} must be a four-number normalized box")
    box = tuple(float(value) for value in raw)
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise DynamicVisualProfileError(f"{label} must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return box


def load_dynamic_visual_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DynamicVisualProfileError("dynamic visual profile must be a JSON object")

    participants = payload.get("participants")
    if not isinstance(participants, list) or not participants or not all(isinstance(item, str) and item.strip() for item in participants):
        raise DynamicVisualProfileError("participants must be a non-empty list of names")
    normalized_participants = [str(item).strip() for item in participants]
    if len({item.casefold() for item in normalized_participants}) != len(normalized_participants):
        raise DynamicVisualProfileError("participants must not contain duplicate names")

    raw_settings = payload.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise DynamicVisualProfileError("settings must be a JSON object")
    settings = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if key in raw_settings:
            settings[key] = raw_settings[key]
    settings["samples_per_segment"] = int(settings["samples_per_segment"])
    settings["horizontal_run_min_pixels"] = int(settings["horizontal_run_min_pixels"])
    for key in (
        "min_active_score",
        "minimum_tile_width",
        "minimum_tile_height",
        "minimum_segment_vote_share",
        "short_segment_seconds",
        "time_offset_seconds",
        "nameplate_lower_fraction",
        "strong_single_score",
    ):
        settings[key] = float(settings[key])
    settings["search_region"] = _box(settings["search_region"], "settings.search_region")
    if not 1 <= settings["samples_per_segment"] <= 5:
        raise DynamicVisualProfileError("settings.samples_per_segment must be in 1..5")
    if settings["horizontal_run_min_pixels"] < 20:
        raise DynamicVisualProfileError("settings.horizontal_run_min_pixels must be at least 20")
    for key in (
        "min_active_score",
        "minimum_tile_width",
        "minimum_tile_height",
        "minimum_segment_vote_share",
        "nameplate_lower_fraction",
        "strong_single_score",
    ):
        if not 0 < settings[key] <= 1:
            raise DynamicVisualProfileError(f"settings.{key} must be in (0, 1]")
    if settings["short_segment_seconds"] <= 0:
        raise DynamicVisualProfileError("settings.short_segment_seconds must be positive")
    return {"participants": normalized_participants, "settings": settings}


def build_dynamic_sample_requests(
    segments: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    duration: float,
) -> list[dict[str, Any]]:
    settings = profile["settings"]
    max_samples = int(settings["samples_per_segment"])
    short_seconds = float(settings["short_segment_seconds"])
    offset = float(settings["time_offset_seconds"])
    requests: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        segment_duration = max(0.0, end - start)
        if segment_duration <= short_seconds:
            fractions = [0.5]
        elif max_samples == 1:
            fractions = [0.5]
        elif max_samples == 2 or segment_duration <= short_seconds * 2:
            fractions = [0.3, 0.7]
        else:
            fractions = [0.2, 0.5, 0.8]
        for fraction in fractions:
            timeline_time = start + segment_duration * fraction
            video_time = min(max(0.0, timeline_time + offset), duration)
            requests.append(
                {
                    "segment_index": segment_index,
                    "timeline_time": round(timeline_time, 3),
                    "video_time": round(video_time, 3),
                }
            )
    return requests


def unique_dynamic_video_times(sample_requests: list[dict[str, Any]]) -> list[float]:
    return sorted({round(float(request["video_time"]), 3) for request in sample_requests})


def _green_mask(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = np.moveaxis(pixels, -1, 0)
    return (
        (green >= 0.34)
        & (green >= red * 1.28)
        & (green >= blue * 1.18)
        & ((green - red) >= 0.10)
        & ((green - blue) >= 0.06)
    )


def _runs(values: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            if index - start >= minimum:
                result.append((start, index - 1))
            start = None
    if start is not None and len(values) - start >= minimum:
        result.append((start, len(values) - 1))
    return result


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def detect_active_tiles(image: Image.Image, profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Find green Discord-like active-speaker outlines without using a person-to-coordinate map."""
    settings = profile["settings"]
    mask = _green_mask(image)
    height, width = mask.shape
    region_x1, region_y1, region_x2, region_y2 = settings["search_region"]
    min_x = int(math.floor(region_x1 * width))
    max_x = int(math.ceil(region_x2 * width))
    min_y = int(math.floor(region_y1 * height))
    max_y = int(math.ceil(region_y2 * height))
    run_minimum = max(int(settings["horizontal_run_min_pixels"]), int(math.ceil(width * settings["minimum_tile_width"] * 0.7)))
    horizontal_edges: list[tuple[int, int, int]] = []
    for y in range(min_y, max_y):
        for x1, x2 in _runs(mask[y, min_x:max_x], run_minimum):
            horizontal_edges.append((y, min_x + x1, min_x + x2))

    minimum_height = max(1, int(math.ceil(height * settings["minimum_tile_height"])))
    maximum_height = max(minimum_height, int(math.floor((region_y2 - region_y1) * height)))
    padding = max(4, int(round(min(width, height) * 0.007)))
    raw: list[dict[str, Any]] = []
    for top_index, (top_y, top_x1, top_x2) in enumerate(horizontal_edges):
        top_width = top_x2 - top_x1 + 1
        for bottom_y, bottom_x1, bottom_x2 in horizontal_edges[top_index + 1 :]:
            tile_height = bottom_y - top_y
            if tile_height < minimum_height:
                continue
            if tile_height > maximum_height:
                break
            bottom_width = bottom_x2 - bottom_x1 + 1
            overlap = max(0, min(top_x2, bottom_x2) - max(top_x1, bottom_x1) + 1)
            if overlap < min(top_width, bottom_width) * 0.85:
                continue
            if abs(top_width - bottom_width) > max(top_width, bottom_width) * 0.16:
                continue
            box = (
                max(0.0, (min(top_x1, bottom_x1) - padding) / width),
                max(0.0, (top_y - 1) / height),
                min(1.0, (max(top_x2, bottom_x2) + padding + 1) / width),
                min(1.0, (bottom_y + 2) / height),
            )
            if box[2] - box[0] < settings["minimum_tile_width"] or box[3] - box[1] < settings["minimum_tile_height"]:
                continue
            score = score_green_highlight_border(image, box)
            if score < settings["min_active_score"]:
                continue
            raw.append({"tile": [round(value, 6) for value in box], "score": round(float(score), 4)})

    deduplicated: list[dict[str, Any]] = []
    for candidate in sorted(raw, key=lambda item: float(item["score"]), reverse=True):
        box = tuple(float(value) for value in candidate["tile"])
        if any(_iou(box, tuple(float(value) for value in item["tile"])) >= 0.82 for item in deduplicated):
            continue
        deduplicated.append(candidate)
    return sorted(deduplicated, key=lambda item: (float(item["tile"][1]), float(item["tile"][0])))


def detect_dynamic_visual_frames(frames: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame in frames:
        base = {
            "time": float(frame.get("time", 0.0)),
            "actualTime": float(frame.get("actualTime", frame.get("time", 0.0))),
            "path": frame.get("path"),
        }
        path = frame.get("path")
        if not path:
            records.append({**base, "active_tiles": [], "reason": "missing_frame"})
            continue
        try:
            with Image.open(str(path)) as image:
                active_tiles = detect_active_tiles(image, profile)
        except Exception as exc:
            records.append({**base, "active_tiles": [], "reason": f"frame_read_failed:{type(exc).__name__}"})
            continue
        reason = "single_active_tile" if len(active_tiles) == 1 else "multiple_active_tiles" if active_tiles else "no_active_tile"
        records.append({**base, "active_tiles": active_tiles, "reason": reason})
    return records


def build_dynamic_ocr_manifest(detected_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"time": frame["time"], "actualTime": frame["actualTime"], "path": frame["path"]}
        for frame in detected_frames
        if len(frame.get("active_tiles", [])) == 1 and frame.get("path")
    ]


def _ocr_box(text: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = text.get("bbox")
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        width = float(raw["width"])
        height = float(raw["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    # Vision uses a lower-left origin; pipeline boxes use a top-left origin.
    return (x, 1.0 - y - height, x + width, 1.0 - y)


def _resolve_nameplate_candidate(value: str, participants: list[str]) -> str | None:
    normalized = re.sub(r"\s+", " ", value.strip()).casefold()
    for participant in sorted(participants, key=len, reverse=True):
        expected = participant.casefold()
        if normalized == expected or normalized.startswith(expected + " "):
            return participant
    resolved = _resolve_candidate(value, participants)
    if resolved:
        return resolved
    scores = sorted(
        ((SequenceMatcher(None, normalized, participant.casefold()).ratio(), participant) for participant in participants), reverse=True
    )
    return scores[0][1] if scores and scores[0][0] >= 0.90 else None


def _match_nameplate(tile: dict[str, Any], texts: list[dict[str, Any]], profile: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    x1, y1, x2, y2 = (float(value) for value in tile["tile"])
    lower_fraction = float(profile["settings"]["nameplate_lower_fraction"])
    candidates: list[dict[str, Any]] = []
    for text in texts:
        box = _ocr_box(text)
        if not box:
            continue
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        relative_y = (center_y - y1) / max(1e-9, y2 - y1)
        if not (x1 <= center_x <= x2 and y1 <= center_y <= y2 and relative_y >= lower_fraction):
            continue
        candidate = _resolve_nameplate_candidate(str(text.get("text", "")), profile["participants"])
        if candidate:
            candidates.append(
                {
                    "name": candidate,
                    "text": str(text.get("text", "")),
                    "confidence": round(float(text.get("confidence", 0.0)), 3),
                    "bbox": [round(value, 6) for value in box],
                }
            )
    names = {str(item["name"]) for item in candidates}
    return (next(iter(names)) if len(names) == 1 else None), candidates


def attach_dynamic_ocr(
    detected_frames: list[dict[str, Any]],
    ocr_records: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    texts_by_path = {str(record.get("path")): list(record.get("texts", [])) for record in ocr_records if record.get("path")}
    scored: list[dict[str, Any]] = []
    for frame in detected_frames:
        active_tiles = list(frame.get("active_tiles", []))
        base = {key: frame.get(key) for key in ("time", "actualTime", "path")}
        if len(active_tiles) != 1:
            scored.append({**base, "active_tiles": active_tiles, "active": False, "name": None, "reason": frame.get("reason")})
            continue
        tile = dict(active_tiles[0])
        name, candidates = _match_nameplate(tile, texts_by_path.get(str(frame.get("path")), []), profile)
        reason = "active_named_tile_nameplate_ocr" if name else "active_tile_nameplate_unresolved"
        scored.append(
            {
                **base,
                "active_tiles": [tile],
                "active": True,
                "name": name,
                "name_source": "dynamic_visual_in_tile_nameplate_ocr" if name else None,
                "nameplate_candidates": candidates,
                "reason": reason,
            }
        )
    return scored


def _clear_stale_visual_identity(segment: dict[str, Any]) -> None:
    source = str(segment.get("name_source") or "")
    if source.startswith("visual_") or source.startswith("dynamic_visual_"):
        segment["name"] = None
        segment["name_source"] = None
        segment["name_confidence"] = 0.0
    for key in (
        "visual_identity_evidence",
        "visual_identity_conflict",
        "visual_identity_candidate_names",
        "dynamic_visual_identity_evidence",
        "dynamic_visual_identity_conflict",
    ):
        segment.pop(key, None)


def attach_dynamic_visual_identity(
    segments: list[dict[str, Any]],
    sample_requests: list[dict[str, Any]],
    scored_frames: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    frame_by_time = {round(float(frame["time"]), 3): frame for frame in scored_frames}
    requests_by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for request in sample_requests:
        requests_by_segment[int(request["segment_index"])].append(request)
    summary: dict[str, Any] = {
        "segments": len(segments),
        "sampled_frames": len(scored_frames),
        "active_frames": sum(1 for frame in scored_frames if frame.get("active")),
        "named_active_frames": sum(1 for frame in scored_frames if frame.get("name")),
        "assigned": 0,
        "unresolved": 0,
        "conflicts": 0,
        "preserved_trusted_identity": 0,
        "by_reason": Counter(),
    }
    vote_share_required = float(profile["settings"]["minimum_segment_vote_share"])
    short_seconds = float(profile["settings"]["short_segment_seconds"])
    strong_single_score = float(profile["settings"]["strong_single_score"])
    for index, segment in enumerate(segments):
        sample_frames = [frame_by_time.get(round(float(request["video_time"]), 3)) for request in requests_by_segment.get(index, [])]
        sample_frames = [frame for frame in sample_frames if frame]
        trusted = has_trusted_identity(segment)
        _clear_stale_visual_identity(segment)
        evidence = [
            {
                "time": round(float(frame.get("actualTime", frame.get("time", 0.0))), 3),
                "frame": frame.get("path"),
                "active_tiles": frame.get("active_tiles", []),
                "name": frame.get("name"),
                "nameplate_candidates": frame.get("nameplate_candidates", []),
                "reason": frame.get("reason"),
            }
            for frame in sample_frames
        ]
        if evidence:
            segment["dynamic_visual_identity_evidence"] = evidence
        if trusted:
            summary["preserved_trusted_identity"] += 1
            named = {str(frame["name"]) for frame in sample_frames if frame.get("name")}
            if named and str(segment.get("name")) not in named:
                segment["dynamic_visual_identity_conflict"] = {
                    "reason": "trusted_identity_disagrees_with_dynamic_visual_evidence",
                    "visual_names": sorted(named),
                }
                summary["conflicts"] += 1
                summary["by_reason"]["trusted_identity_disagrees_with_dynamic_visual_evidence"] += 1
            continue
        if any(frame.get("reason") == "multiple_active_tiles" for frame in sample_frames):
            segment["name"] = None
            segment["name_source"] = "dynamic_visual_identity_conflict"
            segment["name_confidence"] = 0.0
            segment["dynamic_visual_identity_conflict"] = {"reason": "multiple_active_tiles_in_segment"}
            summary["conflicts"] += 1
            summary["unresolved"] += 1
            summary["by_reason"]["multiple_active_tiles_in_segment"] += 1
            continue
        named_frames = [frame for frame in sample_frames if frame.get("name")]
        votes = Counter(str(frame["name"]) for frame in named_frames)
        if len(votes) > 1:
            segment["name"] = None
            segment["name_source"] = "dynamic_visual_identity_conflict"
            segment["name_confidence"] = 0.0
            segment["dynamic_visual_identity_conflict"] = {
                "reason": "multiple_names_in_segment",
                "visual_names": sorted(votes),
            }
            summary["conflicts"] += 1
            summary["unresolved"] += 1
            summary["by_reason"]["multiple_names_in_segment"] += 1
            continue
        if not votes:
            segment["name"] = None
            segment["name_source"] = "dynamic_visual_identity_unresolved"
            segment["name_confidence"] = 0.0
            summary["unresolved"] += 1
            reason = "no_named_active_visual_evidence" if sample_frames else "no_visual_samples"
            summary["by_reason"][reason] += 1
            continue
        name, count = votes.most_common(1)[0]
        vote_share = count / len(sample_frames) if sample_frames else 0.0
        duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        required_count = 1 if duration <= short_seconds else min(2, len(sample_frames))
        matched = [frame for frame in named_frames if frame.get("name") == name]
        mean_score = sum(float(frame["active_tiles"][0]["score"]) for frame in matched) / len(matched)
        if count < required_count or vote_share < vote_share_required or (required_count == 1 and mean_score < strong_single_score):
            segment["name"] = None
            segment["name_source"] = "dynamic_visual_identity_unresolved"
            segment["name_confidence"] = 0.0
            summary["unresolved"] += 1
            summary["by_reason"]["insufficient_dynamic_visual_consensus"] += 1
            continue
        confidence = min(0.94, 0.64 + 0.16 * vote_share + 0.20 * mean_score)
        segment["name"] = name
        segment["name_source"] = "dynamic_visual_in_tile_nameplate_ocr"
        segment["name_confidence"] = round(confidence, 3)
        segment["frame_refs"] = sorted({str(frame["path"]) for frame in matched if frame.get("path")})[:4]
        summary["assigned"] += 1
    summary["by_reason"] = dict(summary["by_reason"])
    return summary


def write_dynamic_visual_identity_report(
    path: Path,
    *,
    profile_path: Path,
    profile: dict[str, Any],
    scored_frames: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    reasons = Counter(str(frame.get("reason") or "unknown") for frame in scored_frames)
    lines = [
        "# Dynamic Visual Identity Report",
        "",
        "## Method",
        f"- Profile: `{profile_path}`",
        "- A name is assigned only when one green active-speaker outline and a participant-whitelisted nameplate are both present in the same frame.",
        "- Tile positions are detected independently in every sampled frame; no coordinate-to-person mapping is reused across frames.",
        "- Multiple active outlines, missing nameplates, and OCR ambiguity remain unresolved.",
        "",
        "## Calibration",
        f"- Participants: {', '.join(profile['participants'])}",
        f"- Active-border threshold: {float(profile['settings']['min_active_score']):.2f}",
        f"- Nameplate lower-tile fraction: {float(profile['settings']['nameplate_lower_fraction']):.2f}",
        f"- Sampled frames: {len(scored_frames)}",
        f"- Frames with one active outline: {reasons.get('active_named_tile_nameplate_ocr', 0) + reasons.get('active_tile_nameplate_unresolved', 0)}",
        f"- Frames with a named active tile: {summary['named_active_frames']}",
        f"- Frames with multiple active outlines: {reasons.get('multiple_active_tiles', 0)}",
        "",
        "## Segment Results",
        f"- Direct visual assignments: {summary['assigned']}",
        f"- Unresolved segments: {summary['unresolved']}",
        f"- Conflicting segments: {summary['conflicts']}",
        f"- Preserved trusted identities: {summary['preserved_trusted_identity']}",
        "",
        "## Limits",
        "- Sidebar names, static layout profiles, and voice-cluster labels are not sufficient by themselves for a direct visual real-name assignment.",
        "- This stage does not infer a name across frames or propagate a name from a diarization cluster.",
        "- Each assigned segment includes the source frame, active-tile box, green-border score, and OCR nameplate evidence in transcript.json.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
