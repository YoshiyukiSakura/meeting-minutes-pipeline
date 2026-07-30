"""Evidence-gated identity matching for Discord-style roster avatar UIs.

This is intentionally a sibling of nameplate and avatar-template identity.  A
roster label never names a speaker by itself.  A segment can be named only
when the same sampled frame contains one detected active-speaker tile, a
profile-bounded roster row with an exact whitelisted label, and a distinctive
avatar match between those two UI elements.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .avatar_template_identity import extract_avatar_signature_from_image
from .dynamic_visual_identity import build_dynamic_sample_requests, detect_active_tiles
from .identity_authority import (
    DIRECT_VISUAL_CLUSTER_SOURCE,
    ROSTER_AVATAR_SOURCE,
    SAME_SESSION_VISUAL_VOICE_SOURCE,
)
from .sample_evidence import unique_frames_for_requests


class RosterAvatarProfileError(ValueError):
    """Raised when a reviewed roster-avatar profile is invalid."""


SOURCE = ROSTER_AVATAR_SOURCE
_CORRECTABLE_EXISTING_SOURCES = frozenset(
    {DIRECT_VISUAL_CLUSTER_SOURCE, SAME_SESSION_VISUAL_VOICE_SOURCE}
)
_DEFAULTS: dict[str, Any] = {
    "samples_per_segment": 3,
    "short_segment_seconds": 1.5,
    "time_offset_seconds": 0.0,
    "minimum_segment_vote_share": 2 / 3,
    "minimum_supporting_frames": 2,
    "min_active_score": 0.75,
    "minimum_tile_width": 0.06,
    "minimum_tile_height": 0.06,
    "horizontal_run_min_pixels": 60,
    "search_region": [0.20, 0.10, 0.98, 1.0],
    "minimum_ocr_confidence": 0.75,
    "minimum_roster_identities": 3,
    "minimum_similarity": 0.78,
    "minimum_margin": 0.14,
    "minimum_roster_avatar_pixels": 14,
    "minimum_avatar_stddev": 0.025,
    "duplicate_template_similarity": 0.985,
    "minimum_tile_avatar_area_ratio": 0.002,
    "maximum_tile_avatar_area_ratio": 0.48,
    "maximum_tile_background_stddev": 0.18,
    "foreground_distance": 0.14,
    "signature_size": 64,
    "active_border_inset_fraction": 0.025,
    "active_outer_crop_fraction": 0.65,
    "active_outer_circle_radius": 0.20,
    "active_inner_crop_fraction": 0.55,
    "active_inner_circle_radius": 0.28,
    "anchor_only_identities": True,
    "minimum_reviewed_anchors": 3,
    "minimum_anchor_identities": 3,
    "minimum_anchor_seconds_separation": 15.0,
}


def _box(raw: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise RosterAvatarProfileError(f"{field} must be a four-number normalized box")
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise RosterAvatarProfileError(f"{field} must contain numbers") from exc
    if not 0.0 <= x1 < x2 <= 1.0 or not 0.0 <= y1 < y2 <= 1.0:
        raise RosterAvatarProfileError(f"{field} must be a normalized top-left box")
    return (x1, y1, x2, y2)


def _normalized_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def _validate_layout_windows(layouts: list[dict[str, Any]]) -> None:
    ordered = sorted(layouts, key=lambda item: (float(item["start"]), float(item["end"])))
    for previous, current in pairwise(ordered):
        if float(current["start"]) < float(previous["end"]):
            raise RosterAvatarProfileError("layouts must not overlap in time")


def load_roster_avatar_profile(path: Path) -> dict[str, Any]:
    """Load a time-bounded, human-reviewed roster-avatar profile."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RosterAvatarProfileError("roster avatar profile must be a JSON object")

    raw_participants = payload.get("participants")
    if not isinstance(raw_participants, list) or not raw_participants:
        raise RosterAvatarProfileError("participants must be a non-empty list")
    participants = [str(item).strip() for item in raw_participants if isinstance(item, str) and item.strip()]
    if len(participants) != len(raw_participants):
        raise RosterAvatarProfileError("participants must contain non-empty strings")
    participant_by_key = {_normalized_name(name): name for name in participants}
    if len(participant_by_key) != len(participants):
        raise RosterAvatarProfileError("participants must not contain duplicate normalized names")

    raw_settings = payload.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise RosterAvatarProfileError("settings must be an object")
    settings = dict(_DEFAULTS)
    settings.update({key: raw_settings[key] for key in _DEFAULTS if key in raw_settings})
    if not isinstance(settings["anchor_only_identities"], bool):
        raise RosterAvatarProfileError("settings.anchor_only_identities must be a boolean")
    for key in (
        "samples_per_segment",
        "minimum_supporting_frames",
        "horizontal_run_min_pixels",
        "minimum_roster_identities",
        "minimum_roster_avatar_pixels",
        "signature_size",
        "minimum_reviewed_anchors",
        "minimum_anchor_identities",
    ):
        settings[key] = int(settings[key])
    for key in (
        "short_segment_seconds",
        "time_offset_seconds",
        "minimum_segment_vote_share",
        "min_active_score",
        "minimum_tile_width",
        "minimum_tile_height",
        "minimum_ocr_confidence",
        "minimum_similarity",
        "minimum_margin",
        "minimum_avatar_stddev",
        "duplicate_template_similarity",
        "minimum_tile_avatar_area_ratio",
        "maximum_tile_avatar_area_ratio",
        "maximum_tile_background_stddev",
        "foreground_distance",
        "active_outer_crop_fraction",
        "active_outer_circle_radius",
        "active_inner_crop_fraction",
        "active_inner_circle_radius",
        "minimum_anchor_seconds_separation",
    ):
        settings[key] = float(settings[key])
    settings["search_region"] = _box(settings["search_region"], "settings.search_region")
    if not 1 <= settings["samples_per_segment"] <= 5:
        raise RosterAvatarProfileError("settings.samples_per_segment must be in 1..5")
    if settings["minimum_supporting_frames"] < 2:
        raise RosterAvatarProfileError("settings.minimum_supporting_frames must be at least 2")
    if settings["horizontal_run_min_pixels"] < 20:
        raise RosterAvatarProfileError("settings.horizontal_run_min_pixels must be at least 20")
    for key in (
        "minimum_segment_vote_share",
        "min_active_score",
        "minimum_tile_width",
        "minimum_tile_height",
        "minimum_ocr_confidence",
        "minimum_similarity",
        "minimum_margin",
        "minimum_avatar_stddev",
        "duplicate_template_similarity",
        "minimum_tile_avatar_area_ratio",
        "maximum_tile_avatar_area_ratio",
        "maximum_tile_background_stddev",
        "active_outer_crop_fraction",
        "active_outer_circle_radius",
        "active_inner_crop_fraction",
        "active_inner_circle_radius",
    ):
        if not 0.0 < settings[key] <= 1.0:
            raise RosterAvatarProfileError(f"settings.{key} must be in (0, 1]")
    settings["active_border_inset_fraction"] = float(settings["active_border_inset_fraction"])
    if not 0.0 <= settings["active_border_inset_fraction"] < 0.25:
        raise RosterAvatarProfileError("settings.active_border_inset_fraction must be in [0, 0.25)")
    if settings["minimum_tile_avatar_area_ratio"] >= settings["maximum_tile_avatar_area_ratio"]:
        raise RosterAvatarProfileError("tile avatar area limits must be ordered")
    if settings["minimum_roster_identities"] < 3:
        raise RosterAvatarProfileError("settings.minimum_roster_identities must be at least 3")
    if settings["minimum_reviewed_anchors"] < 3 or settings["minimum_anchor_identities"] < 3:
        raise RosterAvatarProfileError("reviewed roster calibration must require at least three anchors and identities")
    if settings["short_segment_seconds"] <= 0.0 or settings["minimum_anchor_seconds_separation"] <= 0.0:
        raise RosterAvatarProfileError("duration settings must be positive")

    raw_layouts = payload.get("layouts")
    if not isinstance(raw_layouts, list) or not raw_layouts:
        raise RosterAvatarProfileError("layouts must be a non-empty list")
    layouts: list[dict[str, Any]] = []
    for index, raw_layout in enumerate(raw_layouts):
        if not isinstance(raw_layout, dict):
            raise RosterAvatarProfileError(f"layouts[{index}] must be an object")
        name = str(raw_layout.get("name", "")).strip()
        if not name:
            raise RosterAvatarProfileError(f"layouts[{index}].name is required")
        start = float(raw_layout.get("start", 0.0))
        end_raw = raw_layout.get("end")
        end = float(end_raw) if end_raw is not None else math.inf
        if start < 0.0 or end <= start:
            raise RosterAvatarProfileError(f"layouts[{index}] has an invalid time window")
        raw_avatar = raw_layout.get("avatar", {})
        if not isinstance(raw_avatar, dict):
            raise RosterAvatarProfileError(f"layouts[{index}].avatar must be an object")
        size_multiplier = float(raw_avatar.get("size_multiplier", 1.45))
        gap_multiplier = float(raw_avatar.get("gap_multiplier", 0.24))
        vertical_offset_multiplier = float(raw_avatar.get("vertical_offset_multiplier", 0.0))
        if not 0.75 <= size_multiplier <= 3.0:
            raise RosterAvatarProfileError(f"layouts[{index}].avatar.size_multiplier must be in 0.75..3")
        if not 0.0 <= gap_multiplier <= 2.0:
            raise RosterAvatarProfileError(f"layouts[{index}].avatar.gap_multiplier must be in 0..2")
        if not -1.0 <= vertical_offset_multiplier <= 1.0:
            raise RosterAvatarProfileError(f"layouts[{index}].avatar.vertical_offset_multiplier must be in -1..1")
        layouts.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "roster_region": _box(raw_layout.get("roster_region"), f"layouts[{index}].roster_region"),
                "avatar": {
                    "size_multiplier": size_multiplier,
                    "gap_multiplier": gap_multiplier,
                    "vertical_offset_multiplier": vertical_offset_multiplier,
                },
            }
        )
    _validate_layout_windows(layouts)

    raw_anchors = payload.get("reviewed_anchors")
    if not isinstance(raw_anchors, list):
        raise RosterAvatarProfileError("reviewed_anchors must be a list")
    anchors: list[dict[str, Any]] = []
    for index, raw_anchor in enumerate(raw_anchors):
        if not isinstance(raw_anchor, dict):
            raise RosterAvatarProfileError(f"reviewed_anchors[{index}] must be an object")
        try:
            time = float(raw_anchor["time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RosterAvatarProfileError(f"reviewed_anchors[{index}].time must be a non-negative number") from exc
        if time < 0.0:
            raise RosterAvatarProfileError(f"reviewed_anchors[{index}].time must be non-negative")
        name = raw_anchor.get("name")
        if not isinstance(name, str) or _normalized_name(name) not in participant_by_key:
            raise RosterAvatarProfileError(f"reviewed_anchors[{index}].name must exactly name a participant")
        if raw_anchor.get("reviewed") is not True:
            raise RosterAvatarProfileError(f"reviewed_anchors[{index}].reviewed must be true")
        anchors.append({"time": time, "name": participant_by_key[_normalized_name(name)]})
    if len(anchors) < settings["minimum_reviewed_anchors"]:
        raise RosterAvatarProfileError("reviewed_anchors does not meet the configured minimum")
    if len({anchor["name"] for anchor in anchors}) < settings["minimum_anchor_identities"]:
        raise RosterAvatarProfileError("reviewed_anchors must cover the configured number of distinct identities")
    anchor_times = sorted(float(anchor["time"]) for anchor in anchors)
    if any(
        right - left < float(settings["minimum_anchor_seconds_separation"])
        for left, right in pairwise(anchor_times)
    ):
        raise RosterAvatarProfileError("reviewed_anchors must be time-separated by the configured minimum")

    return {
        "participants": participants,
        "participant_by_key": participant_by_key,
        "settings": settings,
        "layouts": layouts,
        "reviewed_anchors": sorted(anchors, key=lambda item: (float(item["time"]), str(item["name"]))),
    }


def layout_at(profile: dict[str, Any], time: float) -> dict[str, Any] | None:
    for layout in profile["layouts"]:
        if float(layout["start"]) <= time < float(layout["end"]):
            return layout
    return None


def build_roster_sample_requests(
    segments: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    duration: float,
) -> list[dict[str, Any]]:
    return build_dynamic_sample_requests(segments, {"settings": profile["settings"]}, duration=duration)


def build_reviewed_anchor_requests(profile: dict[str, Any], *, duration: float) -> list[dict[str, Any]]:
    return [
        {
            "kind": "reviewed_anchor",
            "expected_name": anchor["name"],
            "video_time": round(min(max(0.0, float(anchor["time"])), max(0.0, duration - 0.001)), 3),
        }
        for anchor in profile["reviewed_anchors"]
    ]


def unique_roster_video_times(requests: list[dict[str, Any]]) -> list[float]:
    return sorted({round(float(request["video_time"]), 3) for request in requests})


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
    if width <= 0.0 or height <= 0.0:
        return None
    return (x, 1.0 - y - height, x + width, 1.0 - y)


def _globalize(local_box: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    return (
        left + local_box[0] * width,
        top + local_box[1] * height,
        left + local_box[2] * width,
        top + local_box[3] * height,
    )


def build_roster_ocr_manifest(frames: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Limit OCR to the calibrated roster region for each layout window."""

    manifest: list[dict[str, Any]] = []
    for frame in frames:
        path = frame.get("path")
        if not path:
            continue
        time = float(frame.get("actualTime", frame.get("time", 0.0)))
        layout = layout_at(profile, time)
        if not layout:
            continue
        manifest.append(
            {
                "time": float(frame.get("time", 0.0)),
                "actualTime": time,
                "path": str(path),
                "regions": [{"label": f"{layout['name']}::roster", "box": list(layout["roster_region"])}],
            }
        )
    return manifest


def detect_roster_active_frames(frames: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Detect only unique active tiles and bind every frame to a roster layout."""

    active_profile = {"settings": profile["settings"]}
    records: list[dict[str, Any]] = []
    for frame in frames:
        base = {
            "time": float(frame.get("time", 0.0)),
            "actualTime": float(frame.get("actualTime", frame.get("time", 0.0))),
            "path": frame.get("path"),
        }
        layout = layout_at(profile, base["actualTime"])
        if not layout:
            records.append({**base, "layout": None, "active_tiles": [], "reason": "no_roster_layout"})
            continue
        path = frame.get("path")
        if not path:
            records.append({**base, "layout": layout["name"], "active_tiles": [], "reason": "missing_frame"})
            continue
        try:
            with Image.open(str(path)) as source:
                active_tiles = _deduplicate_nested_active_tiles(
                    detect_active_tiles(source.convert("RGB"), active_profile)
                )
        except Exception as exc:  # noqa: BLE001 - unreadable UI frames must abstain rather than abort the run
            records.append(
                {
                    **base,
                    "layout": layout["name"],
                    "active_tiles": [],
                    "reason": f"frame_read_failed:{type(exc).__name__}",
                }
            )
            continue
        reason = "single_active_tile" if len(active_tiles) == 1 else "multiple_active_tiles" if active_tiles else "no_active_tile"
        records.append({**base, "layout": layout["name"], "active_tiles": active_tiles, "reason": reason})
    return records


def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left = max(0, min(width - 1, round(box[0] * width)))
    top = max(0, min(height - 1, round(box[1] * height)))
    right = min(width, max(left + 1, round(box[2] * width)))
    bottom = min(height, max(top + 1, round(box[3] * height)))
    return image.crop((left, top, right, bottom))


def _circle_vector(image: Image.Image, *, signature_size: int, radius_fraction: float = 0.422) -> np.ndarray:
    normalized = image.convert("RGB").resize((signature_size, signature_size))
    pixels = np.asarray(normalized, dtype=np.float32) / 255.0
    ys, xs = np.mgrid[:signature_size, :signature_size]
    radius = signature_size * radius_fraction
    circle = (xs - (signature_size - 1) / 2.0) ** 2 + (ys - (signature_size - 1) / 2.0) ** 2 < radius**2
    return pixels[circle].reshape(-1)


def _entropy(image: Image.Image) -> float:
    pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    if pixels.size == 0:
        return 0.0
    counts = np.bincount(pixels.reshape(-1), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0.0] / float(pixels.size)
    return float(-(probabilities * np.log2(probabilities)).sum() / 8.0)


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    normalized_left = left - float(left.mean())
    normalized_right = right - float(right.mean())
    denominator = float(np.linalg.norm(normalized_left) * np.linalg.norm(normalized_right))
    if denominator <= 0.0:
        return -1.0
    return float(normalized_left @ normalized_right / denominator)


def _inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _tile_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap_of_smaller(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(1e-9, min(_tile_area(left), _tile_area(right)))


def _deduplicate_nested_active_tiles(active_tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest border when a green tile fill creates nested false outlines."""

    retained: list[dict[str, Any]] = []
    for candidate in sorted(active_tiles, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        try:
            box = tuple(float(value) for value in candidate["tile"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(box) != 4 or _tile_area(box) <= 0.0:
            continue
        if any(
            _overlap_of_smaller(box, tuple(float(value) for value in existing["tile"])) >= 0.86
            for existing in retained
        ):
            continue
        retained.append(candidate)
    return sorted(retained, key=lambda item: (float(item["tile"][1]), float(item["tile"][0])))


def _roster_avatar_box(
    text_box: tuple[float, float, float, float],
    layout: dict[str, Any],
) -> tuple[float, float, float, float]:
    height = text_box[3] - text_box[1]
    avatar = layout["avatar"]
    side = height * float(avatar["size_multiplier"])
    gap = height * float(avatar["gap_multiplier"])
    center_y = (text_box[1] + text_box[3]) / 2.0 + height * float(avatar["vertical_offset_multiplier"])
    right = text_box[0] - gap
    return (right - side, center_y - side / 2.0, right, center_y + side / 2.0)


def _roster_rows(
    image: Image.Image,
    texts: list[dict[str, Any]],
    profile: dict[str, Any],
    layout: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    region = tuple(float(value) for value in layout["roster_region"])
    for text in texts:
        value = str(text.get("text", ""))
        key = _normalized_name(value)
        name = profile["participant_by_key"].get(key)
        local_box = _ocr_box(text)
        confidence = float(text.get("confidence", 0.0))
        if not name or not local_box:
            continue
        if confidence < float(profile["settings"]["minimum_ocr_confidence"]):
            rejected.append({"text": value, "name": name, "reason": "low_roster_ocr_confidence"})
            continue
        text_box = _globalize(local_box, region)
        avatar_box = _roster_avatar_box(text_box, layout)
        if not _inside(avatar_box, region):
            rejected.append({"text": value, "name": name, "reason": "avatar_box_outside_roster_region"})
            continue
        by_name[name].append(
            {
                "name": name,
                "text": value,
                "ocr_confidence": round(confidence, 4),
                "text_box": [round(value, 6) for value in text_box],
                "avatar_box": [round(value, 6) for value in avatar_box],
            }
        )

    rows: list[dict[str, Any]] = []
    for name, candidates in sorted(by_name.items()):
        if len(candidates) != 1:
            rejected.append({"name": name, "reason": "ambiguous_roster_name_row", "count": len(candidates)})
            continue
        row = dict(candidates[0])
        try:
            crop = _crop(image, tuple(float(value) for value in row["avatar_box"]))
            if min(crop.size) < int(profile["settings"]["minimum_roster_avatar_pixels"]):
                raise RosterAvatarProfileError("roster avatar crop is too small")
            standard_deviation = _spatial_stddev(crop)
            entropy = _entropy(crop)
            if standard_deviation < float(profile["settings"]["minimum_avatar_stddev"]):
                row["reason"] = "roster_avatar_low_variance"
                row["avatar_stddev"] = round(standard_deviation, 6)
                row["avatar_entropy"] = round(entropy, 6)
                rejected.append(row)
                continue
            row["_vectors"] = {
                "outer": _circle_vector(
                    crop,
                    signature_size=int(profile["settings"]["signature_size"]),
                    radius_fraction=float(profile["settings"]["active_outer_circle_radius"]),
                ),
                "inner": _circle_vector(
                    crop,
                    signature_size=int(profile["settings"]["signature_size"]),
                    radius_fraction=float(profile["settings"]["active_inner_circle_radius"]),
                ),
            }
            row["avatar_stddev"] = round(standard_deviation, 6)
            row["avatar_entropy"] = round(entropy, 6)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - malformed roster crops are explicit abstentions
            row["reason"] = f"roster_avatar_signature_error:{type(exc).__name__}"
            rejected.append(row)
    return rows, rejected


def _template_collisions(rows: list[dict[str, Any]], profile: dict[str, Any]) -> set[str]:
    threshold = float(profile["settings"]["duplicate_template_similarity"])
    ambiguous: set[str] = set()
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            active = {"vectors": left["_vectors"]} if left.get("_vectors") is not None else {"vector": left.get("_vector")}
            score = _ensemble_similarity(active, right)
            if score >= threshold:
                ambiguous.add(str(left["name"]))
                ambiguous.add(str(right["name"]))
    return ambiguous


def _spatial_stddev(image: Image.Image) -> float:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if pixels.size == 0:
        return 0.0
    return float(pixels.reshape(-1, 3).std(axis=0).mean())


def _corner_stddev(image: Image.Image, tile: tuple[float, float, float, float]) -> float:
    crop = _crop(image, tile)
    pixels = np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = pixels.shape
    extent_y = max(1, round(height * 0.22))
    extent_x = max(1, round(width * 0.22))
    samples = np.concatenate(
        [
            pixels[:extent_y, :extent_x],
            pixels[:extent_y, -extent_x:],
            pixels[-extent_y:, :extent_x],
            pixels[-extent_y:, -extent_x:],
        ]
    ).reshape(-1, 3)
    return float(samples.std(axis=0).mean())


def _inset_tile(tile: tuple[float, float, float, float], fraction: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = tile
    inset_x = (x2 - x1) * fraction
    inset_y = (y2 - y1) * fraction
    return (x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y)


def _centered_avatar_crop(
    image: Image.Image,
    tile: tuple[float, float, float, float],
    fraction: float,
) -> Image.Image:
    x1, y1, x2, y2 = tile
    side = min(x2 - x1, y2 - y1) * fraction
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return _crop(image, (center_x - side / 2.0, center_y - side / 2.0, center_x + side / 2.0, center_y + side / 2.0))


def _active_avatar_vectors(image: Image.Image, tile: tuple[float, float, float, float], profile: dict[str, Any]) -> dict[str, np.ndarray]:
    settings = profile["settings"]
    return {
        "outer": _circle_vector(
            _centered_avatar_crop(image, tile, float(settings["active_outer_crop_fraction"])),
            signature_size=int(settings["signature_size"]),
            radius_fraction=float(settings["active_outer_circle_radius"]),
        ),
        "inner": _circle_vector(
            _centered_avatar_crop(image, tile, float(settings["active_inner_crop_fraction"])),
            signature_size=int(settings["signature_size"]),
            radius_fraction=float(settings["active_inner_circle_radius"]),
        ),
    }


def _ensemble_similarity(active: dict[str, Any], row: dict[str, Any]) -> float:
    active_vectors = active.get("vectors") or ({"outer": active["vector"], "inner": active["vector"]} if active.get("vector") is not None else {})
    row_vectors = row.get("_vectors") or ({"outer": row["_vector"], "inner": row["_vector"]} if row.get("_vector") is not None else {})
    scores = [
        _similarity(np.asarray(active_vectors[name]), np.asarray(row_vectors[name]))
        for name in ("outer", "inner")
        if name in active_vectors and name in row_vectors
    ]
    return float(sum(scores) / len(scores)) if scores else -1.0


def _active_avatar(
    image: Image.Image,
    tile: tuple[float, float, float, float],
    profile: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    settings = profile["settings"]
    inner_tile = _inset_tile(tile, float(settings["active_border_inset_fraction"]))
    try:
        signature = extract_avatar_signature_from_image(image, inner_tile, settings)
    except Exception as exc:  # noqa: BLE001 - an unreadable active tile must not produce a name
        return None, f"active_avatar_signature_error:{type(exc).__name__}"
    inset = float(settings["active_border_inset_fraction"])
    scale = 1.0 - inset * 2.0
    box = [inset + float(value) * scale for value in signature["avatar_box_in_tile"]]
    area = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))
    if area < float(settings["minimum_tile_avatar_area_ratio"]):
        return None, "active_tile_avatar_too_small"
    if area > float(settings["maximum_tile_avatar_area_ratio"]):
        return None, "active_tile_not_avatar_like"
    background_stddev = _corner_stddev(image, inner_tile)
    if background_stddev > float(settings["maximum_tile_background_stddev"]):
        return None, "active_tile_not_avatar_like"
    outer_crop = _centered_avatar_crop(image, tile, float(settings["active_outer_crop_fraction"]))
    inner_crop = _centered_avatar_crop(image, tile, float(settings["active_inner_crop_fraction"]))
    avatar_stddev = max(_spatial_stddev(outer_crop), _spatial_stddev(inner_crop))
    if avatar_stddev < float(settings["minimum_avatar_stddev"]):
        return None, "active_tile_default_or_low_variance_avatar"
    return {
        "vectors": _active_avatar_vectors(image, tile, profile),
        "avatar_box_in_tile": box,
        "avatar_area_ratio": round(area, 6),
        "background_stddev": round(background_stddev, 6),
        "avatar_stddev": round(avatar_stddev, 6),
    }, None


def _ocr_texts(record: dict[str, Any]) -> list[dict[str, Any]]:
    regions = record.get("regions")
    if not isinstance(regions, list):
        return []
    for region in regions:
        if isinstance(region, dict) and str(region.get("label", "")).endswith("::roster"):
            texts = region.get("texts")
            return list(texts) if isinstance(texts, list) else []
    return []


def _serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"_vector", "_vectors"}}


def score_roster_avatar_frames(
    detected_frames: list[dict[str, Any]],
    roster_ocr_records: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    anchor_names_by_time: dict[float, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Score same-frame roster-to-active-tile matches without mutating the transcript."""

    texts_by_path = {str(record.get("path")): _ocr_texts(record) for record in roster_ocr_records if record.get("path")}
    scored: list[dict[str, Any]] = []
    anchor_names_by_time = anchor_names_by_time or {}
    for frame in detected_frames:
        time = float(frame.get("time", 0.0))
        actual_time = float(frame.get("actualTime", time))
        layout = layout_at(profile, actual_time)
        base: dict[str, Any] = {
            "time": time,
            "actualTime": actual_time,
            "path": frame.get("path"),
            "layout": frame.get("layout"),
            "active_tiles": frame.get("active_tiles", []),
            "reason": frame.get("reason"),
            "reviewed_anchor_names": sorted(anchor_names_by_time.get(round(time, 3), [])),
        }
        active_tiles = list(frame.get("active_tiles", []))
        if not layout:
            scored.append({**base, "decision": "no_roster_layout", "candidate_name": None, "roster_rows": []})
            continue
        if len(active_tiles) != 1:
            decision = "multiple_active_tiles" if len(active_tiles) > 1 else "no_active_tile"
            scored.append({**base, "decision": decision, "candidate_name": None, "roster_rows": []})
            continue
        path = frame.get("path")
        if not path:
            scored.append({**base, "decision": "missing_frame", "candidate_name": None, "roster_rows": []})
            continue
        try:
            with Image.open(str(path)) as source:
                image = source.convert("RGB")
                rows, rejected_rows = _roster_rows(image, texts_by_path.get(str(path), []), profile, layout)
                ambiguous_templates = _template_collisions(rows, profile)
                usable_rows = [row for row in rows if str(row["name"]) not in ambiguous_templates]
                row_details = [_serializable_row(row) for row in rows]
                if len(usable_rows) < int(profile["settings"]["minimum_roster_identities"]):
                    scored.append(
                        {
                            **base,
                            "decision": "insufficient_distinct_roster_avatars",
                            "candidate_name": None,
                            "roster_rows": row_details,
                            "rejected_roster_rows": rejected_rows,
                            "ambiguous_template_names": sorted(ambiguous_templates),
                        }
                    )
                    continue
                tile = tuple(float(value) for value in active_tiles[0]["tile"])
                active_avatar, active_reason = _active_avatar(image, tile, profile)
                if not active_avatar:
                    scored.append(
                        {
                            **base,
                            "decision": active_reason or "active_tile_not_avatar_like",
                            "candidate_name": None,
                            "roster_rows": row_details,
                            "rejected_roster_rows": rejected_rows,
                            "ambiguous_template_names": sorted(ambiguous_templates),
                        }
                    )
                    continue
                ranked = sorted(
                    (
                        (_ensemble_similarity(active_avatar, row), str(row["name"]))
                        for row in usable_rows
                    ),
                    key=lambda item: (-item[0], item[1]),
                )
                top_score, top_name = ranked[0]
                runner_score, runner_name = ranked[1]
                margin = top_score - runner_score
                decision = "matched"
                if top_score < float(profile["settings"]["minimum_similarity"]):
                    decision = "similarity_below_threshold"
                elif margin < float(profile["settings"]["minimum_margin"]):
                    decision = "similarity_margin_below_threshold"
                scored.append(
                    {
                        **base,
                        "decision": decision,
                        "candidate_name": top_name if decision == "matched" else None,
                        "top_candidate_name": top_name,
                        "top_score": round(float(top_score), 6),
                        "runner_candidate_name": runner_name,
                        "runner_score": round(float(runner_score), 6),
                        "margin": round(float(margin), 6),
                        "scores": {name: round(float(score), 6) for score, name in ranked},
                        "active_avatar": {
                            "avatar_box_in_tile": active_avatar["avatar_box_in_tile"],
                            "avatar_area_ratio": active_avatar["avatar_area_ratio"],
                            "background_stddev": active_avatar["background_stddev"],
                            "avatar_stddev": active_avatar.get("avatar_stddev"),
                        },
                        "roster_rows": row_details,
                        "rejected_roster_rows": rejected_rows,
                        "ambiguous_template_names": sorted(ambiguous_templates),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - per-frame scoring failures are recorded as abstentions
            scored.append(
                {
                    **base,
                    "decision": f"frame_score_failed:{type(exc).__name__}",
                    "candidate_name": None,
                    "roster_rows": [],
                }
            )
    return scored


def calibrate_roster_avatar_identity(scored_frames: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    """Require reviewed, time-separated same-frame anchors before name assignment."""

    anchors: list[dict[str, Any]] = []
    for frame in scored_frames:
        for expected_name in frame.get("reviewed_anchor_names", []):
            candidate = frame.get("candidate_name")
            decision = "accepted" if frame.get("decision") == "matched" and candidate == expected_name else "rejected"
            anchors.append(
                {
                    "time": round(float(frame.get("actualTime", frame.get("time", 0.0))), 3),
                    "expected_name": expected_name,
                    "candidate_name": candidate,
                    "frame_decision": frame.get("decision"),
                    "top_candidate_name": frame.get("top_candidate_name"),
                    "top_score": frame.get("top_score"),
                    "margin": frame.get("margin"),
                    "decision": decision,
                    "path": frame.get("path"),
                }
            )
    accepted = [anchor for anchor in anchors if anchor["decision"] == "accepted"]
    accepted_names = {str(anchor["expected_name"]) for anchor in accepted}
    times = sorted(float(anchor["time"]) for anchor in anchors)
    unique_times = len(set(times)) == len(times)
    separated = unique_times and all(
        right - left >= float(profile["settings"]["minimum_anchor_seconds_separation"])
        for left, right in pairwise(times)
    )
    requirements = {
        "enough_reviewed_anchors": len(anchors) >= int(profile["settings"]["minimum_reviewed_anchors"]),
        "enough_distinct_anchor_identities": len(accepted_names)
        >= int(profile["settings"]["minimum_anchor_identities"]),
        "anchor_sample_times_unique": unique_times,
        "anchors_time_separated": separated,
        "all_reviewed_anchors_matched": len(accepted) == len(anchors) and bool(anchors),
        "no_reviewed_anchor_false_accepts": all(
            anchor["candidate_name"] in {None, anchor["expected_name"]} for anchor in anchors
        ),
    }
    return {
        "gate": {"status": "passed" if all(requirements.values()) else "blocked", "requirements": requirements},
        "anchors": anchors,
        "accepted_anchors": len(accepted),
        "distinct_anchor_identities": sorted(accepted_names),
        "eligible_identities": sorted(accepted_names) if all(requirements.values()) else [],
    }


def _clear_stale_roster_identity(segment: dict[str, Any], *, clear_identity: bool = True) -> None:
    if clear_identity and str(segment.get("name_source") or "") == SOURCE:
        segment["name"] = None
        segment["name_source"] = None
        segment["name_confidence"] = 0.0
    for key in (
        "roster_avatar_identity_evidence",
        "roster_avatar_identity_conflict",
        "roster_avatar_identity_corroborates",
        "roster_avatar_identity_corrected_prior",
        "roster_avatar_identity_replaced_prior",
        "roster_avatar_identity_rerun_conflict",
    ):
        segment.pop(key, None)


def attach_roster_avatar_identity(
    segments: list[dict[str, Any]],
    sample_requests: list[dict[str, Any]],
    scored_frames: list[dict[str, Any]],
    profile: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Attach direct same-frame roster evidence while preserving all prior names."""

    frame_by_time = {round(float(frame.get("time", 0.0)), 3): frame for frame in scored_frames}
    requests_by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for request in sample_requests:
        requests_by_segment[int(request["segment_index"])].append(request)
    gate_passed = calibration.get("gate", {}).get("status") == "passed"
    summary: dict[str, Any] = {
        "segments": len(segments),
        "assigned": 0,
        "unresolved": 0,
        "gate_blocked": 0,
        "preserved_existing_identity": 0,
        "corroborated_existing_identity": 0,
        "conflicts": 0,
        "unanchored_identity": 0,
        "corrected_weaker_identity": 0,
        "preserved_prior_roster_identity": 0,
        "replaced_prior_roster_identity": 0,
        "by_decision": Counter(),
    }
    if not gate_passed:
        summary["gate_blocked"] = len(segments)
        summary["preserved_prior_roster_identity"] = sum(
            1 for segment in segments if str(segment.get("name_source") or "") == SOURCE and segment.get("name")
        )
        summary["unresolved"] = sum(1 for segment in segments if not segment.get("name"))
        summary["by_decision"] = {}
        return summary
    for index, segment in enumerate(segments):
        existing_name = segment.get("name")
        existing_source = str(segment.get("name_source") or "")
        existing_confidence = segment.get("name_confidence")
        prior_roster_identity = bool(existing_name) and existing_source == SOURCE
        if not prior_roster_identity:
            _clear_stale_roster_identity(segment)
        sample_frames = unique_frames_for_requests(requests_by_segment.get(index, []), frame_by_time)
        evidence = [
            {
                "time": round(float(frame.get("actualTime", frame.get("time", 0.0))), 3),
                "frame": frame.get("path"),
                "active_tiles": frame.get("active_tiles", []),
                "decision": frame.get("decision"),
                "candidate_name": frame.get("candidate_name"),
                "top_candidate_name": frame.get("top_candidate_name"),
                "top_score": frame.get("top_score"),
                "runner_candidate_name": frame.get("runner_candidate_name"),
                "runner_score": frame.get("runner_score"),
                "margin": frame.get("margin"),
                "active_avatar": frame.get("active_avatar"),
                "roster_rows": frame.get("roster_rows", []),
            }
            for frame in sample_frames
        ]
        if evidence and not prior_roster_identity:
            segment["roster_avatar_identity_evidence"] = evidence
        for frame in sample_frames:
            summary["by_decision"][str(frame.get("decision", "missing"))] += 1
        accepted = [frame for frame in sample_frames if frame.get("decision") == "matched" and frame.get("candidate_name")]
        votes = Counter(str(frame["candidate_name"]) for frame in accepted)
        correctable_prior = (
            {"name": str(existing_name), "source": existing_source}
            if existing_name and existing_source in _CORRECTABLE_EXISTING_SOURCES
            else None
        )
        if existing_name and existing_source != SOURCE and not correctable_prior:
            summary["preserved_existing_identity"] += 1
            if votes:
                if len(votes) == 1 and next(iter(votes)) == str(existing_name):
                    segment["roster_avatar_identity_corroborates"] = SOURCE
                    summary["corroborated_existing_identity"] += 1
                elif any(name != str(existing_name) for name in votes):
                    segment["roster_avatar_identity_conflict"] = {
                        "reason": "existing_identity_disagrees_with_roster_avatar",
                        "existing_name": existing_name,
                        "existing_source": existing_source,
                        "roster_avatar_names": sorted(votes),
                    }
                    summary["conflicts"] += 1
            continue
        if not votes:
            if prior_roster_identity:
                summary["preserved_prior_roster_identity"] += 1
                continue
            summary["unresolved"] += 1
            continue
        if len(votes) > 1:
            if prior_roster_identity:
                segment["roster_avatar_identity_rerun_conflict"] = {
                    "reason": "multiple_roster_avatar_names_in_segment",
                    "prior_roster_avatar_name": existing_name,
                    "roster_avatar_names": sorted(votes),
                }
                summary["preserved_prior_roster_identity"] += 1
                summary["conflicts"] += 1
                continue
            segment["roster_avatar_identity_conflict"] = {
                "reason": "multiple_roster_avatar_names_in_segment",
                "roster_avatar_names": sorted(votes),
            }
            summary["conflicts"] += 1
            summary["unresolved"] += 1
            continue
        name, count = votes.most_common(1)[0]
        eligible_identities = set(calibration.get("eligible_identities", []))
        if bool(profile["settings"]["anchor_only_identities"]) and name not in eligible_identities:
            if prior_roster_identity:
                segment["roster_avatar_identity_rerun_conflict"] = {
                    "reason": "candidate_not_covered_by_accepted_anchor",
                    "prior_roster_avatar_name": existing_name,
                    "roster_avatar_name": name,
                }
                summary["preserved_prior_roster_identity"] += 1
                summary["unanchored_identity"] += 1
                continue
            summary["unanchored_identity"] += 1
            summary["unresolved"] += 1
            continue
        vote_share = count / len(sample_frames) if sample_frames else 0.0
        if count < int(profile["settings"]["minimum_supporting_frames"]) or vote_share < float(
            profile["settings"]["minimum_segment_vote_share"]
        ):
            if prior_roster_identity:
                segment["roster_avatar_identity_rerun_conflict"] = {
                    "reason": "insufficient_roster_avatar_consensus",
                    "prior_roster_avatar_name": existing_name,
                    "roster_avatar_name": name,
                    "supporting_frames": count,
                    "vote_share": round(vote_share, 3),
                }
                summary["preserved_prior_roster_identity"] += 1
                continue
            summary["unresolved"] += 1
            continue
        matching = [frame for frame in accepted if frame.get("candidate_name") == name]
        mean_score = sum(float(frame.get("top_score", 0.0)) for frame in matching) / len(matching)
        mean_margin = sum(float(frame.get("margin", 0.0)) for frame in matching) / len(matching)
        confidence = min(0.90, 0.42 + 0.26 * mean_score + 0.18 * mean_margin + 0.14 * vote_share)
        if prior_roster_identity:
            _clear_stale_roster_identity(segment)
        if evidence:
            segment["roster_avatar_identity_evidence"] = evidence
        segment["name"] = name
        segment["name_source"] = SOURCE
        segment["name_confidence"] = round(confidence, 3)
        segment["frame_refs"] = sorted({str(frame["path"]) for frame in matching if frame.get("path")})[:4]
        if correctable_prior:
            if correctable_prior["name"] == name:
                segment["roster_avatar_identity_corroborates"] = SOURCE
                summary["corroborated_existing_identity"] += 1
            else:
                segment["roster_avatar_identity_corrected_prior"] = correctable_prior
                summary["corrected_weaker_identity"] += 1
        elif prior_roster_identity and str(existing_name) != name:
            segment["roster_avatar_identity_replaced_prior"] = {
                "name": existing_name,
                "source": existing_source,
                "confidence": existing_confidence,
            }
            summary["replaced_prior_roster_identity"] += 1
        summary["assigned"] += 1
    summary["by_decision"] = dict(summary["by_decision"])
    return summary


def write_roster_avatar_identity_report(
    path: Path,
    *,
    profile_path: Path,
    calibration: dict[str, Any],
    attachment_summary: dict[str, Any],
) -> None:
    gate = calibration["gate"]
    lines = [
        "# 侧栏头像实名映射报告",
        "",
        f"- 配置文件：`{profile_path}`",
        f"- 校准门禁：`{gate['status']}`",
        f"- 人工复核锚点数：{len(calibration['anchors'])}",
        f"- 通过锚点数：{calibration['accepted_anchors']}",
        f"- 不同锚点身份：{', '.join(calibration['distinct_anchor_identities']) or '无'}",
        f"- 自动映射身份范围：{', '.join(calibration.get('eligible_identities', [])) or '无'}",
        "",
        "## 门禁检查",
    ]
    for key, passed in gate["requirements"].items():
        lines.append(f"- {key}：{'通过' if passed else '阻断'}")
    lines.extend(
        [
            "",
            "## 片段结果",
            f"- 已实名映射：{attachment_summary.get('assigned', 0)}",
            f"- 未解析：{attachment_summary.get('unresolved', 0)}",
            f"- 被校准门禁阻断：{attachment_summary.get('gate_blocked', 0)}",
            f"- 校准失败时保留的既有实名：{attachment_summary.get('preserved_prior_roster_identity', 0)}",
            f"- 新片段证据通过后替换的既有实名：{attachment_summary.get('replaced_prior_roster_identity', 0)}",
            f"- 保留已有身份：{attachment_summary.get('preserved_existing_identity', 0)}",
            f"- 与已有身份一致：{attachment_summary.get('corroborated_existing_identity', 0)}",
            f"- 已纠正较弱身份：{attachment_summary.get('corrected_weaker_identity', 0)}",
            f"- 身份冲突：{attachment_summary.get('conflicts', 0)}",
            f"- 未经锚点验证的身份：{attachment_summary.get('unanchored_identity', 0)}",
            "",
            "## 证据边界",
            "- 单独的侧栏文字绝不会直接给说话人实名。",
            "- 每次实名映射都需要精确侧栏姓名、至少三名可用侧栏头像、唯一的绿色活动卡片、同帧有区分度的头像匹配以及片段内多帧一致。",
            "- 默认头像、重复头像、摄像头画面、屏幕共享、缺失布局和候选不明确的帧都会弃权。",
            "- 默认只给通过人工锚点校准的身份自动实名；其他候选保留为待复核。",
            "- 此来源不能用于声纹注册或说话人聚类传播。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
