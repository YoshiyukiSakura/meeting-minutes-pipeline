"""Evidence-gated identity matching for Discord-style avatar tiles.

This stage never assumes that a participant stays in a particular grid position.
It learns only from reviewed, in-tile nameplate references and evaluates each
green-highlighted tile independently.  A failed match is an explicit abstention.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label

from .identity_authority import (
    DYNAMIC_NAMEPLATE_SOURCE,
    ROSTER_AVATAR_SOURCE,
    SAME_SESSION_VISUAL_VOICE_SOURCE,
)
from .sample_evidence import unique_frames_for_requests
from .visual_identity import has_trusted_identity


class AvatarTemplateProfileError(ValueError):
    """Raised when a reviewed avatar-template profile is invalid."""


_DIRECT_NAMEPLATE_SOURCE = DYNAMIC_NAMEPLATE_SOURCE
_ROSTER_AVATAR_SOURCE = ROSTER_AVATAR_SOURCE
_VOICEPRINT_SOURCE = SAME_SESSION_VISUAL_VOICE_SOURCE
_DEFAULTS: dict[str, Any] = {
    "minimum_similarity": 0.60,
    "minimum_margin": 0.30,
    "minimum_segment_vote_share": 0.67,
    "short_segment_seconds": 5.0,
    "minimum_validation_frames": 10,
    "minimum_validation_per_identity": 3,
    "minimum_negative_samples": 3,
    "minimum_layout_area_ratio_delta": 0.20,
    "foreground_distance": 0.14,
    "signature_size": 64,
}


def _box(raw: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise AvatarTemplateProfileError(f"{field} must be a four-number list")
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise AvatarTemplateProfileError(f"{field} must contain numbers") from exc
    if not 0.0 <= x1 < x2 <= 1.0 or not 0.0 <= y1 < y2 <= 1.0:
        raise AvatarTemplateProfileError(f"{field} must be a normalized top-left box")
    return (x1, y1, x2, y2)


def _profile_path(value: Any, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AvatarTemplateProfileError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_avatar_template_profile(path: Path) -> dict[str, Any]:
    """Load reviewed avatar references and calibration constraints.

    Template coordinates belong only to their named reference screenshot.  They
    are never reused as coordinates for target frames.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AvatarTemplateProfileError("avatar template profile must be a JSON object")
    templates = payload.get("templates")
    if not isinstance(templates, list) or not templates:
        raise AvatarTemplateProfileError("templates must be a non-empty list")

    base_dir = path.parent
    normalized_templates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(templates):
        if not isinstance(item, dict):
            raise AvatarTemplateProfileError(f"templates[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AvatarTemplateProfileError(f"templates[{index}].name must be a non-empty string")
        normalized_name = name.strip()
        if normalized_name.casefold() in seen_names:
            raise AvatarTemplateProfileError("templates must not contain duplicate names")
        seen_names.add(normalized_name.casefold())
        evidence = item.get("evidence")
        if evidence != "in_tile_nameplate":
            raise AvatarTemplateProfileError(
                f"templates[{index}].evidence must be in_tile_nameplate; sidebar or inferred names are insufficient"
            )
        normalized_templates.append(
            {
                "name": normalized_name,
                "path": _profile_path(item.get("path"), base_dir, f"templates[{index}].path"),
                "box": _box(item.get("box"), f"templates[{index}].box"),
                "evidence": evidence,
            }
        )

    negatives = payload.get("negative_samples", [])
    if not isinstance(negatives, list):
        raise AvatarTemplateProfileError("negative_samples must be a list")
    normalized_negatives: list[dict[str, Any]] = []
    for index, item in enumerate(negatives):
        if not isinstance(item, dict):
            raise AvatarTemplateProfileError(f"negative_samples[{index}] must be an object")
        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            raise AvatarTemplateProfileError(f"negative_samples[{index}].description must be a non-empty string")
        normalized_negatives.append(
            {
                "description": description.strip(),
                "path": _profile_path(item.get("path"), base_dir, f"negative_samples[{index}].path"),
                "box": _box(item.get("box"), f"negative_samples[{index}].box"),
            }
        )

    raw_settings = payload.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise AvatarTemplateProfileError("settings must be a JSON object")
    settings = dict(_DEFAULTS)
    settings.update({key: raw_settings[key] for key in _DEFAULTS if key in raw_settings})
    for key in (
        "minimum_similarity",
        "minimum_margin",
        "minimum_segment_vote_share",
        "short_segment_seconds",
        "minimum_layout_area_ratio_delta",
        "foreground_distance",
    ):
        settings[key] = float(settings[key])
    for key in (
        "minimum_validation_frames",
        "minimum_validation_per_identity",
        "minimum_negative_samples",
        "signature_size",
    ):
        settings[key] = int(settings[key])
    for key in ("minimum_similarity", "minimum_margin", "minimum_segment_vote_share", "foreground_distance"):
        if not 0.0 < settings[key] <= 1.0:
            raise AvatarTemplateProfileError(f"settings.{key} must be in (0, 1]")
    if settings["short_segment_seconds"] <= 0.0:
        raise AvatarTemplateProfileError("settings.short_segment_seconds must be positive")
    if settings["minimum_layout_area_ratio_delta"] <= 0.0:
        raise AvatarTemplateProfileError("settings.minimum_layout_area_ratio_delta must be positive")
    for key in ("minimum_validation_frames", "minimum_validation_per_identity", "minimum_negative_samples"):
        if settings[key] < 1:
            raise AvatarTemplateProfileError(f"settings.{key} must be at least 1")
    if settings["signature_size"] < 16:
        raise AvatarTemplateProfileError("settings.signature_size must be at least 16")
    return {"templates": normalized_templates, "negative_samples": normalized_negatives, "settings": settings}


def extract_avatar_signature_from_image(
    image: Image.Image,
    box: tuple[float, float, float, float],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Extract a centered, scale-normalized avatar signature from an RGB frame."""

    image = image.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    tile = image.crop((left, top, right, bottom))
    pixels = np.asarray(tile, dtype=np.float32) / 255.0
    tile_height, tile_width, _ = pixels.shape
    if tile_width < 16 or tile_height < 16:
        raise AvatarTemplateProfileError("tile is too small to isolate an avatar")

    top_start = max(1, round(0.08 * tile_height))
    top_end = max(top_start + 1, round(0.30 * tile_height))
    left_start = max(1, round(0.05 * tile_width))
    left_end = max(left_start + 1, round(0.24 * tile_width))
    right_start = min(tile_width - 1, round(0.76 * tile_width))
    right_end = min(tile_width, round(0.95 * tile_width))
    middle_start = round(0.38 * tile_height)
    middle_end = max(middle_start + 1, round(0.68 * tile_height))
    background_samples = np.concatenate(
        [
            pixels[top_start:top_end, left_start:left_end],
            pixels[top_start:top_end, right_start:right_end],
            pixels[middle_start:middle_end, left_start:left_end],
            pixels[middle_start:middle_end, right_start:right_end],
        ]
    ).reshape(-1, 3)
    background = np.median(background_samples, axis=0)
    foreground = np.linalg.norm(pixels - background, axis=2) > float(settings["foreground_distance"])
    center_window = np.zeros_like(foreground, dtype=bool)
    center_window[round(0.10 * tile_height) : round(0.82 * tile_height), round(0.12 * tile_width) : round(0.88 * tile_width)] = True
    foreground &= center_window
    foreground = binary_dilation(foreground, iterations=max(1, round(min(tile_width, tile_height) * 0.025)))
    components, component_count = label(foreground)
    center_y, center_x = tile_height * 0.48, tile_width * 0.50
    best: tuple[float, int, int, int, int] | None = None
    for component_id in range(1, component_count + 1):
        ys, xs = np.where(components == component_id)
        area = len(xs)
        if not area:
            continue
        min_y, max_y = int(ys.min()), int(ys.max()) + 1
        min_x, max_x = int(xs.min()), int(xs.max()) + 1
        distance = math.hypot((float(xs.mean()) - center_x) / tile_width, (float(ys.mean()) - center_y) / tile_height)
        score = area / (1.0 + 10.0 * distance)
        candidate = (score, min_x, min_y, max_x, max_y)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise AvatarTemplateProfileError("cannot isolate a centered avatar from tile")

    _, avatar_left, avatar_top, avatar_right, avatar_bottom = best
    side = max(avatar_right - avatar_left, avatar_bottom - avatar_top) * 1.25
    center_avatar_x = (avatar_left + avatar_right) / 2.0
    center_avatar_y = (avatar_top + avatar_bottom) / 2.0
    crop_left = max(0, round(center_avatar_x - side / 2.0))
    crop_top = max(0, round(center_avatar_y - side / 2.0))
    crop_right = min(tile_width, round(center_avatar_x + side / 2.0))
    crop_bottom = min(tile_height, round(center_avatar_y + side / 2.0))
    if crop_right - crop_left < 8 or crop_bottom - crop_top < 8:
        raise AvatarTemplateProfileError("isolated avatar crop is too small")
    signature_size = int(settings["signature_size"])
    normalized = tile.crop((crop_left, crop_top, crop_right, crop_bottom)).resize((signature_size, signature_size))
    normalized_pixels = np.asarray(normalized, dtype=np.float32) / 255.0
    ys, xs = np.mgrid[:signature_size, :signature_size]
    radius = signature_size * 0.422
    circle = (xs - (signature_size - 1) / 2.0) ** 2 + (ys - (signature_size - 1) / 2.0) ** 2 < radius**2
    vector = normalized_pixels[circle].reshape(-1)
    return {
        "vector": vector,
        "avatar_box_in_tile": [
            round(crop_left / tile_width, 6),
            round(crop_top / tile_height, 6),
            round(crop_right / tile_width, 6),
            round(crop_bottom / tile_height, 6),
        ],
    }


def extract_avatar_signature(path: Path, box: tuple[float, float, float, float], settings: dict[str, Any]) -> dict[str, Any]:
    """Extract a centered, scale-normalized avatar signature from one tile box."""

    if not path.exists():
        raise FileNotFoundError(f"Avatar source frame does not exist: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    return extract_avatar_signature_from_image(image, box, settings)


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    normalized_left = left - float(left.mean())
    normalized_right = right - float(right.mean())
    denominator = float(np.linalg.norm(normalized_left) * np.linalg.norm(normalized_right))
    if denominator <= 0.0:
        return -1.0
    return float(normalized_left @ normalized_right / denominator)


def build_avatar_templates(profile: dict[str, Any]) -> dict[str, Any]:
    templates: dict[str, dict[str, Any]] = {}
    for reference in profile["templates"]:
        signature = extract_avatar_signature(reference["path"], reference["box"], profile["settings"])
        templates[reference["name"]] = {
            "vector": signature["vector"],
            "path": reference["path"],
            "box": reference["box"],
            "avatar_box_in_tile": signature["avatar_box_in_tile"],
        }
    return templates


def _rank_avatar(vector: np.ndarray, templates: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    return sorted(
        ((name, _similarity(vector, np.asarray(template["vector"]))) for name, template in templates.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def _tile_area(box: Any) -> float | None:
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x2 - x1) * (y2 - y1)


def calibrate_avatar_templates(
    frames: list[dict[str, Any]],
    profile: dict[str, Any],
    templates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate fixed thresholds on direct nameplates and explicit negative tiles."""

    settings = profile["settings"]
    reference_paths = {Path(item["path"]).resolve() for item in profile["templates"]}
    validation: list[dict[str, Any]] = []
    excluded_reference_frames = 0
    for frame in frames:
        if frame.get("name_source") != _DIRECT_NAMEPLATE_SOURCE or not frame.get("name"):
            continue
        path_value = frame.get("path")
        if not path_value or len(frame.get("active_tiles", [])) != 1:
            continue
        path = Path(str(path_value)).resolve()
        if path in reference_paths:
            excluded_reference_frames += 1
            continue
        expected_name = str(frame["name"])
        if expected_name not in templates:
            continue
        try:
            signature = extract_avatar_signature(path, tuple(float(value) for value in frame["active_tiles"][0]["tile"]), settings)
            ranked = _rank_avatar(np.asarray(signature["vector"]), templates)
        except Exception as exc:  # noqa: BLE001 - calibration failures are retained as evidence
            validation.append(
                {
                    "path": str(path),
                    "time": float(frame.get("actualTime", frame.get("time", 0.0))),
                    "expected_name": expected_name,
                    "decision": "signature_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        top_name, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = top_score - second_score
        accepted = top_score >= float(settings["minimum_similarity"]) and margin >= float(settings["minimum_margin"])
        validation.append(
            {
                "path": str(path),
                "time": round(float(frame.get("actualTime", frame.get("time", 0.0))), 3),
                "expected_name": expected_name,
                "candidate_name": top_name,
                "score": round(top_score, 6),
                "margin": round(margin, 6),
                "decision": "accepted" if accepted else "rejected",
                "correct": top_name == expected_name,
                "tile": frame["active_tiles"][0]["tile"],
            }
        )

    negatives: list[dict[str, Any]] = []
    for sample in profile["negative_samples"]:
        try:
            signature = extract_avatar_signature(sample["path"], sample["box"], settings)
            ranked = _rank_avatar(np.asarray(signature["vector"]), templates)
            top_name, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -1.0
            margin = top_score - second_score
            accepted = top_score >= float(settings["minimum_similarity"]) and margin >= float(settings["minimum_margin"])
            negatives.append(
                {
                    "description": sample["description"],
                    "path": str(sample["path"]),
                    "box": list(sample["box"]),
                    "candidate_name": top_name,
                    "score": round(top_score, 6),
                    "margin": round(margin, 6),
                    "decision": "false_accept" if accepted else "rejected",
                }
            )
        except Exception as exc:  # noqa: BLE001 - invalid negative evidence blocks the calibration gate
            negatives.append(
                {
                    "description": sample["description"],
                    "path": str(sample["path"]),
                    "box": list(sample["box"]),
                    "decision": "signature_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    by_identity: dict[str, dict[str, Any]] = {}
    for name in templates:
        records = [record for record in validation if record.get("expected_name") == name]
        accepted_records = [record for record in records if record.get("decision") == "accepted"]
        false_accepts = [record for record in accepted_records if not record.get("correct")]
        by_identity[name] = {
            "validation_frames": len(records),
            "accepted": len(accepted_records),
            "false_accepts": len(false_accepts),
            "eligible_for_auto_assignment": len(accepted_records) >= int(settings["minimum_validation_per_identity"])
            and not false_accepts,
        }

    accepted_validation = [record for record in validation if record.get("decision") == "accepted"]
    validation_false_accepts = [record for record in accepted_validation if not record.get("correct")]
    validation_errors = [record for record in validation if record.get("decision") == "signature_error"]
    negative_false_accepts = [record for record in negatives if record.get("decision") == "false_accept"]
    negative_errors = [record for record in negatives if record.get("decision") == "signature_error"]
    layout_diverse = False
    layout_ratios: list[float] = []
    for record in accepted_validation:
        reference_area = _tile_area(templates[str(record["expected_name"])]["box"])
        target_area = _tile_area(record.get("tile"))
        if not reference_area or not target_area:
            continue
        ratio = target_area / reference_area
        layout_ratios.append(ratio)
        if abs(math.log(ratio)) >= math.log(1.0 + float(settings["minimum_layout_area_ratio_delta"])):
            layout_diverse = True
    eligible_names = sorted(name for name, values in by_identity.items() if values["eligible_for_auto_assignment"])
    requirements = {
        "enough_validation_frames": len(validation) >= int(settings["minimum_validation_frames"]),
        "no_validation_false_accepts": not validation_false_accepts,
        "no_validation_errors": not validation_errors,
        "enough_negative_samples": len(negatives) >= int(settings["minimum_negative_samples"]),
        "no_negative_false_accepts": not negative_false_accepts,
        "no_negative_errors": not negative_errors,
        "layout_diversity_observed": layout_diverse,
        "at_least_one_eligible_identity": bool(eligible_names),
    }
    status = "passed" if all(requirements.values()) else "blocked"
    return {
        "gate": {"status": status, "requirements": requirements},
        "reference_paths": sorted(str(path) for path in reference_paths),
        "reference_frames_excluded_from_validation": excluded_reference_frames,
        "validation": validation,
        "negative_samples": negatives,
        "by_identity": by_identity,
        "eligible_names": eligible_names,
        "audit_only_names": sorted(set(templates) - set(eligible_names)),
        "layout_area_ratios": [round(value, 6) for value in layout_ratios],
    }


def score_avatar_template_frames(
    frames: list[dict[str, Any]],
    profile: dict[str, Any],
    templates: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score each dynamically detected active tile without mutating nameplate data."""

    settings = profile["settings"]
    eligible_names = set(calibration.get("eligible_names", []))
    gate_passed = calibration.get("gate", {}).get("status") == "passed"
    scored: list[dict[str, Any]] = []
    for frame in frames:
        record = dict(frame)
        active_tiles = record.get("active_tiles", [])
        detail: dict[str, Any] = {"decision": "no_active_tile"}
        if len(active_tiles) > 1:
            detail = {"decision": "multiple_active_tiles"}
        elif len(active_tiles) == 1 and record.get("path"):
            try:
                signature = extract_avatar_signature(
                    Path(str(record["path"])),
                    tuple(float(value) for value in active_tiles[0]["tile"]),
                    settings,
                )
                ranked = _rank_avatar(np.asarray(signature["vector"]), templates)
                top_name, top_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else -1.0
                margin = top_score - second_score
                accepted = (
                    gate_passed
                    and top_name in eligible_names
                    and top_score >= float(settings["minimum_similarity"])
                    and margin >= float(settings["minimum_margin"])
                )
                if not gate_passed:
                    decision = "gate_blocked"
                elif top_name not in eligible_names:
                    decision = "unvalidated_template"
                elif accepted:
                    decision = "accepted"
                else:
                    decision = "rejected"
                detail = {
                    "decision": decision,
                    "candidate_name": top_name,
                    "score": round(top_score, 6),
                    "margin": round(margin, 6),
                    "second_candidate_name": ranked[1][0] if len(ranked) > 1 else None,
                    "avatar_box_in_tile": signature["avatar_box_in_tile"],
                    "reference_frame": str(templates[top_name]["path"]),
                    "reference_tile": list(templates[top_name]["box"]),
                }
            except Exception as exc:  # noqa: BLE001 - per-frame template failures must abstain
                detail = {"decision": "signature_error", "error": f"{type(exc).__name__}: {exc}"}
        record["avatar_template_identity"] = detail
        scored.append(record)
    return scored


def attach_avatar_template_identity(
    segments: list[dict[str, Any]],
    sample_requests: list[dict[str, Any]],
    scored_frames: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Attach per-segment visual evidence while preserving stronger direct evidence."""

    frame_by_time = {round(float(frame["time"]), 3): frame for frame in scored_frames}
    requests_by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for request in sample_requests:
        requests_by_segment[int(request["segment_index"])].append(request)
    settings = profile["settings"]
    summary: dict[str, Any] = {
        "segments": len(segments),
        "assigned": 0,
        "corroborated_voiceprint": 0,
        "voiceprint_conflicts_preserved": 0,
        "roster_avatar_conflicts_preserved": 0,
        "direct_nameplate_conflicts": 0,
        "reviewed_identity_conflicts_preserved": 0,
        "unresolved": 0,
        "by_decision": Counter(),
    }
    for index, segment in enumerate(segments):
        sample_frames = unique_frames_for_requests(requests_by_segment.get(index, []), frame_by_time)
        evidence = []
        for frame in sample_frames:
            detail = dict(frame.get("avatar_template_identity") or {})
            evidence.append(
                {
                    "time": round(float(frame.get("actualTime", frame.get("time", 0.0))), 3),
                    "frame": frame.get("path"),
                    "active_tiles": frame.get("active_tiles", []),
                    **detail,
                }
            )
            summary["by_decision"][str(detail.get("decision", "missing"))] += 1
        if evidence:
            segment["avatar_template_identity_evidence"] = evidence

        direct_nameplate = str(segment.get("name_source") or "") == _DIRECT_NAMEPLATE_SOURCE and bool(segment.get("name"))
        accepted = [item for item in evidence if item.get("decision") == "accepted" and item.get("candidate_name")]
        votes = Counter(str(item["candidate_name"]) for item in accepted)
        if direct_nameplate:
            if votes and any(name != str(segment["name"]) for name in votes):
                segment["avatar_template_identity_conflict"] = {
                    "reason": "direct_nameplate_disagrees_with_avatar_template",
                    "direct_name": segment["name"],
                    "avatar_names": sorted(votes),
                }
                summary["direct_nameplate_conflicts"] += 1
            continue
        if not votes:
            if not segment.get("name"):
                summary["unresolved"] += 1
            continue
        if len(votes) > 1:
            segment["avatar_template_identity_conflict"] = {
                "reason": "multiple_avatar_names_in_segment",
                "avatar_names": sorted(votes),
            }
            if not segment.get("name"):
                summary["unresolved"] += 1
            continue
        name, count = votes.most_common(1)[0]
        active_samples = [item for item in evidence if item.get("decision") not in {"no_active_tile", "multiple_active_tiles"}]
        duration = max(0.0, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
        required_count = 1 if duration <= float(settings["short_segment_seconds"]) else min(2, len(active_samples))
        vote_share = count / len(active_samples) if active_samples else 0.0
        if count < required_count or vote_share < float(settings["minimum_segment_vote_share"]):
            if not segment.get("name"):
                summary["unresolved"] += 1
            continue
        matching = [item for item in accepted if item.get("candidate_name") == name]
        mean_score = sum(float(item["score"]) for item in matching) / len(matching)
        mean_margin = sum(float(item["margin"]) for item in matching) / len(matching)
        existing_name = segment.get("name")
        existing_source = str(segment.get("name_source") or "")
        if has_trusted_identity(segment):
            if str(existing_name) != name:
                segment["avatar_template_identity_conflict"] = {
                    "reason": "reviewed_or_registered_identity_disagrees_with_avatar_template",
                    "existing_name": existing_name,
                    "existing_source": existing_source,
                    "avatar_name": name,
                }
                summary["reviewed_identity_conflicts_preserved"] += 1
            else:
                segment["avatar_template_identity_corroborates"] = existing_source
            continue
        if existing_source == _ROSTER_AVATAR_SOURCE and existing_name:
            if str(existing_name) != name:
                segment["avatar_template_identity_conflict"] = {
                    "reason": "roster_avatar_disagrees_with_avatar_template",
                    "roster_avatar_name": existing_name,
                    "avatar_name": name,
                }
                summary["roster_avatar_conflicts_preserved"] += 1
            else:
                segment["avatar_template_identity_corroborates"] = _ROSTER_AVATAR_SOURCE
            continue
        if existing_source == _VOICEPRINT_SOURCE and existing_name and str(existing_name) != name:
            segment["avatar_template_identity_conflict"] = {
                "reason": "voiceprint_disagrees_with_avatar_template",
                "voiceprint_name": existing_name,
                "avatar_name": name,
            }
            summary["voiceprint_conflicts_preserved"] += 1
            continue
        confidence = min(0.94, 0.52 + 0.25 * mean_score + 0.20 * mean_margin + 0.10 * vote_share)
        segment["name"] = name
        segment["name_source"] = "visual_avatar_template_match"
        segment["name_confidence"] = round(confidence, 3)
        segment["frame_refs"] = sorted({str(item["frame"]) for item in matching if item.get("frame")})[:4]
        if existing_source == _VOICEPRINT_SOURCE and existing_name == name:
            segment["avatar_template_identity_corroborates"] = _VOICEPRINT_SOURCE
            summary["corroborated_voiceprint"] += 1
        summary["assigned"] += 1
    summary["by_decision"] = dict(summary["by_decision"])
    return summary


def serializable_template_library(templates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "reference_frame": str(value["path"]),
            "reference_tile": list(value["box"]),
            "avatar_box_in_tile": value["avatar_box_in_tile"],
        }
        for name, value in sorted(templates.items())
    ]


def write_avatar_template_identity_report(
    path: Path,
    *,
    profile_path: Path,
    calibration: dict[str, Any],
    attachment_summary: dict[str, Any],
) -> None:
    gate = calibration["gate"]
    lines = [
        "# 头像模板身份识别报告",
        "",
        f"- 配置文件：`{profile_path}`",
        f"- 校准闸门：`{gate['status']}`",
        f"- 直接姓名牌验证帧：{len(calibration['validation'])}",
        f"- 参考帧已从验证集中排除：{calibration['reference_frames_excluded_from_validation']}",
        f"- 负样本：{len(calibration['negative_samples'])}",
        f"- 可自动实名模板：{', '.join(calibration['eligible_names']) or '无'}",
        f"- 仅审计模板：{', '.join(calibration['audit_only_names']) or '无'}",
        "",
        "## 闸门检查",
    ]
    for key, passed in gate["requirements"].items():
        lines.append(f"- {key}：{'通过' if passed else '未通过'}")
    lines.extend(["", "## 应用结果"])
    for key in (
        "assigned",
        "corroborated_voiceprint",
        "voiceprint_conflicts_preserved",
        "roster_avatar_conflicts_preserved",
        "direct_nameplate_conflicts",
        "reviewed_identity_conflicts_preserved",
        "unresolved",
    ):
        lines.append(f"- {key}：{attachment_summary.get(key, 0)}")
    lines.extend(["", "## 模板验证覆盖"])
    for name, values in calibration["by_identity"].items():
        lines.append(
            f"- {name}：验证帧 {values['validation_frames']}，接受 {values['accepted']}，错误接受 {values['false_accepts']}，"
            f"{'可自动实名' if values['eligible_for_auto_assignment'] else '仅审计'}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
