from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .visual_highlight import Box, score_green_highlight_border, score_green_speaker_cue, score_highlight_border


class VisualProfileError(ValueError):
    pass


_DEFAULTS: dict[str, Any] = {
    "samples_per_segment": 3,
    "nameplate_samples_per_layout": 6,
    "min_active_score": 0.20,
    "min_active_margin": 0.06,
    "minimum_nameplate_observations": 2,
    "minimum_nameplate_share": 0.67,
    "minimum_segment_vote_share": 0.66,
    "short_segment_seconds": 1.5,
    "time_offset_seconds": 0.0,
    "allow_direct_assignment": False,
}

_NON_NAME_WORDS = {
    "active",
    "audio",
    "camera",
    "chat",
    "guest",
    "meeting",
    "microphone",
    "mute",
    "muted",
    "presenting",
    "screen",
    "share",
    "speaking",
    "unmute",
}


def _box(raw: Any, label: str) -> Box:
    if not isinstance(raw, list) or len(raw) != 4:
        raise VisualProfileError(f"{label} must be a four-number normalized box")
    box = tuple(float(value) for value in raw)
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise VisualProfileError(f"{label} must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return box  # type: ignore[return-value]


def load_visual_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VisualProfileError("visual profile must be a JSON object")
    raw_settings = payload.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise VisualProfileError("settings must be a JSON object")
    settings = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if key in raw_settings:
            settings[key] = raw_settings[key]
    if not isinstance(settings["allow_direct_assignment"], bool):
        raise VisualProfileError("settings.allow_direct_assignment must be a boolean")
    settings["samples_per_segment"] = int(settings["samples_per_segment"])
    settings["nameplate_samples_per_layout"] = int(settings["nameplate_samples_per_layout"])
    settings["minimum_nameplate_observations"] = int(settings["minimum_nameplate_observations"])
    for key in (
        "min_active_score",
        "min_active_margin",
        "minimum_nameplate_share",
        "minimum_segment_vote_share",
        "short_segment_seconds",
        "time_offset_seconds",
    ):
        settings[key] = float(settings[key])
    if not 1 <= settings["samples_per_segment"] <= 3:
        raise VisualProfileError("settings.samples_per_segment must be in 1..3")
    if not 1 <= settings["nameplate_samples_per_layout"] <= 12:
        raise VisualProfileError("settings.nameplate_samples_per_layout must be in 1..12")
    if settings["minimum_nameplate_observations"] < 1:
        raise VisualProfileError("settings.minimum_nameplate_observations must be at least 1")
    for key in ("min_active_score", "min_active_margin", "minimum_nameplate_share", "minimum_segment_vote_share"):
        if not 0 <= settings[key] <= 1:
            raise VisualProfileError(f"settings.{key} must be in 0..1")
    if settings["short_segment_seconds"] <= 0:
        raise VisualProfileError("settings.short_segment_seconds must be positive")

    participants = payload.get("participants", [])
    if not isinstance(participants, list) or not all(isinstance(item, str) and item.strip() for item in participants):
        raise VisualProfileError("participants must be an optional list of non-empty names")
    normalized_participants = [str(item).strip() for item in participants]
    if len({item.casefold() for item in normalized_participants}) != len(normalized_participants):
        raise VisualProfileError("participants must not contain duplicate names")

    raw_layouts = payload.get("layouts")
    if not isinstance(raw_layouts, list) or not raw_layouts:
        raise VisualProfileError("layouts must be a non-empty list")
    layouts: list[dict[str, Any]] = []
    for index, raw_layout in enumerate(raw_layouts):
        if not isinstance(raw_layout, dict):
            raise VisualProfileError(f"layouts[{index}] must be an object")
        name = str(raw_layout.get("name", "")).strip()
        if not name:
            raise VisualProfileError(f"layouts[{index}].name is required")
        start = float(raw_layout.get("start", 0.0))
        end_raw = raw_layout.get("end")
        end = float(end_raw) if end_raw is not None else math.inf
        if start < 0 or end <= start:
            raise VisualProfileError(f"layouts[{index}] has an invalid time window")
        raw_slots = raw_layout.get("slots")
        if not isinstance(raw_slots, dict) or not raw_slots:
            raise VisualProfileError(f"layouts[{index}].slots must be a non-empty object")
        slots: dict[str, dict[str, Any]] = {}
        for slot_name, raw_slot in raw_slots.items():
            if not isinstance(raw_slot, dict):
                raise VisualProfileError(f"layouts[{index}].slots.{slot_name} must be an object")
            label = str(slot_name).strip()
            if not label:
                raise VisualProfileError(f"layouts[{index}] has an empty slot name")
            person = raw_slot.get("person")
            if person is not None and (not isinstance(person, str) or not person.strip()):
                raise VisualProfileError(f"layouts[{index}].slots.{label}.person must be a non-empty string")
            if person is not None and normalized_participants and str(person).strip().casefold() not in {
                participant.casefold() for participant in normalized_participants
            }:
                raise VisualProfileError(f"layouts[{index}].slots.{label}.person must be listed in participants")
            slots[label] = {
                "tile": _box(raw_slot.get("tile"), f"layouts[{index}].slots.{label}.tile"),
                "speaker_cue": _box(raw_slot["speaker_cue"], f"layouts[{index}].slots.{label}.speaker_cue")
                if "speaker_cue" in raw_slot
                else None,
                "nameplate": _box(raw_slot["nameplate"], f"layouts[{index}].slots.{label}.nameplate")
                if "nameplate" in raw_slot
                else None,
                "active_signal": str(raw_slot["active_signal"]).strip() if "active_signal" in raw_slot else None,
                "person": str(person).strip() if person is not None else None,
            }
            signal = slots[label]["active_signal"]
            if signal not in {None, "green_speaker_cue", "green_highlight_border", "highlight_border"}:
                raise VisualProfileError(
                    f"layouts[{index}].slots.{label}.active_signal must be green_speaker_cue, "
                    "green_highlight_border, or highlight_border"
                )
            if settings["allow_direct_assignment"] and signal is None:
                raise VisualProfileError(
                    f"layouts[{index}].slots.{label}.active_signal is required when direct assignment is enabled"
                )
            if signal == "green_speaker_cue" and slots[label]["speaker_cue"] is None:
                raise VisualProfileError(
                    f"layouts[{index}].slots.{label}.speaker_cue is required for active_signal=green_speaker_cue"
                )
        layouts.append({"name": name, "start": start, "end": end, "slots": slots})

    layouts.sort(key=lambda item: float(item["start"]))
    if len({layout["name"] for layout in layouts}) != len(layouts):
        raise VisualProfileError("layout names must be unique")
    for previous, current in zip(layouts, layouts[1:]):
        if float(current["start"]) < float(previous["end"]):
            raise VisualProfileError("layout time windows must not overlap")
    return {"settings": settings, "participants": normalized_participants, "layouts": layouts}


def layout_at(profile: dict[str, Any], time_seconds: float) -> dict[str, Any] | None:
    for layout in profile["layouts"]:
        if float(layout["start"]) <= time_seconds < float(layout["end"]):
            return layout
    return None


def build_segment_sample_requests(
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


def unique_video_times(requests: list[dict[str, Any]]) -> list[float]:
    return sorted({float(item["video_time"]) for item in requests})


def _evenly_spaced(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) <= count:
        return items
    indexes = {round(index * (len(items) - 1) / (count - 1)) for index in range(count)} if count > 1 else {len(items) // 2}
    return [item for index, item in enumerate(items) if index in indexes]


def build_nameplate_manifest(frames: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    count = int(profile["settings"]["nameplate_samples_per_layout"])
    result: list[dict[str, Any]] = []
    for layout in profile["layouts"]:
        candidates = [
            frame
            for frame in frames
            if frame.get("path") and layout_at(profile, float(frame.get("actualTime", frame.get("time", 0.0)))) == layout
        ]
        for frame in _evenly_spaced(candidates, count):
            regions = [
                {"label": f"{layout['name']}::{slot_name}", "box": list(slot["nameplate"])}
                for slot_name, slot in layout["slots"].items()
                if slot.get("nameplate") is not None and not slot.get("person")
            ]
            if regions:
                result.append(
                    {
                        "time": float(frame["time"]),
                        "actualTime": float(frame.get("actualTime", frame["time"])),
                        "path": str(frame["path"]),
                        "regions": regions,
                    }
                )
    return result


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip(" .,:;|[](){}©"))
    return re.sub(r"^[oO0]\s+(?=[A-Za-z])", "", normalized)


def _looks_like_name(value: str) -> bool:
    if not value or len(value) > 32 or value.casefold() in _NON_NAME_WORDS:
        return False
    if re.fullmatch(r"[\u4e00-\u9fff]{2,8}", value):
        return True
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*(?: [A-Za-z][A-Za-z.'-]*){0,2}", value))


def _resolve_candidate(value: str, participants: list[str]) -> str | None:
    candidate = _normalize_name(value)
    if not candidate or len(candidate) > 32 or candidate.casefold() in _NON_NAME_WORDS:
        return None
    if participants:
        exact = next((name for name in participants if name.casefold() == candidate.casefold()), None)
        if exact:
            return exact
        corrected = candidate.translate(str.maketrans({"0": "o", "1": "l", "5": "s"}))
        scored = sorted(
            ((SequenceMatcher(None, corrected.casefold(), name.casefold()).ratio(), name) for name in participants), reverse=True
        )
        if scored and scored[0][0] >= 0.82:
            return scored[0][1]
        return None
    return candidate if _looks_like_name(candidate) else None


def resolve_slot_names(profile: dict[str, Any], ocr_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ocr_records:
        for region in record.get("regions", []):
            label = str(region.get("label", ""))
            for text in region.get("texts", []):
                candidate = _resolve_candidate(str(text.get("text", "")), profile["participants"])
                if candidate:
                    observations[label].append(
                        {
                            "name": candidate,
                            "confidence": float(text.get("confidence", 0.0)),
                            "time": float(record.get("actualTime", record.get("time", 0.0))),
                            "path": record.get("path"),
                        }
                    )

    resolved: dict[str, dict[str, Any]] = {}
    minimum = int(profile["settings"]["minimum_nameplate_observations"])
    minimum_share = float(profile["settings"]["minimum_nameplate_share"])
    for layout in profile["layouts"]:
        for slot_name, slot in layout["slots"].items():
            label = f"{layout['name']}::{slot_name}"
            if slot.get("person"):
                resolved[label] = {
                    "name": slot["person"],
                    "source": "visual_profile_reviewed_slot",
                    "observations": 0,
                    "share": 1.0,
                    "frame_refs": [],
                    "candidates": [{"name": slot["person"], "count": 0}],
                }
                continue
            slot_observations = observations.get(label, [])
            counter = Counter(item["name"] for item in slot_observations)
            candidates = [{"name": name, "count": count} for name, count in counter.most_common(5)]
            if not counter:
                resolved[label] = {
                    "name": None,
                    "source": "nameplate_ocr_unresolved",
                    "observations": 0,
                    "share": 0.0,
                    "frame_refs": [],
                    "candidates": candidates,
                }
                continue
            name, count = counter.most_common(1)[0]
            share = count / len(slot_observations)
            refs = sorted({str(item["path"]) for item in slot_observations if item.get("path")})[:6]
            is_whitelisted = bool(profile["participants"])
            resolved[label] = {
                "name": name if is_whitelisted and count >= minimum and share >= minimum_share else None,
                "source": "visual_nameplate_ocr_whitelisted"
                if is_whitelisted and count >= minimum and share >= minimum_share
                else "nameplate_ocr_unresolved",
                "observations": len(slot_observations),
                "share": round(share, 3),
                "frame_refs": refs,
                "candidates": candidates,
            }
    return resolved


def score_visual_frames(
    frames: list[dict[str, Any]],
    profile: dict[str, Any],
    slot_names: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = profile["settings"]
    records: list[dict[str, Any]] = []
    for frame in frames:
        frame_time = float(frame.get("actualTime", frame.get("time", 0.0)))
        layout = layout_at(profile, frame_time)
        base = {
            "time": float(frame.get("time", frame_time)),
            "actualTime": frame_time,
            "path": frame.get("path"),
            "layout": layout["name"] if layout else None,
        }
        if not layout or not frame.get("path"):
            records.append({**base, "active": False, "reason": "outside_calibrated_layout", "scores": {}})
            continue
        from PIL import Image

        with Image.open(str(frame["path"])) as image:
            signals = {
                slot_name: str(slot.get("active_signal") or ("green_speaker_cue" if slot.get("speaker_cue") else "highlight_border"))
                for slot_name, slot in layout["slots"].items()
            }
            scores = {
                slot_name: (
                    score_green_speaker_cue(image, slot["speaker_cue"])
                    if signals[slot_name] == "green_speaker_cue"
                    else score_green_highlight_border(image, slot["tile"])
                    if signals[slot_name] == "green_highlight_border"
                    else score_highlight_border(image, slot["tile"])
                )
                for slot_name, slot in layout["slots"].items()
            }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_slot, best_score = ordered[0]
        runner_score = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = best_score - runner_score
        label = f"{layout['name']}::{best_slot}"
        name_data = slot_names.get(label, {})
        active = best_score >= float(settings["min_active_score"]) and margin >= float(settings["min_active_margin"])
        if not active:
            reason = "below_active_score" if best_score < float(settings["min_active_score"]) else "active_score_margin_too_small"
        elif not name_data.get("name"):
            reason = "active_slot_name_unresolved"
        else:
            reason = "active_named_slot"
        records.append(
            {
                **base,
                "active": active,
                "reason": reason,
                "slot": best_slot,
                "score": round(best_score, 4),
                "runner_score": round(runner_score, 4),
                "margin": round(margin, 4),
                "scores": {name: round(score, 4) for name, score in scores.items()},
                "active_signal": signals[best_slot],
                "name": name_data.get("name"),
                "name_source": name_data.get("source"),
            }
        )
    return records


_TRUSTED_SOURCES = {
    "voice_enrollment",
    "voice_registry",
    "participant_map",
    "user_confirmed_speaker_volume_mapping",
    "manual_review",
    "human_review",
}
_REVIEWED_SOURCE_PREFIXES = ("reviewed_", "manual_", "user_confirmed_")


def has_trusted_identity(segment: dict[str, Any]) -> bool:
    """Return whether a reviewed, enrolled, or registry identity must not be replaced by weaker visual evidence."""

    source = str(segment.get("name_source") or "")
    return bool(segment.get("name")) and (
        source in _TRUSTED_SOURCES or source.startswith(_REVIEWED_SOURCE_PREFIXES)
    )


def has_direct_visual_authority_identity(segment: dict[str, Any]) -> bool:
    """Return whether a direct active-speaker cue must defer to a human-reviewed identity.

    A calibrated profile combines a stable reviewed slot with a same-time speaker cue.
    That is stronger than cross-recording voice-registry inference, but not stronger
    than a direct human map or explicitly reviewed enrollment.
    """

    return str(segment.get("name_source") or "") != "voice_registry" and has_trusted_identity(segment)


def attach_visual_identity(
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
        "assigned": 0,
        "unresolved": 0,
        "conflicts": 0,
        "preserved_trusted_identity": 0,
        "overridden_voice_registry": 0,
        "preserved_voice_registry_without_visual": 0,
        "corroborated_voice_registry": 0,
        "unvalidated_candidates": 0,
        "assignment_mode": "direct" if profile["settings"]["allow_direct_assignment"] else "audit_only",
        "by_source": Counter(),
        "by_reason": Counter(),
    }
    vote_share_required = float(profile["settings"]["minimum_segment_vote_share"])
    short_seconds = float(profile["settings"]["short_segment_seconds"])
    for index, segment in enumerate(segments):
        segment.pop("visual_identity_evidence", None)
        segment.pop("visual_identity_conflict", None)
        sample_frames = [frame_by_time.get(round(float(request["video_time"]), 3)) for request in requests_by_segment.get(index, [])]
        sample_frames = [frame for frame in sample_frames if frame]
        named_frames = [frame for frame in sample_frames if frame.get("active") and frame.get("name")]
        votes = Counter(str(frame["name"]) for frame in named_frames)
        conflicting = len(votes) > 1
        evidence = [
            {
                "time": round(float(frame["actualTime"]), 3),
                "frame": frame.get("path"),
                "layout": frame.get("layout"),
                "slot": frame.get("slot"),
                "name": frame.get("name"),
                "score": frame.get("score"),
                "margin": frame.get("margin"),
                "reason": frame.get("reason"),
            }
            for frame in sample_frames
        ]
        if evidence:
            segment["visual_identity_evidence"] = evidence
        existing_source = str(segment.get("name_source") or "")
        existing_name = segment.get("name")
        registry_identity = bool(existing_name) and existing_source == "voice_registry"
        trusted = has_direct_visual_authority_identity(segment)
        if trusted:
            summary["preserved_trusted_identity"] += 1
            if votes and any(name != existing_name for name in votes):
                segment["visual_identity_conflict"] = {
                    "reason": "trusted_identity_disagrees_with_visual_evidence",
                    "visual_names": sorted(votes),
                }
                summary["conflicts"] += 1
                summary["by_reason"]["trusted_identity_disagrees_with_visual_evidence"] += 1
            continue
        if conflicting:
            segment["name"] = None
            segment["name_source"] = "visual_identity_conflict"
            segment["name_confidence"] = 0.0
            segment["visual_identity_conflict"] = {"reason": "multiple_active_names_in_segment", "visual_names": sorted(votes)}
            summary["conflicts"] += 1
            summary["unresolved"] += 1
            summary["by_reason"]["multiple_active_names_in_segment"] += 1
            continue
        if not votes:
            if registry_identity:
                # A registry label remains useful when this UI frame has no active-speaker cue.
                # It is not, however, allowed to veto direct visual evidence in another segment.
                summary["preserved_voice_registry_without_visual"] += 1
                continue
            segment["name"] = None
            segment["name_source"] = "visual_identity_unresolved"
            segment["name_confidence"] = 0.0
            reason = "no_named_active_visual_evidence" if sample_frames else "no_visual_samples"
            summary["unresolved"] += 1
            summary["by_reason"][reason] += 1
            continue
        name, count = votes.most_common(1)[0]
        if not profile["settings"]["allow_direct_assignment"]:
            segment["name"] = None
            segment["name_source"] = "visual_identity_unvalidated_candidate"
            segment["name_confidence"] = 0.0
            segment["visual_identity_candidate_names"] = sorted(votes)
            summary["unresolved"] += 1
            summary["unvalidated_candidates"] += 1
            summary["by_reason"]["direct_visual_assignment_disabled"] += 1
            continue
        # An inactive sample frame is not contradictory identity evidence. In
        # call UIs the highlight can lag a spoken word or briefly disappear,
        # so consensus is measured among frames that actually resolve a named
        # active speaker. The separate required-count gate still demands
        # repeated evidence for longer utterances.
        vote_share = count / len(named_frames) if named_frames else 0.0
        segment_duration = float(segment["end"]) - float(segment["start"])
        required_count = 1 if segment_duration <= short_seconds else min(2, len(sample_frames))
        matched_frames = [frame for frame in named_frames if frame.get("name") == name]
        average_score = sum(float(frame.get("score", 0.0)) for frame in matched_frames) / len(matched_frames)
        strong_single = len(sample_frames) == 1 and average_score >= float(profile["settings"]["min_active_score"]) + float(profile["settings"]["min_active_margin"])
        if count < required_count or vote_share < vote_share_required or (required_count == 1 and not strong_single):
            segment["name"] = None
            segment["name_source"] = "visual_identity_unresolved"
            segment["name_confidence"] = 0.0
            summary["unresolved"] += 1
            summary["by_reason"]["insufficient_visual_consensus"] += 1
            continue
        confidence = min(0.94, 0.62 + 0.16 * vote_share + 0.22 * average_score)
        if registry_identity:
            if existing_name == name:
                summary["corroborated_voice_registry"] += 1
            else:
                segment["visual_identity_override"] = {
                    "reason": "direct_visual_evidence_overrides_voice_registry",
                    "previous_name": existing_name,
                    "previous_source": existing_source,
                    "previous_confidence": segment.get("name_confidence"),
                    "visual_name": name,
                }
                summary["overridden_voice_registry"] += 1
                summary["by_reason"]["direct_visual_evidence_overrides_voice_registry"] += 1
        segment["name"] = name
        segment["name_source"] = "visual_active_speaker_highlight"
        segment["name_confidence"] = round(confidence, 3)
        refs = [str(frame["path"]) for frame in matched_frames if frame.get("path")]
        segment["frame_refs"] = sorted(set(refs))[:4]
        summary["assigned"] += 1
        summary["by_source"]["visual_active_speaker_highlight"] += 1
    summary["by_source"] = dict(summary["by_source"])
    summary["by_reason"] = dict(summary["by_reason"])
    return summary


def write_visual_identity_report(
    path: Path,
    *,
    profile_path: Path,
    profile: dict[str, Any],
    slot_names: dict[str, dict[str, Any]],
    scored_frames: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Visual Identity Report",
        "",
        "## Calibration",
        f"- Profile: `{profile_path}`",
        f"- Layouts: {len(profile['layouts'])}",
        f"- Direct visual assignment: {'enabled' if profile['settings']['allow_direct_assignment'] else 'disabled; audit only'}",
        f"- Visual sample frames: {len(scored_frames)}",
        f"- Active frames passing score and margin gates: {summary['active_frames']}",
        "",
        "## Nameplate Resolution",
    ]
    for label, item in sorted(slot_names.items()):
        candidates = ", ".join(f"{candidate['name']}({candidate['count']})" for candidate in item.get("candidates", [])) or "none"
        lines.append(
            f"- `{label}`: {item.get('name') or 'unresolved'} source={item.get('source')} observations={item.get('observations', 0)} share={float(item.get('share', 0.0)):.2f} candidates={candidates}"
        )
    lines += ["", "## Active Signal Diagnostics"]
    for layout in profile["layouts"]:
        layout_name = str(layout["name"])
        layout_frames = [record for record in scored_frames if record.get("layout") == layout_name]
        for slot_name, slot in sorted(layout["slots"].items()):
            values = [float(record.get("scores", {}).get(slot_name, 0.0)) for record in layout_frames]
            nonzero = sum(value > 0.0 for value in values)
            active = sum(
                1
                for record in layout_frames
                if record.get("active") and record.get("slot") == slot_name
            )
            maximum = max(values, default=0.0)
            average = sum(values) / len(values) if values else 0.0
            signal = slot.get("active_signal") or (
                "green_speaker_cue" if slot.get("speaker_cue") else "highlight_border"
            )
            lines.append(
                f"- `{layout_name}::{slot_name}`: signal={signal} frames={len(values)} nonzero={nonzero} "
                f"max_score={maximum:.4f} mean_score={average:.4f} active_frames={active}"
            )
    lines += [
        "",
        "## Segment Results",
        f"- Direct visual assignments: {summary['assigned']}",
        f"- Unresolved segments: {summary['unresolved']}",
        f"- Visual or trusted-identity conflicts: {summary['conflicts']}",
        f"- Preserved enrollment or reviewed-map identities: {summary['preserved_trusted_identity']}",
        f"- Direct visual corrections of voice-registry identities: {summary['overridden_voice_registry']}",
        f"- Voice-registry identities retained without an active visual cue: {summary['preserved_voice_registry_without_visual']}",
        f"- Voice-registry identities corroborated by direct visual evidence: {summary['corroborated_voice_registry']}",
        f"- Unvalidated visual candidates retained for audit only: {summary['unvalidated_candidates']}",
        "",
        "## Limits",
        "- A layout is used only inside its configured time window.",
        "- Colored borders are audit-only until the profile explicitly enables direct assignment after semantic calibration against reviewed speech.",
        "- A real name requires an active-speaker score, a leading margin over other tiles, a reviewed profile slot or a participant-whitelisted nameplate OCR result, and segment-level agreement.",
        "- Visual conflicts, unresolved nameplates, and low-consensus segments stay anonymous and are placed in the review queue.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
