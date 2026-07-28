from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .diarization import (
    SPEECHBRAIN_ECAPA_MODEL,
    _encode_windows_with_classifier,
    _load_speechbrain_classifier,
    _normalize_rows,
    _read_wav,
)


class VisualVoiceIdentityError(ValueError):
    pass


_DEFAULTS: dict[str, Any] = {
    "window_seconds": 1.6,
    "frame_min_separation_seconds": 20.0,
    "minimum_enrollment_samples": 4,
    "minimum_time_span_seconds": 120.0,
    "minimum_score": 0.50,
    "minimum_margin": 0.12,
    "calibration_score_buffer": 0.02,
    "minimum_held_out_accepts": 3,
    "minimum_impostor_trials": 20,
    "minimum_segment_vote_share": 0.80,
    "short_segment_seconds": 1.8,
}

_DYNAMIC_NAMEPLATE_SOURCE = "dynamic_visual_in_tile_nameplate_ocr"
_REVIEWED_SLOT_SOURCE = "visual_profile_reviewed_slot"
_DIRECT_VISUAL_FRAME_SOURCES = frozenset({_DYNAMIC_NAMEPLATE_SOURCE, _REVIEWED_SLOT_SOURCE})
_DIRECT_VISUAL_SEGMENT_SOURCES = frozenset({_DYNAMIC_NAMEPLATE_SOURCE, "visual_active_speaker_highlight"})
_PROPAGATED_CLUSTER_SOURCE = "direct_visual_voice_cluster_consensus"
_VISUAL_VOICE_SOURCE = "same_session_visual_voiceprint"
_PRESERVED_SOURCES = {
    "voice_enrollment",
    "voice_registry",
    "participant_map",
    "user_confirmed_speaker_volume_mapping",
    *_DIRECT_VISUAL_SEGMENT_SOURCES,
}


def load_visual_voice_config(path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise VisualVoiceIdentityError("visual voice config must be a JSON object")
        payload = raw.get("settings", raw)
        if not isinstance(payload, dict):
            raise VisualVoiceIdentityError("visual voice config settings must be a JSON object")
    settings = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if key in payload:
            settings[key] = payload[key]
    for key in ("minimum_enrollment_samples", "minimum_held_out_accepts", "minimum_impostor_trials"):
        settings[key] = int(settings[key])
    for key in _DEFAULTS:
        if key not in {"minimum_enrollment_samples", "minimum_held_out_accepts", "minimum_impostor_trials"}:
            settings[key] = float(settings[key])
    if settings["window_seconds"] < 0.8:
        raise VisualVoiceIdentityError("settings.window_seconds must be at least 0.8")
    if settings["frame_min_separation_seconds"] <= 0:
        raise VisualVoiceIdentityError("settings.frame_min_separation_seconds must be positive")
    if settings["minimum_enrollment_samples"] < 3:
        raise VisualVoiceIdentityError("settings.minimum_enrollment_samples must be at least 3")
    if settings["minimum_held_out_accepts"] < 1:
        raise VisualVoiceIdentityError("settings.minimum_held_out_accepts must be at least 1")
    if settings["minimum_impostor_trials"] < 1:
        raise VisualVoiceIdentityError("settings.minimum_impostor_trials must be at least 1")
    if settings["minimum_time_span_seconds"] < 0:
        raise VisualVoiceIdentityError("settings.minimum_time_span_seconds must be non-negative")
    for key in ("minimum_score", "minimum_margin", "minimum_segment_vote_share", "calibration_score_buffer"):
        if not 0 < settings[key] <= 1:
            raise VisualVoiceIdentityError(f"settings.{key} must be in (0, 1]")
    if settings["short_segment_seconds"] <= 0:
        raise VisualVoiceIdentityError("settings.short_segment_seconds must be positive")
    return settings


def select_visual_voice_enrollment(
    visual_payload: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in visual_payload.get("frames", []):
        if (
            frame.get("name_source") not in _DIRECT_VISUAL_FRAME_SOURCES
            or not frame.get("name")
            or not frame.get("active", True)
        ):
            continue
        try:
            time = float(frame.get("actualTime", frame.get("time", 0.0)))
            score = float(frame.get("score", frame.get("active_tiles", [{}])[0].get("score", 0.0)))
        except (IndexError, TypeError, ValueError):
            continue
        active_tiles = frame.get("active_tiles", [{}])
        tile = active_tiles[0].get("tile") if active_tiles else frame.get("slot")
        grouped[str(frame["name"])].append(
            {
                "name": str(frame["name"]),
                "time": time,
                "frame": frame.get("path"),
                "active_score": score,
                "tile": tile,
                "visual_source": frame.get("name_source"),
                "nameplate_candidates": frame.get("nameplate_candidates", []),
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, dict[str, Any]] = {}
    minimum_samples = int(settings["minimum_enrollment_samples"])
    minimum_span = float(settings["minimum_time_span_seconds"])
    minimum_separation = float(settings["frame_min_separation_seconds"])
    for name, frames in sorted(grouped.items()):
        deduplicated: list[dict[str, Any]] = []
        for frame in sorted(frames, key=lambda item: float(item["time"])):
            if not deduplicated or float(frame["time"]) - float(deduplicated[-1]["time"]) >= minimum_separation:
                deduplicated.append(frame)
        span = float(deduplicated[-1]["time"]) - float(deduplicated[0]["time"]) if len(deduplicated) > 1 else 0.0
        reasons: list[str] = []
        if len(deduplicated) < minimum_samples:
            reasons.append("insufficient_time_separated_visual_samples")
        if span < minimum_span:
            reasons.append("insufficient_visual_sample_time_span")
        if reasons:
            rejected[name] = {
                "reason": reasons,
                "direct_visual_frames": len(frames),
                "time_separated_frames": len(deduplicated),
                "time_span_seconds": round(span, 3),
            }
            continue
        selected[name] = deduplicated
    return selected, rejected


def direct_visual_enrollment_frame_count(visual_payload: dict[str, Any]) -> int:
    """Count usable direct visual enrollment frames for source selection."""

    return sum(
        1
        for frame in visual_payload.get("frames", [])
        if frame.get("name_source") in _DIRECT_VISUAL_FRAME_SOURCES
        and frame.get("name")
        and frame.get("active", True)
    )


def clear_visual_voice_identity(segments: list[dict[str, Any]]) -> int:
    """Retract only labels produced by this stage before a rerun.

    A recalibrated voiceprint that no longer passes its held-out gate must not
    leave the prior voice-derived name in the transcript.  In particular, do
    not revive a previous cluster label here: the voice stage may have been
    correcting that cluster label, and the cluster stage must be rerun from
    its current visual evidence if it is still valid.
    """

    cleared = 0
    for segment in segments:
        if str(segment.get("name_source") or "") != _VISUAL_VOICE_SOURCE:
            continue
        for key in (
            "name",
            "name_source",
            "name_confidence",
            "visual_voice_identity_evidence",
            "direct_visual_cluster_identity_evidence",
        ):
            segment.pop(key, None)
        cleared += 1
    return cleared


def _enrollment_windows(
    samples: dict[str, list[dict[str, Any]]],
    *,
    sample_rate: int,
    audio_size: int,
    window_seconds: float,
) -> list[dict[str, Any]]:
    half_window = window_seconds / 2
    windows: list[dict[str, Any]] = []
    for name, items in sorted(samples.items()):
        for ordinal, item in enumerate(items):
            center = float(item["time"])
            start = max(0, int(round((center - half_window) * sample_rate)))
            end = min(audio_size, int(round((center + half_window) * sample_rate)))
            windows.append(
                {
                    "name": name,
                    "ordinal": ordinal,
                    "time": center,
                    "frame": item.get("frame"),
                    "active_score": item.get("active_score"),
                    "tile": item.get("tile"),
                    "sample_start": start,
                    "sample_end": end,
                    "start": start / sample_rate,
                    "end": end / sample_rate,
                }
            )
    return windows


def _three_way_split(indexes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Split time-ordered visual anchors into profile, calibration, and held-out sets."""

    held_out = [index for position, index in enumerate(indexes) if position % 5 == 0]
    calibration = [index for position, index in enumerate(indexes) if position % 5 == 1]
    training = [
        index
        for position, index in enumerate(indexes)
        if position % 5 not in {0, 1}
    ]
    return training, calibration, held_out


def _score_trials(
    embeddings: np.ndarray,
    windows: list[dict[str, Any]],
    matrix: np.ndarray,
    names: list[str],
    indexes_by_name: dict[str, list[int]],
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for name in names:
        actual_index = names.index(name)
        for index in indexes_by_name[name]:
            scores = embeddings[index] @ matrix.T
            order = np.argsort(scores)[::-1]
            best_index = int(order[0])
            runner_index = int(order[1])
            actual_score = float(scores[actual_index])
            best_score = float(scores[best_index])
            runner_score = float(scores[runner_index])
            trials.append(
                {
                    "name": name,
                    "time": round(float(windows[index]["time"]), 3),
                    "actual_score": round(actual_score, 4),
                    "best_name": names[best_index],
                    "best_score": round(best_score, 4),
                    "margin": round(actual_score - max(float(value) for item_index, value in enumerate(scores) if item_index != actual_index), 4),
                    "best_margin": round(best_score - runner_score, 4),
                    "correct_best": best_index == actual_index,
                    "frame": windows[index].get("frame"),
                }
            )
    return trials


def _calibrate_precision_gate(
    embeddings: np.ndarray,
    windows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    names = sorted({str(window["name"]) for window in windows})
    by_name = {name: [index for index, window in enumerate(windows) if window["name"] == name] for name in names}
    training: dict[str, list[int]] = {}
    calibration: dict[str, list[int]] = {}
    held_out: dict[str, list[int]] = {}
    for name, indexes in by_name.items():
        training[name], calibration[name], held_out[name] = _three_way_split(indexes)
    if len(names) < 2 or any(
        len(training[name]) < 3 or not calibration[name] or not held_out[name]
        for name in names
    ):
        return {
            "status": "insufficient_for_precision_calibration",
            "profiles": names,
            "training_samples": {name: len(training[name]) for name in names},
            "calibration_samples": {name: len(calibration[name]) for name in names},
            "held_out_samples": {name: len(held_out[name]) for name in names},
            "split": "time_ordered_modulo_5",
            "training_indexes": training,
        }

    centroids = {
        name: _normalize_rows(embeddings[training[name]].mean(axis=0, keepdims=True))[0] for name in names
    }
    matrix = np.stack([centroids[name] for name in names])
    calibration_trials = _score_trials(embeddings, windows, matrix, names, calibration)
    calibration_negatives = [
        float(trial["best_score"])
        for trial in calibration_trials
        if not trial["correct_best"]
    ]
    threshold = max(
        float(settings["minimum_score"]),
        max(calibration_negatives, default=-1.0) + float(settings["calibration_score_buffer"]),
    )
    margin = float(settings["minimum_margin"])
    trials = _score_trials(embeddings, windows, matrix, names, held_out)
    accepted = [
        trial
        for trial in trials
        if trial["correct_best"]
        and float(trial["actual_score"]) >= threshold
        and float(trial["margin"]) >= margin
    ]
    false_accepts = [
        trial
        for trial in trials
        if not trial["correct_best"]
        and float(trial["best_score"]) >= threshold
        and float(trial["best_margin"]) >= margin
    ]
    accepted_by_name = Counter(str(trial["name"]) for trial in accepted)
    false_accepts_by_name = Counter(str(trial["best_name"]) for trial in false_accepts)
    impostor_trials_by_name = {
        name: sum(1 for trial in trials if trial["name"] != name)
        for name in names
    }
    eligible_profiles = [
        name
        for name in names
        if accepted_by_name[name] >= int(settings["minimum_held_out_accepts"])
        and false_accepts_by_name[name] == 0
        and impostor_trials_by_name[name] >= int(settings["minimum_impostor_trials"])
    ]
    ineligible_profiles: dict[str, list[str]] = {}
    for name in names:
        reasons: list[str] = []
        if accepted_by_name[name] < int(settings["minimum_held_out_accepts"]):
            reasons.append("insufficient_held_out_accepts")
        if false_accepts_by_name[name]:
            reasons.append("held_out_false_accept")
        if impostor_trials_by_name[name] < int(settings["minimum_impostor_trials"]):
            reasons.append("insufficient_impostor_trials")
        if reasons:
            ineligible_profiles[name] = reasons
    status = (
        "precision_calibrated"
        if len(eligible_profiles) == len(names)
        else "partial_precision_calibrated"
        if eligible_profiles
        else "insufficient_for_precision_calibration"
    )
    return {
        "status": status,
        "profiles": names,
        "training_samples": {name: len(training[name]) for name in names},
        "calibration_samples": {name: len(calibration[name]) for name in names},
        "held_out_samples": {name: len(held_out[name]) for name in names},
        "split": "time_ordered_modulo_5",
        "training_indexes": training,
        "threshold_source": "calibration_impostor_max_plus_buffer",
        "threshold": round(threshold, 4),
        "margin": round(margin, 4),
        "calibration_trials": calibration_trials,
        "trials": trials,
        "accepted_trials": len(accepted),
        "false_accepts": len(false_accepts),
        "accepted_by_name": dict(accepted_by_name),
        "false_accepts_by_name": dict(false_accepts_by_name),
        "impostor_trials_by_name": impostor_trials_by_name,
        "eligible_profiles": eligible_profiles,
        "ineligible_profiles": ineligible_profiles,
    }


def build_visual_voice_registry(
    audio_path: Path,
    visual_payload: dict[str, Any],
    output_path: Path,
    *,
    settings: dict[str, Any],
    speechbrain_cache: Path,
    model: str = SPEECHBRAIN_ECAPA_MODEL,
) -> dict[str, Any]:
    samples, rejected = select_visual_voice_enrollment(visual_payload, settings)
    if len(samples) < 2:
        raise VisualVoiceIdentityError("at least two participants need time-separated direct visual evidence for same-session voice verification")
    sample_rate, data = _read_wav(audio_path)
    windows = _enrollment_windows(
        samples,
        sample_rate=sample_rate,
        audio_size=len(data),
        window_seconds=float(settings["window_seconds"]),
    )
    classifier = _load_speechbrain_classifier(model, speechbrain_cache)
    embeddings, valid_windows = _encode_windows_with_classifier(classifier, data, sample_rate, windows)
    if len(valid_windows) != len(windows):
        raise VisualVoiceIdentityError("one or more direct visual enrollment windows were too short for voice embedding")
    calibration = _calibrate_precision_gate(embeddings, valid_windows, settings)
    profiles: dict[str, dict[str, Any]] = {}
    for name in sorted(samples):
        indexes = list(calibration.get("training_indexes", {}).get(name, []))
        if not indexes:
            indexes = [index for index, window in enumerate(valid_windows) if window["name"] == name]
        centroid = _normalize_rows(embeddings[indexes].mean(axis=0, keepdims=True))[0]
        profiles[name] = {
            "centroid": [round(float(value), 8) for value in centroid],
            "samples": len(indexes),
            "time_span_seconds": round(float(samples[name][-1]["time"]) - float(samples[name][0]["time"]), 3),
        }
    payload = {
        "format": "same-session-visual-voice-registry/v2",
        "audio": str(audio_path),
        "model": model,
        "settings": settings,
        "profiles": profiles,
        "enrollment": valid_windows,
        "rejected_visual_names": rejected,
        "calibration": calibration,
        "note": "Profiles are built only from direct same-frame active-tile evidence with either in-tile nameplate OCR or a reviewed visual profile slot. The registry is valid only for this recording.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _segment_windows(
    segments: list[dict[str, Any]],
    *,
    sample_rate: int,
    audio_size: int,
    window_seconds: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        if duration < 0.45:
            continue
        usable_start = float(segment["start"]) + min(0.05, duration / 8)
        usable_end = float(segment["end"]) - min(0.05, duration / 8)
        usable_duration = max(0.0, usable_end - usable_start)
        if usable_duration < 0.45:
            continue
        centers = [0.5] if usable_duration <= window_seconds * 1.45 else [0.30, 0.70]
        for fraction in centers:
            width = min(window_seconds, usable_duration)
            center = usable_start + usable_duration * fraction
            start = max(usable_start, center - width / 2)
            end = min(usable_end, start + width)
            start = max(usable_start, end - width)
            sample_start = max(0, int(round(start * sample_rate)))
            sample_end = min(audio_size, int(round(end * sample_rate)))
            result.append(
                {
                    "segment_index": segment_index,
                    "start": sample_start / sample_rate,
                    "end": sample_end / sample_rate,
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                }
            )
    return result


def attach_visual_voice_scores(
    segments: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    scores: np.ndarray,
    names: list[str],
    registry: dict[str, Any],
) -> dict[str, Any]:
    if len(windows) != len(scores):
        raise VisualVoiceIdentityError("voice windows and scores must have the same length")
    calibration = registry.get("calibration", {})
    if calibration.get("status") not in {"precision_calibrated", "partial_precision_calibrated"}:
        return {
            "status": "audit_only_insufficient_calibration",
            "assigned_segments": 0,
            "accepted_windows": 0,
            "reason": calibration.get("status"),
        }
    threshold = float(calibration["threshold"])
    margin_threshold = float(calibration["margin"])
    eligible_profiles = {str(name) for name in calibration.get("eligible_profiles", names)}
    if not eligible_profiles:
        return {
            "status": "audit_only_insufficient_calibration",
            "assigned_segments": 0,
            "accepted_windows": 0,
            "reason": "no_eligible_profiles",
        }
    evidence_by_segment: list[dict[str, list[dict[str, Any]]]] = [{} for _ in segments]
    accepted_windows = 0
    abstained_ineligible_top = 0
    for window, row in zip(windows, scores):
        order = np.argsort(row)[::-1]
        best_index = int(order[0])
        runner_index = int(order[1])
        best_score = float(row[best_index])
        margin = best_score - float(row[runner_index])
        # All profiles remain competitors. An ineligible top result must abstain,
        # never fall through to a lower-scoring eligible profile.
        if names[best_index] not in eligible_profiles:
            abstained_ineligible_top += 1
            continue
        if best_score < threshold or margin < margin_threshold:
            continue
        accepted_windows += 1
        evidence_by_segment[int(window["segment_index"])].setdefault(names[best_index], []).append(
            {
                "start": round(float(window["start"]), 3),
                "end": round(float(window["end"]), 3),
                "score": round(best_score, 4),
                "margin": round(margin, 4),
            }
        )

    assigned = 0
    preserved = 0
    confirmed_cluster_assignments = 0
    corrected_cluster_assignments = 0
    vote_share_required = float(registry["settings"]["minimum_segment_vote_share"])
    short_seconds = float(registry["settings"]["short_segment_seconds"])
    for segment, candidates in zip(segments, evidence_by_segment):
        if str(segment.get("name_source") or "") in _PRESERVED_SOURCES and segment.get("name"):
            preserved += 1
            continue
        if not candidates:
            continue
        ranked = sorted(candidates.items(), key=lambda item: len(item[1]), reverse=True)
        name, records = ranked[0]
        total = sum(len(value) for value in candidates.values())
        duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        required = 1 if duration <= short_seconds else 2
        vote_share = len(records) / total
        if len(records) < required or vote_share < vote_share_required:
            continue
        mean_score = sum(float(record["score"]) for record in records) / len(records)
        mean_margin = sum(float(record["margin"]) for record in records) / len(records)
        confidence = min(0.90, 0.40 + mean_score * 0.42 + mean_margin * 0.50)
        if segment.get("speaker_confidence") is not None:
            confidence = min(confidence, float(segment["speaker_confidence"]))
        prior_cluster_name = None
        if str(segment.get("name_source") or "") == _PROPAGATED_CLUSTER_SOURCE:
            prior_cluster_name = str(segment.get("name") or "") or None
            if prior_cluster_name == name:
                confirmed_cluster_assignments += 1
            else:
                corrected_cluster_assignments += 1
        segment["name"] = name
        segment["name_source"] = _VISUAL_VOICE_SOURCE
        segment["name_confidence"] = round(confidence, 3)
        segment["visual_voice_identity_evidence"] = {
            "registry_format": registry["format"],
            "profile": name,
            "accepted_windows": len(records),
            "vote_share": round(vote_share, 3),
            "mean_score": round(mean_score, 4),
            "mean_margin": round(mean_margin, 4),
            "threshold": round(threshold, 4),
            "margin_threshold": round(margin_threshold, 4),
            "windows": records,
        }
        if prior_cluster_name:
            segment["visual_voice_identity_evidence"]["prior_cluster_name"] = prior_cluster_name
        assigned += 1
    return {
        "status": "ok",
        "threshold": threshold,
        "margin": margin_threshold,
        "accepted_windows": accepted_windows,
        "assigned_segments": assigned,
        "preserved_direct_visual_segments": preserved,
        "confirmed_cluster_assignments": confirmed_cluster_assignments,
        "corrected_cluster_assignments": corrected_cluster_assignments,
        "eligible_profiles": sorted(eligible_profiles),
        "abstained_ineligible_top_windows": abstained_ineligible_top,
    }


def apply_visual_voice_registry(
    audio_path: Path,
    segments: list[dict[str, Any]],
    registry_path: Path,
    *,
    speechbrain_cache: Path,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("format") not in {
        "same-session-visual-voice-registry/v1",
        "same-session-visual-voice-registry/v2",
    }:
        raise VisualVoiceIdentityError("unsupported same-session visual voice registry format")
    names = sorted(registry.get("profiles", {}))
    if len(names) < 2:
        raise VisualVoiceIdentityError("same-session visual voice registry needs at least two profiles")
    sample_rate, data = _read_wav(audio_path)
    windows = _segment_windows(
        segments,
        sample_rate=sample_rate,
        audio_size=len(data),
        window_seconds=float(registry["settings"]["window_seconds"]),
    )
    classifier = _load_speechbrain_classifier(str(registry["model"]), speechbrain_cache)
    embeddings, valid_windows = _encode_windows_with_classifier(classifier, data, sample_rate, windows)
    if not valid_windows:
        return {"status": "empty_audio_windows", "assigned_segments": 0}
    profile_matrix = _normalize_rows(
        np.asarray([registry["profiles"][name]["centroid"] for name in names], dtype="float32")
    )
    if embeddings.shape[1] != profile_matrix.shape[1]:
        raise VisualVoiceIdentityError("same-session voice embedding dimensions do not match the registry")
    return attach_visual_voice_scores(segments, valid_windows, embeddings @ profile_matrix.T, names, registry)


def write_visual_voice_report(path: Path, *, registry: dict[str, Any], status: dict[str, Any]) -> None:
    calibration = registry.get("calibration", {})
    lines = [
        "# Same-Session Visual Voiceprint Report",
        "",
        "## Method",
        "- Enrollment comes only from direct same-frame active-speaker evidence with in-tile nameplate OCR or a reviewed visual-profile slot.",
        "- A participant needs multiple time-separated direct visual samples before a voice profile is eligible.",
        "- Profile centroids, threshold calibration, and final validation use separate time-ordered direct-visual samples.",
        "- A transcript segment is named only when its own audio windows pass the independent held-out precision gate and agree with each other.",
        "- Ineligible profiles remain scoring competitors; an ineligible top score always abstains instead of falling through to another name.",
        "- Participants without enough cross-time visual enrollment remain anonymous; this voiceprint stage does not perform cluster-wide propagation.",
        "",
        "## Enrollment",
    ]
    for name, profile in sorted(registry.get("profiles", {}).items()):
        lines.append(
            f"- `{name}`: samples={profile.get('samples')} time_span_seconds={float(profile.get('time_span_seconds', 0.0)):.1f}"
        )
    rejected = registry.get("rejected_visual_names", {})
    if rejected:
        lines.append("")
        lines.append("## Not Enrolled")
        for name, reason in sorted(rejected.items()):
            lines.append(
                f"- `{name}`: reasons={','.join(reason.get('reason', []))} direct_frames={reason.get('direct_visual_frames', 0)} "
                f"time_separated_frames={reason.get('time_separated_frames', 0)} time_span_seconds={float(reason.get('time_span_seconds', 0.0)):.1f}"
            )
    lines += [
        "",
        "## Calibration",
        f"- Status: {calibration.get('status', 'unknown')}",
        f"- Threshold source: {calibration.get('threshold_source', 'not available')}",
        f"- Held-out precision threshold: {calibration.get('threshold', 'not available')}",
        f"- Held-out margin threshold: {calibration.get('margin', 'not available')}",
        f"- Accepted held-out trials: {calibration.get('accepted_trials', 0)}",
        f"- Held-out false accepts: {calibration.get('false_accepts', 0)}",
        f"- Eligible profiles: {', '.join(calibration.get('eligible_profiles', [])) or 'none'}",
        "",
        "## Results",
        f"- Status: {status.get('status', 'unknown')}",
        f"- Accepted audio windows: {status.get('accepted_windows', 0)}",
        f"- Additional named transcript segments: {status.get('assigned_segments', 0)}",
        f"- Preserved direct visual segments: {status.get('preserved_direct_visual_segments', 0)}",
        f"- Cluster-propagated assignments independently confirmed by voiceprint: {status.get('confirmed_cluster_assignments', 0)}",
        f"- Cluster-propagated assignments corrected by voiceprint: {status.get('corrected_cluster_assignments', 0)}",
        f"- Abstained ineligible-top windows: {status.get('abstained_ineligible_top_windows', 0)}",
        f"- Cleared prior same-session voiceprint assignments before rerun: {status.get('cleared_prior_visual_voice_assignments', 0)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
