from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .identity_authority import (
    DIRECT_VISUAL_CLUSTER_SOURCE,
    DYNAMIC_NAMEPLATE_SOURCE,
    REVIEWED_SLOT_SOURCE,
)


class DirectVisualClusterIdentityError(ValueError):
    pass


_DEFAULTS: dict[str, Any] = {
    "boundary_erosion_seconds": 0.8,
    "erosion_stability_seconds": [0.6, 0.8, 1.0],
    "minimum_turn_frame_votes": 2,
    "minimum_turn_vote_share": 0.80,
    "minimum_training_turns": 12,
    "minimum_validation_turns": 8,
    "minimum_training_vote_share": 0.90,
    "minimum_validation_vote_share": 0.85,
    "minimum_wilson_lower_bound": 0.75,
    "minimum_time_span_seconds": 120.0,
    "maximum_support_gap_seconds": 120.0,
    "minimum_segment_speaker_confidence": 0.80,
}

_DIRECT_FRAME_SOURCES = frozenset(
    {
        DYNAMIC_NAMEPLATE_SOURCE,
        REVIEWED_SLOT_SOURCE,
    }
)

_PROPAGATED_SOURCE = DIRECT_VISUAL_CLUSTER_SOURCE


def load_direct_visual_cluster_config(path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path is not None:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise DirectVisualClusterIdentityError("direct visual cluster config must be a JSON object")
        payload = raw.get("settings", raw)
        if not isinstance(payload, dict):
            raise DirectVisualClusterIdentityError("direct visual cluster config settings must be a JSON object")

    settings = dict(_DEFAULTS)
    for key in _DEFAULTS:
        if key in payload:
            settings[key] = payload[key]
    for key in ("minimum_turn_frame_votes", "minimum_training_turns", "minimum_validation_turns"):
        settings[key] = int(settings[key])
    for key in (
        "boundary_erosion_seconds",
        "minimum_turn_vote_share",
        "minimum_training_vote_share",
        "minimum_validation_vote_share",
        "minimum_wilson_lower_bound",
        "minimum_time_span_seconds",
        "maximum_support_gap_seconds",
        "minimum_segment_speaker_confidence",
    ):
        settings[key] = float(settings[key])
    raw_stability = settings["erosion_stability_seconds"]
    if not isinstance(raw_stability, list) or not raw_stability:
        raise DirectVisualClusterIdentityError("settings.erosion_stability_seconds must be a non-empty list")
    settings["erosion_stability_seconds"] = sorted({float(value) for value in raw_stability})

    if settings["boundary_erosion_seconds"] <= 0:
        raise DirectVisualClusterIdentityError("settings.boundary_erosion_seconds must be positive")
    if any(value <= 0 for value in settings["erosion_stability_seconds"]):
        raise DirectVisualClusterIdentityError("settings.erosion_stability_seconds must contain positive values")
    if settings["minimum_turn_frame_votes"] < 1:
        raise DirectVisualClusterIdentityError("settings.minimum_turn_frame_votes must be at least 1")
    if settings["minimum_training_turns"] < 2 or settings["minimum_validation_turns"] < 2:
        raise DirectVisualClusterIdentityError("settings minimum turn counts must be at least 2")
    for key in (
        "minimum_turn_vote_share",
        "minimum_training_vote_share",
        "minimum_validation_vote_share",
        "minimum_wilson_lower_bound",
        "minimum_segment_speaker_confidence",
    ):
        if not 0 < settings[key] <= 1:
            raise DirectVisualClusterIdentityError(f"settings.{key} must be in (0, 1]")
    for key in ("minimum_time_span_seconds", "maximum_support_gap_seconds"):
        if settings[key] < 0:
            raise DirectVisualClusterIdentityError(f"settings.{key} must be non-negative")
    return settings


def _wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    adjustment = z * math.sqrt((proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total)
    return max(0.0, (center - adjustment) / denominator)


def _frame_time(frame: dict[str, Any]) -> float | None:
    try:
        return float(frame.get("actualTime", frame.get("time")))
    except (TypeError, ValueError):
        return None


def direct_visual_active_frame_count(visual_payload: dict[str, Any]) -> int:
    return sum(
        1
        for frame in visual_payload.get("frames", [])
        if frame.get("active")
        and frame.get("name")
        and frame.get("name_source") in _DIRECT_FRAME_SOURCES
    )


def _turn_evidence(
    frames: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    erosion_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    starts = [float(turn["start"]) for turn in turns]
    votes: dict[int, Counter[str]] = defaultdict(Counter)
    status: Counter[str] = Counter()
    for frame in frames:
        if (
            not frame.get("active")
            or not frame.get("name")
            or frame.get("name_source") not in _DIRECT_FRAME_SOURCES
        ):
            continue
        status["direct_active_frames"] += 1
        time = _frame_time(frame)
        if time is None:
            status["invalid_time"] += 1
            continue
        index = bisect_right(starts, time) - 1
        if index < 0 or index >= len(turns):
            status["outside_turns"] += 1
            continue
        turn = turns[index]
        start = float(turn["start"])
        end = float(turn["end"])
        if time > end:
            status["outside_turns"] += 1
            continue
        if time < start + erosion_seconds or time > end - erosion_seconds:
            status["boundary_eroded_frames"] += 1
            continue
        votes[index][str(frame["name"])] += 1
        status["aligned_frames"] += 1

    records: list[dict[str, Any]] = []
    for index, counter in sorted(votes.items()):
        total = sum(counter.values())
        name, count = counter.most_common(1)[0]
        share = count / total
        if total < int(settings["minimum_turn_frame_votes"]):
            status["insufficient_turn_frames"] += 1
            continue
        if share < float(settings["minimum_turn_vote_share"]):
            status["ambiguous_turn_votes"] += 1
            continue
        turn = turns[index]
        records.append(
            {
                "turn_index": index,
                "speaker": str(turn.get("speaker") or "Speaker Unknown"),
                "start": round(float(turn["start"]), 3),
                "end": round(float(turn["end"]), 3),
                "time": round((float(turn["start"]) + float(turn["end"])) / 2.0, 3),
                "name": name,
                "frame_votes": total,
                "winner_frame_votes": count,
                "winner_frame_vote_share": round(share, 3),
                "votes": dict(counter),
            }
        )
        status["accepted_turn_evidence"] += 1
    return records, dict(status)


def _name_stats(records: list[dict[str, Any]], name: str | None = None) -> dict[str, Any]:
    votes = Counter(str(record["name"]) for record in records)
    total = len(records)
    if not votes:
        return {
            "turns": 0,
            "winner": None,
            "winner_turns": 0,
            "winner_share": 0.0,
            "wilson_lower_bound": 0.0,
            "runner_up_turns": 0,
            "runner_ratio": None,
            "time_span_seconds": 0.0,
            "votes": {},
        }
    winner, winner_turns = votes.most_common(1)[0]
    candidate = str(name) if name is not None else winner
    candidate_turns = int(votes.get(candidate, 0))
    candidate_share = candidate_turns / total
    other_counts = [count for item_name, count in votes.items() if item_name != candidate]
    runner_up = max(other_counts, default=0)
    candidate_records = [record for record in records if str(record["name"]) == candidate]
    span = (
        max(float(record["time"]) for record in candidate_records)
        - min(float(record["time"]) for record in candidate_records)
        if len(candidate_records) > 1
        else 0.0
    )
    return {
        "turns": total,
        "winner": winner,
        "winner_turns": candidate_turns,
        "winner_share": round(candidate_share, 3),
        "wilson_lower_bound": round(_wilson_lower_bound(candidate_turns, total), 3),
        "runner_up_turns": runner_up,
        "runner_ratio": round(candidate_turns / runner_up, 3) if runner_up else None,
        "time_span_seconds": round(span, 3),
        "votes": dict(votes),
    }


def _passes_vote_gate(
    stats: dict[str, Any],
    *,
    min_turns: int,
    min_share: float,
    min_wilson: float,
) -> list[str]:
    reasons: list[str] = []
    if int(stats["turns"]) < min_turns:
        reasons.append("insufficient_distinct_turns")
    if float(stats["winner_share"]) < min_share:
        reasons.append("insufficient_turn_vote_share")
    if float(stats["wilson_lower_bound"]) < min_wilson:
        reasons.append("insufficient_wilson_lower_bound")
    return reasons


def _directional_gate(
    training: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    training_stats = _name_stats(training)
    candidate = training_stats["winner"]
    if not candidate:
        return {"accepted": False, "candidate": None, "reason": ["no_training_turn_evidence"]}
    training_reasons = _passes_vote_gate(
        training_stats,
        min_turns=int(settings["minimum_training_turns"]),
        min_share=float(settings["minimum_training_vote_share"]),
        min_wilson=float(settings["minimum_wilson_lower_bound"]),
    )
    held_out_stats = _name_stats(held_out, str(candidate))
    held_out_reasons = _passes_vote_gate(
        held_out_stats,
        min_turns=int(settings["minimum_validation_turns"]),
        min_share=float(settings["minimum_validation_vote_share"]),
        min_wilson=float(settings["minimum_wilson_lower_bound"]),
    )
    if held_out_stats["winner"] != candidate:
        held_out_reasons.append("held_out_winner_disagrees")
    return {
        "accepted": not training_reasons and not held_out_reasons,
        "candidate": candidate,
        "training": training_stats,
        "held_out": held_out_stats,
        "training_reasons": training_reasons,
        "held_out_reasons": held_out_reasons,
    }


def _quartile_stats(records: list[dict[str, Any]], candidate: str, settings: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    count = len(records)
    for index in range(4):
        start = math.floor(index * count / 4)
        end = math.floor((index + 1) * count / 4)
        stats = _name_stats(records[start:end], candidate)
        reasons = _passes_vote_gate(
            stats,
            min_turns=2,
            min_share=float(settings["minimum_validation_vote_share"]),
            min_wilson=0.0,
        )
        values.append({"quartile": index + 1, "stats": stats, "reasons": reasons})
    return values


def _stability_gate(
    candidate: str,
    speaker: str,
    by_erosion: dict[float, list[dict[str, Any]]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    values: list[dict[str, Any]] = []
    reasons: list[str] = []
    for erosion, records in sorted(by_erosion.items()):
        speaker_records = [record for record in records if record["speaker"] == speaker]
        stats = _name_stats(speaker_records, candidate)
        item_reasons = _passes_vote_gate(
            stats,
            min_turns=int(settings["minimum_training_turns"]),
            min_share=float(settings["minimum_training_vote_share"]),
            min_wilson=float(settings["minimum_wilson_lower_bound"]),
        )
        if stats["winner"] != candidate:
            item_reasons.append("erosion_winner_disagrees")
        if item_reasons:
            reasons.append(f"erosion_{erosion:g}_unstable")
        values.append({"erosion_seconds": erosion, "stats": stats, "reasons": item_reasons})
    return values, reasons


def _support_intervals(
    records: list[dict[str, Any]],
    candidate: str | None,
    *,
    maximum_gap_seconds: float,
) -> list[dict[str, float]]:
    """Return contiguous direct-visual support without filling long visual gaps."""

    if not candidate:
        return []
    candidate_records = sorted(
        (record for record in records if str(record["name"]) == candidate),
        key=lambda record: (float(record["start"]), float(record["end"])),
    )
    if not candidate_records:
        return []
    intervals: list[dict[str, float]] = []
    start = float(candidate_records[0]["start"])
    end = float(candidate_records[0]["end"])
    for record in candidate_records[1:]:
        next_start = float(record["start"])
        next_end = float(record["end"])
        if next_start - end <= maximum_gap_seconds:
            end = max(end, next_end)
            continue
        intervals.append({"start": round(start, 3), "end": round(end, 3)})
        start = next_start
        end = next_end
    intervals.append({"start": round(start, 3), "end": round(end, 3)})
    return intervals


def _evaluate_cluster(
    speaker: str,
    records: list[dict[str, Any]],
    by_erosion: dict[float, list[dict[str, Any]]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda record: (float(record["time"]), int(record["turn_index"])))
    split_index = math.ceil(len(ordered) * 0.60)
    early = ordered[:split_index]
    late = ordered[split_index:]
    forward = _directional_gate(early, late, settings)
    reverse = _directional_gate(late, early, settings)
    reasons: list[str] = []
    candidate = forward.get("candidate")
    if not forward.get("accepted"):
        reasons.append("early_to_late_validation_failed")
    if not reverse.get("accepted"):
        reasons.append("late_to_early_validation_failed")
    if not candidate or candidate != reverse.get("candidate"):
        reasons.append("directional_candidates_disagree")
        candidate = None

    all_stats = _name_stats(ordered, candidate)
    quartiles = _quartile_stats(ordered, str(candidate), settings) if candidate else []
    if candidate and any(item["reasons"] for item in quartiles):
        reasons.append("quartile_stability_failed")
    if candidate and float(all_stats["time_span_seconds"]) < float(settings["minimum_time_span_seconds"]):
        reasons.append("insufficient_time_span")
    stability, stability_reasons = _stability_gate(str(candidate), speaker, by_erosion, settings) if candidate else ([], ["no_candidate"])
    reasons.extend(stability_reasons)

    accepted = bool(candidate) and not reasons
    support_intervals = _support_intervals(
        ordered,
        candidate if accepted else None,
        maximum_gap_seconds=float(settings["maximum_support_gap_seconds"]),
    )
    return {
        "speaker": speaker,
        "status": "accepted" if accepted else "rejected",
        "candidate": candidate,
        "all": all_stats,
        "early_to_late": forward,
        "late_to_early": reverse,
        "quartiles": quartiles,
        "erosion_stability": stability,
        "reasons": sorted(set(reasons)),
        "support_intervals": support_intervals,
        "support_start": round(min((float(record["start"]) for record in ordered if record["name"] == candidate), default=0.0), 3),
        "support_end": round(max((float(record["end"]) for record in ordered if record["name"] == candidate), default=0.0), 3),
    }


def build_direct_visual_cluster_identity(
    visual_payload: dict[str, Any],
    turns: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not turns:
        raise DirectVisualClusterIdentityError("speaker turns are required for direct visual cluster identity")
    frames = list(visual_payload.get("frames", []))
    by_erosion: dict[float, list[dict[str, Any]]] = {}
    alignment: dict[str, dict[str, int]] = {}
    erosion_values = sorted(set([float(settings["boundary_erosion_seconds"]), *settings["erosion_stability_seconds"]]))
    for erosion in erosion_values:
        records, status = _turn_evidence(frames, turns, settings=settings, erosion_seconds=erosion)
        by_erosion[erosion] = records
        alignment[f"{erosion:g}"] = status

    primary_erosion = float(settings["boundary_erosion_seconds"])
    primary_records = by_erosion[primary_erosion]
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in primary_records:
        if record["speaker"] != "Speaker Unknown":
            by_speaker[str(record["speaker"])].append(record)

    clusters = {
        speaker: _evaluate_cluster(speaker, records, by_erosion, settings)
        for speaker, records in sorted(by_speaker.items())
    }
    name_to_speakers: dict[str, list[str]] = defaultdict(list)
    for speaker, cluster in clusters.items():
        if cluster["status"] == "accepted" and cluster["candidate"]:
            name_to_speakers[str(cluster["candidate"])].append(speaker)
    for name, speakers in name_to_speakers.items():
        if len(speakers) < 2:
            continue
        for speaker in speakers:
            cluster = clusters[speaker]
            cluster["status"] = "rejected"
            cluster["reasons"] = sorted(set([*cluster["reasons"], "name_collision"]))
            cluster["name_collision_with"] = sorted(item for item in speakers if item != speaker)

    accepted = {
        speaker: cluster
        for speaker, cluster in clusters.items()
        if cluster["status"] == "accepted" and cluster["candidate"]
    }
    return {
        "format": "direct-visual-voice-cluster-identity/v2",
        "settings": settings,
        "direct_frame_sources": sorted(_DIRECT_FRAME_SOURCES),
        "primary_erosion_seconds": primary_erosion,
        "alignment": alignment,
        "clusters": clusters,
        "accepted_clusters": accepted,
        "note": "Names are propagated only from turn-level direct visual evidence after boundary erosion, bidirectional held-out validation, quartile stability, one-name-to-one-cluster checks, and contiguous direct-visual support intervals.",
    }


def clear_direct_visual_cluster_identity(segments: list[dict[str, Any]]) -> int:
    """Remove only this stage's prior propagation before a configuration rerun."""

    cleared = 0
    for segment in segments:
        if str(segment.get("name_source") or "") != _PROPAGATED_SOURCE:
            continue
        for key in (
            "name",
            "name_source",
            "name_confidence",
            "direct_visual_cluster_identity_evidence",
        ):
            segment.pop(key, None)
        cleared += 1
    return cleared


def _support_interval_for_segment(
    cluster: dict[str, Any],
    *,
    start: float,
    end: float,
) -> dict[str, float] | None:
    intervals = cluster.get("support_intervals")
    if not isinstance(intervals, list):
        # Legacy full-span bounds can bridge an unobserved visual gap.  A
        # current propagation artifact must carry the explicit intervals.
        return None
    for interval in intervals:
        try:
            interval_start = float(interval["start"])
            interval_end = float(interval["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start >= interval_start and end <= interval_end:
            return {"start": round(interval_start, 3), "end": round(interval_end, 3)}
    return None


def apply_direct_visual_cluster_identity(
    segments: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    accepted = dict(payload.get("accepted_clusters", {}))
    settings = payload.get("settings", {})
    min_confidence = float(settings.get("minimum_segment_speaker_confidence", 1.0))
    assigned = 0
    preserved = 0
    skipped_low_confidence = 0
    skipped_outside_support = 0
    for segment in segments:
        if segment.get("name"):
            preserved += 1
            continue
        speaker = str(segment.get("speaker") or "")
        cluster = accepted.get(speaker)
        if not cluster:
            continue
        try:
            speaker_confidence = float(segment.get("speaker_confidence", 0.0))
        except (TypeError, ValueError):
            speaker_confidence = 0.0
        if speaker_confidence < min_confidence:
            skipped_low_confidence += 1
            continue
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        support_interval = _support_interval_for_segment(cluster, start=start, end=end)
        if support_interval is None:
            skipped_outside_support += 1
            continue
        held_out = cluster["early_to_late"]["held_out"]
        reverse_held_out = cluster["late_to_early"]["held_out"]
        confidence = min(
            0.95,
            0.62
            + 0.18 * min(float(held_out["winner_share"]), float(reverse_held_out["winner_share"]))
            + 0.14 * min(float(held_out["wilson_lower_bound"]), float(reverse_held_out["wilson_lower_bound"])),
        )
        confidence = min(confidence, speaker_confidence)
        segment["name"] = str(cluster["candidate"])
        segment["name_source"] = _PROPAGATED_SOURCE
        segment["name_confidence"] = round(confidence, 3)
        segment["direct_visual_cluster_identity_evidence"] = {
            "format": payload["format"],
            "speaker_cluster": speaker,
            "candidate": cluster["candidate"],
            "primary_erosion_seconds": payload["primary_erosion_seconds"],
            "held_out_turn_share": held_out["winner_share"],
            "held_out_wilson_lower_bound": held_out["wilson_lower_bound"],
            "reverse_held_out_turn_share": reverse_held_out["winner_share"],
            "reverse_held_out_wilson_lower_bound": reverse_held_out["wilson_lower_bound"],
            "support_interval": support_interval,
        }
        assigned += 1
    return {
        "status": "ok" if accepted else "audit_only_no_accepted_clusters",
        "accepted_clusters": len(accepted),
        "assigned_segments": assigned,
        "preserved_named_segments": preserved,
        "skipped_low_speaker_confidence": skipped_low_confidence,
        "skipped_outside_visual_support": skipped_outside_support,
    }


def write_direct_visual_cluster_report(path: Path, *, payload: dict[str, Any], status: dict[str, Any]) -> None:
    lines = [
        "# Direct Visual Voice Cluster Identity Report",
        "",
        "## Method",
        "- Direct active-speaker visual frames are assigned to diarization turns only after fixed boundary erosion.",
        "- A turn contributes one vote after its own frame votes agree; repeated frames do not increase the effective sample size.",
        "- Each cluster must pass both early-to-late and late-to-early held-out checks, four-quartile stability, erosion stability, and name uniqueness.",
        f"- Propagation is limited to contiguous visual-support intervals; gaps above {float(payload.get('settings', {}).get('maximum_support_gap_seconds', 0.0)):.1f} seconds remain unnamed.",
        "- Mixed clusters and Speaker Unknown remain unnamed.",
        "",
        "## Cluster Results",
    ]
    for speaker, cluster in sorted(payload.get("clusters", {}).items()):
        all_stats = cluster.get("all", {})
        lines.append(
            f"- `{speaker}`: status={cluster.get('status')} candidate={cluster.get('candidate')} "
            f"turns={all_stats.get('turns', 0)} share={float(all_stats.get('winner_share', 0.0)):.3f} "
            f"wilson_lower={float(all_stats.get('wilson_lower_bound', 0.0)):.3f} "
            f"support_intervals={len(cluster.get('support_intervals', []))} "
            f"reasons={','.join(cluster.get('reasons', [])) or 'none'}"
        )
    lines += [
        "",
        "## Application",
        f"- Status: {status.get('status', 'unknown')}",
        f"- Accepted clusters: {status.get('accepted_clusters', 0)}",
        f"- Additional named transcript segments: {status.get('assigned_segments', 0)}",
        f"- Cleared prior cluster-propagated segments before rerun: {status.get('cleared_prior_cluster_assignments', 0)}",
        f"- Preserved named segments: {status.get('preserved_named_segments', 0)}",
        f"- Skipped for low diarization confidence: {status.get('skipped_low_speaker_confidence', 0)}",
        f"- Skipped outside direct visual support: {status.get('skipped_outside_visual_support', 0)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
