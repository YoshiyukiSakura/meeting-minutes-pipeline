from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_items import build_action_intent_recall, stable_segment_id, transcript_fingerprint
from .jsonio import read_json, write_json
from .minutes_contract import (
    UNASSIGNED_ACTION_OWNER,
    ShareableActionRow,
    ShareableProjectUpdateRow,
)

ACTION_EVIDENCE_FORMAT = "meeting-minutes/reviewed-action-evidence-v1"
ACTION_INTENT_REVIEW_FORMAT = "meeting-minutes/reviewed-action-intents-v1"
PROJECT_EVIDENCE_FORMAT = "meeting-minutes/reviewed-project-evidence-v1"
PUBLICATION_FORMAT = "meeting-minutes/shareable-minutes-v4"
CANONICAL_MINUTES_FILENAMES = {
    "zh": "minutes.md",
    "en": "minutes.en.md",
}
SHARE_BUNDLE_DIRNAME = "share"
SHARE_BUNDLE_FILENAMES = {
    "zh": "minutes.md",
    "en": "minutes.en.md",
    "transcript": "transcript.md",
}
_IDENTITY_CONFIDENCE = 0.6
PROJECT_UPDATE_COVERAGE_MIN_SECONDS = 60.0
PROJECT_UPDATE_EVIDENCE_MAX_TIME_PADDING_SECONDS = 5.0
DISPLAY_TIMESTAMP_ROUNDING_TOLERANCE_SECONDS = 1.0
ACTION_INTENT_REJECTION_REASONS = frozenset(
    {
        "not_an_action",
        "conditional_or_hypothetical",
        "owner_unresolved",
        "duplicate_of_published_action",
        "superseded",
        "insufficient_transcript_context",
    }
)


@dataclass(frozen=True)
class ParticipantCoverage:
    """Named speech coverage that must be accounted for before publication."""

    participant: str
    covered_seconds: float
    identity_sources: tuple[str, ...]


def payload_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def action_ledger_fingerprint(ledger: dict[str, Any]) -> str:
    return payload_fingerprint(ledger)


def _normalize_participant_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _trusted_named_segment(segment: dict[str, Any]) -> bool:
    name = _normalize_participant_name(segment.get("name"))
    if not name:
        return False
    if name.casefold().startswith("speaker ") or name.casefold() in {"unknown", "unresolved"}:
        return False
    return float(segment.get("name_confidence", 0.0) or 0.0) >= _IDENTITY_CONFIDENCE


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    ordered_intervals = sorted(intervals)
    start, end = ordered_intervals[0]
    for next_start, next_end in ordered_intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += end - start
        start, end = next_start, next_end
    return total + end - start


def recompute_project_update_coverage(segments: list[dict[str, Any]]) -> dict[str, ParticipantCoverage]:
    """Return the release-blocking participant set from the current transcript.

    Coverage is the union of high-confidence named speech intervals, rather
    than a sum of potentially overlapping diarization or ASR fragments.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for segment in segments:
        if not _trusted_named_segment(segment):
            continue
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        participant = _normalize_participant_name(segment.get("name"))
        item = grouped.setdefault(participant, {"intervals": [], "sources": set()})
        item["intervals"].append((start, end))
        source = str(segment.get("name_source") or "").strip()
        if source:
            item["sources"].add(source)

    coverage: dict[str, ParticipantCoverage] = {}
    for participant, item in grouped.items():
        covered_seconds = _union_duration(item["intervals"])
        if covered_seconds < PROJECT_UPDATE_COVERAGE_MIN_SECONDS:
            continue
        coverage[participant] = ParticipantCoverage(
            participant=participant,
            covered_seconds=covered_seconds,
            identity_sources=tuple(sorted(item["sources"])),
        )
    return dict(sorted(coverage.items()))


def project_update_coverage_snapshot(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize deterministic coverage for publication status and internal audit."""

    return [
        {
            "participant": coverage.participant,
            "covered_seconds": round(coverage.covered_seconds, 3),
            "identity_sources": list(coverage.identity_sources),
        }
        for coverage in recompute_project_update_coverage(segments).values()
    ]


def canonical_minutes_paths(output_dir: Path) -> dict[str, Path]:
    return {
        language: output_dir / filename
        for language, filename in CANONICAL_MINUTES_FILENAMES.items()
    }


def share_bundle_paths(output_dir: Path) -> dict[str, Path]:
    share_dir = output_dir / SHARE_BUNDLE_DIRNAME
    return {
        artifact: share_dir / filename
        for artifact, filename in SHARE_BUNDLE_FILENAMES.items()
    }


def _next_archive_path(path: Path, kind: str, fingerprint: str | None = None) -> Path:
    suffix = f".{fingerprint[:12]}" if fingerprint else ""
    candidate = path.with_name(f"{path.stem}.{kind}{suffix}{path.suffix}")
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{kind}{suffix}-{index}{path.suffix}")
        index += 1
    return candidate


def _write_publication_status(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "minutes.publish-status.json", payload)


def _archive_canonical_minutes(
    paths: dict[str, Path],
    *,
    kind: str,
    fingerprints: dict[str, str] | None = None,
) -> dict[str, str]:
    archived: dict[str, str] = {}
    for language, path in paths.items():
        if not path.exists():
            continue
        archive = _next_archive_path(path, kind, (fingerprints or {}).get(language))
        path.replace(archive)
        archived[language] = archive.name
    return archived


def _archive_share_bundle(output_dir: Path, *, kind: str, fingerprint: str | None = None) -> str | None:
    share_dir = output_dir / SHARE_BUNDLE_DIRNAME
    if not share_dir.exists():
        return None
    archive = _next_archive_path(share_dir, kind, fingerprint)
    share_dir.replace(archive)
    return archive.name


def _bound_evidence_stale_reason(published: dict[str, Any], evidence_kind: str) -> str | None:
    expected_fingerprint = published.get(f"{evidence_kind}_sha256")
    raw_path = published.get(f"{evidence_kind}_path")
    if expected_fingerprint is None and raw_path is None:
        return None
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        return f"{evidence_kind}_fingerprint_missing"
    if not isinstance(raw_path, str) or not raw_path:
        return f"{evidence_kind}_path_missing"
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return f"{evidence_kind}_missing"
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return f"{evidence_kind}_invalid"
    if payload_fingerprint(payload) != expected_fingerprint:
        return f"{evidence_kind}_changed"
    return None


def sync_publication_status(
    output_dir: Path,
    segments: list[dict[str, Any]],
    action_ledger: dict[str, Any],
) -> dict[str, Any]:
    """Invalidate canonical minutes when their bound source state changes.

    This is deliberately conservative. An unknown or malformed publication
    record is never allowed to keep the canonical filename after a pipeline
    stage has regenerated transcript-derived artifacts.
    """

    destinations = canonical_minutes_paths(output_dir)
    share_paths = share_bundle_paths(output_dir)
    status_path = output_dir / "minutes.publish-status.json"
    transcript_sha256 = transcript_fingerprint(segments)
    ledger_sha256 = action_ledger_fingerprint(action_ledger)
    existing_languages = [language for language, path in destinations.items() if path.exists()]
    if not existing_languages:
        archived_share = _archive_share_bundle(output_dir, kind="untracked", fingerprint=transcript_sha256)
        result = {"status": "pending_review", "reason": "publish_minutes_required", "format": PUBLICATION_FORMAT}
        if archived_share:
            result["archived_share_bundle"] = archived_share
        return result

    if not status_path.is_file():
        archived = _archive_canonical_minutes(destinations, kind="untracked")
        archived_share = _archive_share_bundle(output_dir, kind="untracked", fingerprint=transcript_sha256)
        status = {
            "format": PUBLICATION_FORMAT,
            "status": "untracked",
            "reason": "missing_publication_status",
            "archived_minutes": archived,
            "current_transcript_sha256": transcript_sha256,
            "current_action_ledger_sha256": ledger_sha256,
        }
        if archived_share:
            status["archived_share_bundle"] = archived_share
        _write_publication_status(
            output_dir,
            status,
        )
        result = {"status": "untracked", "reason": "missing_publication_status", "archived_minutes": archived}
        if archived_share:
            result["archived_share_bundle"] = archived_share
        return result

    try:
        published = read_json(status_path)
    except (OSError, ValueError, json.JSONDecodeError):
        archived = _archive_canonical_minutes(destinations, kind="untracked")
        archived_share = _archive_share_bundle(output_dir, kind="untracked", fingerprint=transcript_sha256)
        status = {
            "format": PUBLICATION_FORMAT,
            "status": "untracked",
            "reason": "invalid_publication_status",
            "archived_minutes": archived,
            "current_transcript_sha256": transcript_sha256,
            "current_action_ledger_sha256": ledger_sha256,
        }
        if archived_share:
            status["archived_share_bundle"] = archived_share
        _write_publication_status(
            output_dir,
            status,
        )
        result = {"status": "untracked", "reason": "invalid_publication_status", "archived_minutes": archived}
        if archived_share:
            result["archived_share_bundle"] = archived_share
        return result

    if len(existing_languages) != len(destinations):
        state = "stale" if published.get("status") == "published" else "untracked"
        archived = _archive_canonical_minutes(destinations, kind=state)
        archived_share = _archive_share_bundle(output_dir, kind=state, fingerprint=transcript_sha256)
        status = {
            "format": PUBLICATION_FORMAT,
            "status": state,
            "reason": "missing_canonical_language",
            "archived_minutes": archived,
            "previous_publication": published,
            "current_transcript_sha256": transcript_sha256,
            "current_action_ledger_sha256": ledger_sha256,
        }
        if archived_share:
            status["archived_share_bundle"] = archived_share
        _write_publication_status(
            output_dir,
            status,
        )
        result = {"status": state, "reason": "missing_canonical_language", "archived_minutes": archived}
        if archived_share:
            result["archived_share_bundle"] = archived_share
        return result

    canonical_sha256 = {
        language: hashlib.sha256(path.read_bytes()).hexdigest()
        for language, path in destinations.items()
    }
    stale_reasons: list[str] = []
    if published.get("format") != PUBLICATION_FORMAT or published.get("status") != "published":
        stale_reasons.append("publication_status_not_current")
    if published.get("languages") != list(CANONICAL_MINUTES_FILENAMES):
        stale_reasons.append("publication_languages_not_current")
    if published.get("transcript_sha256") != transcript_sha256:
        stale_reasons.append("transcript_changed")
    if published.get("action_ledger_sha256") != ledger_sha256:
        stale_reasons.append("action_ledger_changed")
    for evidence_kind in ("action_evidence", "action_intent_review", "project_evidence"):
        if reason := _bound_evidence_stale_reason(published, evidence_kind):
            stale_reasons.append(reason)
    published_hashes = published.get("canonical_sha256")
    if not isinstance(published_hashes, dict):
        stale_reasons.append("canonical_hashes_missing")
    else:
        for language, current_hash in canonical_sha256.items():
            if published_hashes.get(language) != current_hash:
                stale_reasons.append(f"canonical_content_changed:{language}")
    published_share_hashes = published.get("share_sha256")
    if not isinstance(published_share_hashes, dict):
        stale_reasons.append("share_hashes_missing")
    else:
        for artifact, path in share_paths.items():
            if not path.is_file():
                stale_reasons.append(f"share_artifact_missing:{artifact}")
                continue
            current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if published_share_hashes.get(artifact) != current_hash:
                stale_reasons.append(f"share_artifact_changed:{artifact}")
    if not stale_reasons:
        return {
            "status": "published",
            "format": PUBLICATION_FORMAT,
            "languages": list(CANONICAL_MINUTES_FILENAMES),
            "share_bundle": SHARE_BUNDLE_DIRNAME,
            "action_row_count": int(published.get("action_row_count", 0) or 0),
            "action_evidence_bound": bool(published.get("action_evidence_sha256")),
            "action_intent_review_bound": bool(published.get("action_intent_review_sha256")),
            "project_update_row_count": int(published.get("project_update_row_count", 0) or 0),
            "project_evidence_bound": bool(published.get("project_evidence_sha256")),
        }

    archived = _archive_canonical_minutes(destinations, kind="stale", fingerprints=canonical_sha256)
    archived_share = _archive_share_bundle(output_dir, kind="stale", fingerprint=transcript_sha256)
    status = {
        "format": PUBLICATION_FORMAT,
        "status": "stale",
        "reasons": stale_reasons,
        "archived_minutes": archived,
        "previous_publication": published,
        "current_transcript_sha256": transcript_sha256,
        "current_action_ledger_sha256": ledger_sha256,
    }
    if archived_share:
        status["archived_share_bundle"] = archived_share
    _write_publication_status(
        output_dir,
        status,
    )
    result = {"status": "stale", "reasons": stale_reasons, "archived_minutes": archived}
    if archived_share:
        result["archived_share_bundle"] = archived_share
    return result


def validate_reviewed_action_evidence(
    *,
    manifest: dict[str, Any],
    rows: list[ShareableActionRow],
    segments: list[dict[str, Any]],
    action_ledger: dict[str, Any],
) -> list[str]:
    """Validate the non-shareable evidence manifest behind published action rows."""

    errors: list[str] = []
    if manifest.get("format") != ACTION_EVIDENCE_FORMAT:
        errors.append("action_evidence_format_invalid")
    transcript_sha256 = transcript_fingerprint(segments)
    ledger_sha256 = action_ledger_fingerprint(action_ledger)
    if manifest.get("transcript_sha256") != transcript_sha256:
        errors.append("action_evidence_transcript_mismatch")
    if manifest.get("action_ledger_sha256") != ledger_sha256:
        errors.append("action_evidence_ledger_mismatch")
    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, list):
        return errors + ["action_evidence_rows_invalid"]

    entries: dict[int, dict[str, Any]] = {}
    for entry in raw_rows:
        if not isinstance(entry, dict) or not isinstance(entry.get("row"), int):
            errors.append("action_evidence_row_invalid")
            continue
        row_index = entry["row"]
        if row_index in entries:
            errors.append(f"action_evidence_row_duplicate:{row_index}")
        entries[row_index] = entry
    expected_indices = {row.index for row in rows}
    if set(entries) != expected_indices:
        errors.append("action_evidence_rows_do_not_match_minutes")

    records = {
        stable_segment_id(segment, index): segment
        for index, segment in enumerate(segments)
    }
    candidates = {
        str(candidate.get("candidate_id")): candidate
        for candidate in action_ledger.get("candidates", [])
    }
    for row in rows:
        entry = entries.get(row.index)
        if entry is None:
            continue
        prefix = f"action_evidence_row:{row.index}"
        source_ids = entry.get("source_segment_ids")
        source_ids_are_valid = (
            isinstance(source_ids, list)
            and bool(source_ids)
            and all(isinstance(item, str) and item for item in source_ids)
        )
        if not source_ids_are_valid:
            errors.append(f"{prefix}:source_segments_invalid")
            continue
        source_segments = [records.get(segment_id) for segment_id in source_ids]
        if any(segment is None for segment in source_segments):
            errors.append(f"{prefix}:source_segment_unknown")
            continue
        if not all(
            float(segment.get("end", 0.0)) > row.start and float(segment.get("start", 0.0)) < row.end
            for segment in source_segments
        ):
            errors.append(f"{prefix}:source_segment_outside_action_range")

        mode = entry.get("evidence_mode")
        owner_segment_id = entry.get("owner_evidence_segment_id")
        if row.owner == UNASSIGNED_ACTION_OWNER:
            if owner_segment_id not in (None, ""):
                errors.append(f"{prefix}:unassigned_owner_evidence_unexpected")
            if mode != "reviewed_context":
                errors.append(f"{prefix}:unassigned_owner_requires_reviewed_context")
        else:
            owner_segment = records.get(str(owner_segment_id or ""))
            if owner_segment is None:
                errors.append(f"{prefix}:owner_evidence_unknown")
            elif float(owner_segment.get("name_confidence", 0.0) or 0.0) < _IDENTITY_CONFIDENCE:
                errors.append(f"{prefix}:owner_identity_low_confidence")
            elif str(owner_segment.get("name") or "").strip() != row.owner:
                errors.append(f"{prefix}:owner_evidence_mismatch")

        if mode == "ledger":
            candidate = candidates.get(str(entry.get("candidate_id") or ""))
            if candidate is None:
                errors.append(f"{prefix}:ledger_candidate_unknown")
            else:
                if candidate.get("status") != "accepted":
                    errors.append(f"{prefix}:ledger_candidate_not_accepted")
                if candidate.get("owner") != row.owner:
                    errors.append(f"{prefix}:ledger_owner_mismatch")
                if candidate.get("commitment_segment_id") not in source_ids:
                    errors.append(f"{prefix}:ledger_commitment_not_linked")
        elif mode == "reviewed_context":
            if not str(entry.get("review_note") or "").strip():
                errors.append(f"{prefix}:review_note_required")
        else:
            errors.append(f"{prefix}:evidence_mode_invalid")
    return sorted(set(errors))


def action_intent_recall_signals(
    *,
    segments: list[dict[str, Any]],
    action_ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recompute the independent weak-intent recall set for a release gate."""

    candidates = action_ledger.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    audit = build_action_intent_recall(segments, candidates)
    signals = audit.get("signals", [])
    return signals if isinstance(signals, list) else []


def validate_reviewed_action_intent_review(
    *,
    manifest: dict[str, Any],
    rows: list[ShareableActionRow],
    action_evidence: dict[str, Any] | None,
    segments: list[dict[str, Any]],
    action_ledger: dict[str, Any],
) -> list[str]:
    """Require a durable disposition for every independently recalled intent.

    A weak phrase is not proof of an action. It must either be linked to a
    reviewed action row with the same named speaker or be explicitly rejected
    using a constrained reason. The manifest is fingerprinted and later bound
    into publication status, so transcript or ledger changes invalidate it.
    """

    errors: list[str] = []
    if manifest.get("format") != ACTION_INTENT_REVIEW_FORMAT:
        errors.append("action_intent_review_format_invalid")
    transcript_sha256 = transcript_fingerprint(segments)
    ledger_sha256 = action_ledger_fingerprint(action_ledger)
    if manifest.get("transcript_sha256") != transcript_sha256:
        errors.append("action_intent_review_transcript_mismatch")
    if manifest.get("action_ledger_sha256") != ledger_sha256:
        errors.append("action_intent_review_ledger_mismatch")

    raw_items = manifest.get("items")
    if not isinstance(raw_items, list):
        return sorted(set(errors + ["action_intent_review_items_invalid"]))
    entries: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict) or not isinstance(item.get("signal_id"), str) or not item["signal_id"]:
            errors.append("action_intent_review_item_invalid")
            continue
        signal_id = item["signal_id"]
        if signal_id in entries:
            errors.append(f"action_intent_review_item_duplicate:{signal_id}")
        entries[signal_id] = item

    signals = action_intent_recall_signals(segments=segments, action_ledger=action_ledger)
    expected_signals = {str(signal["signal_id"]): signal for signal in signals}
    if set(entries) != set(expected_signals):
        errors.append("action_intent_review_items_do_not_match_recall")

    evidence_entries: dict[int, dict[str, Any]] = {}
    if isinstance(action_evidence, dict) and isinstance(action_evidence.get("rows"), list):
        for entry in action_evidence["rows"]:
            if isinstance(entry, dict) and isinstance(entry.get("row"), int):
                evidence_entries[entry["row"]] = entry

    for signal_id, signal in expected_signals.items():
        entry = entries.get(signal_id)
        if entry is None:
            continue
        prefix = f"action_intent_review_signal:{signal_id}"
        disposition = entry.get("disposition")
        if disposition not in {"published", "rejected"}:
            errors.append(f"{prefix}:disposition_invalid")
            continue
        if not str(entry.get("review_note") or "").strip():
            errors.append(f"{prefix}:review_note_required")
        matching_rows = [
            row
            for row in rows
            if row.owner == signal["participant"]
            and signal["segment_id"] in (evidence_entries.get(row.index, {}).get("source_segment_ids") or [])
        ]
        if disposition == "published":
            if not matching_rows:
                errors.append(f"{prefix}:published_source_not_linked")
            continue
        rejection_reason = entry.get("rejection_reason")
        if rejection_reason not in ACTION_INTENT_REJECTION_REASONS:
            errors.append(f"{prefix}:rejection_reason_invalid")
        if matching_rows:
            errors.append(f"{prefix}:rejected_source_still_published")
    return sorted(set(errors))


def validate_reviewed_project_evidence(
    *,
    manifest: dict[str, Any],
    rows: list[ShareableProjectUpdateRow],
    segments: list[dict[str, Any]],
    action_ledger: dict[str, Any],
) -> list[str]:
    """Validate the internal evidence and coverage gate for project updates.

    A project update describes current or completed work. It is deliberately not
    an alternate action-item channel: evidence that is the accepted commitment
    for an action item must stay in the action table and its action manifest.
    """

    errors: list[str] = []
    if manifest.get("format") != PROJECT_EVIDENCE_FORMAT:
        errors.append("project_evidence_format_invalid")
    transcript_sha256 = transcript_fingerprint(segments)
    ledger_sha256 = action_ledger_fingerprint(action_ledger)
    if manifest.get("transcript_sha256") != transcript_sha256:
        errors.append("project_evidence_transcript_mismatch")
    if manifest.get("action_ledger_sha256") != ledger_sha256:
        errors.append("project_evidence_ledger_mismatch")
    if manifest.get("coverage_min_seconds") != PROJECT_UPDATE_COVERAGE_MIN_SECONDS:
        errors.append("project_evidence_coverage_threshold_mismatch")

    raw_rows = manifest.get("rows")
    if not isinstance(raw_rows, list):
        return errors + ["project_evidence_rows_invalid"]
    entries: dict[int, dict[str, Any]] = {}
    for entry in raw_rows:
        if not isinstance(entry, dict) or not isinstance(entry.get("row"), int):
            errors.append("project_evidence_row_invalid")
            continue
        row_index = entry["row"]
        if row_index in entries:
            errors.append(f"project_evidence_row_duplicate:{row_index}")
        entries[row_index] = entry
    expected_indices = {row.index for row in rows}
    if set(entries) != expected_indices:
        errors.append("project_evidence_rows_do_not_match_minutes")

    records = {
        stable_segment_id(segment, index): segment
        for index, segment in enumerate(segments)
    }
    accepted_commitment_ids = {
        str(candidate.get("commitment_segment_id"))
        for candidate in action_ledger.get("candidates", [])
        if candidate.get("status") == "accepted" and candidate.get("commitment_segment_id")
    }
    row_participants: set[str] = set()
    for row in rows:
        entry = entries.get(row.index)
        if entry is None:
            continue
        prefix = f"project_evidence_row:{row.index}"
        participant = _normalize_participant_name(row.participant)
        if not participant:
            errors.append(f"{prefix}:participant_invalid")
            continue
        row_participants.add(participant)
        source_ids = entry.get("source_segment_ids")
        source_ids_are_valid = (
            isinstance(source_ids, list)
            and bool(source_ids)
            and all(isinstance(item, str) and item for item in source_ids)
        )
        if not source_ids_are_valid:
            errors.append(f"{prefix}:source_segments_invalid")
            continue
        source_segments = [records.get(segment_id) for segment_id in source_ids]
        if any(segment is None for segment in source_segments):
            errors.append(f"{prefix}:source_segment_unknown")
            continue
        source_segments = [segment for segment in source_segments if segment is not None]
        if any(
            _normalize_participant_name(segment.get("name")) != participant
            or not _trusted_named_segment(segment)
            for segment in source_segments
        ):
            errors.append(f"{prefix}:source_participant_mismatch")
        source_start = min(float(segment.get("start", 0.0)) for segment in source_segments)
        source_end = max(float(segment.get("end", 0.0)) for segment in source_segments)
        if (
            source_start
            < row.start - DISPLAY_TIMESTAMP_ROUNDING_TOLERANCE_SECONDS
            or source_end
            > row.end + DISPLAY_TIMESTAMP_ROUNDING_TOLERANCE_SECONDS
        ):
            errors.append(f"{prefix}:source_segment_outside_project_range")
        if (
            source_start - row.start > PROJECT_UPDATE_EVIDENCE_MAX_TIME_PADDING_SECONDS
            or row.end - source_end > PROJECT_UPDATE_EVIDENCE_MAX_TIME_PADDING_SECONDS
        ):
            errors.append(f"{prefix}:evidence_time_window_excessive")
        if any(segment_id in accepted_commitment_ids for segment_id in source_ids):
            errors.append(f"{prefix}:shadows_accepted_action_item")

        participant_evidence_id = entry.get("participant_evidence_segment_id")
        participant_evidence = records.get(str(participant_evidence_id or ""))
        if participant_evidence is None:
            errors.append(f"{prefix}:participant_evidence_unknown")
        elif not _trusted_named_segment(participant_evidence):
            errors.append(f"{prefix}:participant_identity_low_confidence")
        elif _normalize_participant_name(participant_evidence.get("name")) != participant:
            errors.append(f"{prefix}:participant_evidence_mismatch")
        if not str(entry.get("review_note") or "").strip():
            errors.append(f"{prefix}:review_note_required")

    coverage = recompute_project_update_coverage(segments)
    raw_exceptions = manifest.get("exceptions", [])
    if not isinstance(raw_exceptions, list):
        return sorted(set(errors + ["project_evidence_exceptions_invalid"]))
    exception_participants: set[str] = set()
    for exception in raw_exceptions:
        if not isinstance(exception, dict):
            errors.append("project_evidence_exception_invalid")
            continue
        participant = _normalize_participant_name(exception.get("participant"))
        if not participant:
            errors.append("project_evidence_exception_participant_invalid")
            continue
        if participant in exception_participants:
            errors.append(f"project_evidence_exception_duplicate:{participant}")
            continue
        exception_participants.add(participant)
        expected_coverage = coverage.get(participant)
        if expected_coverage is None:
            errors.append(f"project_evidence_exception_participant_not_required:{participant}")
            continue
        try:
            reported_seconds = float(exception.get("covered_seconds"))
        except (TypeError, ValueError):
            errors.append(f"project_evidence_exception_seconds_invalid:{participant}")
        else:
            if abs(reported_seconds - expected_coverage.covered_seconds) > 0.01:
                errors.append(f"project_evidence_exception_stats_mismatch:{participant}")
        reported_sources = exception.get("identity_sources")
        if not isinstance(reported_sources, list) or tuple(sorted(map(str, reported_sources))) != expected_coverage.identity_sources:
            errors.append(f"project_evidence_exception_sources_mismatch:{participant}")
        evidence_segment = records.get(str(exception.get("identity_evidence_segment_id") or ""))
        if evidence_segment is None:
            errors.append(f"project_evidence_exception_evidence_unknown:{participant}")
        elif not _trusted_named_segment(evidence_segment):
            errors.append(f"project_evidence_exception_identity_low_confidence:{participant}")
        elif _normalize_participant_name(evidence_segment.get("name")) != participant:
            errors.append(f"project_evidence_exception_evidence_mismatch:{participant}")
        if not str(exception.get("reason") or "").strip():
            errors.append(f"project_evidence_exception_reason_required:{participant}")

    for participant in coverage:
        if participant not in row_participants and participant not in exception_participants:
            errors.append(f"project_coverage_missing:{participant}")
        if participant in row_participants and participant in exception_participants:
            errors.append(f"project_coverage_duplicated:{participant}")
    return sorted(set(errors))


def build_reviewed_evidence_manifests(
    *,
    smart_minutes: dict[str, Any],
    action_rows: list[ShareableActionRow],
    project_rows: list[ShareableProjectUpdateRow],
    segments: list[dict[str, Any]],
    action_ledger: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]] | None, list[str]]:
    """Bind a validated smart-minutes payload to internal publication evidence."""

    minutes = smart_minutes.get("minutes")
    if not isinstance(minutes, dict):
        return None, ["smart_minutes_payload_invalid"]
    raw_actions = minutes.get("actions")
    raw_updates = minutes.get("project_updates")
    if not isinstance(raw_actions, list) or not isinstance(raw_updates, list):
        return None, ["smart_minutes_sections_invalid"]

    records = {
        stable_segment_id(segment, index): segment
        for index, segment in enumerate(segments)
    }
    transcript_sha256 = transcript_fingerprint(segments)
    ledger_sha256 = action_ledger_fingerprint(action_ledger)
    errors: list[str] = []

    def normalized_text(value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "")).strip("。.")

    used_actions: set[int] = set()
    action_entries: list[dict[str, Any]] = []
    for row in action_rows:
        matches = [
            (index, action)
            for index, action in enumerate(raw_actions)
            if index not in used_actions
            and isinstance(action, dict)
            and _normalize_participant_name(action.get("owner")) == row.owner
            and normalized_text(action.get("item_zh")) == normalized_text(row.item)
        ]
        if len(matches) != 1:
            errors.append(f"smart_action_row_match_invalid:{row.index}")
            continue
        action_index, action = matches[0]
        used_actions.add(action_index)
        source_ids = [
            segment_id
            for segment_id in action.get("segment_ids", [])
            if isinstance(segment_id, str)
        ]
        owner_evidence_id: str | None = None
        if row.owner != UNASSIGNED_ACTION_OWNER:
            owner_evidence_id = next(
                (
                    segment_id
                    for segment_id in source_ids
                    if segment_id in records
                    and _trusted_named_segment(records[segment_id])
                    and _normalize_participant_name(
                        records[segment_id].get("name")
                    )
                    == row.owner
                ),
                None,
            )
        action_entries.append(
            {
                "row": row.index,
                "source_segment_ids": source_ids,
                "owner_evidence_segment_id": owner_evidence_id,
                "evidence_mode": "reviewed_context",
                "review_note": (
                    "Bound to the source IDs retained by the validated "
                    "smart-minutes review."
                ),
            }
        )

    action_evidence = {
        "format": ACTION_EVIDENCE_FORMAT,
        "transcript_sha256": transcript_sha256,
        "action_ledger_sha256": ledger_sha256,
        "rows": action_entries,
    }

    intent_items: list[dict[str, Any]] = []
    for signal in action_intent_recall_signals(
        segments=segments,
        action_ledger=action_ledger,
    ):
        matching_rows = [
            row
            for row in action_rows
            if row.owner == signal["participant"]
            and signal["segment_id"]
            in next(
                (
                    entry["source_segment_ids"]
                    for entry in action_entries
                    if entry["row"] == row.index
                ),
                [],
            )
        ]
        if not matching_rows:
            errors.append(
                f"smart_action_intent_unresolved:{signal['signal_id']}"
            )
            continue
        intent_items.append(
            {
                "signal_id": signal["signal_id"],
                "disposition": "published",
                "review_note": (
                    "The reviewed action row retains this independently "
                    "recalled self-intent segment."
                ),
            }
        )
    action_intent_review = {
        "format": ACTION_INTENT_REVIEW_FORMAT,
        "transcript_sha256": transcript_sha256,
        "action_ledger_sha256": ledger_sha256,
        "items": intent_items,
    }

    used_updates: set[int] = set()
    project_entries: list[dict[str, Any]] = []
    for row in project_rows:
        matches = [
            (index, update)
            for index, update in enumerate(raw_updates)
            if index not in used_updates
            and isinstance(update, dict)
            and _normalize_participant_name(update.get("participant"))
            == row.participant
            and normalized_text(update.get("project_zh"))
            == normalized_text(row.project)
            and normalized_text(update.get("update_zh"))
            == normalized_text(row.update)
        ]
        if len(matches) != 1:
            errors.append(f"smart_project_row_match_invalid:{row.index}")
            continue
        update_index, update = matches[0]
        used_updates.add(update_index)
        source_ids = [
            segment_id
            for segment_id in update.get("segment_ids", [])
            if isinstance(segment_id, str)
        ]
        participant_evidence_id = next(
            (
                segment_id
                for segment_id in source_ids
                if segment_id in records
                and _trusted_named_segment(records[segment_id])
                and _normalize_participant_name(
                    records[segment_id].get("name")
                )
                == row.participant
            ),
            None,
        )
        project_entries.append(
            {
                "row": row.index,
                "source_segment_ids": source_ids,
                "participant_evidence_segment_id": participant_evidence_id,
                "review_note": (
                    "Bound to the participant turns retained by the validated "
                    "smart-minutes review."
                ),
            }
        )
    project_evidence = {
        "format": PROJECT_EVIDENCE_FORMAT,
        "transcript_sha256": transcript_sha256,
        "action_ledger_sha256": ledger_sha256,
        "coverage_min_seconds": PROJECT_UPDATE_COVERAGE_MIN_SECONDS,
        "rows": project_entries,
        "exceptions": [],
    }

    errors.extend(
        f"action:{error}"
        for error in validate_reviewed_action_evidence(
            manifest=action_evidence,
            rows=action_rows,
            segments=segments,
            action_ledger=action_ledger,
        )
    )
    errors.extend(
        f"intent:{error}"
        for error in validate_reviewed_action_intent_review(
            manifest=action_intent_review,
            rows=action_rows,
            action_evidence=action_evidence,
            segments=segments,
            action_ledger=action_ledger,
        )
    )
    errors.extend(
        f"project:{error}"
        for error in validate_reviewed_project_evidence(
            manifest=project_evidence,
            rows=project_rows,
            segments=segments,
            action_ledger=action_ledger,
        )
    )
    if errors:
        return None, sorted(set(errors))
    return {
        "action_evidence": action_evidence,
        "action_intent_review": action_intent_review,
        "project_evidence": project_evidence,
    }, []
