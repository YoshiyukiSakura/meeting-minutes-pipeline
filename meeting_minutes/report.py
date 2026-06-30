from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .keyframes import KEYWORDS
from .time_utils import format_ts


def _speaker_label(segment: dict[str, Any]) -> str:
    name = segment.get("name")
    if name and float(segment.get("name_confidence", 0.0)) >= 0.6:
        return str(name)
    return str(segment.get("speaker") or "Speaker Unknown")


def _evidence(segment: dict[str, Any]) -> str:
    return f"{format_ts(float(segment['start']))}-{format_ts(float(segment['end']))}"


def write_transcript_markdown(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = ["# Transcript", ""]
    for segment in segments:
        label = _speaker_label(segment)
        confidence = float(segment.get("name_confidence") or segment.get("speaker_confidence") or 0.0)
        lines.append(f"- `{_evidence(segment)}` **{label}** ({confidence:.2f}): {segment['text']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_minutes(
    path: Path,
    *,
    segments: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    keyword_segments = [
        segment
        for segment in segments
        if any(keyword.lower() in str(segment.get("text", "")).lower() for keyword in KEYWORDS)
    ]
    named_counts = Counter(_speaker_label(segment) for segment in segments)
    lines = [
        "# Meeting Minutes",
        "",
        "## Source",
        f"- Input: `{metadata.get('input')}`",
        f"- Duration: `{format_ts(float(metadata.get('duration', 0.0)))}`",
        f"- Local-first: no SaaS upload performed by this pipeline.",
        "",
        "## Speaker Coverage",
    ]
    if named_counts:
        for label, count in named_counts.most_common():
            lines.append(f"- {label}: {count} transcript segment(s)")
    else:
        lines.append("- No transcript segments were generated.")

    lines += ["", "## Summary"]
    if keyword_segments:
        for segment in keyword_segments[:12]:
            lines.append(f"- `{_evidence(segment)}` **{_speaker_label(segment)}**: {segment['text']}")
    elif segments:
        for segment in segments[:8]:
            lines.append(f"- `{_evidence(segment)}` **{_speaker_label(segment)}**: {segment['text']}")
    else:
        lines.append("- ASR did not produce text; see `quality_report.md`.")

    lines += ["", "## Decisions / Risks / Action Candidates"]
    action_count = 0
    for segment in keyword_segments:
        action_count += 1
        refs = ", ".join(Path(ref).name for ref in segment.get("frame_refs", [])[:2]) or "no frame ref"
        lines.append(f"- `{_evidence(segment)}` **{_speaker_label(segment)}** [{refs}]: {segment['text']}")
    if action_count == 0:
        lines.append("- No explicit decision/action keywords were detected automatically.")

    lines += ["", "## Key Frames"]
    for frame in keyframes[:40]:
        reasons = ", ".join(frame.get("reasons", []))
        lines.append(f"- `{format_ts(float(frame['time']))}` `{Path(frame['path']).name}` {reasons}")
    if not keyframes:
        lines.append("- No key frames selected.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(
    path: Path,
    *,
    segments: list[dict[str, Any]],
    ocr_records: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    statuses: dict[str, Any],
) -> None:
    named = [s for s in segments if s.get("name") and float(s.get("name_confidence", 0.0)) >= 0.6]
    unresolved_name = [s for s in segments if not s.get("name") or float(s.get("name_confidence", 0.0)) < 0.6]
    unknown = [s for s in segments if not s.get("name") and str(s.get("speaker")) == "Speaker Unknown"]
    anonymous = [
        s
        for s in segments
        if not s.get("name")
        and str(s.get("speaker") or "Speaker Unknown") not in {"Speaker Unknown", ""}
    ]
    candidate_only = [s for s in segments if s.get("name_candidates") and not s.get("name")]
    lines = [
        "# Quality Report",
        "",
        "## Pipeline Status",
    ]
    for name, status in statuses.items():
        lines.append(f"- {name}: `{status.get('status')}` {status.get('reason') or status.get('error') or ''}".rstrip())
    lines += [
        "",
        "## Metrics",
        f"- Transcript segments: {len(segments)}",
        f"- OCR frame records: {len(ocr_records)}",
        f"- Selected key frames: {len(keyframes)}",
        f"- Real-name mapped segments: {len(named)}",
        f"- Segments without resolved real name: {len(unresolved_name)}",
        f"- Anonymous speaker-label segments: {len(anonymous)}",
        f"- OCR-candidate-only segments: {len(candidate_only)}",
        f"- Unknown-speaker segments: {len(unknown)}",
        "",
        "## Known Limits",
        "- Real names are assigned only from explicit voice enrollment, participant map, or nearby OCR names; unlabeled voice clusters are never promoted to real names.",
        "- If a video-call recording hides name plates and no voice enrollment is provided, speaker labels remain low-confidence until reviewed.",
        "- Pyannote diarization needs `HF_TOKEN` and the optional `diarization` extra; without it, local SpeechBrain ECAPA clustering is the strongest no-token backend.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_queue(path: Path, segments: list[dict[str, Any]]) -> None:
    lines = ["# Review Queue", ""]
    queued = 0
    for segment in segments:
        reasons: list[str] = []
        if str(segment.get("speaker")) == "Speaker Unknown":
            reasons.append("speaker_unknown")
        if not segment.get("name") or float(segment.get("name_confidence", 0.0)) < 0.6:
            reasons.append("name_low_confidence")
        if reasons:
            queued += 1
            candidates = segment.get("name_candidates") or []
            candidate_text = ""
            if candidates:
                candidate_text = " candidates=" + ", ".join(f"{item['name']}({item['count']})" for item in candidates[:3])
            lines.append(f"- `{_evidence(segment)}` {', '.join(reasons)}{candidate_text}: {segment['text']}")
    if queued == 0:
        lines.append("- Empty.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_speaker_samples(path: Path, segments: list[dict[str, Any]], *, max_per_speaker: int = 12) -> None:
    lines = [
        "# Speaker Samples",
        "",
        "Use this file to map diarized labels to reviewed participant names.",
        "Do not apply a mapping unless these sample lines make the voice identity clear.",
        "",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        speaker = str(segment.get("speaker") or "Speaker Unknown")
        if speaker == "Speaker Unknown":
            continue
        text = str(segment.get("text", "")).strip()
        if len(text) < 12:
            continue
        grouped.setdefault(speaker, []).append(segment)
    if not grouped:
        lines.append("- No diarized speaker samples available.")
    for speaker in sorted(grouped):
        lines += [f"## {speaker}", ""]
        samples = sorted(grouped[speaker], key=lambda item: float(item.get("speaker_confidence", 0.0)), reverse=True)
        for segment in samples[:max_per_speaker]:
            confidence = float(segment.get("speaker_confidence", 0.0))
            lines.append(f"- `{_evidence(segment)}` ({confidence:.2f}) {segment['text']}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
