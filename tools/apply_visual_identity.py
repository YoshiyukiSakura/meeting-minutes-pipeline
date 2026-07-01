from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from meeting_minutes.jsonio import read_json, write_json
from meeting_minutes.report import (
    write_quality_report,
    write_review_queue,
    write_speaker_samples,
    write_transcript_markdown,
)


def _load_string_map(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return {str(key): str(value) for key, value in payload.items()}


def _candidate_mode(time_seconds: float, layout_switch_seconds: float) -> str:
    # This recording uses the 2x2 grid before screen sharing starts, then the
    # right participant rail. Scores for the inactive layout are noisy.
    return "full" if time_seconds < layout_switch_seconds else "side"


def _load_active_frames(
    scores_path: Path,
    *,
    min_score: float,
    ui_to_person: dict[str, str],
    layout_switch_seconds: float,
) -> list[dict[str, Any]]:
    records = read_json(scores_path)
    active = []
    for record in records:
        time_seconds = float(record["time"])
        if time_seconds < 90:
            continue
        if record.get("mode") != _candidate_mode(time_seconds, layout_switch_seconds):
            continue
        score = float(record.get("best_score", 0.0))
        if score < min_score:
            continue
        ui_name = str(record.get("best", ""))
        person = ui_to_person.get(ui_name)
        if not person:
            continue
        active.append(
            {
                "time": time_seconds,
                "person": person,
                "score": score,
                "path": record.get("path"),
                "mode": record.get("mode"),
                "ui_name": ui_name,
            }
        )
    return active


def _segment_evidence(segment: dict[str, Any], active_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = float(segment["start"])
    end = float(segment["end"])
    return [
        frame
        for frame in active_frames
        if start - 0.75 <= float(frame["time"]) <= end + 0.75
    ]


def _apply_visual_identity(
    segments: list[dict[str, Any]],
    active_frames: list[dict[str, Any]],
    *,
    cluster_fallback: dict[str, str],
    mixed_clusters: list[str],
) -> dict[str, Any]:
    summary = {
        "visual_frames": len(active_frames),
        "visual_frames_by_person": dict(Counter(frame["person"] for frame in active_frames)),
        "assigned_by_source": Counter(),
        "assigned_by_person": Counter(),
        "unresolved_by_speaker": Counter(),
        "cluster_notes": {
            "cluster_fallback": cluster_fallback,
            "mixed_clusters": mixed_clusters,
        },
    }

    for segment in segments:
        segment.pop("visual_identity_evidence", None)
        evidence = _segment_evidence(segment, active_frames)
        visual_name = None
        visual_confidence = 0.0
        visual_refs: list[str] = []
        if evidence:
            votes = Counter(frame["person"] for frame in evidence)
            top_name, top_count = votes.most_common(1)[0]
            if top_count >= max(1, len(evidence) * 0.6):
                visual_name = top_name
                top_scores = [float(frame["score"]) for frame in evidence if frame["person"] == top_name]
                # Border scores are ratios, not calibrated probabilities. Keep
                # the confidence useful for filtering without overstating it.
                visual_confidence = min(0.75, 0.50 + max(top_scores) * 0.50)
                visual_refs = [str(frame["path"]) for frame in evidence if frame.get("path")]
                segment["visual_identity_evidence"] = [
                    {
                        "time": round(float(frame["time"]), 3),
                        "person": frame["person"],
                        "score": round(float(frame["score"]), 3),
                        "frame": frame.get("path"),
                        "ui_name": frame.get("ui_name"),
                    }
                    for frame in evidence
                ]

        if visual_name:
            segment["name"] = visual_name
            segment["name_source"] = "visual_active_speaker_highlight"
            segment["name_confidence"] = round(visual_confidence, 3)
            if visual_refs:
                existing_refs = [str(ref) for ref in segment.get("frame_refs", [])]
                segment["frame_refs"] = sorted(set(existing_refs + visual_refs))[:4]
        elif str(segment.get("speaker")) in cluster_fallback:
            segment["name"] = cluster_fallback[str(segment.get("speaker"))]
            segment["name_source"] = "visual_confirmed_cluster_fallback"
            segment["name_confidence"] = 0.72
        else:
            segment["name"] = None
            segment["name_source"] = "needs_review_visual_or_voice_enrollment"
            segment["name_confidence"] = 0.0

        if segment.get("name") and float(segment.get("name_confidence", 0.0)) >= 0.6:
            summary["assigned_by_source"][str(segment.get("name_source"))] += 1
            summary["assigned_by_person"][str(segment.get("name"))] += 1
        else:
            summary["unresolved_by_speaker"][str(segment.get("speaker") or "Speaker Unknown")] += 1

    summary["assigned_by_source"] = dict(summary["assigned_by_source"])
    summary["assigned_by_person"] = dict(summary["assigned_by_person"])
    summary["unresolved_by_speaker"] = dict(summary["unresolved_by_speaker"])
    return summary


def _write_identity_report(path: Path, summary: dict[str, Any], active_frames: list[dict[str, Any]]) -> None:
    frame_counts = Counter(summary.get("visual_frames_by_person", {}))
    assigned_counts = Counter(summary.get("assigned_by_person", {}))
    source_counts = Counter(summary.get("assigned_by_source", {}))
    fallback_counts: Counter[str] = Counter()
    for person, source in source_counts.items():
        _ = person, source
    for source, count in source_counts.items():
        if source == "visual_confirmed_cluster_fallback":
            fallback_counts["fallback"] = count
    lines = [
        "# Visual Speaker Identity Report",
        "",
        "## Method",
        "- OCR or manual review establishes the visible meeting tiles and maps UI labels to reviewed participant names.",
        "- Active speaker evidence uses the highlighted tile border in either the 2x2 grid or right participant rail.",
        "- Segment-level visual evidence overrides voice clusters.",
        "- Cluster fallback should be used only for clusters repeatedly confirmed by visual evidence.",
        "- Mixed clusters should stay unresolved unless segment-level visual evidence is available.",
        "",
        "## Summary",
        f"- Active visual evidence frames: {summary.get('visual_frames', 0)}",
    ]
    for person, count in frame_counts.most_common():
        lines.append(f"- {person}: {count} active-frame evidence point(s)")
    lines += ["", "## Assigned Segments"]
    for person, count in assigned_counts.most_common():
        lines.append(f"- {person}: {count} segment(s)")
    if fallback_counts:
        lines.append(f"- Cluster fallback: {fallback_counts['fallback']} segment(s); treat these as lower confidence than direct visual evidence.")
    lines += ["", "## Sources"]
    for source, count in source_counts.most_common():
        lines.append(f"- {source}: {count} segment(s)")
    lines += ["", "## Unresolved"]
    unresolved = summary.get("unresolved_by_speaker", {})
    if unresolved:
        for speaker, count in Counter(unresolved).most_common():
            lines.append(f"- {speaker}: {count} segment(s)")
    else:
        lines.append("- Empty.")
    lines += ["", "## Representative Visual Evidence"]
    for frame in active_frames[:80]:
        lines.append(
            f"- `{frame['time']:.2f}s` {frame['person']} score={frame['score']:.3f} `{Path(str(frame['path'])).name}`"
        )
    lines += [
        "",
        "## Notes",
        "- Do not attribute speech to visible participants unless there is segment-level visual evidence, voice enrollment, or reviewed cluster fallback.",
        "- Keep unresolved or mixed-cluster segments in the review queue.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_statuses(output_dir: Path, ocr_records: list[dict[str, Any]], keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    if "asr" not in statuses and (output_dir / "transcript.raw.json").exists():
        statuses["asr"] = {"status": "ok", "source": "transcript.raw.json"}
    if "diarization" not in statuses and (output_dir / "diarization_status.json").exists():
        payload = read_json(output_dir / "diarization_status.json")
        statuses["diarization"] = payload.get("status", payload)
    if "ocr" not in statuses and (output_dir / "ocr.json").exists():
        statuses["ocr"] = {"status": "ok", "frames": len(ocr_records)}
    if "keyframes" not in statuses and (output_dir / "keyframes.json").exists():
        statuses["keyframes"] = {"status": "ok", "frames": len(keyframes)}
    if "summary" not in statuses and (output_dir / "meeting_summary_status.json").exists():
        statuses["summary"] = read_json(output_dir / "meeting_summary_status.json")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--ui-map", type=Path, required=True, help="JSON object mapping visible UI labels to participant names.")
    parser.add_argument("--cluster-fallback", type=Path, help="Optional JSON object mapping diarized speaker labels to reviewed names.")
    parser.add_argument("--mixed-clusters", nargs="*", default=[], help="Speaker labels that must not receive cluster-level names.")
    parser.add_argument("--layout-switch-seconds", type=float, default=225.0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    segments = read_json(output_dir / "transcript.json")
    ui_to_person = _load_string_map(args.ui_map.expanduser().resolve())
    cluster_fallback = _load_string_map(args.cluster_fallback.expanduser().resolve() if args.cluster_fallback else None)
    for speaker in args.mixed_clusters:
        cluster_fallback.pop(str(speaker), None)
    active_frames = _load_active_frames(
        args.scores.expanduser().resolve(),
        min_score=args.min_score,
        ui_to_person=ui_to_person,
        layout_switch_seconds=args.layout_switch_seconds,
    )
    summary = _apply_visual_identity(
        segments,
        active_frames,
        cluster_fallback=cluster_fallback,
        mixed_clusters=[str(speaker) for speaker in args.mixed_clusters],
    )

    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    statuses = _load_statuses(output_dir, ocr_records, keyframes)
    statuses["identity_mapping"] = {
        "status": "partial",
        "reason": "Applied visual active-speaker highlights; unresolved or mixed-cluster segments remain in review.",
    }

    write_json(output_dir / "transcript.json", segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments)
    write_json(output_dir / "visual_identity_summary.json", summary)
    _write_identity_report(output_dir / "speaker_identity_visual_report.md", summary, active_frames)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
