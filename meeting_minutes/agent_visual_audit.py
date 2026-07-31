from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from .action_items import transcript_fingerprint
from .identity_authority import ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
from .jsonio import write_json

AGENT_VISUAL_AUDIT_FORMAT = "meeting-minutes/agent-visual-audit-v1"
AGENT_VISUAL_AUDIT_VETO_FORMAT = "meeting-minutes/agent-visual-audit-veto-v1"
DEFAULT_SAMPLES_PER_IDENTITY = 2
DEFAULT_MAX_SAMPLES = 24
_AUDITABLE_VERDICTS = frozenset({"confirm", "reject", "uncertain"})
_AUDITABLE_HIGHLIGHT_STATES = frozenset({"visible", "not_visible", "uncertain"})
_AUDITABLE_IDENTITY_BASES = frozenset(
    {"same_frame_roster_avatar", "calibrated_same_session_slot_avatar"}
)


def _plain(value: object) -> str:
    return str(value or "").strip()


def _segment_id(segment: dict[str, Any], index: int) -> str:
    return _plain(segment.get("segment_id") or segment.get("id")) or f"segment-{index:05d}"


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def transcript_content_fingerprint(segments: list[dict[str, Any]]) -> str:
    """Bind an audit to transcript content while allowing identity enrichment."""

    payload = [
        {
            "id": _segment_id(segment, index),
            "start": round(float(segment.get("start", 0.0)), 3),
            "end": round(float(segment.get("end", 0.0)), 3),
            "speaker": _plain(segment.get("speaker")),
            "text": _plain(segment.get("text")),
        }
        for index, segment in enumerate(segments, start=1)
    ]
    return _fingerprint(payload)


def restore_direct_visual_candidates_from_manifest(
    segments: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Restore previously audited candidates without changing transcript content."""

    copied = deepcopy(segments)
    segment_by_id = {
        _segment_id(segment, index): segment
        for index, segment in enumerate(copied, start=1)
    }
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        segment_id = _plain(sample.get("segment_id"))
        expected_name = _plain(sample.get("expected_name"))
        if not segment_id or not expected_name:
            errors.append("manifest_candidate_invalid")
            continue
        prior = candidates.get(segment_id)
        if prior is not None and _plain(prior.get("expected_name")).casefold() != expected_name.casefold():
            errors.append(f"manifest_candidate_name_conflict:{segment_id}")
            continue
        candidates[segment_id] = sample
    for segment_id, sample in candidates.items():
        segment = segment_by_id.get(segment_id)
        if segment is None:
            errors.append(f"manifest_candidate_segment_missing:{segment_id}")
            continue
        if (
            abs(float(segment.get("start", 0.0)) - float(sample.get("start", 0.0))) > 0.05
            or abs(float(segment.get("end", 0.0)) - float(sample.get("end", 0.0))) > 0.05
        ):
            errors.append(f"manifest_candidate_timeline_mismatch:{segment_id}")
            continue
        segment["name"] = _plain(sample.get("expected_name"))
        segment["name_source"] = ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
        segment["name_confidence"] = max(
            _number(segment.get("name_confidence")),
            _number(sample.get("highlight_score")),
        )
    required_ids = {
        _plain(segment_id)
        for segment_id in (
            manifest.get("coverage", {}).get("required_direct_segment_ids")
            or []
        )
        if _plain(segment_id)
    }
    if required_ids != set(candidates):
        errors.append("manifest_candidate_coverage_mismatch")
    return copied, sorted(set(errors))


def _slot_person(value: object) -> str:
    if isinstance(value, dict):
        return _plain(value.get("name"))
    return _plain(value)


def _choose_spread(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not samples:
        return []
    if len(samples) <= count:
        return samples
    indexes = {
        round(position * (len(samples) - 1) / max(1, count - 1))
        for position in range(count)
    }
    selected = [samples[index] for index in sorted(indexes)]
    if len(selected) < count:
        for sample in samples:
            if sample not in selected:
                selected.append(sample)
            if len(selected) == count:
                break
    return selected


def _fingerprint(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bind_manifest_frame_hashes(manifest: dict[str, Any]) -> None:
    """Bind every agent-visible image to its bytes, not merely its path."""

    for collection in ("calibrations", "samples"):
        entries = manifest.get(collection)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for path_key, hash_key in (
                ("frame", "frame_sha256"),
                ("inspection_frame", "inspection_frame_sha256"),
                ("roster_inspection_frame", "roster_inspection_frame_sha256"),
            ):
                raw_path = _plain(entry.get(path_key))
                if not raw_path:
                    entry.pop(hash_key, None)
                    continue
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise ValueError(f"Agent audit image is missing: {path}")
                entry[hash_key] = _file_sha256(path)


def validate_agent_visual_audit_manifest_content(manifest: dict[str, Any]) -> list[str]:
    """Return deterministic errors when an audited image changed in place."""

    errors: list[str] = []
    claimed_fingerprint = _plain(manifest.get("manifest_sha256"))
    fingerprint_payload = deepcopy(manifest)
    fingerprint_payload.pop("manifest_sha256", None)
    if not claimed_fingerprint or _fingerprint(fingerprint_payload) != claimed_fingerprint:
        errors.append("manifest_sha256_mismatch")
    for collection in ("calibrations", "samples"):
        entries = manifest.get(collection)
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                errors.append(f"{collection}:{index}:not_object")
                continue
            for path_key, hash_key in (
                ("frame", "frame_sha256"),
                ("inspection_frame", "inspection_frame_sha256"),
                ("roster_inspection_frame", "roster_inspection_frame_sha256"),
            ):
                raw_path = _plain(entry.get(path_key))
                expected = _plain(entry.get(hash_key))
                if not raw_path:
                    continue
                path = Path(raw_path).expanduser().resolve()
                if not expected:
                    errors.append(f"{collection}:{index}:{hash_key}:missing")
                    continue
                if not path.is_file():
                    errors.append(f"{collection}:{index}:{path_key}:missing")
                    continue
                if _file_sha256(path) != expected:
                    errors.append(f"{collection}:{index}:{path_key}:content_mismatch")
    return errors


def _known_visual_frame_paths(visual_identity: dict[str, Any]) -> set[Path]:
    """Return frames emitted by this exact visual-identity artifact.

    A calibration frame is valid only when it was produced by the visual pass
    whose recording provenance is embedded in ``visual_identity``. Requiring
    membership in this manifest prevents an arbitrary screenshot from another
    call being used to validate a static slot map.
    """

    paths: set[Path] = set()
    frames = visual_identity.get("frames")
    if not isinstance(frames, list):
        return paths
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        raw_path = _plain(frame.get("path"))
        if raw_path:
            paths.add(Path(raw_path).expanduser().resolve())
    return paths


def validate_same_recording_calibration_frames(
    visual_identity: dict[str, Any],
    calibration_frames: list[Path] | None,
) -> list[Path]:
    """Reject calibration screenshots not emitted by this recording's visual pass."""

    requested = [frame.expanduser().resolve() for frame in calibration_frames or []]
    if not requested:
        return []
    known_frames = _known_visual_frame_paths(visual_identity)
    if not known_frames:
        raise ValueError(
            "Calibration requires visual_identity.frames with same-recording frame provenance"
        )
    invalid = [frame for frame in requested if not frame.is_file() or frame not in known_frames]
    if invalid:
        rendered = ", ".join(str(frame) for frame in invalid)
        raise ValueError(
            "Calibration frames must be existing frames listed by the selected "
            f"visual_identity artifact: {rendered}"
        )
    return requested


def select_default_same_recording_calibration_frames(
    visual_identity: dict[str, Any],
    *,
    layout: str | None = None,
) -> list[Path]:
    """Choose one auditable full-frame calibration only when unambiguous.

    This is an input-selection convenience, not identity evidence. The local
    vision agent must still confirm that visible nameplates match the static
    layout before any cropped active-speaker tile can pass the audit.
    """

    slot_names_raw = visual_identity.get("slot_names")
    if not isinstance(slot_names_raw, dict):
        return []
    known_layouts = {
        _plain(key).split("::", 1)[0]
        for key, value in slot_names_raw.items()
        if "::" in _plain(key) and _slot_person(value)
    }
    selected_layout = _plain(layout)
    if selected_layout:
        if selected_layout not in known_layouts:
            return []
    elif len(known_layouts) == 1:
        selected_layout = next(iter(known_layouts))
    else:
        return []

    candidates: list[tuple[float, float, float, str, Path]] = []
    frames = visual_identity.get("frames")
    if not isinstance(frames, list):
        return []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if _plain(frame.get("layout")) != selected_layout:
            continue
        if _plain(frame.get("reason")) != "active_named_slot":
            continue
        path_raw = _plain(frame.get("path"))
        if not path_raw:
            continue
        path = Path(path_raw).expanduser().resolve()
        if not path.is_file():
            continue
        candidates.append(
            (
                -_number(frame.get("score")),
                -_number(frame.get("margin")),
                _number(frame.get("time")),
                str(path),
                path,
            )
        )
    if not candidates:
        return []
    candidates.sort()
    return [candidates[0][-1]]


def _static_tile_boxes(visual_identity: dict[str, Any]) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    profile_path = Path(_plain(visual_identity.get("profile"))).expanduser()
    if not profile_path.is_file():
        return {}
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    layouts = profile.get("layouts") if isinstance(profile, dict) else None
    if not isinstance(layouts, list):
        return {}
    boxes: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for layout in layouts:
        if not isinstance(layout, dict):
            continue
        layout_name = _plain(layout.get("name"))
        slots = layout.get("slots")
        if not layout_name or not isinstance(slots, dict):
            continue
        for slot_name, slot in slots.items():
            if not isinstance(slot, dict):
                continue
            raw_box = slot.get("tile")
            if not isinstance(raw_box, list) or len(raw_box) != 4:
                continue
            try:
                left, top, right, bottom = (float(value) for value in raw_box)
            except (TypeError, ValueError):
                continue
            if 0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0:
                boxes[(layout_name, _plain(slot_name))] = (left, top, right, bottom)
    return boxes


def _write_active_tile_crops(
    manifest: dict[str, Any],
    visual_identity: dict[str, Any],
    audit_dir: Path,
) -> int:
    boxes = _static_tile_boxes(visual_identity)
    if not boxes:
        return 0
    crop_dir = audit_dir / "tile_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        frame = Path(_plain(sample.get("frame"))).expanduser()
        box = boxes.get((_plain(sample.get("layout")), _plain(sample.get("expected_slot"))))
        if not frame.is_file() or box is None:
            continue
        try:
            with Image.open(frame) as image:
                width, height = image.size
                left, top, right, bottom = box
                margin_x = (right - left) * 0.035
                # Discord keeps the display name immediately below the green
                # tile border. Preserve substantially more lower context so a
                # reviewer can see both the active cue and its nameplate.
                margin_top_y = (bottom - top) * 0.07
                margin_bottom_y = (bottom - top) * 0.32
                crop_box = (
                    max(0, round((left - margin_x) * width)),
                    max(0, round((top - margin_top_y) * height)),
                    min(width, round((right + margin_x) * width)),
                    min(height, round((bottom + margin_bottom_y) * height)),
                )
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    continue
                crop = image.convert("RGB").crop(crop_box)
                largest_side = max(crop.size)
                scale = min(4, max(1, 900 // max(1, largest_side)))
                if scale > 1:
                    crop = crop.resize(
                        (crop.width * scale, crop.height * scale),
                        Image.Resampling.LANCZOS,
                    )
                crop_path = crop_dir / f"{_plain(sample.get('sample_id'))}.png"
                crop.save(crop_path, format="PNG", optimize=True)
        except (OSError, ValueError):
            continue
        sample["inspection_frame"] = str(crop_path)
        sample["inspection_kind"] = "active_tile_crop"
        created += 1
    return created


def _write_same_frame_roster_crops(
    manifest: dict[str, Any],
    visual_identity: dict[str, Any],
    audit_dir: Path,
) -> int:
    """Extract the voice-member roster beside a static Discord grid.

    Discord call tiles do not render a textual nameplate. The same source frame
    does render a voice roster containing each participant's avatar and name,
    so a reviewer can match the highlighted tile avatar to the roster without
    relying on a persistent tile position or an earlier recording.
    """

    boxes = _static_tile_boxes(visual_identity)
    if not boxes:
        return 0
    roster_dir = audit_dir / "roster_crops"
    roster_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for sample in manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        layout = _plain(sample.get("layout"))
        frame = Path(_plain(sample.get("frame"))).expanduser()
        layout_boxes = [box for (box_layout, _slot), box in boxes.items() if box_layout == layout]
        if not frame.is_file() or not layout_boxes:
            continue
        left_edge = min(box[0] for box in layout_boxes)
        top_edge = min(box[1] for box in layout_boxes)
        bottom_edge = max(box[3] for box in layout_boxes)
        roster_left = max(0.0, left_edge - max(0.23, left_edge * 0.8))
        roster_right = max(roster_left + 0.04, left_edge - 0.006)
        try:
            with Image.open(frame) as image:
                source = image.convert("RGB")
                width, height = source.size
                crop_box = (
                    round(roster_left * width),
                    round(max(0.0, top_edge - 0.03) * height),
                    round(min(1.0, roster_right) * width),
                    round(min(1.0, bottom_edge + 0.02) * height),
                )
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    continue
                crop = source.crop(crop_box)
                largest_side = max(crop.size)
                scale = min(3, max(1, 1200 // max(1, largest_side)))
                if scale > 1:
                    crop = crop.resize(
                        (crop.width * scale, crop.height * scale),
                        Image.Resampling.LANCZOS,
                    )
                crop_path = roster_dir / f"{_plain(sample.get('sample_id'))}.png"
                crop.save(crop_path, format="PNG", optimize=True)
        except (OSError, ValueError):
            continue
        sample["roster_inspection_frame"] = str(crop_path)
        sample["roster_inspection_kind"] = "same_frame_voice_roster_crop"
        created += 1
    return created


def _write_calibration_tile_contacts(
    manifest: dict[str, Any],
    visual_identity: dict[str, Any],
    audit_dir: Path,
) -> int:
    """Create unannotated enlarged tile contact sheets for calibration review."""

    boxes = _static_tile_boxes(visual_identity)
    if not boxes:
        return 0
    contact_dir = audit_dir / "calibration_tile_contacts"
    contact_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for calibration in manifest.get("calibrations", []):
        if not isinstance(calibration, dict):
            continue
        layout = _plain(calibration.get("layout"))
        frame = Path(_plain(calibration.get("frame"))).expanduser()
        if not layout or not frame.is_file():
            continue
        slots = sorted(
            (
                (slot_name, box)
                for (box_layout, slot_name), box in boxes.items()
                if box_layout == layout
            ),
            key=lambda item: (item[1][1], item[1][0], item[0]),
        )
        if not slots:
            continue
        try:
            with Image.open(frame) as image:
                source = image.convert("RGB")
                tiles: list[Image.Image] = []
                for _slot_name, (left, top, right, bottom) in slots:
                    width, height = source.size
                    margin_x = (right - left) * 0.035
                    margin_y = (bottom - top) * 0.07
                    crop_box = (
                        max(0, round((left - margin_x) * width)),
                        max(0, round((top - margin_y) * height)),
                        min(width, round((right + margin_x) * width)),
                        min(height, round((bottom + margin_y) * height)),
                    )
                    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                        continue
                    tile = source.crop(crop_box)
                    tile.thumbnail((480, 300), Image.Resampling.LANCZOS)
                    tiles.append(tile)
                if len(tiles) != len(slots):
                    continue
                columns = min(3, len(tiles))
                cell_width = max(tile.width for tile in tiles)
                cell_height = max(tile.height for tile in tiles)
                rows = (len(tiles) + columns - 1) // columns
                contact = Image.new(
                    "RGB",
                    (cell_width * columns, cell_height * rows),
                    (0, 0, 0),
                )
                for index, tile in enumerate(tiles):
                    left = (index % columns) * cell_width
                    top = (index // columns) * cell_height
                    contact.paste(tile, (left, top))
                contact_path = contact_dir / (
                    f"{_plain(calibration.get('calibration_id'))}.png"
                )
                contact.save(contact_path, format="PNG", optimize=True)
        except (OSError, ValueError):
            continue
        calibration["inspection_frame"] = str(contact_path)
        calibration["inspection_kind"] = "unannotated_tile_contact_sheet"
        calibration["inspection_slots"] = [slot_name for slot_name, _box in slots]
        created += 1
    return created


def build_agent_visual_audit_manifest(
    segments: list[dict[str, Any]],
    visual_identity: dict[str, Any],
    *,
    samples_per_identity: int = DEFAULT_SAMPLES_PER_IDENTITY,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    full_coverage: bool = False,
    priority_segment_ids: set[str] | None = None,
    calibration_frames: list[Path] | None = None,
    calibration_layout: str | None = None,
) -> dict[str, Any]:
    """Select direct visual assignments that a vision-capable coding agent can audit.

    This is intentionally a calibration and veto surface. It never invents a name,
    and it does not sample anonymous, conflict, or voice-only identity records.
    """

    if samples_per_identity < 1:
        raise ValueError("samples_per_identity must be positive")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    slot_names_raw = visual_identity.get("slot_names")
    slot_names = {
        _plain(slot): _slot_person(name)
        for slot, name in slot_names_raw.items()
        if _plain(slot) and _slot_person(name)
    } if isinstance(slot_names_raw, dict) else {}

    priority_ids = {_plain(segment_id) for segment_id in priority_segment_ids or set() if _plain(segment_id)}
    direct_segments = 0
    unavailable_frames = 0
    required_direct_segment_ids: list[str] = []
    candidates_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_frames: set[tuple[str, str, str]] = set()
    for index, segment in enumerate(segments, start=1):
        expected_name = _plain(segment.get("name"))
        if (
            _plain(segment.get("name_source")) != ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
            or not expected_name
        ):
            continue
        direct_segments += 1
        segment_id = _segment_id(segment, index)
        required_direct_segment_ids.append(segment_id)
        evidence_list = segment.get("visual_identity_evidence")
        if not isinstance(evidence_list, list):
            continue
        for evidence in evidence_list:
            if not isinstance(evidence, dict):
                continue
            if _plain(evidence.get("reason")) != "active_named_slot":
                continue
            if _plain(evidence.get("name")) != expected_name:
                continue
            frame_raw = _plain(evidence.get("frame"))
            if not frame_raw:
                continue
            frame = Path(frame_raw).expanduser().resolve()
            if not frame.is_file():
                unavailable_frames += 1
                continue
            frame_key = (
                expected_name.casefold(),
                str(frame),
                segment_id if full_coverage else "",
            )
            if frame_key in seen_frames:
                continue
            seen_frames.add(frame_key)
            candidate = {
                "sample_id": "",
                "segment_id": segment_id,
                "start": round(_number(segment.get("start")), 3),
                "end": round(_number(segment.get("end")), 3),
                "expected_name": expected_name,
                "layout": _plain(evidence.get("layout")) or None,
                "expected_slot": _plain(evidence.get("slot")) or None,
                "frame": str(frame),
                "highlight_score": round(_number(evidence.get("score")), 4),
                "highlight_margin": round(_number(evidence.get("margin")), 4),
                "priority": segment_id in priority_ids,
            }
            candidates_by_name[expected_name].append(candidate)
            candidates_by_segment[segment_id].append(candidate)

    baseline_samples: list[dict[str, Any]] = []
    for name in sorted(candidates_by_name, key=str.casefold):
        ordered = sorted(
            candidates_by_name[name],
            key=lambda sample: (sample["start"], sample["end"], sample["frame"]),
        )
        baseline_samples.extend(_choose_spread(ordered, samples_per_identity))
    all_candidates = [
        sample
        for name in sorted(candidates_by_name, key=str.casefold)
        for sample in sorted(
            candidates_by_name[name],
            key=lambda candidate: (candidate["start"], candidate["end"], candidate["frame"]),
        )
    ]
    priority_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in all_candidates:
        if sample["priority"]:
            priority_by_segment[_plain(sample["segment_id"])].append(sample)
    priority_samples = [
        max(
            candidates,
            key=lambda sample: (
                sample["highlight_score"],
                sample["highlight_margin"],
                -sample["start"],
                sample["frame"],
            ),
        )
        for _segment_id, candidates in sorted(priority_by_segment.items())
        if candidates
    ]
    priority_samples.sort(key=lambda sample: (sample["start"], sample["end"], sample["frame"]))
    selected_keys = {
        (sample["expected_name"].casefold(), sample["frame"])
        for sample in priority_samples
    }
    if full_coverage:
        selected = [
            max(
                candidates,
                key=lambda sample: (
                    sample["highlight_score"],
                    sample["highlight_margin"],
                    -sample["start"],
                    sample["frame"],
                ),
            )
            for segment_id, candidates in sorted(candidates_by_segment.items())
            if candidates
        ]
        selected.sort(key=lambda sample: (sample["start"], sample["end"], sample["frame"]))
    else:
        selected = [*priority_samples, *[
            sample
            for sample in baseline_samples
            if (sample["expected_name"].casefold(), sample["frame"]) not in selected_keys
        ]]
        selected = selected[:max_samples]
    for index, sample in enumerate(selected, start=1):
        sample["sample_id"] = f"sample-{index:03d}"

    layouts = sorted(
        {
            _plain(sample.get("layout"))
            for sample in selected
            if _plain(sample.get("layout"))
        }
    )
    selected_layout = _plain(calibration_layout)
    if not selected_layout and len(layouts) == 1:
        selected_layout = layouts[0]
    calibrations: list[dict[str, Any]] = []
    if selected_layout:
        expected_slot_names = {
            key.split("::", 1)[1]: name
            for key, name in slot_names.items()
            if key.startswith(f"{selected_layout}::")
        }
        for index, raw_frame in enumerate(calibration_frames or [], start=1):
            frame = raw_frame.expanduser().resolve()
            if not frame.is_file() or not expected_slot_names:
                continue
            calibrations.append(
                {
                    "calibration_id": f"calibration-{index:03d}",
                    "layout": selected_layout,
                    "frame": str(frame),
                    "expected_slot_names": expected_slot_names,
                }
            )

    coverage = {
        "direct_named_segments": direct_segments,
        "required_direct_segment_ids": sorted(set(required_direct_segment_ids)),
        "selected_direct_segment_ids": sorted(
            {_plain(sample.get("segment_id")) for sample in selected if _plain(sample.get("segment_id"))}
        ),
        "candidate_frames": sum(len(samples) for samples in candidates_by_name.values()),
        "selected_frames": len(selected),
        "unavailable_evidence_frames": unavailable_frames,
        "selection_mode": "full_coverage" if full_coverage else "sampled",
        "selected_by_name": {
            name: sum(1 for sample in selected if sample["expected_name"] == name)
            for name in sorted(candidates_by_name, key=str.casefold)
        },
        "calibration_frames": len(calibrations),
        "priority_action_frames": sum(1 for sample in selected if sample["priority"]),
    }
    manifest = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "identity_source": ACTIVE_SPEAKER_HIGHLIGHT_SOURCE,
        "recording": visual_identity.get("recording") if isinstance(visual_identity.get("recording"), dict) else {},
        "slot_names": slot_names,
        "roster": sorted({*slot_names.values(), *(sample["expected_name"] for sample in selected)}, key=str.casefold),
        "coverage": coverage,
        "calibrations": calibrations,
        "samples": selected,
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    return manifest


def agent_visual_audit_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["format", "agent", "overall_verdict", "calibrations", "samples"],
        "properties": {
            "format": {
                "type": "string",
                "const": AGENT_VISUAL_AUDIT_FORMAT,
            },
            "agent": {"type": "string", "enum": ["codex", "cursor"]},
            "overall_verdict": {"type": "string", "enum": ["pass", "needs_review"]},
            "calibrations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "calibration_id",
                        "layout",
                        "observed_slot_names",
                        "verdict",
                    ],
                    "properties": {
                        "calibration_id": {"type": "string"},
                        "layout": {"type": "string"},
                        "observed_slot_names": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["slot", "name"],
                                "properties": {
                                    "slot": {"type": "string"},
                                    "name": {"type": "string"},
                                },
                            },
                        },
                        "verdict": {"type": "string", "enum": sorted(_AUDITABLE_VERDICTS)},
                    },
                },
            },
            "samples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sample_id", "green_highlight", "observed_name", "verdict"],
                    "properties": {
                        "sample_id": {"type": "string"},
                        "green_highlight": {
                            "type": "string",
                            "enum": sorted(_AUDITABLE_HIGHLIGHT_STATES),
                        },
                        "observed_name": {"type": ["string", "null"]},
                        "identity_basis": {
                            "type": ["string", "null"],
                            "enum": [*sorted(_AUDITABLE_IDENTITY_BASES), None],
                        },
                        "verdict": {"type": "string", "enum": sorted(_AUDITABLE_VERDICTS)},
                    },
                },
            },
        },
    }


def render_agent_visual_audit_prompt(manifest: dict[str, Any], *, agent: str) -> str:
    calibrations = [
        {
            "calibration_id": calibration["calibration_id"],
            "layout": calibration["layout"],
            "frame": calibration["frame"],
            "inspection_frame": calibration.get("inspection_frame"),
            "inspection_slots": calibration.get("inspection_slots", []),
            "expected_slot_names": calibration["expected_slot_names"],
        }
        for calibration in manifest.get("calibrations", [])
        if isinstance(calibration, dict)
    ]
    samples = [
        {
            "sample_id": sample["sample_id"],
            "active_tile_frame": sample.get("inspection_frame") or sample["frame"],
            "same_frame_roster_frame": sample.get("roster_inspection_frame"),
            "source_frame": sample["frame"],
            "timestamp_seconds": sample["start"],
            "claimed_name": sample["expected_name"],
            "claimed_slot": sample["expected_slot"],
        }
        for sample in manifest.get("samples", [])
        if isinstance(sample, dict)
    ]
    request = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": agent,
        "roster": manifest.get("roster", []),
        "calibrations": calibrations,
        "samples": samples,
        "rules": [
            "逐张查看提供的本地截图。active_tile_frame 是放大的高亮 tile 裁图；same_frame_roster_frame 是同一 source_frame 的语音成员列表裁图；source_frame 保留原始整帧出处。",
            "先审计 calibrations。只有当校准整帧内可见姓名与 expected_slot_names 的静态格位完全一致时，该 calibration 的 verdict 才能是 confirm。",
            "每个 calibration 都必须返回 observed_slot_names，逐项列出画面中直接可见的 slot 和 name；缺任一预期 slot、同一 slot 重复或姓名不一致都不能 confirm。",
            "若 calibration 有 inspection_frame，它是从同一原始整帧裁出的、未加文字标注的 tile 放大拼图；inspection_slots 给出拼图的阅读顺序。用它逐格核对姓名，并以 frame 确认这是同一完整布局。",
            "Discord tile 不显示文字姓名。sample 优先使用 same_frame_roster_avatar：绿色边框清晰包围 claimed_slot；active_tile_frame 的头像可与 same_frame_roster_frame 中头像旁的 observed_name 一一匹配；observed_name 与 claimed_name 一致。",
            "若当前成员列表被通话栏遮挡或 claimed_name 位于裁图可视区域之外，可使用 calibrated_same_session_slot_avatar 作为受限回退，但必须同时满足：对应 calibration 已完整确认全部 slot/name；当前 source_frame 的宫格布局和 claimed_slot 与 calibration 完全一致；绿色边框清晰包围 claimed_slot；active_tile_frame 的头像与 calibration 同一 slot 的头像明显一致；当前帧不存在冲突姓名或头像证据。",
            "声音、转写文本、其他会议或未经完整确认的静态布局不能作为 sample 实名依据。",
            "看不清姓名或高亮时使用 uncertain，不得猜测。",
            "confirm 时 observed_name 必须等于 claimed_name，并填写 identity_basis 为 same_frame_roster_avatar 或 calibrated_same_session_slot_avatar。看不清则 observed_name 为 null、identity_basis 为 null，verdict 必须是 uncertain 或 reject。",
            "不得修改文件、不得访问网络、不得根据当前静态布局推断姓名。",
            "返回一个 JSON 对象，且 samples 必须恰好覆盖所有 sample_id，不要使用 Markdown。",
        ],
    }
    return json.dumps(request, ensure_ascii=False, separators=(",", ":"))


def write_agent_visual_audit_bundle(
    output_dir: Path,
    segments: list[dict[str, Any]],
    visual_identity: dict[str, Any],
    *,
    samples_per_identity: int = DEFAULT_SAMPLES_PER_IDENTITY,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    full_coverage: bool = False,
    priority_segment_ids: set[str] | None = None,
    calibration_frames: list[Path] | None = None,
    calibration_layout: str | None = None,
) -> dict[str, Path]:
    audit_dir = output_dir / "work" / "agent_visual_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    verified_calibration_frames = validate_same_recording_calibration_frames(
        visual_identity,
        calibration_frames,
    )
    if not verified_calibration_frames:
        verified_calibration_frames = select_default_same_recording_calibration_frames(
            visual_identity,
            layout=calibration_layout,
        )
    manifest = build_agent_visual_audit_manifest(
        segments,
        visual_identity,
        samples_per_identity=samples_per_identity,
        max_samples=max_samples,
        full_coverage=full_coverage,
        priority_segment_ids=priority_segment_ids,
        calibration_frames=verified_calibration_frames,
        calibration_layout=calibration_layout,
    )
    manifest["coverage"]["active_tile_crops_created"] = _write_active_tile_crops(
        manifest,
        visual_identity,
        audit_dir,
    )
    manifest["coverage"]["same_frame_roster_crops_created"] = _write_same_frame_roster_crops(
        manifest,
        visual_identity,
        audit_dir,
    )
    manifest["coverage"]["calibration_tile_contacts_created"] = (
        _write_calibration_tile_contacts(
            manifest,
            visual_identity,
            audit_dir,
        )
    )
    _bind_manifest_frame_hashes(manifest)
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _fingerprint(manifest)
    manifest_path = audit_dir / "request.json"
    schema_path = audit_dir / "response.schema.json"
    codex_prompt_path = audit_dir / "codex.prompt.json"
    cursor_prompt_path = audit_dir / "cursor.prompt.json"
    write_json(manifest_path, manifest)
    write_json(schema_path, agent_visual_audit_schema())
    codex_prompt_path.write_text(render_agent_visual_audit_prompt(manifest, agent="codex") + "\n", encoding="utf-8")
    cursor_prompt_path.write_text(render_agent_visual_audit_prompt(manifest, agent="cursor") + "\n", encoding="utf-8")
    return {
        "directory": audit_dir,
        "manifest": manifest_path,
        "schema": schema_path,
        "codex_prompt": codex_prompt_path,
        "cursor_prompt": cursor_prompt_path,
    }


def parse_agent_visual_audit_response(text: str) -> dict[str, Any] | None:
    """Find the final JSON object in an agent response without trusting prose."""

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    for value in reversed(candidates):
        if isinstance(value.get("samples"), list):
            return value
    return None


def validate_agent_visual_audit_response(
    response: dict[str, Any] | None,
    manifest: dict[str, Any],
    *,
    expected_agent: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(response, dict):
        return None, ["response_not_json_object"]
    if response.get("format") != AGENT_VISUAL_AUDIT_FORMAT:
        return None, ["format_mismatch"]
    if _plain(response.get("agent")) != expected_agent:
        return None, ["agent_mismatch"]
    if _plain(response.get("overall_verdict")) not in {"pass", "needs_review"}:
        return None, ["overall_verdict_invalid"]

    requested_calibrations = {
        _plain(calibration.get("calibration_id")): calibration
        for calibration in manifest.get("calibrations", [])
        if isinstance(calibration, dict) and _plain(calibration.get("calibration_id"))
    }
    raw_calibrations = response.get("calibrations")
    if not isinstance(raw_calibrations, list):
        return None, ["calibrations_not_list"]
    observed_calibrations: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for raw in raw_calibrations:
        if not isinstance(raw, dict):
            errors.append("calibration_not_object")
            continue
        calibration_id = _plain(raw.get("calibration_id"))
        expected = requested_calibrations.get(calibration_id)
        if expected is None:
            errors.append(f"unexpected_calibration:{calibration_id or 'empty'}")
            continue
        if calibration_id in observed_calibrations:
            errors.append(f"duplicate_calibration:{calibration_id}")
            continue
        layout = _plain(raw.get("layout"))
        verdict = _plain(raw.get("verdict"))
        raw_slot_names = raw.get("observed_slot_names")
        observed_slot_names: dict[str, str] = {}
        if layout != _plain(expected.get("layout")):
            errors.append(f"calibration_layout_mismatch:{calibration_id}")
        if verdict not in _AUDITABLE_VERDICTS:
            errors.append(f"invalid_calibration_verdict:{calibration_id}")
        if not isinstance(raw_slot_names, list):
            errors.append(f"calibration_slot_names_not_list:{calibration_id}")
        else:
            for item in raw_slot_names:
                if not isinstance(item, dict):
                    errors.append(f"calibration_slot_name_not_object:{calibration_id}")
                    continue
                slot = _plain(item.get("slot"))
                name = _plain(item.get("name"))
                if not slot or not name:
                    errors.append(f"calibration_slot_name_invalid:{calibration_id}")
                    continue
                if slot in observed_slot_names:
                    errors.append(f"duplicate_calibration_slot:{calibration_id}:{slot}")
                    continue
                observed_slot_names[slot] = name
        expected_slot_names = {
            _plain(slot): _plain(name)
            for slot, name in (expected.get("expected_slot_names") or {}).items()
            if _plain(slot) and _plain(name)
        }
        names_match = (
            set(observed_slot_names) == set(expected_slot_names)
            and all(
                observed_slot_names[slot].casefold() == expected_slot_names[slot].casefold()
                for slot in expected_slot_names
            )
        )
        if verdict == "confirm" and not names_match:
            errors.append(f"unsupported_calibration_confirmation:{calibration_id}")
        observed_calibrations[calibration_id] = {
            "calibration_id": calibration_id,
            "layout": layout,
            "observed_slot_names": [
                {"slot": slot, "name": observed_slot_names[slot]}
                for slot in sorted(observed_slot_names)
            ],
            "verdict": verdict,
            "confirmed": (
                verdict == "confirm"
                and layout == _plain(expected.get("layout"))
                and names_match
            ),
        }
    for calibration_id in requested_calibrations:
        if calibration_id not in observed_calibrations:
            errors.append(f"missing_calibration:{calibration_id}")
    if errors:
        return None, sorted(set(errors))
    requested = {
        _plain(sample.get("sample_id")): sample
        for sample in manifest.get("samples", [])
        if isinstance(sample, dict) and _plain(sample.get("sample_id"))
    }
    raw_samples = response.get("samples")
    if not isinstance(raw_samples, list):
        return None, ["samples_not_list"]
    observed: dict[str, dict[str, Any]] = {}
    for raw in raw_samples:
        if not isinstance(raw, dict):
            errors.append("sample_not_object")
            continue
        sample_id = _plain(raw.get("sample_id"))
        if sample_id not in requested:
            errors.append(f"unexpected_sample:{sample_id or 'empty'}")
            continue
        if sample_id in observed:
            errors.append(f"duplicate_sample:{sample_id}")
            continue
        highlight = _plain(raw.get("green_highlight"))
        verdict = _plain(raw.get("verdict"))
        observed_name = raw.get("observed_name")
        raw_identity_basis = raw.get("identity_basis")
        if highlight not in _AUDITABLE_HIGHLIGHT_STATES:
            errors.append(f"invalid_highlight:{sample_id}")
        if verdict not in _AUDITABLE_VERDICTS:
            errors.append(f"invalid_verdict:{sample_id}")
        if observed_name is not None and not isinstance(observed_name, str):
            errors.append(f"invalid_observed_name:{sample_id}")
        if raw_identity_basis is not None and not isinstance(raw_identity_basis, str):
            errors.append(f"invalid_identity_basis:{sample_id}")
        observed_name_text = _plain(observed_name)
        identity_basis = _plain(raw_identity_basis)
        if not identity_basis and verdict == "confirm":
            # Responses produced by the earlier, stricter policy could only
            # confirm through a same-frame roster match.
            identity_basis = "same_frame_roster_avatar"
        if identity_basis and identity_basis not in _AUDITABLE_IDENTITY_BASES:
            errors.append(f"invalid_identity_basis:{sample_id}")
        expected_name = _plain(requested[sample_id].get("expected_name"))
        direct_name_support = observed_name_text.casefold() == expected_name.casefold()
        calibrated_layout_support = any(
            calibration.get("confirmed") is True
            and _plain(calibration.get("layout"))
            == _plain(requested[sample_id].get("layout"))
            for calibration in observed_calibrations.values()
        )
        if (
            identity_basis == "calibrated_same_session_slot_avatar"
            and not calibrated_layout_support
        ):
            errors.append(f"unsupported_calibrated_identity_basis:{sample_id}")
        confirmed = (
            verdict == "confirm"
            and highlight == "visible"
            and direct_name_support
            and identity_basis in _AUDITABLE_IDENTITY_BASES
            and (
                identity_basis != "calibrated_same_session_slot_avatar"
                or calibrated_layout_support
            )
        )
        if verdict == "confirm" and not confirmed:
            errors.append(f"unsupported_confirmation:{sample_id}")
        observed[sample_id] = {
            "sample_id": sample_id,
            "green_highlight": highlight,
            "observed_name": observed_name_text or None,
            "verdict": verdict,
            "identity_basis": identity_basis if confirmed else None,
            "confirmed": confirmed,
        }
    for sample_id in requested:
        if sample_id not in observed:
            errors.append(f"missing_sample:{sample_id}")
    if errors:
        return None, sorted(set(errors))
    normalized = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": expected_agent,
        "overall_verdict": response["overall_verdict"],
        "manifest_sha256": _plain(manifest.get("manifest_sha256")),
        "calibrations": [
            observed_calibrations[calibration_id]
            for calibration_id in sorted(observed_calibrations)
        ],
        "samples": [observed[sample_id] for sample_id in sorted(observed)],
    }
    return normalized, []


def summarize_agent_visual_audits(
    manifest: dict[str, Any],
    agent_results: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    sample_by_id = {
        _plain(sample.get("sample_id")): sample
        for sample in manifest.get("samples", [])
        if isinstance(sample, dict) and _plain(sample.get("sample_id"))
    }
    sample_ids = [
        sample_id for sample_id in sample_by_id
    ]
    requested_agents = sorted(agent_results)
    calibration_ids = [
        _plain(calibration.get("calibration_id"))
        for calibration in manifest.get("calibrations", [])
        if isinstance(calibration, dict) and _plain(calibration.get("calibration_id"))
    ]
    calibrations: list[dict[str, Any]] = []
    for calibration_id in calibration_ids:
        entries: list[dict[str, Any]] = []
        for agent in requested_agents:
            result = agent_results.get(agent)
            match = next(
                (
                    calibration
                    for calibration in (result or {}).get("calibrations", [])
                    if calibration.get("calibration_id") == calibration_id
                ),
                None,
            )
            if isinstance(match, dict):
                entries.append({"agent": agent, **match})
            else:
                entries.append({"agent": agent, "verdict": "missing", "confirmed": False})
        confirmations = sum(1 for entry in entries if entry.get("confirmed") is True)
        calibrations.append(
            {
                "calibration_id": calibration_id,
                "confirmations": confirmations,
                "required_confirmations": len(requested_agents),
                "status": (
                    "confirmed"
                    if requested_agents and confirmations == len(requested_agents)
                    else "needs_review"
                ),
                "agents": entries,
            }
        )
    samples: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        entries = []
        for agent in requested_agents:
            result = agent_results.get(agent)
            match = next(
                (
                    sample
                    for sample in (result or {}).get("samples", [])
                    if sample.get("sample_id") == sample_id
                ),
                None,
            )
            if isinstance(match, dict):
                entries.append({"agent": agent, **match})
            else:
                entries.append({"agent": agent, "verdict": "missing", "confirmed": False})
        confirmations = sum(1 for entry in entries if entry.get("confirmed") is True)
        fully_audited = bool(requested_agents) and all(
            _plain(entry.get("verdict")) in _AUDITABLE_VERDICTS
            for entry in entries
        )
        samples.append(
            {
                "sample_id": sample_id,
                "confirmations": confirmations,
                "required_confirmations": len(requested_agents),
                "status": (
                    "confirmed"
                    if requested_agents and confirmations == len(requested_agents)
                    else "abstained"
                    if fully_audited
                    else "missing"
                ),
                "agents": entries,
            }
        )
    required_direct_ids = {
        _plain(segment_id)
        for segment_id in (manifest.get("coverage", {}).get("required_direct_segment_ids") or [])
        if _plain(segment_id)
    }
    confirmed_direct_ids = {
        _plain(sample_by_id[sample["sample_id"]].get("segment_id"))
        for sample in samples
        if sample.get("status") == "confirmed"
        and _plain(sample_by_id[sample["sample_id"]].get("segment_id"))
    }
    audited_direct_ids = {
        _plain(sample_by_id[sample["sample_id"]].get("segment_id"))
        for sample in samples
        if sample.get("status") in {"confirmed", "abstained"}
        and _plain(sample_by_id[sample["sample_id"]].get("segment_id"))
    }
    abstained_direct_ids = audited_direct_ids - confirmed_direct_ids
    uncovered_direct_ids = sorted(required_direct_ids - audited_direct_ids)
    calibration_passed = all(item["status"] == "confirmed" for item in calibrations)
    passed = (
        bool(sample_ids and requested_agents)
        and calibration_passed
        and all(sample["status"] in {"confirmed", "abstained"} for sample in samples)
        and not uncovered_direct_ids
    )
    return {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "manifest_sha256": _plain(manifest.get("manifest_sha256")),
        "requested_agents": requested_agents,
        "status": "passed" if passed else "needs_review",
        "calibrations": calibrations,
        "samples": samples,
        "coverage_complete": not uncovered_direct_ids and bool(required_direct_ids),
        "audited_direct_segment_ids": sorted(audited_direct_ids),
        "confirmed_direct_segment_ids": sorted(confirmed_direct_ids),
        "abstained_direct_segment_ids": sorted(abstained_direct_ids),
        "uncovered_direct_segment_ids": uncovered_direct_ids,
        "confirmed_sample_count": sum(
            sample["status"] == "confirmed" for sample in samples
        ),
        "abstained_sample_count": sum(
            sample["status"] == "abstained" for sample in samples
        ),
        "guardrail": "Agent results may veto a direct visual assignment but never create or propagate a real-name assignment.",
    }


def build_agent_visual_audit_veto(
    manifest: dict[str, Any],
    agent_results: dict[str, dict[str, Any] | None],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a fail-closed allowlist for direct active-speaker identities.

    Agents can never create a name. They can only preserve a name that already
    exists in a direct visual segment after every requested agent confirms the
    same highlighted tile and its same-frame roster-avatar or inline-name match. Every unreviewed,
    uncertain, rejected, partial, or stale candidate is cleared downstream.
    """

    sample_by_id = {
        _plain(sample.get("sample_id")): sample
        for sample in manifest.get("samples", [])
        if isinstance(sample, dict) and _plain(sample.get("sample_id"))
    }
    required_direct_ids = {
        _segment_id(segment, index)
        for index, segment in enumerate(segments, start=1)
        if (
            _plain(segment.get("name_source")) == ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
            and _plain(segment.get("name"))
        )
    }
    expected_required_ids = {
        _plain(segment_id)
        for segment_id in (manifest.get("coverage", {}).get("required_direct_segment_ids") or [])
        if _plain(segment_id)
    }
    requested_agents = sorted(agent_results)
    vetoed_segments: dict[str, list[dict[str, str]]] = defaultdict(list)
    vetoed_layouts: dict[str, list[dict[str, str]]] = defaultdict(list)

    calibration_ids = [
        _plain(calibration.get("calibration_id"))
        for calibration in manifest.get("calibrations", [])
        if isinstance(calibration, dict) and _plain(calibration.get("calibration_id"))
    ]
    calibrations_confirmed = True
    for calibration_id in calibration_ids:
        for agent in requested_agents:
            result = agent_results.get(agent)
            calibration = next(
                (
                    item
                    for item in (result or {}).get("calibrations", [])
                    if isinstance(item, dict) and _plain(item.get("calibration_id")) == calibration_id
                ),
                None,
            )
            if not isinstance(calibration, dict) or calibration.get("confirmed") is not True:
                calibrations_confirmed = False

    confirmed_segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audited_segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for agent, result in sorted(agent_results.items()):
        if not isinstance(result, dict):
            continue
        if _plain(result.get("manifest_sha256")) != _plain(manifest.get("manifest_sha256")):
            continue
        for calibration in result.get("calibrations", []):
            if not isinstance(calibration, dict) or _plain(calibration.get("verdict")) != "reject":
                continue
            layout = _plain(calibration.get("layout"))
            if layout:
                vetoed_layouts[layout].append(
                    {
                        "agent": agent,
                        "calibration_id": _plain(calibration.get("calibration_id")),
                        "reason": "agent_rejected_calibration",
                    }
                )
        for reviewed_sample in result.get("samples", []):
            if not isinstance(reviewed_sample, dict) or _plain(reviewed_sample.get("verdict")) != "reject":
                continue
            sample = sample_by_id.get(_plain(reviewed_sample.get("sample_id")))
            if not isinstance(sample, dict):
                continue
            segment_id = _plain(sample.get("segment_id"))
            if segment_id:
                vetoed_segments[segment_id].append(
                    {
                        "agent": agent,
                        "sample_id": _plain(reviewed_sample.get("sample_id")),
                        "reason": "agent_rejected_active_tile",
                    }
                )

    if requested_agents and calibrations_confirmed:
        for sample_id, sample in sample_by_id.items():
            if not isinstance(sample, dict):
                continue
            confirmations: list[dict[str, str]] = []
            dispositions: list[dict[str, str]] = []
            for agent in requested_agents:
                result = agent_results.get(agent)
                reviewed = next(
                    (
                        item
                        for item in (result or {}).get("samples", [])
                        if isinstance(item, dict) and _plain(item.get("sample_id")) == sample_id
                    ),
                    None,
                )
                if (
                    not isinstance(reviewed, dict)
                    or _plain(reviewed.get("verdict")) not in _AUDITABLE_VERDICTS
                ):
                    dispositions = []
                    confirmations = []
                    break
                dispositions.append(
                    {
                        "agent": agent,
                        "sample_id": sample_id,
                        "verdict": _plain(reviewed.get("verdict")),
                    }
                )
                if reviewed.get("confirmed") is not True:
                    confirmations = []
                    continue
                confirmations.append(
                    {
                        "agent": agent,
                        "sample_id": sample_id,
                        "reason": "agent_confirmed_same_frame_visual_identity",
                    }
                )
            segment_id = _plain(sample.get("segment_id"))
            if dispositions and len(dispositions) == len(requested_agents) and segment_id:
                audited_segments[segment_id].extend(dispositions)
            if (
                len(confirmations) == len(requested_agents)
                and requested_agents
                and segment_id
            ):
                confirmed_segments[segment_id].extend(confirmations)

    for segment_id in sorted(set(audited_segments) - set(confirmed_segments)):
        vetoed_segments[segment_id].append(
            {
                "agent": "gate",
                "sample_id": "",
                "reason": "agent_abstained_visual_identity",
            }
        )
    for segment_id in sorted(required_direct_ids - set(audited_segments)):
        vetoed_segments[segment_id].append(
            {
                "agent": "gate",
                "sample_id": "",
                "reason": "missing_all_agent_disposition",
            }
        )
    if expected_required_ids != required_direct_ids:
        for segment_id in sorted(required_direct_ids):
            vetoed_segments[segment_id].append(
                {
                    "agent": "gate",
                    "sample_id": "",
                    "reason": "manifest_transcript_direct_identity_set_mismatch",
                }
            )

    coverage_complete = (
        bool(required_direct_ids)
        and expected_required_ids == required_direct_ids
        and required_direct_ids == set(audited_segments)
        and calibrations_confirmed
        and not vetoed_layouts
    )

    return {
        "format": AGENT_VISUAL_AUDIT_VETO_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "transcript_content_sha256": transcript_content_fingerprint(segments),
        "manifest_sha256": _plain(manifest.get("manifest_sha256")),
        "status": "passed" if coverage_complete else "needs_review",
        "requested_agents": requested_agents,
        "calibrations_confirmed": calibrations_confirmed,
        "coverage_complete": coverage_complete,
        "required_direct_segment_ids": sorted(required_direct_ids),
        "audited_segment_ids": {
            segment_id: reasons
            for segment_id, reasons in sorted(audited_segments.items())
        },
        "confirmed_segment_ids": {
            segment_id: reasons
            for segment_id, reasons in sorted(confirmed_segments.items())
        },
        "vetoed_segment_ids": {
            segment_id: reasons
            for segment_id, reasons in sorted(vetoed_segments.items())
        },
        "vetoed_layouts": {
            layout: reasons
            for layout, reasons in sorted(vetoed_layouts.items())
        },
        "guardrail": (
            "This overlay can only clear an existing direct active-speaker visual "
            "identity. It never adds, propagates, or replaces a real name."
        ),
    }


def apply_agent_visual_audit_veto(
    segments: list[dict[str, Any]],
    veto: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a derived transcript with unconfirmed visual names cleared."""

    copied = deepcopy(segments)

    def clear_all_direct(reason: str) -> tuple[list[dict[str, Any]], int]:
        cleared = 0
        for index, segment in enumerate(copied, start=1):
            if _plain(segment.get("name_source")) != ACTIVE_SPEAKER_HIGHLIGHT_SOURCE:
                continue
            segment["name"] = None
            segment["name_confidence"] = 0.0
            segment["name_source"] = "visual_agent_audit_unconfirmed"
            segment["visual_identity_agent_audit"] = {
                "status": "unconfirmed",
                "reason": reason,
                "segment_id": _segment_id(segment, index),
            }
            cleared += 1
        return copied, cleared

    has_direct_names = any(
        _plain(segment.get("name_source")) == ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
        and _plain(segment.get("name"))
        for segment in copied
    )
    if not has_direct_names:
        return copied, {"status": "not_required", "cleared_segments": 0, "publishable": True}
    if not isinstance(veto, dict) or veto.get("format") != AGENT_VISUAL_AUDIT_VETO_FORMAT:
        copied, cleared = clear_all_direct("audit_not_available")
        return copied, {"status": "not_available_fail_closed", "cleared_segments": cleared, "publishable": False}
    expected_content_sha256 = _plain(veto.get("transcript_content_sha256"))
    transcript_is_current = (
        expected_content_sha256 == transcript_content_fingerprint(segments)
        if expected_content_sha256
        else _plain(veto.get("transcript_sha256"))
        == transcript_fingerprint(segments)
    )
    if not transcript_is_current:
        copied, cleared = clear_all_direct("audit_transcript_content_stale")
        return copied, {
            "status": "stale_transcript_fail_closed",
            "cleared_segments": cleared,
            "publishable": False,
        }
    raw_segments = veto.get("vetoed_segment_ids")
    raw_layouts = veto.get("vetoed_layouts")
    raw_confirmed = veto.get("confirmed_segment_ids")
    vetoed_segment_ids = raw_segments if isinstance(raw_segments, dict) else {}
    vetoed_layouts = raw_layouts if isinstance(raw_layouts, dict) else {}
    confirmed_segment_ids = raw_confirmed if isinstance(raw_confirmed, dict) else {}
    cleared_segments = 0
    confirmed_segments = 0
    for index, segment in enumerate(copied, start=1):
        if _plain(segment.get("name_source")) != ACTIVE_SPEAKER_HIGHLIGHT_SOURCE:
            continue
        segment_id = _segment_id(segment, index)
        reasons: list[dict[str, Any]] = []
        direct_reasons = vetoed_segment_ids.get(segment_id)
        if isinstance(direct_reasons, list):
            reasons.extend(reason for reason in direct_reasons if isinstance(reason, dict))
        evidence = segment.get("visual_identity_evidence")
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                layout_reasons = vetoed_layouts.get(_plain(item.get("layout")))
                if isinstance(layout_reasons, list):
                    reasons.extend(reason for reason in layout_reasons if isinstance(reason, dict))
        confirmations = confirmed_segment_ids.get(segment_id)
        if isinstance(confirmations, list) and confirmations and not reasons:
            segment["visual_identity_agent_audit"] = {
                "status": "confirmed",
                "manifest_sha256": _plain(veto.get("manifest_sha256")),
                "confirmations": confirmations,
            }
            confirmed_segments += 1
            continue
        segment["name"] = None
        segment["name_confidence"] = 0.0
        segment["name_source"] = "visual_agent_audit_unconfirmed"
        segment["visual_identity_agent_audit"] = {
            "status": "unconfirmed",
            "manifest_sha256": _plain(veto.get("manifest_sha256")),
            "reasons": reasons or [
                {
                    "agent": "gate",
                    "sample_id": "",
                    "reason": "missing_all_agent_same_frame_confirmation",
                }
            ],
        }
        cleared_segments += 1
    publishable = bool(veto.get("coverage_complete")) and confirmed_segments > 0
    return copied, {
        "status": (
            "passed_with_abstentions"
            if publishable and cleared_segments
            else "passed"
            if publishable
            else "partial_coverage_fail_closed"
        ),
        "cleared_segments": cleared_segments,
        "confirmed_segments": confirmed_segments,
        "vetoed_segment_count": len(vetoed_segment_ids),
        "vetoed_layout_count": len(vetoed_layouts),
        "publishable": publishable,
    }


def write_agent_visual_audit_report(
    path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# 本地代理视觉审计",
        "",
        f"- 结果：`{summary.get('status', 'needs_review')}`",
        f"- 审计代理：{', '.join(summary.get('requested_agents', [])) or '未运行'}",
        f"- 直接高亮具名片段：{manifest.get('coverage', {}).get('direct_named_segments', 0)}",
        f"- 代理抽样帧：{manifest.get('coverage', {}).get('selected_frames', 0)}",
        "- 规则：代理只能否决已有的同帧高亮实名证据，不能通过头像、座位、声音或历史信息创建实名映射。",
        "",
        "## 抽样结果",
        "",
        "| 样本 | 确认数 | 要求数 | 结果 |",
        "| --- | ---: | ---: | --- |",
    ]
    for sample in summary.get("samples", []):
        if not isinstance(sample, dict):
            continue
        lines.append(
            "| {sample_id} | {confirmations} | {required} | {status} |".format(
                sample_id=_plain(sample.get("sample_id")),
                confirmations=sample.get("confirmations", 0),
                required=sample.get("required_confirmations", 0),
                status=_plain(sample.get("status")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
