import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from meeting_minutes.agent_visual_audit import (
    AGENT_VISUAL_AUDIT_FORMAT,
    AGENT_VISUAL_AUDIT_VETO_FORMAT,
    agent_visual_audit_schema,
    apply_agent_visual_audit_veto,
    build_agent_visual_audit_veto,
    build_agent_visual_audit_manifest,
    parse_agent_visual_audit_response,
    restore_direct_visual_candidates_from_manifest,
    select_default_same_recording_calibration_frames,
    summarize_agent_visual_audits,
    validate_agent_visual_audit_response,
    validate_agent_visual_audit_manifest_content,
    validate_same_recording_calibration_frames,
    write_agent_visual_audit_bundle,
)
from meeting_minutes.cli import visual_agent_audit_existing
from meeting_minutes.identity_authority import ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
from meeting_minutes.jsonio import read_json, write_json


def _segment(name: str, frame: Path, *, start: float, source: str = ACTIVE_SPEAKER_HIGHLIGHT_SOURCE):
    return {
        "id": f"segment-{start}",
        "start": start,
        "end": start + 2.0,
        "speaker": "Speaker 1",
        "name": name,
        "name_source": source,
        "text": "Visible active-speaker sample.",
        "visual_identity_evidence": [
            {
                "frame": str(frame),
                "name": name,
                "slot": "top_center",
                "score": 0.95,
                "margin": 0.9,
                "reason": "active_named_slot",
            }
        ],
    }


def _manifest(tmp_path: Path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    ignored = tmp_path / "ignored.jpg"
    for path in (first, second, ignored):
        path.write_bytes(b"frame")
    return build_agent_visual_audit_manifest(
        [
            _segment("John", first, start=1.0),
            _segment("John", second, start=10.0),
            _segment("Billy", ignored, start=20.0, source="voice_cluster"),
        ],
        {"slot_names": {"top_center": {"name": "John", "source": "reviewed_slot"}}},
        samples_per_identity=2,
        max_samples=8,
    )


def test_agent_visual_audit_manifest_selects_only_direct_visual_frames(tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest["format"] == AGENT_VISUAL_AUDIT_FORMAT
    assert manifest["coverage"]["direct_named_segments"] == 2
    assert manifest["coverage"]["selected_frames"] == 2
    assert [sample["expected_name"] for sample in manifest["samples"]] == ["John", "John"]
    assert manifest["roster"] == ["John"]
    assert all(sample["frame"].endswith(".jpg") for sample in manifest["samples"])


def test_agent_visual_audit_full_coverage_selects_one_frame_for_every_named_segment(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    third = tmp_path / "third.jpg"
    for path in (first, second, third):
        path.write_bytes(b"frame")
    segments = [
        _segment("John", first, start=1.0),
        _segment("Billy", second, start=10.0),
        _segment("John", third, start=20.0),
    ]

    manifest = build_agent_visual_audit_manifest(
        segments,
        {"slot_names": {"top_center": {"name": "John"}}},
        samples_per_identity=1,
        max_samples=1,
        full_coverage=True,
    )

    assert manifest["coverage"]["selection_mode"] == "full_coverage"
    assert manifest["coverage"]["selected_frames"] == 3
    assert set(manifest["coverage"]["required_direct_segment_ids"]) == {
        segment["id"] for segment in segments
    }
    assert set(manifest["coverage"]["selected_direct_segment_ids"]) == {
        segment["id"] for segment in segments
    }


def test_agent_visual_audit_schema_is_compatible_with_codex_structured_output():
    schema = agent_visual_audit_schema()

    assert schema["type"] == "object"
    assert schema["properties"]["format"] == {
        "type": "string",
        "const": AGENT_VISUAL_AUDIT_FORMAT,
    }


def test_agent_visual_audit_prioritizes_frames_backing_existing_actions(tmp_path):
    action_frame = tmp_path / "action.jpg"
    unrelated_frame = tmp_path / "unrelated.jpg"
    action_frame.write_bytes(b"frame")
    unrelated_frame.write_bytes(b"frame")
    action_segment = _segment("John", action_frame, start=10.0)
    unrelated_segment = _segment("Billy", unrelated_frame, start=20.0)

    manifest = build_agent_visual_audit_manifest(
        [action_segment, unrelated_segment],
        {"slot_names": {"top_center": {"name": "John", "source": "reviewed_slot"}}},
        samples_per_identity=1,
        max_samples=1,
        priority_segment_ids={action_segment["id"]},
    )

    assert manifest["coverage"]["priority_action_frames"] == 1
    assert manifest["samples"][0]["segment_id"] == action_segment["id"]
    assert manifest["samples"][0]["priority"] is True


def test_agent_visual_audit_uses_one_best_priority_frame_per_action_segment(tmp_path):
    first_action_frame = tmp_path / "action-first.jpg"
    best_action_frame = tmp_path / "action-best.jpg"
    other_identity_frame = tmp_path / "other.jpg"
    for frame in (first_action_frame, best_action_frame, other_identity_frame):
        frame.write_bytes(b"frame")
    action_segment = _segment("John", first_action_frame, start=10.0)
    action_segment["visual_identity_evidence"].append(
        {
            "frame": str(best_action_frame),
            "name": "John",
            "slot": "top_center",
            "score": 0.99,
            "margin": 0.98,
            "reason": "active_named_slot",
        }
    )
    other_segment = _segment("Billy", other_identity_frame, start=20.0)

    manifest = build_agent_visual_audit_manifest(
        [action_segment, other_segment],
        {"slot_names": {"top_center": {"name": "John", "source": "reviewed_slot"}}},
        samples_per_identity=1,
        max_samples=3,
        priority_segment_ids={action_segment["id"]},
    )

    priority_samples = [sample for sample in manifest["samples"] if sample["priority"]]
    assert manifest["coverage"]["priority_action_frames"] == 1
    assert len(priority_samples) == 1
    assert priority_samples[0]["frame"] == str(best_action_frame.resolve())
    assert {sample["expected_name"] for sample in manifest["samples"]} == {"Billy", "John"}


def test_calibration_frame_must_belong_to_selected_visual_identity_manifest(tmp_path):
    recorded_frame = tmp_path / "recorded.jpg"
    unrelated_frame = tmp_path / "unrelated.jpg"
    recorded_frame.write_bytes(b"frame")
    unrelated_frame.write_bytes(b"frame")
    visual_identity = {"frames": [{"path": str(recorded_frame)}]}

    assert validate_same_recording_calibration_frames(
        visual_identity,
        [recorded_frame],
    ) == [recorded_frame.resolve()]
    with pytest.raises(ValueError, match="listed by the selected visual_identity artifact"):
        validate_same_recording_calibration_frames(visual_identity, [unrelated_frame])


def test_agent_visual_audit_automatically_selects_one_unambiguous_calibration(
    tmp_path,
):
    calibration = tmp_path / "calibration.jpg"
    weaker = tmp_path / "weaker.jpg"
    active = tmp_path / "active.jpg"
    for frame in (calibration, weaker, active):
        Image.new("RGB", (100, 100), (30, 30, 30)).save(frame)
    segment = _segment("John", active, start=10.0)
    segment["visual_identity_evidence"][0]["layout"] = "grid"
    visual_identity = {
        "profile": str(tmp_path / "profile.json"),
        "slot_names": {"grid::top_center": {"name": "John"}},
        "frames": [
            {
                "path": str(weaker),
                "layout": "grid",
                "reason": "active_named_slot",
                "score": 0.85,
                "margin": 0.75,
                "time": 1.0,
            },
            {
                "path": str(calibration),
                "layout": "grid",
                "reason": "active_named_slot",
                "score": 0.95,
                "margin": 0.9,
                "time": 2.0,
            },
        ],
    }

    assert select_default_same_recording_calibration_frames(visual_identity) == [
        calibration.resolve()
    ]
    paths = write_agent_visual_audit_bundle(
        tmp_path,
        [segment],
        visual_identity,
    )
    manifest = read_json(paths["manifest"])

    assert manifest["coverage"]["calibration_frames"] == 1
    assert manifest["calibrations"][0]["frame"] == str(calibration.resolve())
    assert manifest["calibrations"][0]["layout"] == "grid"


def test_auto_calibration_requires_one_known_layout(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    visual_identity = {
        "slot_names": {
            "grid-a::top_left": {"name": "Alice"},
            "grid-b::top_left": {"name": "Bob"},
        },
        "frames": [
            {
                "path": str(frame),
                "layout": "grid-a",
                "reason": "active_named_slot",
                "score": 0.95,
                "margin": 0.9,
                "time": 1.0,
            }
        ],
    }

    assert select_default_same_recording_calibration_frames(visual_identity) == []
    assert select_default_same_recording_calibration_frames(
        visual_identity,
        layout="grid-a",
    ) == [frame.resolve()]


def test_agent_visual_audit_rejection_veto_clears_derived_identity_only(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    segments = [_segment("John", frame, start=1.0)]
    manifest = build_agent_visual_audit_manifest(
        segments,
        {"slot_names": {"top_center": {"name": "John", "source": "reviewed_slot"}}},
        samples_per_identity=1,
        max_samples=1,
    )
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "cursor",
        "overall_verdict": "needs_review",
        "calibrations": [],
        "samples": [
            {
                "sample_id": manifest["samples"][0]["sample_id"],
                "green_highlight": "not_visible",
                "observed_name": None,
                "verdict": "reject",
            }
        ],
    }
    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="cursor",
    )

    assert errors == []
    assert normalized is not None
    veto = build_agent_visual_audit_veto(manifest, {"cursor": normalized}, segments)
    gated, status = apply_agent_visual_audit_veto(segments, veto)

    assert veto["format"] == AGENT_VISUAL_AUDIT_VETO_FORMAT
    assert veto["status"] == "passed"
    assert status["status"] == "partial_coverage_fail_closed"
    assert status["publishable"] is False
    assert segments[0]["name"] == "John"
    assert gated[0]["name"] is None
    assert gated[0]["name_source"] == "visual_agent_audit_unconfirmed"


def test_agent_visual_audit_allows_publish_after_clearing_explicit_abstention(
    tmp_path,
):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    segments = [
        _segment("John", first, start=1.0),
        _segment("Billy", second, start=10.0),
    ]
    manifest = build_agent_visual_audit_manifest(
        segments,
        {
            "slot_names": {
                "top_center": {"name": "John", "source": "reviewed_slot"},
                "top_left": {"name": "Billy", "source": "reviewed_slot"},
            }
        },
        full_coverage=True,
    )
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "needs_review",
        "calibrations": [],
        "samples": [
            {
                "sample_id": manifest["samples"][0]["sample_id"],
                "green_highlight": "visible",
                "observed_name": manifest["samples"][0]["expected_name"],
                "verdict": "confirm",
            },
            {
                "sample_id": manifest["samples"][1]["sample_id"],
                "green_highlight": "visible",
                "observed_name": None,
                "verdict": "uncertain",
            },
        ],
    }
    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )
    assert errors == []
    assert normalized is not None

    veto = build_agent_visual_audit_veto(
        manifest,
        {"codex": normalized},
        segments,
    )
    gated, status = apply_agent_visual_audit_veto(segments, veto)

    assert veto["coverage_complete"] is True
    assert status["status"] == "passed_with_abstentions"
    assert status["publishable"] is True
    assert sum(segment.get("name") is not None for segment in gated) == 1
    assert sum(
        segment.get("name_source") == "visual_agent_audit_unconfirmed"
        for segment in gated
    ) == 1


def test_agent_visual_audit_stays_current_after_identity_only_enrichment(
    tmp_path,
):
    direct_frame = tmp_path / "direct.jpg"
    anonymous_frame = tmp_path / "anonymous.jpg"
    direct_frame.write_bytes(b"direct")
    anonymous_frame.write_bytes(b"anonymous")
    direct = _segment("John", direct_frame, start=1.0)
    anonymous = _segment("Billy", anonymous_frame, start=10.0)
    anonymous["name"] = None
    anonymous["name_source"] = None
    segments = [direct, anonymous]
    manifest = build_agent_visual_audit_manifest(
        segments,
        {"slot_names": {"top_center": {"name": "John"}}},
        full_coverage=True,
    )
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "pass",
        "calibrations": [],
        "samples": [
            {
                "sample_id": manifest["samples"][0]["sample_id"],
                "green_highlight": "visible",
                "observed_name": "John",
                "verdict": "confirm",
            }
        ],
    }
    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )
    assert errors == []
    assert normalized is not None
    veto = build_agent_visual_audit_veto(
        manifest,
        {"codex": normalized},
        segments,
    )

    enriched = [dict(segment) for segment in segments]
    enriched[1]["name"] = "Billy"
    enriched[1]["name_source"] = "voice_registry"
    _, status = apply_agent_visual_audit_veto(enriched, veto)

    assert status["publishable"] is True

    enriched[1]["text"] = "Different transcript content."
    _, stale_status = apply_agent_visual_audit_veto(enriched, veto)

    assert stale_status["status"] == "stale_transcript_fail_closed"
    assert stale_status["publishable"] is False


def test_agent_visual_audit_restores_candidates_from_content_bound_manifest(
    tmp_path,
):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    source = [
        _segment("John", first, start=1.0),
        _segment("Billy", second, start=10.0),
    ]
    manifest = build_agent_visual_audit_manifest(
        source,
        {
            "slot_names": {
                "top_center": {"name": "John"},
                "top_left": {"name": "Billy"},
            }
        },
        full_coverage=True,
    )
    gated = [dict(segment) for segment in source]
    gated[1]["name"] = None
    gated[1]["name_source"] = "visual_agent_audit_unconfirmed"

    restored, errors = restore_direct_visual_candidates_from_manifest(
        gated,
        manifest,
    )

    assert errors == []
    assert [segment["name"] for segment in restored] == [
        source[0]["name"],
        source[1]["name"],
    ]
    assert all(
        segment["name_source"] == ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
        for segment in restored
    )


def test_agent_visual_audit_writes_upscaled_active_tile_crops(tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (200, 100), (30, 30, 30)).save(frame)
    profile = tmp_path / "visual-profile.json"
    write_json(
        profile,
        {
            "layouts": [
                {
                    "name": "grid",
                    "slots": {"top_center": {"tile": [0.2, 0.1, 0.8, 0.9]}},
                }
            ]
        },
    )
    segment = _segment("John", frame, start=1.0)
    segment["visual_identity_evidence"][0]["layout"] = "grid"

    paths = write_agent_visual_audit_bundle(
        tmp_path,
        [segment],
        {
            "profile": str(profile),
            "slot_names": {"grid::top_center": {"name": "John"}},
        },
    )
    manifest = read_json(paths["manifest"])
    crop = Path(manifest["samples"][0]["inspection_frame"])
    roster_crop = Path(manifest["samples"][0]["roster_inspection_frame"])

    assert crop.is_file()
    assert roster_crop.is_file()
    assert manifest["coverage"]["active_tile_crops_created"] == 1
    assert manifest["coverage"]["same_frame_roster_crops_created"] == 1
    with Image.open(crop) as image:
        assert max(image.size) > 200


def test_agent_visual_audit_writes_unannotated_calibration_tile_contact_sheet(
    tmp_path,
):
    frame = tmp_path / "calibration.jpg"
    Image.new("RGB", (300, 200), (30, 30, 30)).save(frame)
    profile = tmp_path / "visual-profile.json"
    write_json(
        profile,
        {
            "layouts": [
                {
                    "name": "grid",
                    "slots": {
                        "top_left": {"tile": [0.0, 0.0, 0.5, 1.0]},
                        "top_right": {"tile": [0.5, 0.0, 1.0, 1.0]},
                    },
                }
            ]
        },
    )
    segment = _segment("John", frame, start=1.0)
    segment["visual_identity_evidence"][0]["layout"] = "grid"

    paths = write_agent_visual_audit_bundle(
        tmp_path,
        [segment],
        {
            "profile": str(profile),
            "slot_names": {
                "grid::top_left": {"name": "John"},
                "grid::top_right": {"name": "Billy"},
            },
            "frames": [{"path": str(frame)}],
        },
        calibration_frames=[frame],
        calibration_layout="grid",
    )
    manifest = read_json(paths["manifest"])
    contact = Path(manifest["calibrations"][0]["inspection_frame"])

    assert contact.is_file()
    assert manifest["calibrations"][0]["inspection_kind"] == (
        "unannotated_tile_contact_sheet"
    )
    assert manifest["coverage"]["calibration_tile_contacts_created"] == 1


def test_agent_visual_audit_confirmation_requires_visible_matching_name(tmp_path):
    manifest = _manifest(tmp_path)
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "pass",
        "calibrations": [],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "green_highlight": "visible",
                "observed_name": "John",
                "verdict": "confirm",
            }
            for sample in manifest["samples"]
        ],
    }

    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )

    assert errors == []
    assert normalized is not None
    assert all(sample["confirmed"] for sample in normalized["samples"])

    response["samples"][0]["observed_name"] = "Billy"
    rejected, rejection_errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )

    assert rejected is None
    assert rejection_errors == ["unsupported_confirmation:sample-001"]


def test_agent_visual_audit_requires_direct_nameplate_even_after_layout_calibration(tmp_path):
    manifest = _manifest(tmp_path)
    for sample in manifest["samples"]:
        sample["layout"] = "grid"
    manifest["calibrations"] = [
        {
            "calibration_id": "calibration-001",
            "layout": "grid",
            "frame": str(tmp_path / "calibration.jpg"),
            "expected_slot_names": {"top_center": "John"},
        }
    ]
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "cursor",
        "overall_verdict": "pass",
        "calibrations": [
            {
                "calibration_id": "calibration-001",
                "layout": "grid",
                "observed_slot_names": [{"slot": "top_center", "name": "John"}],
                "verdict": "confirm",
            }
        ],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "green_highlight": "visible",
                "observed_name": None,
                "verdict": "confirm",
            }
            for sample in manifest["samples"]
        ],
    }

    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="cursor",
    )

    assert normalized is None
    assert errors == [
        "unsupported_confirmation:sample-001",
        "unsupported_confirmation:sample-002",
    ]

    for sample in response["samples"]:
        sample["observed_name"] = "John"
    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="cursor",
    )

    assert errors == []
    assert normalized is not None
    assert {sample["identity_basis"] for sample in normalized["samples"]} == {
        "same_frame_roster_avatar"
    }

    for sample in response["samples"]:
        sample["identity_basis"] = "calibrated_same_session_slot_avatar"
    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="cursor",
    )

    assert errors == []
    assert normalized is not None
    assert {sample["identity_basis"] for sample in normalized["samples"]} == {
        "calibrated_same_session_slot_avatar"
    }


def test_agent_visual_audit_rejects_calibrated_slot_fallback_without_matching_calibration(
    tmp_path,
):
    manifest = _manifest(tmp_path)
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "pass",
        "calibrations": [],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "green_highlight": "visible",
                "observed_name": "John",
                "identity_basis": "calibrated_same_session_slot_avatar",
                "verdict": "confirm",
            }
            for sample in manifest["samples"]
        ],
    }

    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )

    assert normalized is None
    assert errors == [
        "unsupported_calibrated_identity_basis:sample-001",
        "unsupported_calibrated_identity_basis:sample-002",
        "unsupported_confirmation:sample-001",
        "unsupported_confirmation:sample-002",
    ]


def test_agent_visual_audit_rejects_calibration_when_any_slot_name_is_wrong(tmp_path):
    manifest = _manifest(tmp_path)
    for sample in manifest["samples"]:
        sample["layout"] = "grid"
    manifest["calibrations"] = [
        {
            "calibration_id": "calibration-001",
            "layout": "grid",
            "frame": str(tmp_path / "calibration.jpg"),
            "expected_slot_names": {"top_center": "John"},
        }
    ]
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "needs_review",
        "calibrations": [
            {
                "calibration_id": "calibration-001",
                "layout": "grid",
                "observed_slot_names": [{"slot": "top_center", "name": "Billy"}],
                "verdict": "confirm",
            }
        ],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "green_highlight": "visible",
                "observed_name": "John",
                "verdict": "confirm",
            }
            for sample in manifest["samples"]
        ],
    }

    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )

    assert normalized is None
    assert errors == ["unsupported_calibration_confirmation:calibration-001"]


def test_agent_visual_audit_fail_closes_uncovered_direct_names(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    segments = [
        _segment("John", first, start=1.0),
        _segment("Billy", second, start=10.0),
    ]
    manifest = build_agent_visual_audit_manifest(
        segments,
        {"slot_names": {"top_center": {"name": "John"}}},
        samples_per_identity=1,
        max_samples=1,
    )
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "pass",
        "calibrations": [],
        "samples": [
            {
                "sample_id": manifest["samples"][0]["sample_id"],
                "green_highlight": "visible",
                "observed_name": manifest["samples"][0]["expected_name"],
                "verdict": "confirm",
            }
        ],
    }
    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )

    assert errors == []
    assert normalized is not None
    veto = build_agent_visual_audit_veto(manifest, {"codex": normalized}, segments)
    gated, status = apply_agent_visual_audit_veto(segments, veto)

    assert status["status"] == "partial_coverage_fail_closed"
    assert status["publishable"] is False
    assert sum(segment.get("name") is not None for segment in gated) == 1
    assert any(
        segment.get("name_source") == "visual_agent_audit_unconfirmed"
        for segment in gated
    )


def test_agent_visual_audit_manifest_detects_in_place_frame_mutation(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"first")
    paths = write_agent_visual_audit_bundle(
        tmp_path,
        [_segment("John", frame, start=1.0)],
        {"slot_names": {"top_center": {"name": "John"}}},
    )
    manifest = read_json(paths["manifest"])

    assert validate_agent_visual_audit_manifest_content(manifest) == []
    frame.write_bytes(b"second")

    assert validate_agent_visual_audit_manifest_content(manifest) == [
        "samples:1:frame:content_mismatch"
    ]


def test_agent_visual_audit_summary_requires_every_requested_agent(tmp_path):
    manifest = _manifest(tmp_path)
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "pass",
        "calibrations": [],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "green_highlight": "visible",
                "observed_name": "John",
                "verdict": "confirm",
            }
            for sample in manifest["samples"]
        ],
    }
    codex, errors = validate_agent_visual_audit_response(response, manifest, expected_agent="codex")
    assert errors == []
    cursor_response = {**response, "agent": "cursor"}
    cursor, errors = validate_agent_visual_audit_response(cursor_response, manifest, expected_agent="cursor")
    assert errors == []

    summary = summarize_agent_visual_audits(manifest, {"codex": codex, "cursor": cursor})
    assert summary["status"] == "passed"

    summary = summarize_agent_visual_audits(manifest, {"codex": codex, "cursor": None})
    assert summary["status"] == "needs_review"


def test_agent_visual_audit_summary_accepts_explicit_abstentions(tmp_path):
    manifest = _manifest(tmp_path)
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "needs_review",
        "calibrations": [],
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "green_highlight": "visible",
                "observed_name": None,
                "identity_basis": None,
                "verdict": "uncertain",
            }
            for sample in manifest["samples"]
        ],
    }
    codex, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent="codex",
    )
    assert errors == []

    summary = summarize_agent_visual_audits(manifest, {"codex": codex})

    assert summary["status"] == "passed"
    assert summary["coverage_complete"] is True
    assert summary["confirmed_sample_count"] == 0
    assert summary["abstained_sample_count"] == len(manifest["samples"])


def test_agent_visual_audit_parses_final_json_object(tmp_path):
    manifest = _manifest(tmp_path)
    response = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": "codex",
        "overall_verdict": "needs_review",
        "calibrations": [],
        "samples": [],
    }

    parsed = parse_agent_visual_audit_response(f"tool output\n{response}\n")

    assert parsed is None
    parsed = parse_agent_visual_audit_response("tool output\n" + json.dumps(response) + "\n")
    assert parsed == response
    assert manifest["samples"]


def test_visual_agent_audit_builds_pack_without_running_agents(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    write_json(tmp_path / "transcript.json", [_segment("John", frame, start=1.0)])
    write_json(
        tmp_path / "visual_identity.json",
        {"recording": {}, "slot_names": {"top_center": "John"}},
    )

    result = visual_agent_audit_existing(
        SimpleNamespace(
            output_dir=tmp_path,
            visual_identity_path=None,
            samples_per_identity=2,
            max_samples=24,
            run_agent=[],
            workspace=tmp_path,
            codex_bin="codex",
            cursor_bin="cursor-agent",
            cursor_model="cursor-grok-4.5-high",
            timeout=1,
            require_consensus=False,
        )
    )

    assert result == 0
    audit = read_json(tmp_path / "agent_visual_audit.json")
    assert audit["status"] == "pack_ready"
    assert (tmp_path / "work" / "agent_visual_audit" / "request.json").is_file()
