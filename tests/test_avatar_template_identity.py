from __future__ import annotations

from pathlib import Path

import numpy as np

import meeting_minutes.avatar_template_identity as avatar_identity
from meeting_minutes.avatar_template_identity import (
    attach_avatar_template_identity,
    calibrate_avatar_templates,
    score_avatar_template_frames,
)


def _settings(**overrides: object) -> dict:
    values = {
        "minimum_similarity": 0.60,
        "minimum_margin": 0.30,
        "minimum_segment_vote_share": 0.67,
        "short_segment_seconds": 5.0,
        "minimum_validation_frames": 6,
        "minimum_validation_per_identity": 3,
        "minimum_negative_samples": 2,
        "minimum_layout_area_ratio_delta": 0.20,
        "foreground_distance": 0.14,
        "signature_size": 64,
    }
    values.update(overrides)
    return values


def _profile() -> dict:
    return {
        "templates": [
            {"name": "Billy", "path": Path("/tmp/reference-billy.jpg"), "box": (0.0, 0.0, 0.3, 0.3)},
            {"name": "Xin", "path": Path("/tmp/reference-xin.jpg"), "box": (0.3, 0.0, 0.6, 0.3)},
            {"name": "Aleksei", "path": Path("/tmp/reference-aleksei.jpg"), "box": (0.6, 0.0, 0.9, 0.3)},
        ],
        "negative_samples": [
            {"description": "screenshare", "path": Path("/tmp/negative-a.jpg"), "box": (0.0, 0.0, 0.5, 0.5)},
            {"description": "sixth-participant", "path": Path("/tmp/negative-b.jpg"), "box": (0.5, 0.0, 1.0, 0.5)},
        ],
        "settings": _settings(),
    }


def _templates() -> dict:
    return {
        "Billy": {"vector": np.asarray([1.0, 0.0, 0.0]), "path": Path("/tmp/reference-billy.jpg"), "box": (0.0, 0.0, 0.3, 0.3)},
        "Xin": {"vector": np.asarray([0.0, 1.0, 0.0]), "path": Path("/tmp/reference-xin.jpg"), "box": (0.3, 0.0, 0.6, 0.3)},
        "Aleksei": {"vector": np.asarray([0.0, 0.0, 1.0]), "path": Path("/tmp/reference-aleksei.jpg"), "box": (0.6, 0.0, 0.9, 0.3)},
    }


def _frame(path: str, *, name: str | None = None, tile: list[float] | None = None) -> dict:
    frame = {
        "time": float(len(path)),
        "actualTime": float(len(path)),
        "path": path,
        "active_tiles": [{"tile": tile or [0.1, 0.1, 0.7, 0.7], "score": 0.99}],
    }
    if name:
        frame["name"] = name
        frame["name_source"] = "dynamic_visual_in_tile_nameplate_ocr"
    return frame


def _mock_signatures(monkeypatch):
    vectors = {
        "reference-billy.jpg": np.asarray([1.0, 0.0, 0.0]),
        "reference-xin.jpg": np.asarray([0.0, 1.0, 0.0]),
        "reference-aleksei.jpg": np.asarray([0.0, 0.0, 1.0]),
        "billy-1.jpg": np.asarray([1.0, 0.0, 0.0]),
        "billy-2.jpg": np.asarray([1.0, 0.0, 0.0]),
        "billy-3.jpg": np.asarray([1.0, 0.0, 0.0]),
        "xin-1.jpg": np.asarray([0.0, 1.0, 0.0]),
        "xin-2.jpg": np.asarray([0.0, 1.0, 0.0]),
        "xin-3.jpg": np.asarray([0.0, 1.0, 0.0]),
        "active-aleksei.jpg": np.asarray([0.0, 0.0, 1.0]),
        "negative-a.jpg": np.asarray([1.0, 1.0, 1.0]),
        "negative-b.jpg": np.asarray([1.0, 1.0, 1.0]),
    }

    def fake_extract(path: Path, _box: tuple[float, float, float, float], _settings: dict) -> dict:
        return {"vector": vectors[path.name], "avatar_box_in_tile": [0.2, 0.2, 0.8, 0.8]}

    monkeypatch.setattr(avatar_identity, "extract_avatar_signature", fake_extract)


def test_calibration_excludes_reference_frame_and_requires_open_set_negatives(monkeypatch):
    _mock_signatures(monkeypatch)
    profile = _profile()
    frames = [
        _frame("/tmp/reference-billy.jpg", name="Billy"),
        _frame("/tmp/billy-1.jpg", name="Billy"),
        _frame("/tmp/billy-2.jpg", name="Billy"),
        _frame("/tmp/billy-3.jpg", name="Billy"),
        _frame("/tmp/xin-1.jpg", name="Xin"),
        _frame("/tmp/xin-2.jpg", name="Xin"),
        _frame("/tmp/xin-3.jpg", name="Xin"),
    ]

    calibration = calibrate_avatar_templates(frames, profile, _templates())

    assert calibration["gate"]["status"] == "passed"
    assert calibration["reference_frames_excluded_from_validation"] == 1
    assert calibration["eligible_names"] == ["Billy", "Xin"]
    assert calibration["audit_only_names"] == ["Aleksei"]
    assert all(sample["decision"] == "rejected" for sample in calibration["negative_samples"])
    assert calibration["gate"]["requirements"]["layout_diversity_observed"] is True


def test_unvalidated_template_is_audit_only_and_multi_active_abstains(monkeypatch):
    _mock_signatures(monkeypatch)
    profile = _profile()
    calibration = {"gate": {"status": "passed"}, "eligible_names": ["Billy", "Xin"]}
    frames = [
        _frame("/tmp/active-aleksei.jpg"),
        {"time": 2.0, "path": "/tmp/none.jpg", "active_tiles": []},
        {
            "time": 3.0,
            "path": "/tmp/multiple.jpg",
            "active_tiles": [
                {"tile": [0.1, 0.1, 0.4, 0.5], "score": 0.99},
                {"tile": [0.5, 0.1, 0.8, 0.5], "score": 0.99},
            ],
        },
    ]

    scored = score_avatar_template_frames(frames, profile, _templates(), calibration)

    assert scored[0]["avatar_template_identity"]["decision"] == "unvalidated_template"
    assert scored[0]["avatar_template_identity"]["candidate_name"] == "Aleksei"
    assert scored[1]["avatar_template_identity"]["decision"] == "no_active_tile"
    assert scored[2]["avatar_template_identity"]["decision"] == "multiple_active_tiles"


def test_avatar_evidence_does_not_overwrite_direct_nameplate_voiceprint_or_roster_identity():
    profile = {"settings": _settings()}
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1"},
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "Speaker 2",
            "name": "Billy",
            "name_source": "same_session_visual_voiceprint",
        },
        {
            "start": 4.0,
            "end": 5.0,
            "speaker": "Speaker 3",
            "name": "Billy",
            "name_source": "dynamic_visual_in_tile_nameplate_ocr",
        },
        {
            "start": 6.0,
            "end": 7.0,
            "speaker": "Speaker 4",
            "name": "Billy",
            "name_source": "visual_roster_avatar_match",
        },
    ]
    scored_frames = [
        {
            "time": 0.5,
            "actualTime": 0.5,
            "path": "/tmp/billy.jpg",
            "active_tiles": [{"tile": [0.1, 0.1, 0.5, 0.5]}],
            "avatar_template_identity": {"decision": "accepted", "candidate_name": "Billy", "score": 0.90, "margin": 0.50},
        },
        {
            "time": 2.5,
            "actualTime": 2.5,
            "path": "/tmp/xin-voice-conflict.jpg",
            "active_tiles": [{"tile": [0.1, 0.1, 0.5, 0.5]}],
            "avatar_template_identity": {"decision": "accepted", "candidate_name": "Xin", "score": 0.90, "margin": 0.50},
        },
        {
            "time": 4.5,
            "actualTime": 4.5,
            "path": "/tmp/xin-nameplate-conflict.jpg",
            "active_tiles": [{"tile": [0.1, 0.1, 0.5, 0.5]}],
            "avatar_template_identity": {"decision": "accepted", "candidate_name": "Xin", "score": 0.90, "margin": 0.50},
        },
        {
            "time": 6.5,
            "actualTime": 6.5,
            "path": "/tmp/xin-roster-conflict.jpg",
            "active_tiles": [{"tile": [0.1, 0.1, 0.5, 0.5]}],
            "avatar_template_identity": {"decision": "accepted", "candidate_name": "Xin", "score": 0.90, "margin": 0.50},
        },
    ]
    requests = [
        {"segment_index": 0, "video_time": 0.5},
        {"segment_index": 1, "video_time": 2.5},
        {"segment_index": 2, "video_time": 4.5},
        {"segment_index": 3, "video_time": 6.5},
    ]

    summary = attach_avatar_template_identity(segments, requests, scored_frames, profile)

    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "visual_avatar_template_match"
    assert segments[1]["name"] == "Billy"
    assert segments[1]["name_source"] == "same_session_visual_voiceprint"
    assert segments[1]["avatar_template_identity_conflict"]["reason"] == "voiceprint_disagrees_with_avatar_template"
    assert segments[2]["name"] == "Billy"
    assert segments[2]["name_source"] == "dynamic_visual_in_tile_nameplate_ocr"
    assert segments[2]["avatar_template_identity_conflict"]["reason"] == "direct_nameplate_disagrees_with_avatar_template"
    assert segments[3]["name"] == "Billy"
    assert segments[3]["name_source"] == "visual_roster_avatar_match"
    assert segments[3]["avatar_template_identity_conflict"]["reason"] == "roster_avatar_disagrees_with_avatar_template"
    assert summary["voiceprint_conflicts_preserved"] == 1
    assert summary["roster_avatar_conflicts_preserved"] == 1
    assert summary["direct_nameplate_conflicts"] == 1


def test_avatar_evidence_does_not_overwrite_reviewed_or_registered_identity():
    profile = {"settings": _settings()}
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "Speaker 1",
            "name": "Xin",
            "name_source": "participant_map",
            "name_confidence": 0.95,
        },
        {
            "start": 2.0,
            "end": 3.0,
            "speaker": "Speaker 2",
            "name": "Nick Burin",
            "name_source": "voice_registry",
            "name_confidence": 0.51,
        },
        {
            "start": 4.0,
            "end": 5.0,
            "speaker": "Speaker 3",
            "name": "Billy",
            "name_source": "reviewed_participant_map",
        },
    ]
    scored_frames = [
        {
            "time": 0.5,
            "actualTime": 0.5,
            "path": "/tmp/billy-participant-map-conflict.jpg",
            "active_tiles": [{"tile": [0.1, 0.1, 0.5, 0.5]}],
            "avatar_template_identity": {"decision": "accepted", "candidate_name": "Billy", "score": 0.90, "margin": 0.50},
        },
        {
            "time": 2.5,
            "actualTime": 2.5,
            "path": "/tmp/billy-registry-conflict.jpg",
            "active_tiles": [{"tile": [0.1, 0.1, 0.5, 0.5]}],
            "avatar_template_identity": {"decision": "accepted", "candidate_name": "Billy", "score": 0.90, "margin": 0.50},
        },
        {
            "time": 4.5,
            "actualTime": 4.5,
            "path": "/tmp/billy-reviewed-match.jpg",
            "active_tiles": [{"tile": [0.1, 0.1, 0.5, 0.5]}],
            "avatar_template_identity": {"decision": "accepted", "candidate_name": "Billy", "score": 0.90, "margin": 0.50},
        },
    ]
    requests = [
        {"segment_index": 0, "video_time": 0.5},
        {"segment_index": 1, "video_time": 2.5},
        {"segment_index": 2, "video_time": 4.5},
    ]

    summary = attach_avatar_template_identity(segments, requests, scored_frames, profile)

    assert [segment["name"] for segment in segments] == ["Xin", "Nick Burin", "Billy"]
    assert [segment["name_source"] for segment in segments] == [
        "participant_map",
        "voice_registry",
        "reviewed_participant_map",
    ]
    assert segments[0]["avatar_template_identity_conflict"]["reason"] == "reviewed_or_registered_identity_disagrees_with_avatar_template"
    assert segments[1]["avatar_template_identity_conflict"]["reason"] == "reviewed_or_registered_identity_disagrees_with_avatar_template"
    assert segments[2]["avatar_template_identity_corroborates"] == "reviewed_participant_map"
    assert summary["reviewed_identity_conflicts_preserved"] == 2
