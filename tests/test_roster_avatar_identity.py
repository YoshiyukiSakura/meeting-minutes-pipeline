from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import meeting_minutes.direct_visual_cluster_identity as cluster_identity
import meeting_minutes.roster_avatar_identity as roster_identity
import meeting_minutes.visual_voice_identity as voice_identity
from meeting_minutes.roster_avatar_identity import (
    SOURCE,
    attach_roster_avatar_identity,
    calibrate_roster_avatar_identity,
    load_roster_avatar_profile,
    score_roster_avatar_frames,
)


def _profile(tmp_path: Path) -> dict:
    path = tmp_path / "roster-profile.json"
    path.write_text(
        json.dumps(
            {
                "participants": ["Billy", "John", "Xin"],
                "settings": {
                    "minimum_similarity": 0.70,
                    "minimum_margin": 0.15,
                    "minimum_ocr_confidence": 0.80,
                    "minimum_supporting_frames": 2,
                    "minimum_roster_identities": 3,
                    "minimum_reviewed_anchors": 3,
                    "minimum_anchor_identities": 3,
                    "minimum_anchor_seconds_separation": 10,
                },
                "layouts": [
                    {
                        "name": "discord-left-roster",
                        "start": 0,
                        "end": 100,
                        "roster_region": [0.0, 0.0, 0.4, 1.0],
                        "avatar": {"size_multiplier": 1.45, "gap_multiplier": 0.24},
                    }
                ],
                "reviewed_anchors": [
                    {"time": 10, "name": "Billy", "reviewed": True},
                    {"time": 30, "name": "John", "reviewed": True},
                    {"time": 50, "name": "Xin", "reviewed": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    return load_roster_avatar_profile(path)


def _frame(tmp_path: Path, name: str = "frame.jpg") -> Path:
    path = tmp_path / name
    Image.new("RGB", (640, 360), "black").save(path)
    return path


def _row(name: str, vector: list[float]) -> dict:
    return {
        "name": name,
        "text": name,
        "ocr_confidence": 0.99,
        "text_box": [0.2, 0.2, 0.3, 0.3],
        "avatar_box": [0.05, 0.2, 0.15, 0.3],
        "avatar_stddev": 0.2,
        "avatar_entropy": 0.7,
        "_vector": np.asarray(vector, dtype=np.float32),
    }


def _detected(path: Path, time: float = 10.0) -> list[dict]:
    return [
        {
            "time": time,
            "actualTime": time,
            "path": str(path),
            "layout": "discord-left-roster",
            "active_tiles": [{"tile": [0.45, 0.1, 0.95, 0.9], "score": 0.99}],
            "reason": "single_active_tile",
        }
    ]


def _patch_vectors(monkeypatch, *, rows: list[dict], active: list[float]) -> None:
    monkeypatch.setattr(roster_identity, "_roster_rows", lambda *_args: (rows, []))
    monkeypatch.setattr(
        roster_identity,
        "_active_avatar",
        lambda *_args: (
            {
                "vector": np.asarray(active, dtype=np.float32),
                "avatar_box_in_tile": [0.3, 0.3, 0.7, 0.7],
                "avatar_area_ratio": 0.16,
                "background_stddev": 0.05,
            },
            None,
        ),
    )


def test_roster_name_matching_is_exact_and_does_not_merge_similar_usernames(tmp_path):
    profile = _profile(tmp_path)
    profile["participants"] = ["John", "johnjr0507", "Xin"]
    profile["participant_by_key"] = {name.casefold(): name for name in profile["participants"]}
    pixels = np.zeros((360, 640, 3), dtype=np.uint8)
    pixels[135:190, 20:130, 0] = np.arange(110, dtype=np.uint8)
    pixels[135:190, 20:130, 1] = np.arange(55, dtype=np.uint8)[:, None]
    pixels[135:190, 20:130, 2] = 180
    image = Image.fromarray(pixels, "RGB")
    rows, rejected = roster_identity._roster_rows(
        image,
        [
            {
                "text": "johnjr0507",
                "confidence": 0.99,
                "bbox": {"x": 0.5, "y": 0.5, "width": 0.3, "height": 0.1},
            }
        ],
        profile,
        profile["layouts"][0],
    )
    assert [row["name"] for row in rows] == ["johnjr0507"]
    assert not rejected or all(item.get("name") != "John" for item in rejected)


def test_nested_highlight_candidates_keep_only_the_strongest_tile(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    path = _frame(tmp_path)
    monkeypatch.setattr(
        roster_identity,
        "detect_active_tiles",
        lambda *_args: [
            {"tile": [0.45, 0.10, 0.95, 0.90], "score": 0.95},
            {"tile": [0.46, 0.12, 0.94, 0.89], "score": 0.99},
        ],
    )

    detected = roster_identity.detect_roster_active_frames(
        [{"time": 10.0, "actualTime": 10.0, "path": str(path)}],
        profile,
    )

    assert detected[0]["reason"] == "single_active_tile"
    assert detected[0]["active_tiles"] == [{"tile": [0.46, 0.12, 0.94, 0.89], "score": 0.99}]


def test_distinct_highlight_candidates_remain_ambiguous(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    path = _frame(tmp_path)
    monkeypatch.setattr(
        roster_identity,
        "detect_active_tiles",
        lambda *_args: [
            {"tile": [0.45, 0.10, 0.65, 0.40], "score": 0.99},
            {"tile": [0.70, 0.45, 0.95, 0.90], "score": 0.98},
        ],
    )

    detected = roster_identity.detect_roster_active_frames(
        [{"time": 10.0, "actualTime": 10.0, "path": str(path)}],
        profile,
    )

    assert detected[0]["reason"] == "multiple_active_tiles"
    assert len(detected[0]["active_tiles"]) == 2


def test_layout_windows_are_half_open_at_the_boundary(tmp_path):
    profile = _profile(tmp_path)
    first, second = profile["layouts"][0], dict(profile["layouts"][0])
    first["name"] = "first"
    first["end"] = 50.0
    second["name"] = "second"
    second["start"] = 50.0
    second["end"] = 100.0
    profile["layouts"] = [first, second]

    assert roster_identity.layout_at(profile, 49.999)["name"] == "first"
    assert roster_identity.layout_at(profile, 50.0)["name"] == "second"
    assert roster_identity.layout_at(profile, 100.0) is None


def test_roster_samples_avoid_the_exact_duration_boundary(tmp_path):
    profile = _profile(tmp_path)
    profile["reviewed_anchors"] = [{"time": 100.0, "name": "Billy"}]

    segment_requests = roster_identity.build_roster_sample_requests(
        [{"start": 99.8, "end": 100.0, "speaker": "Speaker 1"}],
        profile,
        duration=100.0,
    )
    anchor_requests = roster_identity.build_reviewed_anchor_requests(profile, duration=100.0)

    assert all(request["video_time"] < 100.0 for request in segment_requests)
    assert anchor_requests[0]["video_time"] == 99.999


def test_active_avatar_rejects_default_and_camera_like_tiles(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    profile["settings"]["maximum_tile_background_stddev"] = 0.04
    monkeypatch.setattr(
        roster_identity,
        "extract_avatar_signature_from_image",
        lambda *_args: {"vector": np.asarray([1.0]), "avatar_box_in_tile": [0.3, 0.3, 0.7, 0.7]},
    )
    flat = Image.new("RGB", (640, 360), "black")
    active, reason = roster_identity._active_avatar(flat, (0.2, 0.2, 0.8, 0.8), profile)
    assert active is None
    assert reason == "active_tile_default_or_low_variance_avatar"

    random_pixels = np.random.default_rng(7).integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
    textured = Image.fromarray(random_pixels, "RGB")
    active, reason = roster_identity._active_avatar(textured, (0.2, 0.2, 0.8, 0.8), profile)
    assert active is None
    assert reason == "active_tile_not_avatar_like"

    monkeypatch.setattr(
        roster_identity,
        "extract_avatar_signature_from_image",
        lambda *_args: {"vector": np.asarray([1.0]), "avatar_box_in_tile": [0.01, 0.01, 0.99, 0.99]},
    )
    active, reason = roster_identity._active_avatar(flat, (0.2, 0.2, 0.8, 0.8), profile)
    assert active is None
    assert reason == "active_tile_not_avatar_like"


def test_score_requires_three_distinct_roster_avatars(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    path = _frame(tmp_path)
    _patch_vectors(monkeypatch, rows=[_row("Billy", [1, 0]), _row("Xin", [0, 1])], active=[1, 0])

    scored = score_roster_avatar_frames(_detected(path), [], profile)

    assert scored[0]["decision"] == "insufficient_distinct_roster_avatars"
    assert scored[0]["candidate_name"] is None


def test_score_requires_similarity_margin_and_rejects_duplicate_templates(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    path = _frame(tmp_path)
    duplicate_rows = [
        _row("Billy", [1, 0, 0]),
        _row("John", [1, 0, 0]),
        _row("Xin", [0, 1, 0]),
    ]
    _patch_vectors(monkeypatch, rows=duplicate_rows, active=[1, 0, 0])

    scored = score_roster_avatar_frames(_detected(path), [], profile)

    assert scored[0]["decision"] == "insufficient_distinct_roster_avatars"
    assert set(scored[0]["ambiguous_template_names"]) == {"Billy", "John"}


def test_score_abstains_on_similarity_and_margin_failures(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    profile["settings"]["minimum_similarity"] = 0.80
    profile["settings"]["minimum_margin"] = 0.20
    path = _frame(tmp_path)
    rows = [_row("Billy", [1, 0, 0]), _row("John", [0, 1, 0]), _row("Xin", [0, 0, 1])]
    _patch_vectors(monkeypatch, rows=rows, active=[1, 0, 0])
    monkeypatch.setattr(roster_identity, "_template_collisions", lambda *_args: set())

    scores = {"Billy": 0.79, "John": 0.20, "Xin": 0.10}
    monkeypatch.setattr(roster_identity, "_ensemble_similarity", lambda _active, row: scores[row["name"]])
    below_similarity = score_roster_avatar_frames(_detected(path), [], profile)
    assert below_similarity[0]["decision"] == "similarity_below_threshold"
    assert below_similarity[0]["candidate_name"] is None

    scores = {"Billy": 0.92, "John": 0.75, "Xin": 0.10}
    below_margin = score_roster_avatar_frames(_detected(path), [], profile)
    assert below_margin[0]["decision"] == "similarity_margin_below_threshold"
    assert below_margin[0]["candidate_name"] is None


def test_score_orders_equal_candidates_by_name_for_reproducibility(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    path = _frame(tmp_path)
    rows = [_row("Billy", [1, 0, 0]), _row("John", [0, 1, 0]), _row("Xin", [0, 0, 1])]
    _patch_vectors(monkeypatch, rows=rows, active=[1, 0, 0])
    monkeypatch.setattr(roster_identity, "_template_collisions", lambda *_args: set())
    scores = {"Billy": 0.95, "John": 0.95, "Xin": 0.10}
    monkeypatch.setattr(roster_identity, "_ensemble_similarity", lambda _active, row: scores[row["name"]])

    scored = score_roster_avatar_frames(_detected(path), [], profile)

    assert scored[0]["decision"] == "similarity_margin_below_threshold"
    assert scored[0]["top_candidate_name"] == "Billy"
    assert scored[0]["runner_candidate_name"] == "John"


def test_reviewed_anchor_gate_requires_correct_time_separated_matches(tmp_path):
    profile = _profile(tmp_path)
    frames = [
        {
            "time": 10.0,
            "actualTime": 10.0,
            "reviewed_anchor_names": ["Billy"],
            "decision": "matched",
            "candidate_name": "Billy",
        },
        {
            "time": 30.0,
            "actualTime": 30.0,
            "reviewed_anchor_names": ["John"],
            "decision": "matched",
            "candidate_name": "John",
        },
        {
            "time": 50.0,
            "actualTime": 50.0,
            "reviewed_anchor_names": ["Xin"],
            "decision": "matched",
            "candidate_name": "Xin",
        },
    ]

    calibration = calibrate_roster_avatar_identity(frames, profile)

    assert calibration["gate"]["status"] == "passed"
    assert calibration["eligible_identities"] == ["Billy", "John", "Xin"]
    frames[2]["candidate_name"] = "Billy"
    failed = calibrate_roster_avatar_identity(frames, profile)
    assert failed["gate"]["status"] == "blocked"
    assert not failed["gate"]["requirements"]["no_reviewed_anchor_false_accepts"]
    assert not failed["gate"]["requirements"]["enough_distinct_anchor_identities"]
    assert failed["distinct_anchor_identities"] == ["Billy", "John"]


def test_reviewed_anchor_gate_blocks_an_ambiguous_anchor(tmp_path):
    profile = _profile(tmp_path)
    frames = [
        {"time": 10.0, "reviewed_anchor_names": ["Billy"], "decision": "matched", "candidate_name": "Billy"},
        {"time": 30.0, "reviewed_anchor_names": ["John"], "decision": "matched", "candidate_name": "John"},
        {
            "time": 50.0,
            "reviewed_anchor_names": ["Xin"],
            "decision": "similarity_margin_below_threshold",
            "candidate_name": None,
        },
    ]

    calibration = calibrate_roster_avatar_identity(frames, profile)

    assert calibration["gate"]["status"] == "blocked"
    assert not calibration["gate"]["requirements"]["all_reviewed_anchors_matched"]
    assert calibration["eligible_identities"] == []


def test_roster_profile_rejects_anchors_without_minimum_time_separation(tmp_path):
    path = tmp_path / "invalid-anchor-profile.json"
    path.write_text(
        json.dumps(
            {
                "participants": ["Billy", "John", "Xin"],
                "settings": {"minimum_anchor_seconds_separation": 10},
                "layouts": [
                    {
                        "name": "discord-left-roster",
                        "start": 0,
                        "end": 100,
                        "roster_region": [0.0, 0.0, 0.4, 1.0],
                        "avatar": {},
                    }
                ],
                "reviewed_anchors": [
                    {"time": 10, "name": "Billy", "reviewed": True},
                    {"time": 15, "name": "John", "reviewed": True},
                    {"time": 50, "name": "Xin", "reviewed": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(roster_identity.RosterAvatarProfileError, match="time-separated"):
        load_roster_avatar_profile(path)


def test_reviewed_anchor_gate_blocks_rounded_duplicate_sample_times(tmp_path):
    profile = _profile(tmp_path)
    frames = [
        {
            "time": 10.0001,
            "actualTime": 10.0001,
            "reviewed_anchor_names": ["Billy"],
            "decision": "matched",
            "candidate_name": "Billy",
        },
        {
            "time": 10.0004,
            "actualTime": 10.0004,
            "reviewed_anchor_names": ["John"],
            "decision": "matched",
            "candidate_name": "John",
        },
        {
            "time": 30.0,
            "actualTime": 30.0,
            "reviewed_anchor_names": ["Xin"],
            "decision": "matched",
            "candidate_name": "Xin",
        },
    ]

    calibration = calibrate_roster_avatar_identity(frames, profile)

    assert calibration["gate"]["status"] == "blocked"
    assert not calibration["gate"]["requirements"]["anchor_sample_times_unique"]
    assert not calibration["gate"]["requirements"]["anchors_time_separated"]


def test_attachment_default_two_of_three_frame_consensus_assigns(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [{"start": 0.0, "end": 4.0, "speaker": "Speaker 1"}]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 2.0},
        {"segment_index": 0, "video_time": 3.0},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.94, "margin": 0.40},
        {"time": 2.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.95, "margin": 0.41},
        {"time": 3.0, "path": "/tmp/three.jpg", "decision": "no_active_tile"},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["assigned"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == SOURCE


def test_attachment_needs_two_matches_and_corrects_weaker_voiceprint_identity(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [
        {"start": 0.0, "end": 4.0, "speaker": "Speaker 1"},
        {
            "start": 5.0,
            "end": 9.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_source": "same_session_visual_voiceprint",
        },
        {"start": 10.0, "end": 11.0, "speaker": "Speaker 3"},
    ]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 3.0},
        {"segment_index": 1, "video_time": 6.0},
        {"segment_index": 1, "video_time": 8.0},
        {"segment_index": 2, "video_time": 10.5},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.94, "margin": 0.40},
        {"time": 3.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.95, "margin": 0.41},
        {"time": 6.0, "path": "/tmp/three.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.95, "margin": 0.42},
        {"time": 8.0, "path": "/tmp/four.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.96, "margin": 0.40},
        {"time": 10.5, "path": "/tmp/five.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.97, "margin": 0.43},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == SOURCE
    assert segments[1]["name"] == "Billy"
    assert segments[1]["name_source"] == SOURCE
    assert segments[1]["roster_avatar_identity_corrected_prior"] == {
        "name": "Xin",
        "source": "same_session_visual_voiceprint",
    }
    assert "name" not in segments[2]
    assert summary["assigned"] == 2
    assert summary["corrected_weaker_identity"] == 1


def test_attachment_corrects_weaker_cluster_identity(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "Speaker 1",
            "name": "Xin",
            "name_source": "direct_visual_voice_cluster_consensus",
        }
    ]
    requests = [{"segment_index": 0, "video_time": 1.0}, {"segment_index": 0, "video_time": 3.0}]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.94, "margin": 0.40},
        {"time": 3.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.95, "margin": 0.41},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["corrected_weaker_identity"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == SOURCE


def test_attachment_preserves_direct_visual_identity_when_roster_disagrees(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "Speaker 1",
            "name": "Xin",
            "name_source": "dynamic_visual_in_tile_nameplate_ocr",
        }
    ]
    requests = [{"segment_index": 0, "video_time": 1.0}, {"segment_index": 0, "video_time": 3.0}]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.94, "margin": 0.40},
        {"time": 3.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.95, "margin": 0.41},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["preserved_existing_identity"] == 1
    assert summary["conflicts"] == 1
    assert segments[0]["name"] == "Xin"
    assert segments[0]["name_source"] == "dynamic_visual_in_tile_nameplate_ocr"


def test_attachment_preserves_confirmed_identity_sources_when_roster_disagrees(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    requests = [{"segment_index": 0, "video_time": 1.0}, {"segment_index": 0, "video_time": 3.0}]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.94, "margin": 0.40},
        {"time": 3.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.95, "margin": 0.41},
    ]

    for source in ("voice_enrollment", "participant_map", "user_confirmed_speaker_volume_mapping"):
        segments = [{"start": 0.0, "end": 4.0, "speaker": "Speaker 1", "name": "Xin", "name_source": source}]
        summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

        assert summary["preserved_existing_identity"] == 1
        assert summary["conflicts"] == 1
        assert segments[0]["name"] == "Xin"
        assert segments[0]["name_source"] == source


def test_attachment_deduplicates_collapsed_sample_times_before_voting(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [{"start": 0.0, "end": 4.0, "speaker": "Speaker 1"}]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 1.0001},
        {"segment_index": 0, "video_time": 1.0},
    ]
    frames = [{"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.94, "margin": 0.40}]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["assigned"] == 0
    assert summary["unresolved"] == 1
    assert "name" not in segments[0]


def test_attachment_cannot_assign_when_calibration_is_blocked(tmp_path):
    profile = _profile(tmp_path)
    segments = [{"start": 0.0, "end": 4.0, "speaker": "Speaker 1"}]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 3.0},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.94, "margin": 0.40},
        {"time": 3.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Billy", "top_score": 0.95, "margin": 0.41},
    ]

    summary = attach_roster_avatar_identity(
        segments,
        requests,
        frames,
        profile,
        {"gate": {"status": "blocked"}},
    )

    assert "name" not in segments[0]
    assert summary["assigned"] == 0
    assert summary["gate_blocked"] == 1


def test_attachment_keeps_unanchored_identity_unresolved(tmp_path):
    profile = _profile(tmp_path)
    segments = [{"start": 0.0, "end": 4.0, "speaker": "Speaker 1"}]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 3.0},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "John", "top_score": 0.94, "margin": 0.40},
        {"time": 3.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "John", "top_score": 0.95, "margin": 0.41},
    ]

    summary = attach_roster_avatar_identity(
        segments,
        requests,
        frames,
        profile,
        {"gate": {"status": "passed"}, "eligible_identities": ["Billy"]},
    )

    assert "name" not in segments[0]
    assert summary["assigned"] == 0
    assert summary["unanchored_identity"] == 1


def test_attachment_gate_blocked_preserves_prior_roster_identity(tmp_path):
    profile = _profile(tmp_path)
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": SOURCE,
            "name_confidence": 0.81,
            "roster_avatar_identity_evidence": [{"time": 1.0, "candidate_name": "Billy"}],
        }
    ]

    summary = attach_roster_avatar_identity(
        segments,
        [{"segment_index": 0, "video_time": 1.0}],
        [{"time": 1.0, "decision": "matched", "candidate_name": "Xin"}],
        profile,
        {"gate": {"status": "blocked"}},
    )

    assert summary["assigned"] == 0
    assert summary["gate_blocked"] == 1
    assert summary["preserved_prior_roster_identity"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == SOURCE
    assert segments[0]["roster_avatar_identity_evidence"] == [{"time": 1.0, "candidate_name": "Billy"}]


def test_attachment_preserves_prior_roster_identity_when_new_consensus_is_insufficient(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": SOURCE,
            "name_confidence": 0.81,
            "roster_avatar_identity_evidence": [{"time": 1.0, "candidate_name": "Billy"}],
        }
    ]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 2.0},
        {"segment_index": 0, "video_time": 3.0},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Xin", "top_score": 0.95, "margin": 0.41},
        {"time": 2.0, "path": "/tmp/two.jpg", "decision": "no_active_tile"},
        {"time": 3.0, "path": "/tmp/three.jpg", "decision": "no_active_tile"},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["preserved_prior_roster_identity"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == SOURCE
    assert segments[0]["roster_avatar_identity_rerun_conflict"]["reason"] == "insufficient_roster_avatar_consensus"


def test_attachment_preserves_prior_roster_identity_when_rerun_has_multiple_candidates(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": SOURCE,
            "name_confidence": 0.81,
        }
    ]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 2.0},
        {"segment_index": 0, "video_time": 3.0},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Xin", "top_score": 0.95, "margin": 0.41},
        {"time": 2.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "John", "top_score": 0.96, "margin": 0.42},
        {"time": 3.0, "path": "/tmp/three.jpg", "decision": "no_active_tile"},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["preserved_prior_roster_identity"] == 1
    assert summary["conflicts"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == SOURCE
    assert segments[0]["roster_avatar_identity_rerun_conflict"] == {
        "reason": "multiple_roster_avatar_names_in_segment",
        "prior_roster_avatar_name": "Billy",
        "roster_avatar_names": ["John", "Xin"],
    }


def test_attachment_preserves_prior_roster_identity_when_rerun_candidate_is_unanchored(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy"]}
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": SOURCE,
            "name_confidence": 0.81,
        }
    ]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 2.0},
        {"segment_index": 0, "video_time": 3.0},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Xin", "top_score": 0.95, "margin": 0.41},
        {"time": 2.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Xin", "top_score": 0.96, "margin": 0.42},
        {"time": 3.0, "path": "/tmp/three.jpg", "decision": "no_active_tile"},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["preserved_prior_roster_identity"] == 1
    assert summary["unanchored_identity"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == SOURCE
    assert segments[0]["roster_avatar_identity_rerun_conflict"] == {
        "reason": "candidate_not_covered_by_accepted_anchor",
        "prior_roster_avatar_name": "Billy",
        "roster_avatar_name": "Xin",
    }


def test_attachment_replaces_prior_roster_identity_only_after_new_consensus_passes(tmp_path):
    profile = _profile(tmp_path)
    calibration = {"gate": {"status": "passed"}, "eligible_identities": ["Billy", "John", "Xin"]}
    segments = [
        {
            "start": 0.0,
            "end": 4.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": SOURCE,
            "name_confidence": 0.81,
        }
    ]
    requests = [
        {"segment_index": 0, "video_time": 1.0},
        {"segment_index": 0, "video_time": 2.0},
        {"segment_index": 0, "video_time": 3.0},
    ]
    frames = [
        {"time": 1.0, "path": "/tmp/one.jpg", "decision": "matched", "candidate_name": "Xin", "top_score": 0.95, "margin": 0.41},
        {"time": 2.0, "path": "/tmp/two.jpg", "decision": "matched", "candidate_name": "Xin", "top_score": 0.96, "margin": 0.42},
        {"time": 3.0, "path": "/tmp/three.jpg", "decision": "no_active_tile"},
    ]

    summary = attach_roster_avatar_identity(segments, requests, frames, profile, calibration)

    assert summary["replaced_prior_roster_identity"] == 1
    assert segments[0]["name"] == "Xin"
    assert segments[0]["name_source"] == SOURCE
    assert segments[0]["roster_avatar_identity_replaced_prior"]["name"] == "Billy"


def test_roster_source_is_not_eligible_for_voice_or_cluster_propagation():
    assert SOURCE not in cluster_identity._DIRECT_FRAME_SOURCES
    assert SOURCE not in voice_identity._DIRECT_VISUAL_FRAME_SOURCES
    assert SOURCE not in voice_identity._DIRECT_VISUAL_SEGMENT_SOURCES
