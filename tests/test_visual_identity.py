import json

import pytest
from PIL import Image, ImageDraw

from meeting_minutes.visual_identity import (
    VisualProfileError,
    attach_visual_identity,
    build_segment_sample_requests,
    load_visual_profile,
    resolve_slot_names,
    score_visual_frames,
)


def _profile(tmp_path, *, people=True):
    slots = {
        "left": {
            "tile": [0.05, 0.08, 0.45, 0.75],
            "nameplate": [0.05, 0.65, 0.25, 0.75],
        },
        "right": {
            "tile": [0.55, 0.08, 0.95, 0.75],
            "nameplate": [0.55, 0.65, 0.75, 0.75],
        },
    }
    if people:
        slots["left"]["person"] = "Alice"
        slots["right"]["person"] = "Bob"
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "settings": {
                    "samples_per_segment": 3,
                    "allow_direct_assignment": True,
                    "min_active_score": 0.1,
                    "min_active_margin": 0.05,
                    "minimum_segment_vote_share": 0.75,
                    "short_segment_seconds": 1.5,
                },
                "participants": ["Alice", "Bob"],
                "layouts": [{"name": "grid", "start": 0, "end": 30, "slots": slots}],
            }
        ),
        encoding="utf-8",
    )
    return load_visual_profile(path)


def _frame(tmp_path, name, active):
    path = tmp_path / name
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    if active == "left":
        draw.rectangle((10, 10, 90, 90), outline=(0, 255, 120), width=5)
    if active == "right":
        draw.rectangle((110, 10, 190, 90), outline=(0, 255, 120), width=5)
    image.save(path)
    return path


def test_profile_rejects_overlapping_layout_windows(tmp_path):
    path = tmp_path / "bad-profile.json"
    path.write_text(
        json.dumps(
            {
                "layouts": [
                    {"name": "one", "start": 0, "end": 10, "slots": {"a": {"tile": [0, 0, 0.5, 0.5]}}},
                    {"name": "two", "start": 9, "end": 20, "slots": {"a": {"tile": [0, 0, 0.5, 0.5]}}},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(VisualProfileError, match="must not overlap"):
        load_visual_profile(path)


def test_profile_rejects_reviewed_slot_outside_participant_whitelist(tmp_path):
    path = tmp_path / "bad-person-profile.json"
    path.write_text(
        json.dumps(
            {
                "participants": ["Alice"],
                "layouts": [
                    {
                        "name": "one",
                        "slots": {"a": {"tile": [0, 0, 0.5, 0.5], "person": "Bob"}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(VisualProfileError, match="must be listed"):
        load_visual_profile(path)


def test_visual_identity_uses_profiled_green_speaker_cue(tmp_path):
    profile = _profile(tmp_path)
    profile["layouts"][0]["slots"]["left"]["speaker_cue"] = (0.05, 0.66, 0.21, 0.93)
    profile["layouts"][0]["slots"]["right"]["speaker_cue"] = (0.55, 0.66, 0.71, 0.93)
    frame = tmp_path / "green-cue.jpg"
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 80, 40, 110), radius=12, fill=(250, 250, 250))
    draw.rectangle((17, 93, 19, 99), fill=(65, 210, 105))
    draw.rectangle((22, 89, 24, 103), fill=(65, 210, 105))
    draw.rectangle((27, 92, 29, 100), fill=(65, 210, 105))
    image.save(frame)

    scored = score_visual_frames(
        [{"time": 3.0, "actualTime": 3.0, "path": str(frame)}],
        profile,
        resolve_slot_names(profile, []),
    )

    assert scored[0]["slot"] == "left"
    assert scored[0]["active_signal"] == "green_speaker_cue"
    assert scored[0]["score"] > 0.9


def test_visual_identity_uses_profiled_green_highlight_border(tmp_path):
    profile = _profile(tmp_path)
    profile["layouts"][0]["slots"]["left"]["active_signal"] = "green_highlight_border"
    profile["layouts"][0]["slots"]["right"]["active_signal"] = "green_highlight_border"
    frame = _frame(tmp_path, "green-border.jpg", "left")

    scored = score_visual_frames(
        [{"time": 3.0, "actualTime": 3.0, "path": str(frame)}],
        profile,
        resolve_slot_names(profile, []),
    )

    assert scored[0]["slot"] == "left"
    assert scored[0]["active_signal"] == "green_highlight_border"
    assert scored[0]["score"] > 0.9


def test_nameplate_ocr_consensus_uses_participant_whitelist(tmp_path):
    profile = _profile(tmp_path, people=False)
    records = [
        {
            "time": 2.0,
            "path": "/tmp/one.jpg",
            "regions": [
                {"label": "grid::left", "texts": [{"text": "alice", "confidence": 0.9}]},
                {"label": "grid::right", "texts": [{"text": "B0b", "confidence": 0.9}]},
            ],
        },
        {
            "time": 4.0,
            "path": "/tmp/two.jpg",
            "regions": [
                {"label": "grid::left", "texts": [{"text": "Alice", "confidence": 0.9}]},
                {"label": "grid::right", "texts": [{"text": "Bob", "confidence": 0.9}]},
            ],
        },
    ]

    names = resolve_slot_names(profile, records)

    assert names["grid::left"]["name"] == "Alice"
    assert names["grid::right"]["name"] == "Bob"


def test_nameplate_ocr_without_whitelist_stays_a_candidate(tmp_path):
    profile = _profile(tmp_path, people=False)
    profile["participants"] = []
    records = [
        {
            "time": 2.0,
            "path": "/tmp/one.jpg",
            "regions": [{"label": "grid::left", "texts": [{"text": "Alice", "confidence": 0.9}]}],
        },
        {
            "time": 4.0,
            "path": "/tmp/two.jpg",
            "regions": [{"label": "grid::left", "texts": [{"text": "Alice", "confidence": 0.9}]}],
        },
    ]

    names = resolve_slot_names(profile, records)

    assert names["grid::left"]["name"] is None
    assert names["grid::left"]["candidates"] == [{"name": "Alice", "count": 2}]


def test_nameplate_candidate_removes_voice_status_marker(tmp_path):
    profile = _profile(tmp_path, people=False)
    records = [
        {
            "time": 2.0,
            "path": "/tmp/one.jpg",
            "regions": [{"label": "grid::left", "texts": [{"text": "o Alice", "confidence": 0.9}]}],
        },
        {
            "time": 4.0,
            "path": "/tmp/two.jpg",
            "regions": [{"label": "grid::left", "texts": [{"text": "Alice", "confidence": 0.9}]}],
        },
    ]

    names = resolve_slot_names(profile, records)

    assert names["grid::left"]["name"] == "Alice"


def test_nameplate_candidate_removes_ui_icon(tmp_path):
    profile = _profile(tmp_path, people=False)
    records = [
        {
            "time": 2.0,
            "path": "/tmp/one.jpg",
            "regions": [{"label": "grid::left", "texts": [{"text": "© Alice", "confidence": 0.9}]}],
        },
        {
            "time": 4.0,
            "path": "/tmp/two.jpg",
            "regions": [{"label": "grid::left", "texts": [{"text": "Alice", "confidence": 0.9}]}],
        },
    ]

    names = resolve_slot_names(profile, records)

    assert names["grid::left"]["name"] == "Alice"


def test_visual_identity_assigns_only_consistent_active_evidence(tmp_path):
    profile = _profile(tmp_path)
    requests = build_segment_sample_requests([{"start": 2.0, "end": 5.0, "speaker": "Speaker 1", "text": "hello"}], profile, duration=30)
    frames = [
        {"time": request["video_time"], "actualTime": request["video_time"], "path": str(_frame(tmp_path, f"frame-{index}.jpg", "left"))}
        for index, request in enumerate(requests)
    ]
    names = resolve_slot_names(profile, [])
    scored = score_visual_frames(frames, profile, names)
    segments = [{"start": 2.0, "end": 5.0, "speaker": "Speaker 1", "text": "hello"}]

    summary = attach_visual_identity(segments, requests, scored, profile)

    assert summary["assigned"] == 1
    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_source"] == "visual_active_speaker_highlight"
    assert len(segments[0]["frame_refs"]) == 2


def test_visual_identity_accepts_two_of_three_samples_and_calculates_confidence(tmp_path):
    profile = _profile(tmp_path)
    profile["settings"]["minimum_segment_vote_share"] = 0.66
    requests = build_segment_sample_requests([{"start": 2.0, "end": 5.2, "speaker": "Speaker 1", "text": "hello"}], profile, duration=30)
    frames = [
        {
            "time": request["video_time"],
            "actualTime": request["video_time"],
            "path": str(_frame(tmp_path, f"frame-{index}.jpg", "left" if index < 2 else None)),
        }
        for index, request in enumerate(requests)
    ]
    names = resolve_slot_names(profile, [])
    scored = score_visual_frames(frames, profile, names)
    segments = [{"start": 2.0, "end": 5.2, "speaker": "Speaker 1", "text": "hello"}]

    summary = attach_visual_identity(segments, requests, scored, profile)

    matched_scores = [float(frame["score"]) for frame in scored if frame.get("active") and frame.get("name") == "Alice"]
    expected = round(min(0.94, 0.62 + 0.16 * (2 / 3) + 0.22 * (sum(matched_scores) / len(matched_scores))), 3)
    assert summary["assigned"] == 1
    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_confidence"] == expected


def test_visual_identity_keeps_conflicting_evidence_anonymous(tmp_path):
    profile = _profile(tmp_path)
    requests = build_segment_sample_requests([{"start": 2.0, "end": 5.0, "speaker": "Speaker 1", "text": "hello"}], profile, duration=30)
    frames = [
        {"time": request["video_time"], "actualTime": request["video_time"], "path": str(_frame(tmp_path, f"frame-{index}.jpg", "left" if index == 0 else "right"))}
        for index, request in enumerate(requests)
    ]
    names = resolve_slot_names(profile, [])
    scored = score_visual_frames(frames, profile, names)
    segments = [{"start": 2.0, "end": 5.0, "speaker": "Speaker 1", "text": "hello"}]

    summary = attach_visual_identity(segments, requests, scored, profile)

    assert summary["conflicts"] == 1
    assert segments[0]["name"] is None
    assert segments[0]["name_source"] == "visual_identity_conflict"


def test_visual_identity_rejects_tied_active_tile_scores(tmp_path):
    profile = _profile(tmp_path)
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), outline=(0, 255, 120), width=5)
    draw.rectangle((110, 10, 190, 90), outline=(0, 255, 120), width=5)
    frame_path = tmp_path / "tied.jpg"
    image.save(frame_path)

    names = resolve_slot_names(profile, [])
    scored = score_visual_frames([{"time": 3.0, "actualTime": 3.0, "path": str(frame_path)}], profile, names)

    assert scored[0]["active"] is False
    assert scored[0]["reason"] == "active_score_margin_too_small"


def test_visual_identity_never_uses_speaker_cluster_as_fallback(tmp_path):
    profile = _profile(tmp_path)
    requests = build_segment_sample_requests([{"start": 2.0, "end": 2.8, "speaker": "Speaker 1", "text": "hello"}], profile, duration=30)
    frames = [
        {"time": request["video_time"], "actualTime": request["video_time"], "path": str(_frame(tmp_path, "inactive.jpg", None))}
        for request in requests
    ]
    names = resolve_slot_names(profile, [])
    scored = score_visual_frames(frames, profile, names)
    segments = [{"start": 2.0, "end": 2.8, "speaker": "Speaker 1", "text": "hello"}]

    summary = attach_visual_identity(segments, requests, scored, profile)

    assert summary["assigned"] == 0
    assert segments[0]["name"] is None
    assert segments[0]["name_source"] == "visual_identity_unresolved"


def test_visual_identity_defaults_to_audit_only(tmp_path):
    profile = _profile(tmp_path)
    profile["settings"]["allow_direct_assignment"] = False
    requests = build_segment_sample_requests([{"start": 2.0, "end": 2.8, "speaker": "Speaker 1", "text": "hello"}], profile, duration=30)
    frames = [
        {"time": request["video_time"], "actualTime": request["video_time"], "path": str(_frame(tmp_path, "active.jpg", "left"))}
        for request in requests
    ]
    names = resolve_slot_names(profile, [])
    scored = score_visual_frames(frames, profile, names)
    segments = [{"start": 2.0, "end": 2.8, "speaker": "Speaker 1", "text": "hello"}]

    summary = attach_visual_identity(segments, requests, scored, profile)

    assert summary["assignment_mode"] == "audit_only"
    assert summary["assigned"] == 0
    assert summary["unvalidated_candidates"] == 1
    assert segments[0]["name"] is None
    assert segments[0]["name_source"] == "visual_identity_unvalidated_candidate"


def test_visual_identity_preserves_voice_registry_identity(tmp_path):
    profile = _profile(tmp_path)
    requests = build_segment_sample_requests(
        [{"start": 2.0, "end": 2.8, "speaker": "Speaker 1", "text": "hello"}],
        profile,
        duration=30,
    )
    frames = [
        {"time": request["video_time"], "actualTime": request["video_time"], "path": str(_frame(tmp_path, "active-right.jpg", "right"))}
        for request in requests
    ]
    names = resolve_slot_names(profile, [])
    scored = score_visual_frames(frames, profile, names)
    segments = [
        {
            "start": 2.0,
            "end": 2.8,
            "speaker": "Speaker 1",
            "name": "Alice",
            "name_source": "voice_registry",
            "name_confidence": 0.8,
            "text": "hello",
        }
    ]

    summary = attach_visual_identity(segments, requests, scored, profile)

    assert summary["preserved_trusted_identity"] == 1
    assert summary["conflicts"] == 1
    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_source"] == "voice_registry"
