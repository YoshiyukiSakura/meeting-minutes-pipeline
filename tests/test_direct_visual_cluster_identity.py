from __future__ import annotations

from meeting_minutes.direct_visual_cluster_identity import (
    _turn_evidence,
    apply_direct_visual_cluster_identity,
    build_direct_visual_cluster_identity,
    clear_direct_visual_cluster_identity,
    load_direct_visual_cluster_config,
)


def _settings() -> dict:
    settings = load_direct_visual_cluster_config()
    settings.update(
        {
            "minimum_training_turns": 5,
            "minimum_validation_turns": 3,
            "minimum_wilson_lower_bound": 0.0,
            "minimum_time_span_seconds": 0.0,
        }
    )
    return settings


def _turn(speaker: str, start: float) -> dict:
    return {"speaker": speaker, "start": start, "end": start + 8.0}


def _frame(name: str, time: float) -> dict:
    return {
        "name": name,
        "name_source": "visual_profile_reviewed_slot",
        "active": True,
        "actualTime": time,
        "path": f"/tmp/{name}-{time}.jpg",
    }


def test_turn_evidence_erodes_boundary_lag_before_voting():
    settings = _settings()
    turns = [_turn("Speaker 3", 0.0), _turn("Speaker 4", 8.0)]
    frames = [
        _frame("Billy", 1.5),
        _frame("Billy", 2.5),
        _frame("Max Kotlov", 7.7),
        _frame("Max Kotlov", 9.5),
        _frame("Max Kotlov", 10.5),
    ]

    records, status = _turn_evidence(frames, turns, settings=settings, erosion_seconds=0.8)

    assert status["boundary_eroded_frames"] == 1
    assert [(record["speaker"], record["name"]) for record in records] == [
        ("Speaker 3", "Billy"),
        ("Speaker 4", "Max Kotlov"),
    ]


def test_cluster_identity_rejects_hundreds_of_correlated_frames_from_two_turns():
    settings = _settings()
    turns = [_turn("Speaker 3", 0.0), _turn("Speaker 3", 12.0)]
    frames = [
        _frame("Billy", 2.0 + index * 0.01)
        for index in range(100)
    ] + [
        _frame("Billy", 14.0 + index * 0.01)
        for index in range(100)
    ]

    payload = build_direct_visual_cluster_identity({"frames": frames}, turns, settings=settings)

    assert payload["accepted_clusters"] == {}
    assert payload["clusters"]["Speaker 3"]["status"] == "rejected"


def test_cluster_identity_accepts_turn_disjoint_bidirectional_visual_confirmation():
    settings = _settings()
    turns: list[dict] = []
    frames: list[dict] = []
    for index in range(48):
        start = index * 20.0
        speaker = "Speaker 3" if index % 2 == 0 else "Speaker 4"
        name = "Billy" if speaker == "Speaker 3" else "Max Kotlov"
        turns.append(_turn(speaker, start))
        frames.extend([_frame(name, start + 2.0), _frame(name, start + 5.0)])

    payload = build_direct_visual_cluster_identity({"frames": frames}, turns, settings=settings)

    assert payload["accepted_clusters"]["Speaker 3"]["candidate"] == "Billy"
    assert payload["accepted_clusters"]["Speaker 4"]["candidate"] == "Max Kotlov"
    assert payload["clusters"]["Speaker 3"]["early_to_late"]["accepted"] is True
    assert payload["clusters"]["Speaker 3"]["late_to_early"]["accepted"] is True

    segments = [
        {"start": 22.0, "end": 23.0, "speaker": "Speaker 4", "speaker_confidence": 0.95},
        {"start": 42.0, "end": 43.0, "speaker": "Speaker 3", "speaker_confidence": 0.95},
        {
            "start": 62.0,
            "end": 63.0,
            "speaker": "Speaker 4",
            "speaker_confidence": 0.95,
            "name": "Billy",
            "name_source": "visual_active_speaker_highlight",
        },
    ]
    status = apply_direct_visual_cluster_identity(segments, payload)

    assert status["assigned_segments"] == 2
    assert segments[0]["name"] == "Max Kotlov"
    assert segments[1]["name"] == "Billy"
    assert segments[2]["name"] == "Billy"
    assert segments[2]["name_source"] == "visual_active_speaker_highlight"


def test_cluster_identity_rejects_drift_and_name_collisions():
    settings = _settings()
    turns: list[dict] = []
    frames: list[dict] = []
    for index in range(24):
        start = index * 20.0
        turns.append(_turn("Speaker 2", start))
        name = "Xin" if index < 12 else "Sebastian"
        frames.extend([_frame(name, start + 2.0), _frame(name, start + 5.0)])
    for speaker, offset in (("Speaker 3", 500.0), ("Speaker 4", 1000.0)):
        for index in range(24):
            start = offset + index * 20.0
            turns.append(_turn(speaker, start))
            frames.extend([_frame("Billy", start + 2.0), _frame("Billy", start + 5.0)])
    turns.sort(key=lambda turn: turn["start"])

    payload = build_direct_visual_cluster_identity({"frames": frames}, turns, settings=settings)

    assert payload["clusters"]["Speaker 2"]["status"] == "rejected"
    assert "directional_candidates_disagree" in payload["clusters"]["Speaker 2"]["reasons"]
    assert payload["clusters"]["Speaker 3"]["status"] == "rejected"
    assert payload["clusters"]["Speaker 4"]["status"] == "rejected"
    assert "name_collision" in payload["clusters"]["Speaker 3"]["reasons"]
    assert "name_collision" in payload["clusters"]["Speaker 4"]["reasons"]


def test_cluster_identity_does_not_fill_a_long_gap_without_direct_visual_support():
    settings = _settings()
    turns: list[dict] = []
    frames: list[dict] = []
    for index in range(20):
        start = index * 20.0 if index < 10 else 600.0 + (index - 10) * 20.0
        turns.append(_turn("Speaker 3", start))
        frames.extend([_frame("Billy", start + 2.0), _frame("Billy", start + 5.0)])

    payload = build_direct_visual_cluster_identity({"frames": frames}, turns, settings=settings)
    cluster = payload["accepted_clusters"]["Speaker 3"]
    assert len(cluster["support_intervals"]) == 2

    segments = [
        {"start": 42.0, "end": 43.0, "speaker": "Speaker 3", "speaker_confidence": 0.95},
        {"start": 300.0, "end": 301.0, "speaker": "Speaker 3", "speaker_confidence": 0.95},
        {"start": 62.0, "end": 63.0, "speaker": "Speaker 3", "speaker_confidence": 0.50},
    ]

    status = apply_direct_visual_cluster_identity(segments, payload)

    assert status["assigned_segments"] == 1
    assert status["skipped_outside_visual_support"] == 1
    assert status["skipped_low_speaker_confidence"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[1].get("name") is None
    assert segments[2].get("name") is None


def test_cluster_identity_rejects_legacy_full_span_support_without_intervals():
    settings = _settings()
    turns = [_turn("Speaker 3", index * 20.0) for index in range(20)]
    frames = [
        frame
        for turn in turns
        for frame in (_frame("Billy", turn["start"] + 2.0), _frame("Billy", turn["start"] + 5.0))
    ]
    payload = build_direct_visual_cluster_identity({"frames": frames}, turns, settings=settings)
    payload["accepted_clusters"]["Speaker 3"].pop("support_intervals")
    segments = [{"start": 42.0, "end": 43.0, "speaker": "Speaker 3", "speaker_confidence": 0.95}]

    status = apply_direct_visual_cluster_identity(segments, payload)

    assert status["assigned_segments"] == 0
    assert status["skipped_outside_visual_support"] == 1
    assert segments[0].get("name") is None


def test_cluster_identity_exercises_wilson_time_span_and_erosion_stability_gates():
    turns = [_turn("Speaker 3", index * 20.0) for index in range(12)]
    stable_frames = [
        frame
        for turn in turns
        for frame in (_frame("Billy", turn["start"] + 2.0), _frame("Billy", turn["start"] + 5.0))
    ]

    wilson_settings = _settings()
    wilson_settings["minimum_wilson_lower_bound"] = 0.95
    wilson_payload = build_direct_visual_cluster_identity({"frames": stable_frames}, turns, settings=wilson_settings)
    assert "insufficient_wilson_lower_bound" in wilson_payload["clusters"]["Speaker 3"]["early_to_late"]["training_reasons"]

    span_settings = _settings()
    span_settings["minimum_time_span_seconds"] = 500.0
    span_payload = build_direct_visual_cluster_identity({"frames": stable_frames}, turns, settings=span_settings)
    assert "insufficient_time_span" in span_payload["clusters"]["Speaker 3"]["reasons"]

    unstable_frames = [
        frame
        for turn in turns
        for frame in (
            _frame("Billy", turn["start"] + 0.85),
            _frame("Billy", turn["start"] + 0.90),
            _frame("Billy", turn["start"] + 1.10),
        )
    ]
    erosion_payload = build_direct_visual_cluster_identity({"frames": unstable_frames}, turns, settings=_settings())
    assert "erosion_1_unstable" in erosion_payload["clusters"]["Speaker 3"]["reasons"]


def test_clear_direct_visual_cluster_identity_retracts_only_prior_propagation():
    segments = [
        {
            "name": "Billy",
            "name_source": "direct_visual_voice_cluster_consensus",
            "name_confidence": 0.9,
            "direct_visual_cluster_identity_evidence": {"candidate": "Billy"},
        },
        {
            "name": "Max Kotlov",
            "name_source": "visual_active_speaker_highlight",
            "name_confidence": 0.94,
        },
    ]

    cleared = clear_direct_visual_cluster_identity(segments)

    assert cleared == 1
    assert segments[0].get("name") is None
    assert segments[0].get("name_source") is None
    assert segments[1]["name"] == "Max Kotlov"
    assert segments[1]["name_source"] == "visual_active_speaker_highlight"
