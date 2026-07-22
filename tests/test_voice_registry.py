from __future__ import annotations

from meeting_minutes.voice_registry import attach_registry_scores, calibrate_voice_registry, load_voice_registry


def test_calibration_uses_cross_source_trials_only():
    sources = [
        {
            "id": "one",
            "profiles": {
                "Alice": {"embeddings": [[1.0, 0.0], [0.99, 0.01]]},
                "Bob": {"embeddings": [[0.0, 1.0], [0.01, 0.99]]},
            },
        },
        {
            "id": "two",
            "profiles": {
                "Alice": {"embeddings": [[0.98, 0.02], [0.97, 0.03]]},
                "Bob": {"embeddings": [[0.02, 0.98], [0.03, 0.97]]},
            },
        },
    ]

    calibration = calibrate_voice_registry(sources, target_far=0.01)

    assert calibration["status"] == "limited"
    assert calibration["positive_trials"] == 8
    assert calibration["negative_trials"] == 8
    assert calibration["suggested_threshold"] is not None


def test_attach_registry_scores_requires_consistent_evidence_for_long_segment():
    segments = [{"start": 0.0, "end": 3.2, "speaker": "Speaker 1", "speaker_confidence": 0.9}]
    windows = [
        {"start": 0.0, "end": 1.6},
        {"start": 0.8, "end": 2.4},
        {"start": 1.6, "end": 3.2},
    ]

    summary = attach_registry_scores(
        segments,
        windows,
        ["Alice", "Bob"],
        [[0.92, 0.1], [0.9, 0.15], [0.91, 0.11]],
        threshold=0.8,
        margin=0.2,
    )

    assert summary["assigned_segments"] == 1
    assert segments[0]["speaker"] == "Speaker 1"
    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_source"] == "voice_registry"
    assert segments[0]["voice_registry_evidence"]["accepted_windows"] == 3


def test_attach_registry_scores_keeps_mixed_long_segment_anonymous():
    segments = [{"start": 0.0, "end": 3.2, "speaker": "Speaker 1"}]
    windows = [
        {"start": 0.0, "end": 1.6},
        {"start": 0.8, "end": 2.4},
        {"start": 1.6, "end": 3.2},
    ]

    summary = attach_registry_scores(
        segments,
        windows,
        ["Alice", "Bob"],
        [[0.92, 0.1], [0.1, 0.92], [0.91, 0.11]],
        threshold=0.8,
        margin=0.2,
    )

    assert summary["assigned_segments"] == 0
    assert "name" not in segments[0]


def test_load_voice_registry_rejects_mismatched_dimensions(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(
        '{"format":"meeting-minutes/voice-registry-v1","profiles":{"Alice":{"centroid":[1,0]},"Bob":{"centroid":[0,1,2]}}}',
        encoding="utf-8",
    )

    try:
        load_voice_registry(registry)
    except ValueError as exc:
        assert "equal dimensions" in str(exc)
    else:  # pragma: no cover - protects a validation invariant
        raise AssertionError("mismatched profile dimensions must be rejected")
