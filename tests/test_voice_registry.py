from __future__ import annotations

from meeting_minutes.voice_registry import (
    attach_registry_scores,
    calibrate_voice_registry,
    enforce_registry_cluster_consensus,
    load_voice_registry,
)


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


def test_attach_registry_scores_preserves_calibrated_roster_avatar_identity():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": "visual_roster_avatar_match",
            "name_confidence": 0.81,
        }
    ]

    summary = attach_registry_scores(
        segments,
        [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}],
        ["Billy", "Xin"],
        [[0.1, 0.95], [0.1, 0.95]],
        threshold=0.8,
        margin=0.2,
    )

    assert summary["assigned_segments"] == 0
    assert summary["skipped_trusted_segments"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "visual_roster_avatar_match"


def test_attach_registry_scores_preserves_direct_active_visual_identity():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": "visual_active_speaker_highlight",
            "name_confidence": 0.9,
        }
    ]

    summary = attach_registry_scores(
        segments,
        [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}],
        ["Billy", "Xin"],
        [[0.1, 0.95], [0.1, 0.95]],
        threshold=0.8,
        margin=0.2,
    )

    assert summary["assigned_segments"] == 0
    assert summary["skipped_trusted_segments"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "visual_active_speaker_highlight"


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


def test_cluster_consensus_expands_only_homogeneous_registry_cluster():
    segments = [
        *[
            {
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_source": "voice_registry",
                "name_confidence": 0.85,
                "voice_registry_evidence": {"accepted_windows": 1},
            }
            for _ in range(18)
        ],
        *[{"speaker": "Speaker 1", "name": None, "name_source": None} for _ in range(2)],
        *[
            {
                "speaker": "Speaker 2",
                "name": "Armando",
                "name_source": "voice_registry",
                "name_confidence": 0.85,
                "voice_registry_evidence": {"accepted_windows": 1},
            }
            for _ in range(8)
        ],
        *[
            {
                "speaker": "Speaker 2",
                "name": "Xin",
                "name_source": "voice_registry",
                "name_confidence": 0.85,
                "voice_registry_evidence": {"accepted_windows": 1},
            }
            for _ in range(8)
        ],
        {
            "speaker": "Speaker 2",
            "name": "Billy",
            "name_source": "ocr_candidates_only",
            "voice_registry_evidence": {"accepted_windows": 1},
        },
        {"speaker": "Speaker 2", "name": "Rob", "name_source": "participant_map", "name_confidence": 0.95},
    ]

    summary = enforce_registry_cluster_consensus(segments, minimum_support=5)

    assert summary["accepted_clusters"]["Speaker 1"]["name"] == "Billy"
    assert summary["rejected_clusters"]["Speaker 2"]["name"] == "Armando"
    assert all(segment["name"] == "Billy" for segment in segments[:20])
    assert all(segment["name"] is None for segment in segments[20:37])
    assert segments[37]["name"] == "Rob"
    assert segments[37]["name_source"] == "participant_map"


def test_cluster_consensus_is_idempotent_and_retains_prior_rejection():
    segments = [
        *[
            {
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_source": "voice_registry",
                "voice_registry_evidence": {"accepted_windows": 1},
            }
            for _ in range(12)
        ],
        *[
            {
                "speaker": "Speaker 2",
                "name": None,
                "name_source": "voice_registry_cluster_inconsistent",
                "voice_registry_cluster_consensus": {
                    "status": "rejected",
                    "name": "Xin",
                    "votes": {"Xin": 4, "Armando": 4},
                },
            }
            for _ in range(2)
        ],
    ]

    first = enforce_registry_cluster_consensus(segments)
    second = enforce_registry_cluster_consensus(segments)

    assert first["expanded_segments"] == 0
    assert second["expanded_segments"] == 0
    assert second["rejected_clusters"]["Speaker 2"]["name"] == "Xin"
