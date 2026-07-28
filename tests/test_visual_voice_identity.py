from __future__ import annotations

import numpy as np

from meeting_minutes.visual_voice_identity import (
    _calibrate_precision_gate,
    attach_visual_voice_scores,
    clear_visual_voice_identity,
    direct_visual_enrollment_frame_count,
    load_visual_voice_config,
    select_visual_voice_enrollment,
)


def _direct_frame(name: str, time: float) -> dict:
    return {
        "name": name,
        "name_source": "dynamic_visual_in_tile_nameplate_ocr",
        "actualTime": time,
        "path": f"/tmp/{name}-{time}.jpg",
        "active_tiles": [{"score": 0.99, "tile": [0.2, 0.2, 0.5, 0.6]}],
    }


def test_select_visual_voice_enrollment_requires_cross_time_visual_evidence():
    settings = load_visual_voice_config()
    payload = {
        "frames": [
            _direct_frame("Billy", 10),
            _direct_frame("Billy", 40),
            _direct_frame("Billy", 160),
            _direct_frame("Billy", 190),
            _direct_frame("Xin", 20),
            _direct_frame("Xin", 30),
            _direct_frame("Xin", 40),
            _direct_frame("Xin", 50),
        ]
    }

    selected, rejected = select_visual_voice_enrollment(payload, settings)

    assert list(selected) == ["Billy"]
    assert rejected["Xin"]["reason"] == [
        "insufficient_time_separated_visual_samples",
        "insufficient_visual_sample_time_span",
    ]


def test_select_visual_voice_enrollment_accepts_reviewed_active_slot_evidence():
    settings = load_visual_voice_config()
    frames = [
        {
            "name": "Billy",
            "name_source": "visual_profile_reviewed_slot",
            "active": True,
            "actualTime": time,
            "path": f"/tmp/billy-{time}.jpg",
            "score": 0.98,
            "slot": "billy",
        }
        for time in (10.0, 40.0, 160.0, 190.0)
    ]

    selected, rejected = select_visual_voice_enrollment({"frames": frames}, settings)

    assert list(selected) == ["Billy"]
    assert rejected == {}
    assert selected["Billy"][0]["visual_source"] == "visual_profile_reviewed_slot"
    assert direct_visual_enrollment_frame_count({"frames": frames}) == 4


def test_voice_scores_need_precision_calibration_and_segment_consensus():
    registry = {
        "format": "same-session-visual-voice-registry/v1",
        "settings": {"minimum_segment_vote_share": 0.8, "short_segment_seconds": 1.8},
        "calibration": {"status": "precision_calibrated", "threshold": 0.5, "margin": 0.12},
    }
    segments = [
        {"start": 0.0, "end": 3.0, "speaker": "Speaker 1"},
        {
            "start": 3.0,
            "end": 4.0,
            "speaker": "Speaker 2",
            "name": "Nick Burin",
            "name_source": "dynamic_visual_in_tile_nameplate_ocr",
        },
    ]
    windows = [
        {"segment_index": 0, "start": 0.2, "end": 1.8},
        {"segment_index": 0, "start": 1.2, "end": 2.8},
        {"segment_index": 1, "start": 3.2, "end": 3.8},
    ]
    scores = np.asarray(
        [
            [0.70, 0.15],
            [0.68, 0.20],
            [0.10, 0.80],
        ],
        dtype="float32",
    )

    status = attach_visual_voice_scores(segments, windows, scores, ["Billy", "Nick Burin"], registry)

    assert status["assigned_segments"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "same_session_visual_voiceprint"
    assert segments[1]["name"] == "Nick Burin"
    assert segments[1]["name_source"] == "dynamic_visual_in_tile_nameplate_ocr"


def test_voice_scores_preserve_direct_static_visual_identity():
    registry = {
        "format": "same-session-visual-voice-registry/v1",
        "settings": {"minimum_segment_vote_share": 0.8, "short_segment_seconds": 1.8},
        "calibration": {"status": "precision_calibrated", "threshold": 0.5, "margin": 0.12},
    }
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": "visual_active_speaker_highlight",
        }
    ]

    status = attach_visual_voice_scores(
        segments,
        [{"segment_index": 0, "start": 0.0, "end": 1.0}],
        np.asarray([[0.10, 0.90]], dtype="float32"),
        ["Billy", "Xin"],
        registry,
    )

    assert status["preserved_direct_visual_segments"] == 1
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "visual_active_speaker_highlight"


def test_voice_scores_can_correct_cluster_propagated_identity_after_precision_gate():
    registry = {
        "format": "same-session-visual-voice-registry/v2",
        "settings": {"minimum_segment_vote_share": 0.8, "short_segment_seconds": 1.8},
        "calibration": {"status": "precision_calibrated", "threshold": 0.5, "margin": 0.12},
    }
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "Speaker 3",
            "name": "Billy",
            "name_source": "direct_visual_voice_cluster_consensus",
        }
    ]

    status = attach_visual_voice_scores(
        segments,
        [{"segment_index": 0, "start": 0.0, "end": 1.0}],
        np.asarray([[0.10, 0.90]], dtype="float32"),
        ["Billy", "Xin"],
        registry,
    )

    assert status["corrected_cluster_assignments"] == 1
    assert status["confirmed_cluster_assignments"] == 0
    assert segments[0]["name"] == "Xin"
    assert segments[0]["name_source"] == "same_session_visual_voiceprint"
    assert segments[0]["visual_voice_identity_evidence"]["prior_cluster_name"] == "Billy"


def test_clear_visual_voice_identity_retracts_only_voiceprint_labels():
    segments = [
        {
            "name": "Xin",
            "name_source": "same_session_visual_voiceprint",
            "name_confidence": 0.88,
            "visual_voice_identity_evidence": {"prior_cluster_name": "Billy"},
            "direct_visual_cluster_identity_evidence": {"candidate": "Billy"},
        },
        {
            "name": "Billy",
            "name_source": "visual_active_speaker_highlight",
            "name_confidence": 0.94,
        },
    ]

    cleared = clear_visual_voice_identity(segments)

    assert cleared == 1
    assert "name" not in segments[0]
    assert "name_source" not in segments[0]
    assert "name_confidence" not in segments[0]
    assert "visual_voice_identity_evidence" not in segments[0]
    assert "direct_visual_cluster_identity_evidence" not in segments[0]
    assert segments[1]["name"] == "Billy"
    assert segments[1]["name_source"] == "visual_active_speaker_highlight"


def test_voice_scores_do_not_attach_when_calibration_is_insufficient():
    registry = {
        "format": "same-session-visual-voice-registry/v1",
        "settings": {"minimum_segment_vote_share": 0.8, "short_segment_seconds": 1.8},
        "calibration": {"status": "insufficient_for_precision_calibration"},
    }
    segments = [{"start": 0.0, "end": 1.0, "speaker": "Speaker 1"}]
    status = attach_visual_voice_scores(
        segments,
        [{"segment_index": 0, "start": 0.0, "end": 1.0}],
        np.asarray([[0.9, 0.1]], dtype="float32"),
        ["Billy", "Nick Burin"],
        registry,
    )

    assert status["status"] == "audit_only_insufficient_calibration"
    assert segments[0].get("name") is None


def test_calibration_makes_eligibility_per_identity_on_independent_held_out_samples():
    settings = load_visual_voice_config()
    settings.update(
        {
            "minimum_held_out_accepts": 1,
            "minimum_impostor_trials": 1,
            "minimum_score": 0.5,
            "minimum_margin": 0.12,
            "calibration_score_buffer": 0.02,
        }
    )
    names = ("Billy", "Sebastian", "Xin")
    windows: list[dict] = []
    vectors: list[list[float]] = []
    base_vectors = {
        "Billy": [1.0, 0.0, 0.0],
        "Sebastian": [0.0, 0.0, 1.0],
        "Xin": [0.0, 1.0, 0.0],
    }
    for name_index, name in enumerate(names):
        for position in range(10):
            windows.append({"name": name, "time": name_index * 100 + position, "frame": f"/{name}-{position}.jpg"})
            vector = list(base_vectors[name])
            # Held-out Sebastian examples (positions 0 and 5) imitate Billy.
            if name == "Sebastian" and position in {0, 5}:
                vector = [0.99, 0.01, 0.0]
            vectors.append(vector)

    calibration = _calibrate_precision_gate(np.asarray(vectors, dtype="float32"), windows, settings)

    assert calibration["split"] == "time_ordered_modulo_5"
    assert calibration["threshold_source"] == "calibration_impostor_max_plus_buffer"
    assert calibration["status"] == "partial_precision_calibrated"
    assert calibration["eligible_profiles"] == ["Xin"]
    assert calibration["false_accepts_by_name"] == {"Billy": 2}
    assert calibration["ineligible_profiles"]["Billy"] == ["held_out_false_accept"]
    assert calibration["ineligible_profiles"]["Sebastian"] == ["insufficient_held_out_accepts"]


def test_partial_calibration_abstains_when_ineligible_profile_scores_highest():
    registry = {
        "format": "same-session-visual-voice-registry/v2",
        "settings": {"minimum_segment_vote_share": 0.8, "short_segment_seconds": 1.8},
        "calibration": {
            "status": "partial_precision_calibrated",
            "threshold": 0.5,
            "margin": 0.12,
            "eligible_profiles": ["Billy"],
        },
    }
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1"},
        {"start": 1.0, "end": 2.0, "speaker": "Speaker 2"},
        {"start": 2.0, "end": 3.0, "speaker": "Speaker 3"},
    ]
    windows = [
        {"segment_index": 0, "start": 0.0, "end": 1.0},
        {"segment_index": 1, "start": 1.0, "end": 2.0},
        {"segment_index": 2, "start": 2.0, "end": 3.0},
    ]
    scores = np.asarray(
        [
            [0.70, 0.90],  # Sebastian is top and ineligible: must abstain.
            [0.70, 0.62],  # Billy is top but ineligible runner-up blocks the margin.
            [0.82, 0.10],  # Billy is safely top and can be assigned.
        ],
        dtype="float32",
    )

    status = attach_visual_voice_scores(segments, windows, scores, ["Billy", "Sebastian"], registry)

    assert status["assigned_segments"] == 1
    assert status["abstained_ineligible_top_windows"] == 1
    assert segments[0].get("name") is None
    assert segments[1].get("name") is None
    assert segments[2]["name"] == "Billy"
