import json

import pytest

from meeting_minutes.diarization import attach_speakers, diarize_audio, load_voice_enrollment


def test_load_voice_enrollment_original_file_offsets_ranges(tmp_path):
    enrollment = tmp_path / "enrollment.json"
    enrollment.write_text(
        json.dumps(
            {
                "enrollment_audio_reference": "original_file",
                "speakers": {
                    "Billy": [{"start": 12.0, "end": 16.0}],
                    "Xin": [[18.0, 22.0]],
                },
            }
        ),
        encoding="utf-8",
    )
    ranges, meta = load_voice_enrollment(enrollment, source_offset=10.0, audio_duration=20.0)
    assert ranges == {"Billy": [(2.0, 6.0)], "Xin": [(8.0, 12.0)]}
    assert meta["enrollment_audio_reference"] == "original_file"


def test_load_voice_enrollment_rejects_overlapping_ranges(tmp_path):
    enrollment = tmp_path / "enrollment.json"
    enrollment.write_text(
        json.dumps(
            {
                "enrollment_audio_reference": "effective_clip",
                "speakers": {
                    "Billy": [{"start": 1.0, "end": 5.0}],
                    "Xin": [{"start": 4.5, "end": 8.0}],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_voice_enrollment(enrollment, audio_duration=30.0)


def test_attach_speakers_propagates_voice_enrollment_name():
    segments = [{"start": 10.0, "end": 12.0, "text": "hello", "speaker": "Speaker Unknown"}]
    turns = [
        {
            "start": 9.5,
            "end": 12.5,
            "speaker": "Billy",
            "confidence": 0.9,
            "name": "Billy",
            "name_source": "voice_enrollment",
            "name_confidence": 0.9,
        }
    ]
    attach_speakers(segments, turns)
    assert segments[0]["speaker"] == "Billy"
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "voice_enrollment"
    assert segments[0]["name_confidence"] >= 0.9


def test_attach_speakers_uses_speech_overlap_not_full_asr_duration():
    segments = [{"start": 100.0, "end": 112.0, "text": "hold up let me put it in", "speaker": "Speaker Unknown"}]
    turns = [
        {
            "start": 101.0,
            "end": 103.5,
            "speaker": "Billy",
            "confidence": 0.8,
            "name": "Billy",
            "name_source": "user_confirmed_speaker_volume_mapping",
            "name_confidence": 0.8,
        }
    ]
    attach_speakers(segments, turns)
    assert segments[0]["speaker"] == "Billy"
    assert segments[0]["speaker_confidence"] == 0.8
    assert segments[0]["speaker_segment_coverage"] < 0.25
    assert segments[0]["speaker_speech_share"] == 1.0


def test_attach_speakers_lowers_confidence_for_mixed_speaker_segments():
    segments = [{"start": 0.0, "end": 10.0, "text": "mixed", "speaker": "Speaker Unknown"}]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "Billy", "confidence": 0.9},
        {"start": 4.0, "end": 10.0, "speaker": "Xin", "confidence": 0.9},
    ]
    attach_speakers(segments, turns)
    assert segments[0]["speaker"] == "Xin"
    assert segments[0]["speaker_confidence"] == 0.54
    assert segments[0]["speaker_speech_share"] == 0.6


def test_explicit_cluster_backend_requires_expected_speakers(tmp_path):
    turns, status = diarize_audio(tmp_path / "missing.wav", backend="speechbrain-cluster")
    assert turns == []
    assert status["status"] == "failed"
    assert "expected_speakers" in status["reason"]
