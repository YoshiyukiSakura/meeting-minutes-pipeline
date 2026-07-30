import json

import numpy as np
import pytest

from meeting_minutes.diarization import (
    _encode_windows_with_classifier,
    attach_speakers,
    diarize_audio,
    load_voice_enrollment,
    split_segments_by_turns,
)


class _RecordingClassifier:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def encode_batch(self, batch, wav_lens=None, normalize=False):
        self.batch_sizes.append(int(batch.shape[0]))
        return batch.new_ones((batch.shape[0], 1, 2))


def test_load_voice_enrollment_original_file_offsets_ranges(tmp_path):
    enrollment = tmp_path / "enrollment.json"
    enrollment.write_text(
        json.dumps(
            {
                "enrollment_audio_reference": "original_file",
                "speakers": {
                    "Alice": [{"start": 12.0, "end": 16.0}],
                    "Bob": [[18.0, 22.0]],
                },
            }
        ),
        encoding="utf-8",
    )
    ranges, meta = load_voice_enrollment(enrollment, source_offset=10.0, audio_duration=20.0)
    assert ranges == {"Alice": [(2.0, 6.0)], "Bob": [(8.0, 12.0)]}
    assert meta["enrollment_audio_reference"] == "original_file"


def test_load_voice_enrollment_rejects_overlapping_ranges(tmp_path):
    enrollment = tmp_path / "enrollment.json"
    enrollment.write_text(
        json.dumps(
            {
                "enrollment_audio_reference": "effective_clip",
                "speakers": {
                    "Alice": [{"start": 1.0, "end": 5.0}],
                    "Bob": [{"start": 4.5, "end": 8.0}],
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
            "speaker": "Alice",
            "confidence": 0.9,
            "name": "Alice",
            "name_source": "voice_enrollment",
            "name_confidence": 0.9,
        }
    ]
    attach_speakers(segments, turns)
    assert segments[0]["speaker"] == "Alice"
    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_source"] == "voice_enrollment"
    assert segments[0]["name_confidence"] >= 0.9


def test_attach_speakers_uses_speech_overlap_not_full_asr_duration():
    segments = [{"start": 100.0, "end": 112.0, "text": "please share the notes", "speaker": "Speaker Unknown"}]
    turns = [
        {
            "start": 101.0,
            "end": 103.5,
            "speaker": "Alice",
            "confidence": 0.8,
            "name": "Alice",
            "name_source": "user_confirmed_speaker_volume_mapping",
            "name_confidence": 0.8,
        }
    ]
    attach_speakers(segments, turns)
    assert segments[0]["speaker"] == "Alice"
    assert segments[0]["speaker_confidence"] == 0.8
    assert segments[0]["speaker_segment_coverage"] < 0.25
    assert segments[0]["speaker_speech_share"] == 1.0


def test_attach_speakers_lowers_confidence_for_mixed_speaker_segments():
    segments = [{"start": 0.0, "end": 10.0, "text": "mixed", "speaker": "Speaker Unknown"}]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "Alice", "confidence": 0.9},
        {"start": 4.0, "end": 10.0, "speaker": "Bob", "confidence": 0.9},
    ]
    attach_speakers(segments, turns)
    assert segments[0]["speaker"] == "Bob"
    assert segments[0]["speaker_confidence"] == 0.54
    assert segments[0]["speaker_speech_share"] == 0.6


def test_split_segments_by_turns_creates_speaker_homogeneous_utterances():
    segments = [
        {
            "id": "seg_00001",
            "start": 0.0,
            "end": 5.0,
            "text": "Hello there. Hi Alice.",
            "words": [
                {"word": "Hello", "start": 0.1, "end": 0.5},
                {"word": " there.", "start": 0.5, "end": 1.1},
                {"word": " Hi", "start": 3.0, "end": 3.3},
                {"word": " Alice.", "start": 3.3, "end": 4.0},
            ],
            "speaker": "Speaker Unknown",
        }
    ]
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "Alice", "confidence": 0.9},
        {"start": 2.5, "end": 4.5, "speaker": "Bob", "confidence": 0.8},
    ]

    split = split_segments_by_turns(segments, turns)
    attach_speakers(split, turns)

    assert [segment["id"] for segment in split] == ["seg_00001_01", "seg_00001_02"]
    assert [segment["split_from"] for segment in split] == ["seg_00001", "seg_00001"]
    assert [segment["text"] for segment in split] == ["Hello there.", "Hi Alice."]
    assert [segment["speaker"] for segment in split] == ["Alice", "Bob"]
    assert [segment["speaker_assignment"] for segment in split] == ["word_turn_overlap", "word_turn_overlap"]


def test_split_segments_by_turns_preserves_raw_segment_without_usable_word_timing():
    segment = {
        "id": "seg_00001",
        "start": 0.0,
        "end": 2.0,
        "text": "unaligned text",
        "words": [{"word": " unaligned", "start": 0.0}],
        "speaker": "Speaker Unknown",
    }

    split = split_segments_by_turns([segment], [{"start": 0.0, "end": 2.0, "speaker": "Alice"}])

    assert split == [segment]


def test_split_segments_by_turns_abstains_when_turn_tie_cannot_be_resolved():
    segment = {
        "id": "seg_00001",
        "start": 0.0,
        "end": 2.0,
        "text": "Hello",
        "words": [{"word": "Hello", "start": 0.5, "end": 1.5}],
        "speaker": "Speaker Unknown",
    }
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "Alice"},
        {"start": 0.0, "end": 2.0, "speaker": "Bob"},
    ]

    split = split_segments_by_turns([segment], turns)

    assert split == [segment]


def test_explicit_cluster_backend_requires_expected_speakers(tmp_path):
    turns, status = diarize_audio(tmp_path / "missing.wav", backend="speechbrain-cluster")
    assert turns == []
    assert status["status"] == "failed"
    assert "expected_speakers" in status["reason"]


def test_speechbrain_embedding_batches_default_to_small_cpu_safe_size():
    classifier = _RecordingClassifier()
    data = np.zeros(80_000, dtype="float32")
    windows = [
        {"sample_start": index * 8_000, "sample_end": (index + 1) * 8_000}
        for index in range(9)
    ]

    embeddings, valid_windows = _encode_windows_with_classifier(
        classifier,
        data,
        sample_rate=16_000,
        windows=windows,
    )

    assert classifier.batch_sizes == [4, 4, 1]
    assert len(valid_windows) == 9
    assert embeddings.shape == (9, 2)
