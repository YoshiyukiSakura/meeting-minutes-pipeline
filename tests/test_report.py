from meeting_minutes.report import write_quality_report, write_review_queue


def test_quality_report_counts_anonymous_speaker_labels(tmp_path):
    path = tmp_path / "quality_report.md"
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "text": "anonymous"},
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "Billy",
            "name": "Billy",
            "name_confidence": 0.9,
            "text": "named",
        },
        {"start": 2.0, "end": 3.0, "speaker": "Speaker Unknown", "text": "unknown"},
    ]
    write_quality_report(path, segments=segments, ocr_records=[], keyframes=[], statuses={})
    text = path.read_text(encoding="utf-8")
    assert "Real-name mapped segments: 1" in text
    assert "Segments without resolved real name: 2" in text
    assert "Anonymous speaker-label segments: 1" in text
    assert "Unknown-speaker segments: 1" in text


def test_review_queue_skips_unknown_speaker_when_visual_name_is_resolved(tmp_path):
    path = tmp_path / "review_queue.md"
    segments = [
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "Speaker Unknown",
            "name": "Xin",
            "name_confidence": 0.94,
            "text": "resolved by visual evidence",
        },
        {
            "start": 3.0,
            "end": 4.0,
            "speaker": "Speaker Unknown",
            "name": None,
            "name_confidence": 0.0,
            "text": "still unresolved",
        },
    ]

    write_review_queue(path, segments)
    text = path.read_text(encoding="utf-8")

    assert "resolved by visual evidence" not in text
    assert "still unresolved" in text
    assert "speaker_unknown, name_low_confidence" in text
