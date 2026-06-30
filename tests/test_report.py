from meeting_minutes.report import write_quality_report


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
