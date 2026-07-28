from meeting_minutes.report import write_extractive_minutes, write_quality_report, write_review_queue


def test_quality_report_counts_anonymous_speaker_labels(tmp_path):
    path = tmp_path / "quality_report.md"
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "text": "anonymous"},
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "Alice",
            "name": "Alice",
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
            "name": "Alice",
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


def test_minutes_action_section_uses_only_the_action_ledger(tmp_path):
    path = tmp_path / "minutes.md"
    keyword_only_segment = {
        "start": 0.0,
        "end": 2.0,
        "speaker": "Alice",
        "name": "Alice",
        "name_confidence": 0.95,
        "text": "We need to schedule a two-hour MPC maintenance window.",
    }
    ledger = {
        "candidates": [
            {
                "status": "accepted",
                "start": 3.0,
                "end": 5.0,
                "owner": "Bob",
                "source_quote": "I will set the MPC upgrade timing.",
            }
        ]
    }

    write_extractive_minutes(path, segments=[keyword_only_segment], keyframes=[], metadata={}, action_ledger=ledger)
    action_section = path.read_text(encoding="utf-8").split("## Action Items", maxsplit=1)[1].split("## Key Frames", maxsplit=1)[0]

    assert "I will set the MPC upgrade timing." in action_section
    assert "two-hour MPC maintenance window" not in action_section


def test_review_queue_includes_action_candidates_that_need_review(tmp_path):
    path = tmp_path / "review_queue.md"
    ledger = {
        "candidates": [
            {
                "status": "review",
                "start": 1.0,
                "end": 2.0,
                "source_quote": "I will handle it.",
                "review_reasons": ["owner_unresolved", "topic_unresolved"],
            }
        ]
    }

    write_review_queue(path, [], action_ledger=ledger)
    text = path.read_text(encoding="utf-8")

    assert "## Action Items Needing Review" in text
    assert "owner_unresolved, topic_unresolved" in text
    assert "I will handle it." in text


def test_review_queue_includes_independently_recalled_action_intent(tmp_path):
    path = tmp_path / "review_queue.md"
    ledger = {
        "intent_recall": {
            "signals": [
                {
                    "start": 10.0,
                    "end": 12.0,
                    "cue_kind": "self_intent",
                    "candidate_ids": [],
                    "source_quote": "I want to create an issue for zero confirmation.",
                }
            ]
        }
    }

    write_review_queue(path, [], action_ledger=ledger)

    text = path.read_text(encoding="utf-8")
    assert "## Independently Recalled Action Intent" in text
    assert "unmatched_recall" in text
    assert "I want to create an issue" in text
