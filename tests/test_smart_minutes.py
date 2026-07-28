from __future__ import annotations

import copy
import json

from meeting_minutes.deepseek import DeepSeekConfig
from meeting_minutes.minutes_contract import validate_bilingual_minutes
from meeting_minutes.smart_minutes import (
    MAX_ACTIONS,
    MAX_ACTION_SCOUT_ACTIONS,
    SMART_MINUTES_FORMAT,
    build_hierarchical_evidence_records,
    build_review_messages,
    build_synthesis_messages,
    _deterministic_final_review_repair,
    _neutralize_future_owner_claim,
    _normalize_external_delivery_actions,
    _project_update_fallback_action,
    _targeted_final_review_repair_messages,
    build_theme_chunk_messages,
    build_theme_merge_messages,
    build_theme_outline_messages,
    canonical_transcript_records,
    flatten_theme_candidates,
    follow_up_context_hints,
    generate_smart_minutes,
    normalize_theme_chunk_coverage,
    requires_hierarchical_analysis,
    required_action_candidate_groups,
    render_smart_minutes,
    sanitize_reviewed_smart_minutes,
    theme_count_bounds,
    transcript_record_chunks,
    validate_action_scout,
    validate_implicit_follow_up_judgments,
    validate_publication_gate,
    validate_smart_minutes,
    validate_source_minutes,
    validate_theme_chunk,
    validate_theme_merge,
    validate_theme_outline,
    validate_theme_outline_coverage,
)


def _segments() -> list[dict]:
    return [
        {
            "id": "seg_1",
            "start": 0.0,
            "end": 35.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_confidence": 0.94,
            "text": "I will review the latest invoice and calculate the development and support split.",
        },
        {
            "id": "seg_2",
            "start": 35.0,
            "end": 70.0,
            "speaker": "Speaker 1",
            "name": None,
            "name_confidence": 0.0,
            "text": "We also need merge request visibility in GitLab.",
        },
    ]


def _minutes_payload(segment_ids: list[str]) -> dict:
    first, second = segment_ids
    return {
        "themes": [
            {
                "title_zh": "交付可见性与成本复核",
                "title_en": "Delivery Visibility and Cost Review",
                "current_state_zh": "发票构成和日常代码进展缺少可复核信息。",
                "current_state_en": "The invoice composition and daily code progress lack reviewable detail.",
                "outcome_zh": "团队讨论以发票拆分和 GitLab 变更记录提高透明度。",
                "outcome_en": "The team discussed using invoice breakdowns and GitLab change records to improve transparency.",
                "evidence_segment_ids": [first, second],
                "key_points": [
                    {
                        "speaker": "Billy",
                        "text_zh": "Billy 承诺复核发票并计算开发与支持占比。",
                        "text_en": "Billy committed to reviewing the invoice and calculating the development and support split.",
                        "segment_ids": [first],
                    }
                ],
            }
        ],
        "project_updates": [
            {
                "participant": "Billy",
                "project_zh": "团队交付度量",
                "project_en": "Team Delivery Metrics",
                "update_zh": "提出用 GitLab 变更记录提高日常交付可见性。",
                "update_en": "Proposed using GitLab change records to improve day-to-day delivery visibility.",
                "segment_ids": [second],
            }
        ],
        "decisions": [],
        "actions": [
            {
                "owner": "Billy",
                "item_zh": "复核最近一张发票并计算开发与支持占比。",
                "item_en": "Review the latest invoice and calculate the development and support split.",
                "segment_ids": [first],
            }
        ],
    }


def _source_payload(segment_ids: list[str], *, english: bool = False) -> dict:
    bilingual = _minutes_payload(segment_ids)
    suffix = "en" if english else "zh"
    return {
        "themes": [
            {
                "title": theme[f"title_{suffix}"],
                "current_state": theme[f"current_state_{suffix}"],
                "outcome": theme[f"outcome_{suffix}"],
                "evidence_segment_ids": theme["evidence_segment_ids"],
                "key_points": [
                    {
                        "speaker": point["speaker"],
                        "text": point[f"text_{suffix}"],
                        "segment_ids": point["segment_ids"],
                    }
                    for point in theme["key_points"]
                ],
            }
            for theme in bilingual["themes"]
        ],
        "project_updates": [
            {
                "participant": update["participant"],
                "project": update[f"project_{suffix}"],
                "update": update[f"update_{suffix}"],
                "segment_ids": update["segment_ids"],
            }
            for update in bilingual["project_updates"]
        ],
        "decisions": [
            {
                "text": decision[f"text_{suffix}"],
                "segment_ids": decision["segment_ids"],
            }
            for decision in bilingual["decisions"]
        ],
        "actions": [
            {
                "owner": action["owner"],
                "item": action[f"item_{suffix}"],
                "segment_ids": action["segment_ids"],
            }
            for action in bilingual["actions"]
        ],
    }


def _publication_review(
    minutes: dict,
    *,
    findings: list[dict] | None = None,
    prior_finding_dispositions: list[dict] | None = None,
    candidate_dispositions: list[dict] | None = None,
) -> dict:
    return {
        "findings": findings or [],
        "minutes": minutes,
        "prior_finding_dispositions": prior_finding_dispositions or [],
        "candidate_dispositions": (
            candidate_dispositions
            if candidate_dispositions is not None
            else [
                {
                    "candidate_index": index,
                    "disposition": "kept",
                    "action_index": index,
                    "reason_code": "supported",
                    "reason": "The owner makes an explicit grounded commitment.",
                }
                for index, _action in enumerate(minutes["actions"], start=1)
            ]
        ),
        "action_support": [
            {
                "action_index": index,
                "segment_ids": action["segment_ids"],
                "basis": "self_commitment",
            }
            for index, action in enumerate(minutes["actions"], start=1)
        ],
        "decision_support": [
            {
                "decision_index": index,
                "segment_ids": decision["segment_ids"],
                "basis": "explicit_agreement",
            }
            for index, decision in enumerate(minutes["decisions"], start=1)
        ],
        "publishable": True,
    }


def _action_scout(records: list[dict]) -> dict:
    return {
        "actions": [
            {
                "owner": "Billy",
                "item": "复核最新发票并计算开发与支持占比。",
                "segment_ids": [records[0]["segment_id"]],
                "basis": "self_commitment",
            }
        ]
    }


def _implicit_follow_up_scout() -> dict:
    return {"judgments": []}


def _theme_outline(records: list[dict]) -> dict:
    return {
        "themes": [
            {
                "title": "Delivery visibility and invoice review",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[-1]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "The short meeting discusses one coherent topic.",
            }
        ]
    }


def test_theme_outline_accepts_one_coherent_short_meeting_theme():
    records = canonical_transcript_records(_segments())

    messages = build_theme_outline_messages(records, expected_theme_count=1)
    outline, errors = validate_theme_outline(
        _theme_outline(records),
        transcript_records=records,
        expected_theme_count=1,
    )

    assert '"expected_theme_count":1' in messages[1]["content"]
    assert errors == []
    assert outline is not None
    assert outline[0]["start"] == 0.0
    assert outline[0]["end"] == 70.0


def test_long_theme_outline_rejects_count_overlap_and_short_residual_theme():
    records = canonical_transcript_records(
        [
            {
                "id": "topic-a",
                "start": 0.0,
                "end": 240.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "First substantive topic.",
            },
            {
                "id": "topic-b",
                "start": 180.0,
                "end": 420.0,
                "speaker": "Speaker 2",
                "name": "Xin",
                "name_confidence": 0.95,
                "text": "Second overlapping topic.",
            },
            {
                "id": "topic-c",
                "start": 3900.0,
                "end": 4010.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "Tiny closing residual topic.",
            },
        ]
    )
    raw_outline = {
        "themes": [
            {
                "title": "Topic 1",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[1]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "A material business topic.",
            },
            {
                "title": "Topic 2",
                "start_segment_id": records[1]["segment_id"],
                "end_segment_id": records[1]["segment_id"],
                "anchor_segment_ids": [records[1]["segment_id"]],
                "boundary_reason": "A second material business topic.",
            },
            {
                "title": "Topic 3",
                "start_segment_id": records[2]["segment_id"],
                "end_segment_id": records[2]["segment_id"],
                "anchor_segment_ids": [records[2]["segment_id"]],
                "boundary_reason": "A short closing topic.",
            },
        ]
    }

    count_outline, count_errors = validate_theme_outline(
        raw_outline,
        transcript_records=records,
        expected_theme_count=4,
    )
    geometry_outline, geometry_errors = validate_theme_outline(
        raw_outline,
        transcript_records=records,
        expected_theme_count=3,
    )

    assert count_outline is None
    assert count_errors == ["theme_outline_count_invalid:3!=4"]
    assert geometry_outline is None
    assert "theme_outline_ranges_overlap" in geometry_errors
    assert "theme_outline:3:span_too_short" in geometry_errors


def test_theme_outline_accepts_adjacent_segments_with_overlapping_timestamps():
    records = canonical_transcript_records(
        [
            {
                "id": "topic-a",
                "start": 0.0,
                "end": 240.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "First topic.",
            },
            {
                "id": "topic-b",
                "start": 180.0,
                "end": 420.0,
                "speaker": "Speaker 2",
                "name": "Xin",
                "name_confidence": 0.95,
                "text": "Second topic.",
            },
        ]
    )
    raw_outline = {
        "themes": [
            {
                "title": "Topic 1",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[0]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "The first topic ends before the next transcript segment.",
            },
            {
                "title": "Topic 2",
                "start_segment_id": records[1]["segment_id"],
                "end_segment_id": records[1]["segment_id"],
                "anchor_segment_ids": [records[1]["segment_id"]],
                "boundary_reason": "The second topic starts at the next transcript segment.",
            },
        ]
    }

    outline, errors = validate_theme_outline(
        raw_outline,
        transcript_records=records,
        expected_theme_count=2,
    )

    assert errors == []
    assert outline is not None


def test_theme_outline_coverage_rejects_missing_anchor_and_out_of_range_evidence():
    records = canonical_transcript_records(
        _segments()
        + [
            {
                "id": "far-topic",
                "start": 500.0,
                "end": 510.0,
                "speaker": "Speaker 2",
                "name": "Xin",
                "name_confidence": 0.95,
                "text": "A later unrelated topic.",
            }
        ]
    )
    raw_outline = _theme_outline(records[:2])
    outline, outline_errors = validate_theme_outline(
        raw_outline,
        transcript_records=records,
        expected_theme_count=1,
    )
    minutes = _source_payload(
        [records[0]["segment_id"], records[1]["segment_id"]]
    )
    minutes["themes"][0]["evidence_segment_ids"] = [
        records[1]["segment_id"],
        records[2]["segment_id"],
    ]
    minutes["themes"][0]["key_points"][0]["segment_ids"] = [
        records[1]["segment_id"]
    ]

    assert outline_errors == []
    assert outline is not None
    assert validate_theme_outline_coverage(
        minutes,
        theme_outline=outline,
        transcript_records=records,
    ) == [
        "theme:1:outline_anchor_missing",
        f"theme:1:outside_outline_range:{records[2]['segment_id']}",
    ]


def test_transcript_chunks_respect_character_budget_and_core_coverage():
    records = canonical_transcript_records(
        [
            {
                "id": f"chunk-{index}",
                "start": float(index * 10),
                "end": float(index * 10 + 8),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"topic-{index} " + ("x" * 3500),
            }
            for index in range(12)
        ]
    )

    chunks = transcript_record_chunks(
        records,
        target_chars=10_000,
        hard_chars=15_000,
        overlap_records=1,
    )

    assert len(chunks) > 1
    assert chunks[0]["core_start_position"] == 0
    assert chunks[-1]["core_end_position"] == len(records) - 1
    assert all(chunk["input_characters"] <= 15_000 for chunk in chunks)
    assert all(
        current["core_start_position"]
        == previous["core_end_position"] + 1
        for previous, current in zip(chunks, chunks[1:])
    )


def test_theme_chunk_requires_read_marker_and_complete_core_coverage():
    records = canonical_transcript_records(
        [
            {
                "id": f"topic-{index}",
                "start": float(index * 100),
                "end": float(index * 100 + 90),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Substantive topic section {index}.",
            }
            for index in range(6)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": "Substantive topic",
                "summary": "The team discussed one coherent technical topic.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[-1]["segment_id"],
                "anchor_segment_ids": [records[2]["segment_id"]],
            }
        ],
    }

    topics, errors = validate_theme_chunk(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    bad_marker = copy.deepcopy(payload)
    bad_marker["read_marker"]["last_segment_id"] = records[-2]["segment_id"]
    missing_tail = copy.deepcopy(payload)
    missing_tail["topics"][0]["end_segment_id"] = records[-2]["segment_id"]

    assert errors == []
    assert topics is not None
    assert '"last_segment_id"' in build_theme_chunk_messages(chunk)[1]["content"]
    assert validate_theme_chunk(
        bad_marker,
        chunk=chunk,
        transcript_records=records,
    )[1] == ["theme_chunk_read_marker_invalid"]
    assert (
        "theme_chunk:1:end_coverage_missing"
        in validate_theme_chunk(
            missing_tail,
            chunk=chunk,
            transcript_records=records,
        )[1]
    )


def test_theme_chunk_coverage_normalization_preserves_long_omitted_gap():
    records = canonical_transcript_records(
        [
            {
                "id": f"gap-{index}",
                "start": float(index * 20),
                "end": float(index * 20 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}.",
            }
            for index in range(8)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": "Opening topic",
                "summary": "Opening substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[1]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
            },
            {
                "title": "Closing topic",
                "summary": "Closing substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[6]["segment_id"],
                "end_segment_id": records[7]["segment_id"],
                "anchor_segment_ids": [records[6]["segment_id"]],
            },
        ],
    }

    assert "theme_chunk:1:topics_gap" in validate_theme_chunk(
        payload,
        chunk=chunk,
        transcript_records=records,
    )[1]
    normalized, changes = normalize_theme_chunk_coverage(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )

    assert changes == ["inserted_gap_topic_before:2"]
    assert errors == []
    assert topics is not None
    assert len(topics) == 3
    assert topics[1]["importance"] == "transitional"


def test_theme_chunk_normalization_trims_short_overlap_and_misplaced_anchor():
    records = canonical_transcript_records(
        [
            {
                "id": f"overlap-{index}",
                "start": float(index * 20),
                "end": float(index * 20 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}.",
            }
            for index in range(8)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": "First topic",
                "summary": "First substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[3]["segment_id"],
                "anchor_segment_ids": [
                    records[0]["segment_id"],
                    records[6]["segment_id"],
                ],
            },
            {
                "title": "Second topic",
                "summary": "Second substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[3]["segment_id"],
                "end_segment_id": records[7]["segment_id"],
                "anchor_segment_ids": [records[3]["segment_id"]],
            },
        ],
    }

    normalized, changes = normalize_theme_chunk_coverage(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )

    assert errors == []
    assert topics is not None
    assert normalized["topics"][0]["end_segment_id"] == records[2]["segment_id"]
    assert normalized["topics"][0]["anchor_segment_ids"] == [
        records[0]["segment_id"]
    ]
    assert changes == [
        "trimmed_overlap_before:2",
        f"topic:1:dropped_misplaced_anchor:{records[6]['segment_id']}",
    ]


def test_theme_chunk_normalization_expands_into_uncovered_anchor_gap():
    records = canonical_transcript_records(
        [
            {
                "id": f"anchor-gap-{index}",
                "start": float(index * 20),
                "end": float(index * 20 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}.",
            }
            for index in range(6)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": "First topic",
                "summary": "First substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[1]["segment_id"],
                "anchor_segment_ids": [
                    records[0]["segment_id"],
                    records[2]["segment_id"],
                ],
            },
            {
                "title": "Second topic",
                "summary": "Second substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[3]["segment_id"],
                "end_segment_id": records[5]["segment_id"],
                "anchor_segment_ids": [records[3]["segment_id"]],
            },
        ],
    }

    normalized, changes = normalize_theme_chunk_coverage(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )

    assert errors == []
    assert topics is not None
    assert normalized["topics"][0]["end_segment_id"] == records[2]["segment_id"]
    assert changes == [
        f"topic:1:expanded_end_for_anchor:{records[2]['segment_id']}"
    ]


def test_theme_chunk_normalization_merges_substantive_nested_topics():
    records = canonical_transcript_records(
        [
            {
                "id": f"nested-{index}",
                "start": float(index * 20),
                "end": float(index * 20 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}.",
            }
            for index in range(9)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": "Outer topic",
                "summary": "Outer summary.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[5]["segment_id"],
                "anchor_segment_ids": [
                    records[0]["segment_id"],
                    records[1]["segment_id"],
                    records[4]["segment_id"],
                    records[5]["segment_id"],
                ],
            },
            {
                "title": "Nested topic",
                "summary": "Nested summary.",
                "importance": "substantive",
                "start_segment_id": records[2]["segment_id"],
                "end_segment_id": records[3]["segment_id"],
                "anchor_segment_ids": [
                    records[2]["segment_id"],
                    records[3]["segment_id"],
                ],
            },
            {
                "title": "Closing topic",
                "summary": "Closing summary.",
                "importance": "substantive",
                "start_segment_id": records[6]["segment_id"],
                "end_segment_id": records[8]["segment_id"],
                "anchor_segment_ids": [records[6]["segment_id"]],
            },
        ],
    }

    normalized, changes = normalize_theme_chunk_coverage(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )
    normalized_again, repeated_changes = normalize_theme_chunk_coverage(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )

    assert errors == []
    assert topics is not None
    assert len(normalized["topics"]) == 2
    assert normalized["topics"][0]["title"] == "Outer topic / Nested topic"
    assert normalized["topics"][0]["summary"] == (
        "[Outer topic] Outer summary.\n[Nested topic] Nested summary."
    )
    assert normalized["topics"][0]["anchor_segment_ids"] == [
        records[index]["segment_id"]
        for index in range(6)
    ]
    assert changes == ["merged_nested_topics:1:2"]
    assert repeated_changes == []
    assert normalized_again == normalized


def test_theme_chunk_normalization_merges_equal_ranges_and_deduplicates_anchors():
    records = canonical_transcript_records(
        [
            {
                "id": f"equal-nested-{index}",
                "start": float(index * 20),
                "end": float(index * 20 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}.",
            }
            for index in range(7)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": "First view",
                "summary": "First summary.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[3]["segment_id"],
                "anchor_segment_ids": [
                    records[0]["segment_id"],
                    records[2]["segment_id"],
                ],
            },
            {
                "title": "Second view",
                "summary": "Second summary.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[3]["segment_id"],
                "anchor_segment_ids": [
                    records[1]["segment_id"],
                    records[2]["segment_id"],
                ],
            },
            {
                "title": "Closing topic",
                "summary": "Closing summary.",
                "importance": "substantive",
                "start_segment_id": records[4]["segment_id"],
                "end_segment_id": records[6]["segment_id"],
                "anchor_segment_ids": [records[4]["segment_id"]],
            },
        ],
    }

    normalized, changes = normalize_theme_chunk_coverage(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )

    assert errors == []
    assert topics is not None
    assert normalized["topics"][0]["anchor_segment_ids"] == [
        records[index]["segment_id"]
        for index in (0, 1, 2)
    ]
    assert changes == ["merged_nested_topics:1:2"]


def test_theme_chunk_normalization_keeps_long_partial_overlap_invalid():
    records = canonical_transcript_records(
        [
            {
                "id": f"partial-overlap-{index}",
                "start": float(index * 20),
                "end": float(index * 20 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}.",
            }
            for index in range(8)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": "First topic",
                "summary": "First substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[5]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
            },
            {
                "title": "Second topic",
                "summary": "Second substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[3]["segment_id"],
                "end_segment_id": records[7]["segment_id"],
                "anchor_segment_ids": [records[3]["segment_id"]],
            },
        ],
    }

    normalized, changes = normalize_theme_chunk_coverage(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )

    assert topics is None
    assert errors == ["theme_chunk:1:topics_overlap"]
    assert changes == []


def test_theme_chunk_normalization_folds_gap_when_topic_limit_is_reached():
    records = canonical_transcript_records(
        [
            {
                "id": f"limit-{index}",
                "start": float(index * 25),
                "end": float(index * 25 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}.",
            }
            for index in range(16)
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    ranges = [(0, 0), *[(index, index + 1) for index in range(2, 16, 2)]]
    payload = {
        "chunk_index": 1,
        "read_marker": {
            "record_count": len(records),
            "last_segment_id": records[-1]["segment_id"],
        },
        "topics": [
            {
                "title": f"Topic {index}",
                "summary": "Substantive discussion.",
                "importance": "substantive",
                "start_segment_id": records[start]["segment_id"],
                "end_segment_id": records[end]["segment_id"],
                "anchor_segment_ids": [records[start]["segment_id"]],
            }
            for index, (start, end) in enumerate(ranges, start=1)
        ],
    }

    normalized, changes = normalize_theme_chunk_coverage(
        payload,
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        normalized,
        chunk=chunk,
        transcript_records=records,
    )

    assert errors == []
    assert topics is not None
    assert len(topics) == 8
    assert normalized["topics"][1]["start_segment_id"] == records[1]["segment_id"]
    assert "inserted_gap_topic_before:2" in changes
    assert "folded_gap_into_following_topic" in changes


def test_theme_merge_consumes_every_candidate_and_rejects_omitted_tail():
    records = canonical_transcript_records(
        [
            {
                "id": "first-half",
                "start": 0.0,
                "end": 4000.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "First half topic.",
            },
            {
                "id": "tail",
                "start": 8000.0,
                "end": 8506.68,
                "speaker": "Speaker 2",
                "name": "Xin",
                "name_confidence": 0.95,
                "text": "Second half topic and final follow-up.",
            },
        ]
    )
    candidates = flatten_theme_candidates(
        [
            [
                {
                    "chunk_index": 1,
                    "local_topic_index": 1,
                    "title": "First half",
                    "summary": "First substantive discussion.",
                    "importance": "substantive",
                    "start_segment_id": records[0]["segment_id"],
                    "end_segment_id": records[0]["segment_id"],
                    "anchor_segment_ids": [records[0]["segment_id"]],
                    "start_position": 0,
                    "end_position": 0,
                    "start": 0.0,
                    "end": 4000.0,
                }
            ],
            [
                {
                    "chunk_index": 2,
                    "local_topic_index": 1,
                    "title": "Second half",
                    "summary": "Later substantive discussion.",
                    "importance": "substantive",
                    "start_segment_id": records[1]["segment_id"],
                    "end_segment_id": records[1]["segment_id"],
                    "anchor_segment_ids": [records[1]["segment_id"]],
                    "start_position": 1,
                    "end_position": 1,
                    "start": 8000.0,
                    "end": 8506.68,
                }
            ],
        ]
    )
    marker = {"candidate_count": 2, "last_candidate_index": 2}
    valid = {
        "read_marker": marker,
        "themes": [
            {
                "title": "Complete meeting",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[1]["segment_id"],
                "anchor_segment_ids": [
                    records[0]["segment_id"],
                    records[1]["segment_id"],
                ],
                "boundary_reason": "Both adjacent candidates form the complete discussion.",
                "source_candidate_indexes": [1, 2],
            }
        ],
    }
    omitted_tail = {
        "read_marker": marker,
        "themes": [
            {
                "title": "Only the first half",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[0]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "Incorrectly omits the later candidate.",
                "source_candidate_indexes": [1],
            }
        ],
    }

    outline, errors = validate_theme_merge(
        valid,
        candidates=candidates,
        transcript_records=records,
        min_theme_count=1,
        max_theme_count=2,
    )
    omitted_outline, omitted_errors = validate_theme_merge(
        omitted_tail,
        candidates=candidates,
        transcript_records=records,
        min_theme_count=1,
        max_theme_count=2,
    )

    assert errors == []
    assert outline is not None
    assert omitted_outline is None
    assert "theme_merge_candidate_uncovered:2" in omitted_errors
    assert "theme_outline_end_coverage_missing" in omitted_errors
    assert '"candidate_count":2' in build_theme_merge_messages(
        candidates,
        transcript_records=records,
        min_theme_count=1,
        max_theme_count=2,
    )[1]["content"]


def test_long_meeting_theme_count_range_and_compact_evidence_packet():
    records = canonical_transcript_records(
        [
            {
                "id": f"long-{index}",
                "start": float(index * 1000),
                "end": float(index * 1000 + 900),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Long meeting section {index}.",
            }
            for index in range(9)
        ]
    )
    candidates = [
        {
            "candidate_index": index + 1,
            "chunk_index": index + 1,
            "local_topic_index": 1,
            "title": f"Topic {index + 1}",
            "summary": "A substantive topic.",
            "importance": "substantive",
            "start_segment_id": record["segment_id"],
            "end_segment_id": record["segment_id"],
            "anchor_segment_ids": [record["segment_id"]],
            "start_position": index,
            "end_position": index,
            "start": record["start"],
            "end": record["end"],
        }
        for index, record in enumerate(records)
    ]
    outline = [
        {
            "title": candidate["title"],
            "start_segment_id": candidate["start_segment_id"],
            "end_segment_id": candidate["end_segment_id"],
            "anchor_segment_ids": candidate["anchor_segment_ids"],
            "boundary_reason": "Validated local topic.",
            "start": candidate["start"],
            "end": candidate["end"],
            "start_position": candidate["start_position"],
            "end_position": candidate["end_position"],
            "source_candidate_indexes": [candidate["candidate_index"]],
        }
        for candidate in candidates
    ]

    evidence = build_hierarchical_evidence_records(
        records,
        theme_candidates=candidates,
        theme_outline=outline,
        action_scout=[],
        required_action_groups=[],
        required_project_participants=["Billy"],
    )

    assert theme_count_bounds(
        canonical_transcript_records(
            [
                {
                    "id": "full-duration",
                    "start": 0.0,
                    "end": 8506.68,
                    "speaker": "Speaker 1",
                    "name": "Billy",
                    "name_confidence": 0.95,
                    "text": "Full duration marker.",
                }
            ]
        )
    ) == (5, 8)
    assert len(evidence) == len(records)


def test_hierarchical_long_meeting_never_sends_full_transcript_and_resumes(
    monkeypatch,
):
    segment_count = 360
    duration = 8506.68
    step = duration / segment_count
    segments = [
        {
            "id": f"long-{index}",
            "start": index * step,
            "end": min(duration, index * step + 18.0),
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_confidence": 0.95,
            "text": f"technical discussion section {index} " + ("x" * 1100),
        }
        for index in range(segment_count)
    ]
    records = canonical_transcript_records(segments)
    request_sizes: list[int] = []
    checkpoints: list[dict] = []

    def fake_request(*, messages, config, max_tokens=16000):
        request_sizes.append(sum(len(message["content"]) for message in messages))
        system = messages[0]["content"]
        user = json.loads(messages[1]["content"])
        status = {"status": "ok", "requested_model": config.model}
        if "dedicated action-item scout" in system:
            return {"actions": []}, status
        if "focused implicit-follow-up adjudicator" in system:
            return {"judgments": []}, status
        if "local meeting topic-boundary analyst" in system:
            supplied = user["transcript_chunk"]
            supplied_ids = [
                record["segment_id"]
                for record in supplied
            ]
            start_index = supplied_ids.index(user["core_start_segment_id"])
            end_index = supplied_ids.index(user["core_end_segment_id"])
            anchor_id = supplied[(start_index + end_index) // 2]["segment_id"]
            return {
                "chunk_index": user["chunk_index"],
                "read_marker": user["read_marker"],
                "topics": [
                    {
                        "title": f"topic {user['chunk_index']}",
                        "summary": "one substantive technical discussion block",
                        "importance": "substantive",
                        "start_segment_id": user["core_start_segment_id"],
                        "end_segment_id": user["core_end_segment_id"],
                        "anchor_segment_ids": [anchor_id],
                    }
                ],
            }, status
        if "global meeting theme merger" in system:
            candidates = user["topic_candidates"]
            target = min(
                user["max_theme_count"],
                max(user["min_theme_count"], len(candidates)),
            )
            themes = []
            for index in range(target):
                start = index * len(candidates) // target
                end = (index + 1) * len(candidates) // target
                group = candidates[start:end]
                themes.append(
                    {
                        "title": f"merged topic {index + 1}",
                        "start_segment_id": group[0]["start_segment_id"],
                        "end_segment_id": group[-1]["end_segment_id"],
                        "anchor_segment_ids": list(
                            dict.fromkeys(
                                [
                                    group[0]["anchor_segment_ids"][0],
                                    group[-1]["anchor_segment_ids"][-1],
                                ]
                            )
                        ),
                        "boundary_reason": "adjacent local topics form one coherent block",
                        "source_candidate_indexes": [
                            candidate["candidate_index"]
                            for candidate in group
                        ],
                    }
                )
            return {
                "read_marker": user["read_marker"],
                "themes": themes,
            }, status
        if "primary meeting-minutes analyst" in system:
            themes = [
                {
                    "title": f"技术议题 {index}",
                    "current_state": "讨论了当前实现状态。",
                    "outcome": "保留了有证据支持的讨论结论。",
                    "evidence_segment_ids": list(
                        dict.fromkeys(
                            [
                                outline["start_segment_id"],
                                outline["end_segment_id"],
                            ]
                        )
                    ),
                    "key_points": [
                        {
                            "speaker": "Billy",
                            "text": "说明了该技术议题的主要约束。",
                            "segment_ids": [outline["anchor_segment_ids"][0]],
                        }
                    ],
                }
                for index, outline in enumerate(user["theme_outline"], start=1)
            ]
            return {
                "themes": themes,
                "project_updates": [
                    {
                        "participant": "Billy",
                        "project": "技术交付",
                        "update": "说明了当前实现状态与依赖。",
                        "segment_ids": [
                            user["transcript"][0]["segment_id"]
                        ],
                    }
                ],
                "decisions": [],
                "actions": [],
            }, status
        if "independent coverage and evidence reviewer" in system:
            return {"findings": [], "minutes": user["draft_minutes"]}, status
        if "final publication adjudicator" in system:
            return _publication_review(user["draft_minutes"]), status
        if "professional meeting-minutes translator" in system:
            source = copy.deepcopy(user["source_minutes_zh"])
            for index, theme in enumerate(source["themes"], start=1):
                theme["title"] = f"technical topic {index}"
                theme["current_state"] = "current implementation status was discussed"
                theme["outcome"] = "the evidence-backed discussion outcome was retained"
                for point in theme["key_points"]:
                    point["text"] = "the main technical constraint was explained"
            for update in source["project_updates"]:
                update["project"] = "technical delivery"
                update["update"] = "implementation status and dependencies were explained"
            return source, status
        raise AssertionError(f"unexpected model stage: {system[:80]}")

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    assert requires_hierarchical_analysis(records) is True
    result, status = generate_smart_minutes(
        segments=segments,
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert result is not None, status
    assert status["status"] == "reviewed_draft"
    assert result.audit["analysis_mode"] == "hierarchical"
    assert result.audit["model_evidence_record_count"] < len(records) // 2
    assert max(request_sizes) < 120_000
    completed_checkpoint = checkpoints[-1]

    def unexpected_request(**kwargs):
        raise AssertionError("network should not be called for cached hierarchical stages")

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", unexpected_request)
    resumed, resumed_status = generate_smart_minutes(
        segments=segments,
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint=completed_checkpoint,
    )

    assert resumed is not None
    assert resumed_status["status"] == "reviewed_draft"


def test_canonical_transcript_propagates_high_confidence_cluster_name():
    records = canonical_transcript_records(_segments())

    assert [record["speaker"] for record in records] == ["Billy", "Billy"]
    assert records[0]["identity_kind"] == "direct"
    assert records[1]["identity_kind"] == "cluster_consensus"
    assert records[1]["identity_confidence"] == 0.94


def test_canonical_transcript_is_sorted_without_inflating_identity_confidence():
    segments = list(reversed(_segments()))

    records = canonical_transcript_records(segments)

    assert [record["start"] for record in records] == [0.0, 35.0]
    assert records[1]["speaker"] == "Billy"
    assert records[1]["identity_confidence"] == 0.94


def test_action_limits_are_consistent_across_scout_and_minutes():
    assert MAX_ACTIONS == MAX_ACTION_SCOUT_ACTIONS == 24


def test_synthesis_prompt_contains_named_transcript_and_quality_contract():
    records = canonical_transcript_records(_segments())
    groups = required_action_candidate_groups(records)
    messages = build_synthesis_messages(
        records,
        required_project_participants=["Billy"],
        required_action_groups=groups,
    )

    assert '"speaker":"Billy"' in messages[1]["content"]
    assert "exactly one minutes theme per outline entry" in messages[0]["content"]
    assert "An action requires an explicit self-commitment" in messages[0]["content"]
    assert "required_action_candidate_groups" in messages[1]["content"]
    assert '"topic":"invoice_review"' in messages[1]["content"]


def test_required_action_groups_use_canonical_identity_and_deduplicate_topics():
    segments = _segments() + [
        {
            "start": 75.0,
            "end": 90.0,
            "speaker": "Speaker 1",
            "name": None,
            "name_confidence": 0.0,
            "text": "I will review the invoice again.",
        }
    ]
    records = canonical_transcript_records(segments)

    groups = required_action_candidate_groups(records)

    assert len(groups) == 1
    assert groups[0]["owner"] == "Billy"
    assert groups[0]["topic"] == "invoice_review"
    assert len(groups[0]["candidates"]) == 2


def test_follow_up_context_hints_surface_implicit_ownership_language():
    segments = [
        {
            "start": 0.0,
            "end": 12.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_confidence": 0.94,
            "text": "We are working on moving the team from GitHub to GitLab.",
        },
        {
            "start": 60.0,
            "end": 72.0,
            "speaker": "Speaker 2",
            "name": "John",
            "name_confidence": 0.94,
            "text": "They should have the contract by Wednesday.",
        },
    ]
    records = canonical_transcript_records(segments)

    hints = follow_up_context_hints(records)

    assert [hint["anchor_speaker"] for hint in hints] == ["Billy", "John"]
    assert hints[0]["context"][0]["text"].startswith("We are working on")
    assert hints[1]["context"][0]["text"].startswith("They should have")


def test_implicit_follow_up_judgment_becomes_a_grounded_action():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.94,
                "text": "We are working on getting the team source-code access.",
            },
            {
                "start": 13.0,
                "end": 20.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "We want them to move from GitHub to GitLab.",
            },
        ]
    )
    hints = follow_up_context_hints(records)
    payload = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": True,
                "owner": "Billy",
                "item": "推进团队从 GitHub 迁移到 GitLab。",
                "segment_ids": [
                    records[0]["segment_id"],
                    records[1]["segment_id"],
                ],
                "reason": "Billy 说明该迁移工作正在推进。",
            }
        ]
    }

    actions, errors = validate_implicit_follow_up_judgments(
        payload,
        follow_up_hints=hints,
        transcript_records=records,
    )

    assert errors == []
    assert actions == [
        {
            "owner": "Billy",
            "item": "推进团队从 GitHub 迁移到 GitLab。",
            "segment_ids": [
                records[0]["segment_id"],
                records[1]["segment_id"],
            ],
            "basis": "owned_follow_up",
            "must_keep": True,
        }
    ]


def test_implicit_external_delivery_status_is_not_an_action():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.94,
                "text": "I already gave them all the input they needed.",
            },
            {
                "start": 13.0,
                "end": 24.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "They should have the contract by Wednesday.",
            },
        ]
    )
    hints = follow_up_context_hints(records)
    payload = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": True,
                "owner": "John",
                "item": "跟进咨询公司合同进展，合同预计周三完成。",
                "segment_ids": [
                    records[0]["segment_id"],
                    records[1]["segment_id"],
                ],
                "reason": "John 已提供所需输入，并说明合同预计周三完成。",
            }
        ]
    }

    actions, errors = validate_implicit_follow_up_judgments(
        payload,
        follow_up_hints=hints,
        transcript_records=records,
    )

    assert actions is None
    assert errors == [
        "implicit_follow_up:1:external_delivery_status_not_action"
    ]


def test_implicit_follow_up_requires_deterministic_owned_support():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.94,
                "text": "They should have a proposal by Wednesday.",
            }
        ]
    )
    hints = follow_up_context_hints(records)
    payload = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": True,
                "owner": "John",
                "item": "跟进并交付提案。",
                "segment_ids": [records[0]["segment_id"]],
                "reason": "提到了未来交付时间。",
            }
        ]
    }

    actions, errors = validate_implicit_follow_up_judgments(
        payload,
        follow_up_hints=hints,
        transcript_records=records,
    )

    assert actions is None
    assert errors == [
        "implicit_follow_up:1:owned_follow_up_not_grounded"
    ]


def test_started_work_plus_owned_outcome_is_a_hint_not_a_forced_judgment():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.94,
                "text": "We are working on getting the team access.",
            },
            {
                "start": 13.0,
                "end": 20.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "We want them to move from GitHub to GitLab.",
            },
        ]
    )
    hints = follow_up_context_hints(records)
    assert hints[0]["signals"]["strong_local_signal"] is True
    payload = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": False,
                "owner": "",
                "item": "",
                "segment_ids": [],
                "reason": "Incorrectly treated as status only.",
            }
        ]
    }

    actions, errors = validate_implicit_follow_up_judgments(
        payload,
        follow_up_hints=hints,
        transcript_records=records,
    )

    assert actions == []
    assert errors == []


def test_action_scout_ignores_non_public_top_level_metadata(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    scout = _action_scout(records)
    scout["analysis_note"] = "internal model metadata"
    responses = [
        scout,
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    assert status["status"] == "reviewed_draft"


def test_action_scout_narrows_repeated_distant_commitment_evidence():
    segments = _segments() + [
        {
            "start": 500.0,
            "end": 520.0,
            "speaker": "Speaker 1",
            "name": None,
            "name_confidence": 0.0,
            "text": "I will review the invoice again.",
        }
    ]
    records = canonical_transcript_records(segments)
    groups = required_action_candidate_groups(records)
    payload = {
        "actions": [
            {
                "owner": "Billy",
                "item": "复核最新发票。",
                "segment_ids": [
                    groups[0]["candidates"][0]["segment_id"],
                    groups[0]["candidates"][1]["segment_id"],
                ],
                "basis": "self_commitment",
            }
        ]
    }

    actions, errors = validate_action_scout(
        payload,
        transcript_records=records,
        required_action_groups=groups,
    )

    assert errors == []
    assert actions is not None
    assert actions[0]["segment_ids"] == [groups[0]["candidates"][0]["segment_id"]]


def test_final_gate_allows_a_scout_candidate_to_be_explicitly_rejected():
    records = canonical_transcript_records(_segments())
    scout = _action_scout(records)["actions"]
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["actions"] = []
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "rejected",
                "action_index": None,
                "reason_code": "unsupported_item",
                "reason": "The candidate does not survive the final commitment test.",
            }
        ],
    )

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == []


def test_final_gate_rejects_a_scout_candidate_without_a_disposition():
    records = canonical_transcript_records(_segments())
    scout = _action_scout(records)["actions"]
    minutes = _source_payload([record["segment_id"] for record in records])
    review = _publication_review(minutes, candidate_dispositions=[])

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == [
        "publication_gate_candidate_disposition_coverage_mismatch"
    ]


def test_final_gate_rejects_false_unsupported_commitment_reason_for_owned_work():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 20.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.94,
                "text": "We are working on getting the team source-code access.",
            },
            {
                "start": 21.0,
                "end": 35.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "We want them to move from GitHub to GitLab.",
            },
        ]
    )
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["actions"] = []
    scout = [
        {
            "owner": "Billy",
            "item": "推动团队从 GitHub 迁移至 GitLab。",
            "segment_ids": [record["segment_id"] for record in records],
            "basis": "owned_follow_up",
        }
    ]
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "rejected",
                "action_index": None,
                "reason_code": "unsupported_commitment",
                "reason": "This is only a status update, not a new commitment.",
            }
        ],
    )

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == [
        "publication_gate_candidate_disposition:1:"
        "unsupported_commitment_conflicts_with_evidence"
    ]


def test_final_gate_allows_rejection_of_an_incomplete_self_commitment():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "Xin",
                "name_confidence": 0.94,
                "text": "Eventually I will ask AI to move out but send to.",
            }
        ]
    )
    minutes = {
        "themes": [],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }
    scout = [
        {
            "owner": "Xin",
            "item": "将日志迁移到现有中心。",
            "segment_ids": [records[0]["segment_id"]],
            "basis": "self_commitment",
        }
    ]
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "rejected",
                "action_index": None,
                "reason_code": "unsupported_commitment",
                "reason": "The transcript is an incomplete thought, not a concrete follow-up.",
            }
        ],
    )

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == []


def test_final_gate_allows_duplicate_candidates_to_map_to_one_matching_action():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["actions"][0]["item"] = "发送会议记录到频道。"
    scout = [
        {
            "owner": "Billy",
            "item": "会后发送会议记录。",
            "segment_ids": [records[1]["segment_id"]],
            "basis": "self_commitment",
        }
    ]
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "The matching action retains the same owner and outcome.",
            }
        ],
    )

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == []


def test_final_gate_allows_rejection_of_external_delivery_status():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 20.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.94,
                "text": "I already gave them all the input they needed.",
            },
            {
                "start": 21.0,
                "end": 35.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "They should have the contract by Wednesday.",
            },
        ]
    )
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["actions"] = []
    scout = [
        {
            "owner": "John",
            "item": "跟进合同并在周三提供进展。",
            "segment_ids": [record["segment_id"] for record in records],
            "basis": "owned_follow_up",
            "must_keep": True,
        }
    ]
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "rejected",
                "action_index": None,
                "reason_code": "unsupported_item",
                "reason": "The reviewer considers this only a status update.",
            }
        ],
    )

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == []


def test_final_gate_normalizes_external_delivery_completion_guarantee():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.94,
                "text": "I will follow up with them, and I already gave them all the input they needed.",
            },
            {
                "start": 13.0,
                "end": 24.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "They should have the contract by Wednesday.",
            },
        ]
    )
    action = {
        "owner": "John",
        "item": "跟进咨询公司，确保合同在周三前完成",
        "segment_ids": [record["segment_id"] for record in records],
    }
    minutes = {"actions": [action], "decisions": []}
    scout = [
        {
            **action,
            "basis": "owned_follow_up",
            "must_keep": True,
            "external_delivery_update": True,
        }
    ]
    review = {
        "findings": [],
        "minutes": minutes,
        "prior_finding_dispositions": [],
        "candidate_dispositions": [
            {
                "candidate_index": 1,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "John owns the follow-up on the contract status.",
            }
        ],
        "action_support": [
            {
                "action_index": 1,
                "segment_ids": action["segment_ids"],
                "basis": "owned_follow_up",
            }
        ],
        "decision_support": [],
        "publishable": True,
    }

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )
    repaired, changes = _normalize_external_delivery_actions(
        review,
        action_scout=scout,
    )

    assert errors == [
        "publication_gate_candidate_disposition:1:"
        "external_delivery_completion_guarantee"
    ]
    assert repaired is not None
    assert changes == ["normalized_external_delivery_action:1"]
    assert repaired["minutes"]["actions"][0]["item"] == (
        "跟进咨询公司合同进展，合同预计周三完成"
    )
    assert validate_publication_gate(
        repaired,
        repaired["minutes"],
        transcript_records=records,
        action_scout=scout,
    ) == []


def test_deterministic_final_repair_drops_external_delivery_status_action():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.94,
                "text": "I already gave them all the input they needed.",
            },
            {
                "start": 13.0,
                "end": 24.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "They should have the contract by Wednesday.",
            },
        ]
    )
    action = {
        "owner": "John",
        "item": "跟进合同进展，推动咨询公司在周三前交付合同",
        "segment_ids": [record["segment_id"] for record in records],
    }
    review = {
        "minutes": {
            "themes": [],
            "project_updates": [],
            "decisions": [],
            "actions": [action],
        },
        "candidate_dispositions": [
            {
                "candidate_index": 1,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "The status was incorrectly treated as a follow-up.",
            }
        ],
        "action_support": [
            {
                "action_index": 1,
                "segment_ids": action["segment_ids"],
                "basis": "owned_follow_up",
            }
        ],
    }
    scout = [
        {
            **action,
            "basis": "owned_follow_up",
            "must_keep": True,
        }
    ]

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "publication_gate_action_support:1:external_delivery_status_not_action"
        ],
        action_scout=scout,
    )

    assert repaired is not None
    assert changes == ["dropped_external_delivery_status_action:1"]
    assert repaired["minutes"]["actions"] == []
    assert repaired["action_support"] == []
    assert repaired["candidate_dispositions"][0] == {
        "candidate_index": 1,
        "disposition": "rejected",
        "action_index": None,
        "reason_code": "unsupported_commitment",
        "reason": (
            "The cited evidence reports a third-party delivery estimate, not an "
            "owner follow-up commitment."
        ),
    }


def test_final_gate_rejects_negated_self_commitment_support():
    segments = copy.deepcopy(_segments())
    segments[0]["text"] = "I will not review the latest invoice."
    records = canonical_transcript_records(segments)
    minutes = _source_payload([record["segment_id"] for record in records])
    scout = _action_scout(records)["actions"]
    review = _publication_review(minutes)

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == [
        "publication_gate_action_support:1:self_commitment_not_grounded"
    ]


def test_final_gate_rejects_wontfix_for_material_prior_finding():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    scout = _action_scout(records)["actions"]
    prior_findings = [
        {
            "severity": "medium",
            "category": "action_atomicity",
            "description": "The action combines separate outcomes.",
        }
    ]
    review = _publication_review(
        minutes,
        prior_finding_dispositions=[
            {
                "finding_index": 1,
                "disposition": "wontfix",
                "reason": "The final reviewer prefers the combined wording.",
            }
        ],
    )

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
        prior_findings=prior_findings,
    )

    assert errors == [
        "publication_gate_prior_finding_disposition:1:"
        "material_finding_unresolved"
    ]


def test_final_review_prompt_is_a_precision_first_publication_gate():
    records = canonical_transcript_records(_segments())
    draft = _source_payload([record["segment_id"] for record in records])

    messages = build_review_messages(
        records,
        draft,
        required_project_participants=["Billy"],
        pass_index=2,
    )

    assert "final publication adjudicator" in messages[0]["content"]
    assert "There is no desired action count" in messages[0]["content"]
    assert "Never combine two follow-ups" in messages[0]["content"]
    assert "within a 120-second window" in messages[0]["content"]
    assert "candidate_dispositions" in messages[0]["content"]
    assert "prior_finding_dispositions" in messages[0]["content"]


def test_final_review_repair_uses_targeted_evidence_packet():
    records = canonical_transcript_records(_segments())
    draft = _source_payload([record["segment_id"] for record in records])
    outline, errors = validate_theme_outline(
        _theme_outline(records),
        transcript_records=records,
        expected_theme_count=1,
    )
    assert errors == []
    assert outline is not None
    messages = build_review_messages(
        records,
        draft,
        required_project_participants=["Billy"],
        action_scout=_action_scout(records)["actions"],
        theme_outline=outline,
        pass_index=2,
    )

    repair_messages = _targeted_final_review_repair_messages(
        base_messages=messages,
        payload=_publication_review(draft),
        errors=["action:1:named_entity_ungrounded:Xin"],
        transcript_records=records,
        action_scout=_action_scout(records)["actions"],
        prior_findings=[],
        theme_outline=outline,
        required_project_participants=["Billy"],
    )
    repair_payload = json.loads(repair_messages[1]["content"])

    assert len(repair_messages) == 2
    assert repair_payload["invalid_review"]["minutes"] == draft
    assert repair_payload["transcript_evidence"] == records
    assert "draft_minutes" not in repair_payload
    assert "required_action_candidate_groups" not in repair_payload


def test_deterministic_final_repair_drops_invalid_action_and_rebuilds_update():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["actions"].append(
        {
            "owner": "Billy",
            "item": "向 Xin 发送后续报告。",
            "segment_ids": [records[1]["segment_id"]],
        }
    )
    minutes["project_updates"][0]["update"] = "向 Xin 发送后续报告。"
    scout = _action_scout(records)["actions"]
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "Billy explicitly commits to the invoice review.",
            }
        ],
    )

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "publication_gate_action_support:2:self_commitment_not_grounded",
            "project_update:1:named_entity_ungrounded:Xin",
        ],
        action_scout=scout,
    )

    assert repaired is not None
    assert changes == [
        "dropped_unsupported_action:2",
        "rebuilt_project_update:1",
    ]
    assert len(repaired["minutes"]["actions"]) == 1
    assert repaired["minutes"]["project_updates"][0]["update"] == (
        "复核最近一张发票并计算开发与支持占比。"
    )

    cleaned, source_errors = validate_source_minutes(
        repaired["minutes"],
        transcript_records=records,
        required_project_participants=["Billy"],
    )
    assert source_errors == []
    assert cleaned is not None
    assert validate_publication_gate(
        repaired,
        cleaned,
        transcript_records=records,
        action_scout=scout,
    ) == []


def test_deterministic_final_repair_drops_out_of_outline_theme_point():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["key_points"] = [
        {
            "speaker": "Billy",
            "text": "范围内的要点。",
            "segment_ids": [records[0]["segment_id"]],
        },
        {
            "speaker": "Billy",
            "text": "范围外的要点。",
            "segment_ids": [records[1]["segment_id"]],
        },
    ]
    minutes["themes"][0]["evidence_segment_ids"] = [
        records[0]["segment_id"],
        records[1]["segment_id"],
    ]
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "theme:1:outside_outline_range:"
            f"{records[1]['segment_id']}"
        ],
        action_scout=_action_scout(records)["actions"],
    )

    assert repaired is not None
    assert changes == ["dropped_out_of_outline_theme_evidence:1"]
    assert repaired["minutes"]["themes"][0]["evidence_segment_ids"] == [
        records[0]["segment_id"]
    ]
    assert repaired["minutes"]["themes"][0]["key_points"] == [
        {
            "speaker": "Billy",
            "text": "范围内的要点。",
            "segment_ids": [records[0]["segment_id"]],
        }
    ]


def test_neutralize_future_owner_claim_preserves_decimal_versions():
    outcome = _neutralize_future_owner_claim(
        (
            "讨论形成方向，将探索Web Payment Box 2.0的销售策略，"
            "并优先为Xin提供BPS访问权限以进行测试。"
            "业务需求文档将由John Osorno Jr整理。"
        ),
        "Xin",
    )

    assert outcome == (
        "讨论形成方向，将探索Web Payment Box 2.0的销售策略，"
        "相关后续安排待确认。业务需求文档将由John Osorno Jr整理。"
    )


def test_project_update_fallback_prefers_substantive_work_over_notification():
    action = _project_update_fallback_action(
        [
            {
                "owner": "Xin",
                "item": "将演示链接发送到频道。",
                "segment_ids": ["seg-1"],
            },
            {
                "owner": "Xin",
                "item": "以商户身份自行入驻系统进行测试。",
                "segment_ids": ["seg-2"],
            },
        ],
        "Xin",
    )

    assert action is not None
    assert action["segment_ids"] == ["seg-2"]


def test_validator_rejects_action_owner_without_matching_speech():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["actions"][0]["owner"] = "Xin"

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "action:1:owner_unknown" in errors


def test_validator_rejects_anonymous_speaker_reference_in_theme_prose():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["themes"][0]["current_state_zh"] = (
        "Speaker 5 提出了需要补齐的功能。"
    )
    payload["themes"][0]["current_state_en"] = (
        "Speaker 5 proposed the missing features."
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert errors == ["theme:1:current_state_anonymous_speaker_reference"]


def test_validator_rejects_anonymous_speaker_reference_only_in_english_prose():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["themes"][0]["current_state_en"] = (
        "Speaker 5 proposed the missing features."
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert errors == ["theme:1:current_state_anonymous_speaker_reference"]


def test_sanitize_reviewed_minutes_neutralizes_anonymous_labels_without_changing_evidence():
    segments = _segments()
    records = canonical_transcript_records(segments)
    minutes = _minutes_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["current_state_zh"] = "Speaker 5 提出了需要补齐的功能。"
    minutes["themes"][0]["current_state_en"] = "Speaker 5 proposed the missing features."
    source = {"format": SMART_MINUTES_FORMAT, "minutes": minutes}
    original_evidence = copy.deepcopy(minutes["themes"][0]["evidence_segment_ids"])
    original_owner = minutes["actions"][0]["owner"]
    original_action_ids = copy.deepcopy(minutes["actions"][0]["segment_ids"])

    result, errors = sanitize_reviewed_smart_minutes(source, segments=segments)

    assert errors == []
    assert result is not None
    cleaned_theme = result.payload["minutes"]["themes"][0]
    assert cleaned_theme["current_state_zh"] == "一名参会者提出了需要补齐的功能。"
    assert cleaned_theme["current_state_en"] == "An attendee proposed the missing features."
    assert cleaned_theme["evidence_segment_ids"] == original_evidence
    assert result.payload["minutes"]["actions"][0]["owner"] == original_owner
    assert result.payload["minutes"]["actions"][0]["segment_ids"] == original_action_ids
    assert "Speaker 5" not in result.chinese_markdown
    assert "Speaker 5" not in result.english_markdown


def test_sanitize_reviewed_minutes_drops_audited_external_delivery_status_action():
    segments = _segments() + [
        {
            "start": 70.0,
            "end": 82.0,
            "speaker": "Speaker 2",
            "name": "John",
            "name_confidence": 0.94,
            "text": "I already gave them all the input they needed.",
        },
        {
            "start": 83.0,
            "end": 94.0,
            "speaker": "Speaker 2",
            "name": None,
            "name_confidence": 0.0,
            "text": "They should have the contract by Wednesday.",
        },
    ]
    records = canonical_transcript_records(segments)
    minutes = _minutes_payload([record["segment_id"] for record in records[:2]])
    external_action = {
        "owner": "John",
        "item_zh": "跟进合同进展，推动咨询公司在周三前交付合同",
        "item_en": "Follow up and push the consulting firm to deliver by Wednesday",
        "segment_ids": [record["segment_id"] for record in records[2:]],
    }
    minutes["actions"].append(external_action)
    source = {"format": SMART_MINUTES_FORMAT, "minutes": minutes}
    chinese_source = _source_payload([record["segment_id"] for record in records[:2]])
    chinese_source["actions"].append(
        {
            "owner": external_action["owner"],
            "item": external_action["item_zh"],
            "segment_ids": external_action["segment_ids"],
        }
    )
    action_scout = [
        {
            "owner": "Billy",
            "item": chinese_source["actions"][0]["item"],
            "segment_ids": chinese_source["actions"][0]["segment_ids"],
            "basis": "self_commitment",
            "must_keep": True,
        },
        {
            "owner": external_action["owner"],
            "item": external_action["item_zh"],
            "segment_ids": external_action["segment_ids"],
            "basis": "owned_follow_up",
            "must_keep": True,
        },
    ]
    final_review = {
        "findings": [],
        "prior_finding_dispositions": [],
        "candidate_dispositions": [
            {
                "candidate_index": 1,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "Billy explicitly commits to the invoice review.",
            },
            {
                "candidate_index": 2,
                "disposition": "kept",
                "action_index": 2,
                "reason_code": "supported",
                "reason": "John was incorrectly treated as owning the third-party delivery.",
            },
        ],
        "action_support": [
            {
                "action_index": 1,
                "segment_ids": chinese_source["actions"][0]["segment_ids"],
                "basis": "self_commitment",
            },
            {
                "action_index": 2,
                "segment_ids": external_action["segment_ids"],
                "basis": "owned_follow_up",
            },
        ],
        "decision_support": [],
        "publishable": True,
    }
    audit = {
        "reviews": [{"findings": []}, final_review],
        "action_scout": {"actions": action_scout},
    }

    result, errors = sanitize_reviewed_smart_minutes(
        source,
        segments=segments,
        source_audit=audit,
    )

    assert errors == []
    assert result is not None
    assert result.changes == ["dropped_external_delivery_status_action:2"]
    assert result.payload["minutes"]["actions"] == [minutes["actions"][0]]
    assert result.final_review is not None
    assert result.final_review["action_support"] == [
        {
            "action_index": 1,
            "segment_ids": chinese_source["actions"][0]["segment_ids"],
            "basis": "self_commitment",
        }
    ]
    assert result.final_review["candidate_dispositions"][1]["disposition"] == "rejected"
    assert result.final_review["candidate_dispositions"][1]["action_index"] is None


def test_deterministic_final_repair_neutralizes_anonymous_theme_reference():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["current_state"] = "Speaker 5 提出了需要补齐的功能。"
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["theme:1:current_state_anonymous_speaker_reference"],
        action_scout=_action_scout(records)["actions"],
    )

    assert repaired is not None
    assert changes == ["neutralized_anonymous_theme_reference:1"]
    assert repaired["minutes"]["themes"][0]["current_state"] == (
        "一名参会者提出了需要补齐的功能。"
    )


def test_validator_preserves_assignment_context_with_owner_action_evidence():
    segments = _segments() + [
        {
            "start": 70.0,
            "end": 75.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.94,
            "text": "Billy, please review the invoice.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["actions"][0]["segment_ids"].append(records[2]["segment_id"])

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert errors == []
    assert cleaned is not None
    assert cleaned["actions"][0]["segment_ids"] == [
        records[0]["segment_id"],
        records[2]["segment_id"],
    ]


def test_validator_rejects_action_built_from_distant_commitments():
    segments = _segments() + [
        {
            "start": 500.0,
            "end": 520.0,
            "speaker": "Speaker 1",
            "name": None,
            "name_confidence": 0.0,
            "text": "I will send the meeting notes to the channel.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["actions"][0]["item_zh"] = "复核发票并发送会议纪要。"
    payload["actions"][0]["item_en"] = "Review the invoice and send the meeting notes."
    payload["actions"][0]["segment_ids"].append(records[2]["segment_id"])

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert any(error.startswith("action:1:evidence_span_too_wide:") for error in errors)


def test_validator_narrows_project_update_evidence_to_participant_speech():
    segments = _segments() + [
        {
            "start": 70.0,
            "end": 75.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.94,
            "text": "That approach is clear.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["project_updates"][0]["segment_ids"].append(records[2]["segment_id"])

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert errors == []
    assert cleaned is not None
    assert cleaned["project_updates"][0]["segment_ids"] == [records[1]["segment_id"]]


def test_validator_rejects_unknown_extra_evidence_instead_of_silently_dropping_it():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["themes"][0]["key_points"][0]["segment_ids"].append("seg-does-not-exist")

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert errors == [
        "theme:1:point:1:evidence_unknown:seg-does-not-exist"
    ]


def test_validator_rejects_anonymous_key_point_speaker():
    segments = _segments() + [
        {
            "start": 71.0,
            "end": 80.0,
            "speaker": "Speaker 5",
            "name": None,
            "name_confidence": 0.0,
            "text": "We could build the next version from scratch.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["themes"][0]["key_points"][0] = {
        "speaker": "Speaker 5",
        "text_zh": "建议从头构建下一版本。",
        "text_en": "Proposed building the next version from scratch.",
        "segment_ids": [records[2]["segment_id"]],
    }

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "theme:1:point:1:speaker_anonymous" in errors


def test_validator_rejects_named_future_theme_claim_without_nearby_action():
    segments = _segments() + [
        {
            "start": 500.0,
            "end": 510.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.94,
            "text": "This is only an estimate, not a task.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["themes"][0]["outcome_zh"] = "Xin将探索用AI重建系统。"

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "theme:1:future_owner_without_action:Xin" in errors


def test_validator_rejects_entity_rewrite_not_present_in_action_evidence():
    segments = _segments() + [
        {
            "start": 500.0,
            "end": 510.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.94,
            "text": "This is unrelated.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["actions"][0]["item_zh"] = "向Xin发送发票复核结果。"

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "action:1:named_entity_ungrounded:Xin" in errors


def test_validator_rejects_ungrounded_weekday_in_action():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["actions"][0]["item_zh"] = "周三复核最近一张发票。"

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "action:1:weekday_ungrounded:周三" in errors


def test_validator_requires_review_for_multi_outcome_action():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["actions"][0]["item_zh"] = "获取源码访问，并推动团队迁移。"

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "action:1:atomicity_review_required" in errors


def test_validator_rejects_semantically_duplicate_actions_for_same_owner():
    segments = _segments() + [
        {
            "id": "notes-1",
            "start": 75.0,
            "end": 80.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.95,
            "text": "I will send the meeting notes after the meeting.",
        },
        {
            "id": "notes-2",
            "start": 82.0,
            "end": 87.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.95,
            "text": "I will send the meeting notes in the channel.",
        },
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([records[0]["segment_id"], records[1]["segment_id"]])
    payload["actions"].extend(
        [
            {
                "owner": "Xin",
                "item_zh": "会后发送会议记录。",
                "item_en": "Send the meeting notes after the meeting.",
                "segment_ids": [records[2]["segment_id"]],
            },
            {
                "owner": "Xin",
                "item_zh": "发送会议记录到频道。",
                "item_en": "Send the meeting notes in the channel.",
                "segment_ids": [records[3]["segment_id"]],
            },
        ]
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "action:3:semantic_duplicate:2" in errors


def test_long_meeting_theme_cannot_be_a_tiny_residual_bucket():
    segments = _segments() + [
        {
            "start": 4000.0,
            "end": 4010.0,
            "speaker": "Speaker 1",
            "name": None,
            "name_confidence": 0.0,
            "text": "Closing administrative update.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=[],
    )

    assert cleaned is None
    assert any(error.startswith("theme:1:span_too_short:") for error in errors)


def test_long_meeting_theme_rejects_large_internal_evidence_gap():
    segments = _segments() + [
        {
            "start": 4000.0,
            "end": 4010.0,
            "speaker": "Speaker 1",
            "name": None,
            "name_confidence": 0.0,
            "text": "A separate late topic.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["themes"][0]["evidence_segment_ids"].append(
        records[2]["segment_id"]
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=[],
    )

    assert cleaned is None
    assert any(
        error.startswith("theme:1:evidence_gap_too_wide:")
        for error in errors
    )


def test_generate_smart_minutes_runs_synthesis_then_full_review(monkeypatch):
    records = canonical_transcript_records(_segments())
    payload = _source_payload([record["segment_id"] for record in records])
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        payload,
        {
            "findings": [
                {
                    "severity": "material",
                    "category": "action_recall",
                    "description": "The invoice commitment is retained with direct evidence.",
                }
            ],
            "minutes": payload,
        },
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    calls: list[list[dict[str, str]]] = []

    def fake_request(*, messages, config, max_tokens=16000):
        calls.append(messages)
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    assert status["status"] == "reviewed_draft"
    assert status["review_passes"] == 1
    assert len(calls) == 6
    assert "required_action_candidate_groups" in calls[0][1]["content"]
    assert "follow_up_context_hints" in calls[1][1]["content"]
    assert '"expected_theme_count":1' in calls[2][1]["content"]
    assert "draft_minutes" in calls[4][1]["content"]
    assert "source_minutes_zh" in calls[5][1]["content"]
    assert validate_bilingual_minutes(
        result.chinese_markdown,
        result.english_markdown,
        duration=70.0,
    ) == []
    assert "| 00:00:00-00:00:35 |" in result.chinese_markdown
    assert "Billy" in result.english_markdown


def test_review_pass_repairs_a_locally_invalid_synthesis(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    invalid = copy.deepcopy(valid)
    invalid["themes"][0]["unexpected"] = "remove me"
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        invalid,
        {
            "findings": [
                {
                    "severity": "material",
                    "category": "schema",
                    "description": "Shortened an overlong field.",
                }
            ],
            "minutes": valid,
        },
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    review_messages: list[dict[str, str]] = []

    def fake_request(*, messages, config, max_tokens=16000):
        if len(review_messages) == 1:
            review_messages.extend(messages)
        else:
            review_messages.append(messages[0])
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None
    assert status["status"] == "reviewed_draft"
    assert result.audit["synthesis_validation_errors"] == [
        "theme:1:keys_invalid"
    ]


def test_second_review_repairs_first_review_owner_mismatch(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    invalid_review = copy.deepcopy(valid)
    invalid_review["actions"][0]["owner"] = "Xin"
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": invalid_review},
        _publication_review(
            valid,
            findings=[
                {
                    "severity": "blocker",
                    "category": "speaker_attribution",
                    "description": "Restored the action to the speaker supported by the source segment.",
                    "resolution": "fixed",
                }
            ],
        ),
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
    )

    assert result is not None
    assert status["review_passes"] == 2
    assert result.audit["reviews"][0]["validation_errors"] == [
        "action:1:owner_unknown"
    ]
    assert result.payload["minutes"]["actions"][0]["owner"] == "Billy"


def test_second_review_disposes_every_first_review_finding(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    first_finding = {
        "severity": "medium",
        "category": "wording",
        "description": "Keep the action atomic.",
    }
    final_review = _publication_review(
        valid,
        prior_finding_dispositions=[
            {
                "finding_index": 1,
                "disposition": "addressed",
                "reason": "The final action contains one outcome.",
            }
        ],
    )
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [first_finding], "minutes": valid},
        final_review,
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
    )

    assert result is not None
    assert status["status"] == "reviewed_draft"
    assert result.audit["reviews"][1]["prior_finding_dispositions"] == [
        {
            "finding_index": 1,
            "disposition": "addressed",
            "reason": "The final action contains one outcome.",
        }
    ]


def test_final_review_repairs_local_schema_failure_once(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    invalid_final = _publication_review(copy.deepcopy(valid))
    invalid_final["minutes"]["decisions"] = [
        {
            "text": f"Unsupported decision {index}",
            "segment_ids": [records[0]["segment_id"]],
        }
        for index in range(5)
    ]
    invalid_final["decision_support"] = [
        {
            "decision_index": index,
            "segment_ids": [records[0]["segment_id"]],
            "basis": "explicit_agreement",
        }
        for index in range(1, 6)
    ]
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        invalid_final,
        _publication_review(valid),
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
    )

    assert result is not None
    assert status["status"] == "reviewed_draft"
    assert result.audit["reviews"][1]["status"]["repair_attempted"] is True


def test_final_review_deterministically_repairs_source_and_publication_gate_errors(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    invalid = _publication_review(
        copy.deepcopy(valid),
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "rejected",
                "action_index": None,
                "reason_code": "unsupported_commitment",
                "reason": "The reviewer does not consider this a commitment.",
            }
        ],
    )
    invalid["minutes"]["actions"] = []
    invalid["action_support"] = []
    invalid["minutes"]["project_updates"][0]["update"] = (
        "Billy will send Xin the report."
    )
    repaired_english = _source_payload(
        [record["segment_id"] for record in records],
        english=True,
    )
    repaired_english["project_updates"][0].update(
        {
            "project": "Follow-up",
            "update": "Review the latest invoice and calculate the development and support split.",
            "segment_ids": [records[0]["segment_id"]],
        }
    )
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        invalid,
        repaired_english,
    ]
    calls: list[list[dict[str, str]]] = []

    def fake_request(*, messages, config, max_tokens=16000):
        calls.append(messages)
        return responses.pop(0), {
            "status": "ok",
            "requested_model": config.model,
        }

    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.request_deepseek_json",
        fake_request,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
    )

    assert result is not None, status
    assert status["status"] == "reviewed_draft"
    assert len(calls) == 7
    final_status = result.audit["reviews"][1]["status"]
    assert final_status["deterministic_repair_phase"] == "pre_model"
    assert "restored_supported_candidate:1" in final_status[
        "deterministic_repair_changes"
    ]
    assert "rebuilt_project_update:1" in final_status[
        "deterministic_repair_changes"
    ]


def test_final_publication_gate_rejects_unchecked_actions(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    invalid_gate = _publication_review(valid)
    invalid_gate["action_support"] = []
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        invalid_gate,
        copy.deepcopy(invalid_gate),
    ]
    checkpoints: list[dict] = []

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert result is None
    assert status["status"] == "publication_gate_invalid"
    assert status["errors"] == ["publication_gate_action_support_mismatch"]
    assert len(checkpoints[-1]["reviews"]) == 1


def test_final_publication_gate_rejects_unresolved_findings(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    final_review = _publication_review(
        valid,
        findings=[
            {
                "severity": "blocker",
                "category": "unsupported_action",
                "description": "The action still lacks an explicit commitment.",
                "resolution": "unresolved",
            }
        ],
    )
    final_review["publishable"] = False
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        final_review,
        copy.deepcopy(final_review),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
    )

    assert result is None
    assert status["status"] == "publication_gate_invalid"
    assert status["errors"] == [
        "publication_gate_finding:1:unresolved",
        "publication_gate_not_publishable",
    ]


def test_completed_model_stages_resume_from_checkpoint(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    checkpoints: list[dict] = []

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    first, _status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )
    assert first is not None
    completed_checkpoint = checkpoints[-1]
    assert completed_checkpoint["action_scout"]["input_sha256"]
    assert completed_checkpoint["implicit_follow_up_scout"]["input_sha256"]
    assert completed_checkpoint["theme_outline"]["input_sha256"]
    assert completed_checkpoint["synthesis"]["input_sha256"]
    assert completed_checkpoint["reviews"][0]["input_sha256"]
    assert completed_checkpoint["translation"]["input_sha256"]

    def unexpected_request(**kwargs):
        raise AssertionError("network should not be called for cached stages")

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", unexpected_request)
    resumed, resumed_status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint=completed_checkpoint,
    )

    assert resumed is not None
    assert resumed_status["status"] == "reviewed_draft"
    assert resumed.audit["reviews"][0]["status"]["status"] == "cached"


def test_changed_upstream_payload_invalidates_dependent_stage_caches(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    checkpoints: list[dict] = []

    def first_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", first_request)
    first, _status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )
    assert first is not None

    changed_checkpoint = copy.deepcopy(checkpoints[-1])
    changed_checkpoint["action_scout"]["payload"]["actions"][0]["item"] = (
        "复核发票并重新计算开发与支持占比。"
    )
    rerun_responses = [
        _implicit_follow_up_scout(),
        valid,
        {"findings": [], "minutes": valid},
    ]
    rerun_calls: list[list[dict[str, str]]] = []

    def rerun_request(*, messages, config, max_tokens=16000):
        rerun_calls.append(messages)
        return rerun_responses.pop(0), {
            "status": "ok",
            "requested_model": config.model,
        }

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", rerun_request)
    resumed, resumed_status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint=changed_checkpoint,
    )

    assert resumed is not None
    assert resumed_status["status"] == "reviewed_draft"
    assert len(rerun_calls) == 3
    assert resumed.status["action_scout"]["status"] == "cached"
    assert resumed.status["translation"]["status"] == "cached"


def test_renderer_sorts_model_rows_by_source_time():
    segments = [
        {
            "start": 0.0,
            "end": 20.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_confidence": 0.94,
            "text": "I will review the invoice.",
        },
        {
            "start": 40.0,
            "end": 60.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.94,
            "text": "I will test the merchant flow.",
        },
    ]
    records = canonical_transcript_records(segments)
    first, second = [record["segment_id"] for record in records]
    payload = {
        "themes": [
            {
                "title_zh": "行动",
                "title_en": "Actions",
                "current_state_zh": "团队讨论两个后续事项。",
                "current_state_en": "The team discussed two follow-up items.",
                "outcome_zh": "两位负责人分别承诺跟进。",
                "outcome_en": "Two owners committed to follow up.",
                "evidence_segment_ids": [first, second],
                "key_points": [
                    {
                        "speaker": "Billy",
                        "text_zh": "Billy 复核发票。",
                        "text_en": "Billy will review the invoice.",
                        "segment_ids": [first],
                    }
                ],
            }
        ],
        "project_updates": [
            {
                "participant": "Xin",
                "project_zh": "商户测试",
                "project_en": "Merchant Testing",
                "update_zh": "准备测试。",
                "update_en": "Preparing to test.",
                "segment_ids": [second],
            },
            {
                "participant": "Billy",
                "project_zh": "成本复核",
                "project_en": "Cost Review",
                "update_zh": "准备复核发票。",
                "update_en": "Preparing to review the invoice.",
                "segment_ids": [first],
            },
        ],
        "decisions": [],
        "actions": [
            {
                "owner": "Xin",
                "item_zh": "测试商户流程。",
                "item_en": "Test the merchant flow.",
                "segment_ids": [second],
            },
            {
                "owner": "Billy",
                "item_zh": "复核发票。",
                "item_en": "Review the invoice.",
                "segment_ids": [first],
            },
        ],
    }

    chinese, english = render_smart_minutes(payload, transcript_records=records)

    assert chinese.index("| 00:00:00-00:00:20 | Billy") < chinese.index(
        "| 00:00:40-00:01:00 | Xin"
    )
    assert english.index("| 00:00:00-00:00:20 | Review the invoice.") < english.index(
        "| 00:00:40-00:01:00 | Test the merchant flow."
    )
