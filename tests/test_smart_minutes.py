from __future__ import annotations

import copy
import json

import meeting_minutes.smart_minutes as smart_minutes
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
    combine_minutes_languages,
    flatten_theme_candidates,
    follow_up_context_hints,
    generate_smart_minutes,
    normalize_theme_chunk_coverage,
    requires_hierarchical_analysis,
    required_action_candidate_groups,
    render_smart_minutes,
    repair_source_minutes_text_fields,
    sanitize_reviewed_smart_minutes,
    theme_count_bounds,
    transcript_record_chunks,
    validate_action_scout,
    validate_implicit_follow_up_judgments,
    validate_publication_gate,
    validate_smart_minutes,
    validate_source_minutes,
    validate_translation_against_source,
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


def _action_scout_chunk_records(count: int = 8) -> list[dict]:
    return canonical_transcript_records(
        [
            {
                "id": f"action-scout-{index}",
                "start": float(index * 10),
                "end": float(index * 10 + 8),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Technical discussion record {index}.",
            }
            for index in range(count)
        ]
    )


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


def test_translation_alignment_accepts_equivalent_number_weekday_and_compound_tokens():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["themes"][0]["title"] = "Web相关Box于周四（6日）进行41%复核"
    translation["themes"][0]["title"] = (
        "Review Web-related Box on Thursday (6th) after a forty-one percent check"
    )

    errors = validate_translation_against_source(source, translation)

    assert errors == []
    bilingual, combine_errors = combine_minutes_languages(source, translation)
    assert combine_errors == []
    assert bilingual is not None


def test_translation_alignment_accepts_chinese_percentage_numerals_and_lowercase_identifiers():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["project_updates"][0]["project"] = "API网关"
    source["project_updates"][0]["update"] = "价格削减百分之四十一，需提高效率。"
    translation["project_updates"][0]["project"] = "api gateway"
    translation["project_updates"][0]["update"] = (
        "A forty-one percent price cut requires higher efficiency."
    )

    errors = validate_translation_against_source(source, translation)

    assert errors == []


def test_translation_alignment_accepts_chinese_year_duration():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["project_updates"][0]["update"] = "两年降价41%，需要提高效率。"
    translation["project_updates"][0]["update"] = (
        "A 41% price reduction over two years requires higher efficiency."
    )

    errors = validate_translation_against_source(source, translation)

    assert errors == []


def test_translation_alignment_accepts_only_pronoun_for_chinese_single_person():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["project_updates"][0]["update"] = "目前仅由一人维护该系统。"
    translation["project_updates"][0]["update"] = (
        "Currently, the system is maintained only by him."
    )

    errors = validate_translation_against_source(source, translation)

    assert errors == []


def test_translation_alignment_accepts_solely_pronoun_for_chinese_single_person():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["project_updates"][0]["update"] = "目前BPS仅由他一人维护。"
    translation["project_updates"][0]["update"] = "Currently, BPS is maintained solely by him."

    errors = validate_translation_against_source(source, translation)

    assert errors == []


def test_translation_alignment_rejects_changed_chinese_percentage_numerals():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["project_updates"][0]["update"] = "价格削减四十一个百分点，需提高效率。"
    translation["project_updates"][0]["update"] = (
        "A fifty percentage-point price cut requires higher efficiency."
    )

    errors = validate_translation_against_source(source, translation)

    assert "translation_project_update:1:update:number_mismatch" in errors


def test_translation_alignment_rejects_dropped_or_changed_numbers():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["project_updates"][0]["update"] = "价格削减41%，需提高效率。"
    translation["project_updates"][0]["update"] = "A 50% price cut requires higher efficiency."

    errors = validate_translation_against_source(source, translation)

    assert "translation_project_update:1:update:number_mismatch" in errors


def test_translation_alignment_rejects_dropped_source_technical_identifier():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["project_updates"][0]["project"] = "API网关"
    translation["project_updates"][0]["project"] = "Gateway"

    errors = validate_translation_against_source(source, translation)

    assert "translation_project_update:1:project:source_token_missing:api" in errors


def test_translation_alignment_rejects_malformed_target_before_merge():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    del translation["themes"][0]["title"]

    errors = validate_translation_against_source(source, translation)
    bilingual, combine_errors = combine_minutes_languages(source, translation)

    assert "translation_target:themes:1:keys_invalid" in errors
    assert bilingual is None
    assert "translation_target:themes:1:keys_invalid" in combine_errors


def test_publishable_translation_rejects_masking_artifact():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["themes"][0]["current_state"] = "需要通过接口重新测试。"
    translation["themes"][0]["current_state"] = (
        "A retest is needed through the relevant interface."
    )

    bilingual, errors = smart_minutes.combine_publishable_minutes_languages(
        source,
        translation,
    )

    assert bilingual is None
    assert errors == [
        "translation_quality:theme:1:current_state:quality_degraded_placeholder"
    ]


def test_translation_alignment_rejects_pending_operation_as_completed():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    source["themes"][0]["current_state"] = "地址生成需要通过BPS接口重新测试。"
    translation["themes"][0]["current_state"] = (
        "Address generation was retested through the BPS interface."
    )

    errors = validate_translation_against_source(source, translation)

    assert errors == [
        "translation_theme:1:current_state:status_completion_mismatch:retest"
    ]


def test_validate_smart_minutes_rejects_requested_retest_as_completed():
    records = canonical_transcript_records(
        [
            {
                "id": "pending-retest",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": (
                    "I need your help to retest address generation through the BPS "
                    "interface."
                ),
            }
        ]
    )
    minutes = {
        "themes": [
            {
                "title_zh": "地址重试",
                "title_en": "Address Retest",
                "current_state_zh": "地址生成已通过BPS接口重新测试。",
                "current_state_en": (
                    "Address generation was retested through the BPS interface."
                ),
                "outcome_zh": "讨论形成方向，尚未决定。",
                "outcome_en": "Direction discussed, not yet decided.",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    cleaned, errors = validate_smart_minutes(
        minutes,
        transcript_records=records,
        required_project_participants=[],
    )

    assert cleaned is None
    assert errors == [
        "theme:1:current_state:status_unsupported_completion:retest"
    ]


def test_status_fidelity_uses_adjacent_segment_for_split_request():
    records = canonical_transcript_records(
        [
            {
                "id": "pending-retest",
                "start": 0.0,
                "end": 6.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "I need your help to retest that particular issue.",
            },
            {
                "id": "retest-context",
                "start": 6.0,
                "end": 12.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "Use the BPS interface to generate the address once again.",
            },
        ]
    )
    minutes = {
        "themes": [
            {
                "title_zh": "地址重试",
                "title_en": "Address Retest",
                "current_state_zh": "地址生成已通过BPS接口重新测试。",
                "current_state_en": (
                    "Address generation was retested through the BPS interface."
                ),
                "outcome_zh": "讨论形成方向，尚未决定。",
                "outcome_en": "Direction discussed, not yet decided.",
                "evidence_segment_ids": [records[1]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    cleaned, errors = validate_smart_minutes(
        minutes,
        transcript_records=records,
        required_project_participants=[],
    )

    assert cleaned is None
    assert errors == [
        "theme:1:current_state:status_unsupported_completion:retest"
    ]


def test_status_fidelity_rejects_completed_regeneration_for_requested_retest():
    records = canonical_transcript_records(
        [
            {
                "id": "requested-retest",
                "start": 0.0,
                "end": 6.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "I need your help to retest that particular issue.",
            },
            {
                "id": "requested-regeneration",
                "start": 6.0,
                "end": 12.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "Use the BPS interface to try to generate the address once again.",
            },
        ]
    )
    minutes = {
        "themes": [
            {
                "title_zh": "地址重试",
                "title_en": "Address Retest",
                "current_state_zh": "地址生成已通过BPS接口尝试重新生成。",
                "current_state_en": "Address regeneration was attempted through BPS.",
                "outcome_zh": "讨论形成方向，尚未决定。",
                "outcome_en": "Direction discussed, not yet decided.",
                "evidence_segment_ids": [records[1]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    cleaned, errors = validate_smart_minutes(
        minutes,
        transcript_records=records,
        required_project_participants=[],
    )

    assert cleaned is None
    assert errors == [
        "theme:1:current_state:status_unsupported_completion:retest"
    ]


def test_status_fidelity_allows_completed_retest_when_evidence_confirms_it():
    records = canonical_transcript_records(
        [
            {
                "id": "completed-retest",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "We completed the retest through the BPS interface.",
            }
        ]
    )
    minutes = {
        "themes": [
            {
                "title_zh": "地址重试",
                "title_en": "Address Retest",
                "current_state_zh": "地址生成已通过BPS接口重新测试。",
                "current_state_en": (
                    "Address generation was retested through the BPS interface."
                ),
                "outcome_zh": "讨论形成方向，尚未决定。",
                "outcome_en": "Direction discussed, not yet decided.",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    cleaned, errors = validate_smart_minutes(
        minutes,
        transcript_records=records,
        required_project_participants=[],
    )

    assert cleaned is not None
    assert errors == []


def test_status_fidelity_keeps_completion_and_later_request_separate():
    records = canonical_transcript_records(
        [
            {
                "id": "completed-retest-pending-access",
                "start": 0.0,
                "end": 24.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": (
                    "Address generation was retested through BPS. We need to continue "
                    "production network investigation, log review, permission confirmation, "
                    "and incident review, and request Cloudflare email access for Kirill."
                ),
            }
        ]
    )
    minutes = {
        "themes": [
            {
                "title_zh": "地址重试与访问请求",
                "title_en": "Address Retest and Access Request",
                "current_state_zh": (
                    "地址生成已通过BPS接口重新测试。团队需要推进生产网络的只读排查、"
                    "日志核对、权限确认和故障复盘，并请求为Kirill添加Cloudflare邮件访问。"
                ),
                "current_state_en": (
                    "Address generation was retested through BPS. The team needs to continue "
                    "production network investigation, log review, permission confirmation, "
                    "and incident review, and requests Cloudflare email access for Kirill."
                ),
                "outcome_zh": "讨论形成方向，尚未决定。",
                "outcome_en": "Direction discussed, not yet decided.",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    cleaned, errors = validate_smart_minutes(
        minutes,
        transcript_records=records,
        required_project_participants=[],
    )

    assert cleaned is not None
    assert errors == []


def test_translation_alignment_rejects_owner_or_evidence_drift():
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    source = _source_payload(segment_ids)
    translation = _source_payload(segment_ids, english=True)
    translation["actions"][0]["owner"] = "Xin"
    translation["themes"][0]["key_points"][0]["segment_ids"] = [segment_ids[1]]

    errors = validate_translation_against_source(source, translation)

    assert "translation_action:1:owner_mismatch" in errors
    assert "translation_theme:1:point:1:evidence_mismatch" in errors


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


def test_direct_requested_follow_up_is_retained_without_guessing_the_recipient():
    records = canonical_transcript_records(
        [
            {
                "id": "request",
                "start": 117.0,
                "end": 126.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "William might be Sebastian. I need your help here to retest that one particular thing.",
            },
            {
                "id": "scope",
                "start": 126.0,
                "end": 135.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "Use the BPS interface to generate the NPC address again.",
            },
        ]
    )

    candidates = smart_minutes.explicit_requested_follow_up_candidates(records)

    assert len(candidates) == 1
    assert candidates[0]["owner"] == "待确认"
    assert candidates[0]["basis"] == "requested_follow_up"
    assert candidates[0]["must_keep"] is True
    assert candidates[0]["unassigned_explicit_request"] is True
    assert candidates[0]["segment_ids"] == [record["segment_id"] for record in records]

    minutes = {
        "themes": [
            {
                "title": "BPS retest request",
                "current_state": "A BPS retest was requested.",
                "outcome": "The speaker asked for help to retest BPS address generation.",
                "evidence_segment_ids": [record["segment_id"] for record in records],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [
            {
                "owner": "待确认",
                "item": "Retest BPS address generation.",
                "segment_ids": [record["segment_id"] for record in records],
            }
        ],
    }
    cleaned, source_errors = validate_source_minutes(
        minutes,
        transcript_records=records,
        required_project_participants=[],
    )

    assert source_errors == []
    assert cleaned is not None
    review = _publication_review(
        cleaned,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "The direct request requires a BPS retest.",
            }
        ],
    )
    review["action_support"][0]["basis"] = "requested_follow_up"

    assert validate_publication_gate(
        review,
        cleaned,
        transcript_records=records,
        action_scout=candidates,
    ) == []


def _collapsed_must_keep_fixture() -> tuple[list[dict], dict, dict, list[dict]]:
    records = canonical_transcript_records(
        [
            {
                "id": "commitment-prepare",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "text": "I will prepare the product document.",
            },
            {
                "id": "commitment-start",
                "start": 240.0,
                "end": 252.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "text": "I will prepare the technical brief.",
            },
            {
                "id": "commitment-send",
                "start": 480.0,
                "end": 492.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "text": "I will send the document before Thursday.",
            },
        ]
    )
    combined_ids = [record["segment_id"] for record in records]
    minutes = {
        "themes": [],
        "project_updates": [],
        "decisions": [],
        "actions": [
            {
                "owner": "John",
                "item": "准备产品文档、开始编写技术说明，并在周四前发送文档。",
                "segment_ids": combined_ids,
            }
        ],
    }
    scout = [
        {
            "owner": "John",
            "item": "准备产品说明文档。",
            "segment_ids": [records[0]["segment_id"]],
            "basis": "self_commitment",
            "must_keep": True,
        },
        {
            "owner": "John",
            "item": "准备技术说明。",
            "segment_ids": [records[1]["segment_id"]],
            "basis": "self_commitment",
            "must_keep": True,
        },
        {
            "owner": "John",
            "item": "在周四前发送文档。",
            "segment_ids": [records[2]["segment_id"]],
            "basis": "self_commitment",
            "must_keep": True,
        },
    ]
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": index,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "The cited evidence supports this commitment.",
            }
            for index in range(1, 4)
        ],
    )
    return records, minutes, review, scout


def _implicit_follow_up_scout() -> dict:
    return {"judgments": []}


def _negative_implicit_judgment(hint_index: int) -> dict:
    return {
        "hint_index": hint_index,
        "qualifies": False,
        "owner": "",
        "item": "",
        "segment_ids": [],
        "reason": "The supplied context does not establish an owned follow-up.",
    }


def _two_follow_up_hints(records: list[dict]) -> list[dict]:
    return [
        {
            "anchor_segment_id": records[0]["segment_id"],
            "anchor_speaker": "Billy",
            "signals": {"strong_local_signal": False},
            "context": [records[0]],
        },
        {
            "anchor_segment_id": records[1]["segment_id"],
            "anchor_speaker": "Billy",
            "signals": {"strong_local_signal": False},
            "context": [records[1]],
        },
    ]


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


def test_grouped_theme_fallback_refuses_unproven_merge():
    candidates = [
        {
            "candidate_index": 1,
            "title": "Client GitLab migration",
            "start_segment_id": "seg-1",
            "end_segment_id": "seg-1",
            "anchor_segment_ids": ["seg-1"],
        },
        {
            "candidate_index": 2,
            "title": "Operational security review",
            "start_segment_id": "seg-2",
            "end_segment_id": "seg-2",
            "anchor_segment_ids": ["seg-2"],
        },
    ]

    payload = smart_minutes._grouped_theme_merge_fallback(
        candidates,
        target_theme_count=1,
    )

    assert payload is None

    preserved = smart_minutes._grouped_theme_merge_fallback(
        candidates,
        target_theme_count=2,
    )
    assert preserved is not None
    assert [theme["title"] for theme in preserved["themes"]] == [
        "Client GitLab migration",
        "Operational security review",
    ]


def test_theme_merge_candidate_coverage_repair_rebuilds_stale_indexes():
    records = canonical_transcript_records(
        [
            {
                "id": "coverage-1",
                "start": 0.0,
                "end": 100.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "First topic.",
            },
            {
                "id": "coverage-2",
                "start": 110.0,
                "end": 220.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "Second topic.",
            },
        ]
    )
    candidates = [
        {
            "candidate_index": index + 1,
            "chunk_index": 1,
            "local_topic_index": index + 1,
            "title": f"Topic {index + 1}",
            "summary": f"Topic {index + 1} summary.",
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
    payload = {
        "read_marker": {"candidate_count": 2, "last_candidate_index": 2},
        "themes": [
            {
                "title": "Combined delivery question",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[1]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "One parent delivery question.",
                "source_candidate_indexes": [3, 4],
            }
        ],
    }

    repaired, changes = smart_minutes._repair_theme_merge_candidate_coverage(
        payload,
        candidates=candidates,
        transcript_records=records,
    )

    assert repaired is not None
    assert changes == ["repaired_theme_merge_candidate_coverage:1"]
    assert repaired["themes"][0]["source_candidate_indexes"] == [1, 2]
    outline, errors = validate_theme_merge(
        repaired,
        candidates=candidates,
        transcript_records=records,
        min_theme_count=1,
        max_theme_count=1,
    )
    assert errors == []
    assert outline is not None


def test_theme_merge_prompt_fingerprints_the_planner_policy():
    records = canonical_transcript_records(_segments())
    candidates = [
        {
            "candidate_index": 1,
            "chunk_index": 1,
            "local_topic_index": 1,
            "title": "Delivery visibility",
            "summary": "Discussed delivery visibility.",
            "importance": "substantive",
            "start_segment_id": records[0]["segment_id"],
            "end_segment_id": records[1]["segment_id"],
            "anchor_segment_ids": [records[0]["segment_id"]],
            "start": records[0]["start"],
            "end": records[1]["end"],
        }
    ]

    messages = build_theme_merge_messages(
        candidates,
        transcript_records=records,
        min_theme_count=1,
        max_theme_count=1,
    )

    assert json.loads(messages[1]["content"])["planner_policy_version"] == (
        smart_minutes.THEME_MERGE_POLICY_VERSION
    )


def test_cross_chunk_boundary_normalization_assigns_overlap_to_one_theme():
    records = canonical_transcript_records(
        [
            {
                "id": f"boundary-{index}",
                "start": float(index * 10),
                "end": float(index * 10 + 8),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Topic record {index}.",
            }
            for index in range(3)
        ]
    )
    candidates = [
        {
            "title": "First topic",
            "start_segment_id": records[0]["segment_id"],
            "end_segment_id": records[1]["segment_id"],
            "anchor_segment_ids": [records[0]["segment_id"]],
            "start_position": 0,
            "end_position": 1,
            "start": records[0]["start"],
            "end": records[1]["end"],
        },
        {
            "title": "Second topic",
            "start_segment_id": records[1]["segment_id"],
            "end_segment_id": records[2]["segment_id"],
            "anchor_segment_ids": [records[1]["segment_id"]],
            "start_position": 1,
            "end_position": 2,
            "start": records[1]["start"],
            "end": records[2]["end"],
        },
    ]

    normalized, changes = smart_minutes._normalize_cross_chunk_candidate_boundaries(
        candidates,
        transcript_records=records,
    )

    assert changes == ["trimmed_cross_chunk_overlap_before:2"]
    assert normalized[1]["start_position"] == 2
    assert normalized[1]["start_segment_id"] == records[2]["segment_id"]
    assert normalized[1]["anchor_segment_ids"] == [records[2]["segment_id"]]


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


def test_theme_chunk_normalization_partitions_large_crossing_overlap():
    records = canonical_transcript_records(
        [
            {
                "id": f"crossing-overlap-{index}",
                "start": float(index * 40),
                "end": float(index * 40 + 30),
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
                "title": "Earlier topic",
                "summary": "The earlier discussion begins first.",
                "importance": "substantive",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[3]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
            },
            {
                "title": "Later topic",
                "summary": "The later discussion overlaps then continues.",
                "importance": "substantive",
                "start_segment_id": records[2]["segment_id"],
                "end_segment_id": records[5]["segment_id"],
                "anchor_segment_ids": [records[2]["segment_id"]],
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
    assert normalized["topics"][0]["end_segment_id"] == records[1]["segment_id"]
    assert changes == ["partitioned_crossing_overlap_before:2"]


def test_theme_chunk_topology_fallback_covers_the_full_core_range():
    records = canonical_transcript_records(
        [
            {
                "id": f"fallback-topic-{index}",
                "start": float(index * 20),
                "end": float(index * 20 + 10),
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": f"Discussion section {index}." + (" detail" * index),
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

    fallback = smart_minutes._fallback_theme_chunk_payload(
        chunk=chunk,
        transcript_records=records,
    )
    topics, errors = validate_theme_chunk(
        fallback,
        chunk=chunk,
        transcript_records=records,
    )

    assert errors == []
    assert topics is not None
    assert fallback["topics"][0]["title"] == "Operational discussion"
    assert fallback["topics"][0]["start_segment_id"] == records[0]["segment_id"]
    assert fallback["topics"][0]["end_segment_id"] == records[-1]["segment_id"]


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


def test_theme_chunk_normalization_partitions_long_partial_overlap():
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

    assert errors == []
    assert topics is not None
    assert normalized["topics"][0]["end_segment_id"] == records[2]["segment_id"]
    assert changes == ["partitioned_crossing_overlap_before:2"]


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


def test_theme_reduction_relaxes_only_intermediate_long_meeting_span():
    records = canonical_transcript_records(
        [
            {
                "id": "short-local-topic",
                "start": 120.0,
                "end": 170.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "A short local topic that still needs global merging.",
            },
            {
                "id": "meeting-tail",
                "start": 4500.0,
                "end": 4510.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "Later meeting evidence establishes a long meeting duration.",
            },
        ]
    )
    candidate = {
        "candidate_index": 1,
        "chunk_index": 1,
        "local_topic_index": 1,
        "title": "Short local topic",
        "summary": "A compact intermediate topic.",
        "importance": "substantive",
        "start_segment_id": records[0]["segment_id"],
        "end_segment_id": records[0]["segment_id"],
        "anchor_segment_ids": [records[0]["segment_id"]],
        "start_position": 0,
        "end_position": 0,
        "start": records[0]["start"],
        "end": records[0]["end"],
    }
    payload = {
        "read_marker": {"candidate_count": 1, "last_candidate_index": 1},
        "themes": [
            {
                "title": "Short local topic",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[0]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "This is an intermediate local candidate.",
                "source_candidate_indexes": [1],
            }
        ],
    }

    strict_outline, strict_errors = validate_theme_merge(
        payload,
        candidates=[candidate],
        transcript_records=records,
        min_theme_count=1,
        max_theme_count=1,
        require_meeting_edge_coverage=False,
    )
    relaxed_outline, relaxed_errors = validate_theme_merge(
        payload,
        candidates=[candidate],
        transcript_records=records,
        min_theme_count=1,
        max_theme_count=1,
        require_meeting_edge_coverage=False,
        enforce_min_long_theme_span=False,
    )

    assert strict_outline is None
    assert strict_errors == ["theme_outline:1:span_too_short"]
    assert relaxed_errors == []
    assert relaxed_outline is not None


def test_theme_candidate_reduction_accepts_short_intermediate_macro_topic(
    monkeypatch,
):
    records = canonical_transcript_records(
        [
            {
                "id": "short-reduction-topic",
                "start": 120.0,
                "end": 170.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "A short local topic that should remain an intermediate candidate.",
            },
            {
                "id": "reduction-tail",
                "start": 4500.0,
                "end": 4510.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "Long meeting tail.",
            },
        ]
    )
    candidate = {
        "candidate_index": 1,
        "chunk_index": 1,
        "local_topic_index": 1,
        "title": "Short intermediate topic",
        "summary": "A compact local candidate.",
        "importance": "substantive",
        "start_segment_id": records[0]["segment_id"],
        "end_segment_id": records[0]["segment_id"],
        "anchor_segment_ids": [records[0]["segment_id"]],
        "start_position": 0,
        "end_position": 0,
        "start": records[0]["start"],
        "end": records[0]["end"],
    }
    response = {
        "read_marker": {"candidate_count": 1, "last_candidate_index": 1},
        "themes": [
            {
                "title": "Short intermediate topic",
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[0]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "Preserve the local topic for the global merge.",
                "source_candidate_indexes": [1],
            }
        ],
    }

    def fake_request(*, messages, config, max_tokens=16000):
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    macro_candidates, reduced_themes, status = (
        smart_minutes._run_theme_candidate_reductions(
            candidates=[candidate],
            records=records,
            config=DeepSeekConfig(model="test-model"),
            cache={"theme_outline_reductions": []},
            save_checkpoint=lambda: None,
        )
    )

    assert status["status"] == "ok"
    assert macro_candidates is not None
    assert len(macro_candidates) == 1
    assert len(reduced_themes) == 1


def test_theme_candidate_reduction_compacts_grounded_overlong_title(
    monkeypatch,
):
    records = canonical_transcript_records(
        [
            {
                "id": "centralization-topic",
                "start": 120.0,
                "end": 180.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "Centralize services to reduce dependency on the Russian team.",
            }
        ]
    )
    candidate = {
        "candidate_index": 1,
        "chunk_index": 1,
        "local_topic_index": 1,
        "title": "Intermediate operational discussion",
        "summary": "The team discussed service centralization.",
        "importance": "substantive",
        "start_segment_id": records[0]["segment_id"],
        "end_segment_id": records[0]["segment_id"],
        "anchor_segment_ids": [records[0]["segment_id"]],
        "start_position": 0,
        "end_position": 0,
        "start": records[0]["start"],
        "end": records[0]["end"],
    }
    initial_response = {
        "read_marker": {"candidate_count": 1, "last_candidate_index": 1},
        "themes": [
            {
                "title": (
                    "How to centralize services to reduce dependency on Russian team?"
                ),
                "start_segment_id": records[0]["segment_id"],
                "end_segment_id": records[0]["segment_id"],
                "anchor_segment_ids": [records[0]["segment_id"]],
                "boundary_reason": "Service centralization is the specific topic.",
                "source_candidate_indexes": [1],
            }
        ],
    }
    repaired_response = copy.deepcopy(initial_response)
    repaired_response["themes"][0]["title"] = (
        "What is the rationale for centralizing Russian team services?"
    )
    request_count = 0

    def fake_request(*, messages, config, max_tokens=16000):
        nonlocal request_count
        request_count += 1
        response = initial_response if request_count == 1 else repaired_response
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    macro_candidates, reduced_themes, status = (
        smart_minutes._run_theme_candidate_reductions(
            candidates=[candidate],
            records=records,
            config=DeepSeekConfig(model="test-model"),
            cache={"theme_outline_reductions": []},
            save_checkpoint=lambda: None,
        )
    )

    assert request_count == 2
    assert status["status"] == "ok"
    assert macro_candidates is not None
    assert reduced_themes[0]["title"] == (
        "Rationale for centralizing Russian team services"
    )
    assert reduced_themes[0]["start_segment_id"] == records[0]["segment_id"]
    assert reduced_themes[0]["end_segment_id"] == records[0]["segment_id"]
    assert reduced_themes[0]["anchor_segment_ids"] == [records[0]["segment_id"]]
    assert reduced_themes[0]["source_candidate_indexes"] == [1]
    assert status["groups"][0]["status"]["deterministic_title_repair"] == [
        {
            "before": (
                "What is the rationale for centralizing Russian team services?"
            ),
            "after": "Rationale for centralizing Russian team services",
        }
    ]


def test_theme_count_repair_guidance_requires_exact_adjacent_partition():
    guidance = smart_minutes._validation_repair_guidance(
        ["theme_outline_count_out_of_range:25!=10-10"]
    )

    assert guidance == [
        (
            "Return exactly 10 chronological themes. Partition every source "
            "candidate index into non-empty adjacent groups, consuming each index "
            "exactly once. When more candidates than themes exist, merge the "
            "closest neighboring candidates under one truthful parent business "
            "question; do not preserve one theme per candidate, omit an index, "
            "or invent an index."
        )
    ]


def test_theme_merge_topology_repair_lists_exact_allowed_partition():
    messages = smart_minutes._theme_merge_topology_repair_messages(
        [{"role": "system", "content": "merge"}],
        payload={"read_marker": {}, "themes": []},
        errors=[
            "theme_merge_candidate_unknown:4",
            "theme_outline_count_out_of_range:3!=2-2",
        ],
        candidates=[
            {"candidate_index": 1},
            {"candidate_index": 2},
            {"candidate_index": 3},
        ],
        min_theme_count=2,
        max_theme_count=2,
    )
    repair_request = json.loads(messages[-1]["content"])

    assert repair_request["required_theme_count"] == 2
    assert repair_request["allowed_source_candidate_indexes"] == [1, 2, 3]
    assert repair_request["required_source_candidate_partition"] == [1, 2, 3]
    assert "never invent an index" in repair_request["hard_constraints"][0]


def test_one_pass_review_repair_neutralizes_owner_and_narrows_evidence():
    records = canonical_transcript_records(
        [
            {
                "id": "one-pass-action-a",
                "start": 0.0,
                "end": 10.0,
                "speaker": "Speaker 1",
                "text": "We should follow up.",
            },
            {
                "id": "one-pass-action-b",
                "start": 200.0,
                "end": 210.0,
                "speaker": "Speaker 1",
                "text": "Later discussion.",
            },
        ]
    )
    review = {
        "findings": [],
        "minutes": {
            "themes": [],
            "project_updates": [],
            "decisions": [],
            "actions": [
                {
                    "owner": "Unknown Person",
                    "item": "跟进相关事项",
                    "segment_ids": [
                        records[0]["segment_id"],
                        records[1]["segment_id"],
                    ],
                }
            ],
        },
    }

    repaired, changes = smart_minutes._deterministic_coverage_review_repair(
        review,
        errors=[
            "action:1:owner_unknown",
            "action:1:evidence_span_too_wide:210.0>120.0",
        ],
        action_scout=[],
        transcript_records=records,
        theme_outline=[],
    )

    assert repaired is not None
    assert set(repaired) == {"findings", "minutes"}
    assert repaired["minutes"]["actions"][0]["owner"] == "待确认"
    assert repaired["minutes"]["actions"][0]["segment_ids"] == [
        records[0]["segment_id"]
    ]
    assert changes == [
        "neutralized_unknown_action_owner:1",
        "narrowed_action_evidence:1",
    ]


def test_one_pass_review_repair_restores_missing_must_keep_action():
    records = canonical_transcript_records(
        [
            {
                "id": "john-deadline",
                "start": 0.0,
                "end": 8.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.99,
                "name_source": "visual_active_speaker_highlight",
                "text": "I will send the document before Thursday.",
            }
        ],
        allow_cluster_name_consensus=False,
    )
    candidate = {
        "owner": "John",
        "item": "在周四前发送文档",
        "segment_ids": [records[0]["segment_id"]],
        "basis": "self_commitment",
        "must_keep": True,
    }
    review = {
        "findings": [],
        "minutes": {
            "themes": [],
            "project_updates": [],
            "decisions": [],
            "actions": [],
        },
    }

    assert smart_minutes.validate_must_keep_action_coverage(
        review["minutes"],
        [candidate],
    ) == ["must_keep_action_missing:1"]

    repaired, changes = smart_minutes._deterministic_coverage_review_repair(
        review,
        errors=["must_keep_action_missing:1"],
        action_scout=[candidate],
        transcript_records=records,
        theme_outline=[],
    )

    assert repaired is not None
    assert repaired["minutes"]["actions"] == [
        {
            "owner": "John",
            "item": "在周四前发送文档",
            "segment_ids": [records[0]["segment_id"]],
        }
    ]
    assert smart_minutes.validate_must_keep_action_coverage(
        repaired["minutes"],
        [candidate],
    ) == []
    assert changes == ["restored_must_keep_action_candidate:1"]


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
        ),
        macro_candidate_count=12,
    ) == (9, 10)
    assert len(evidence) == len(records)


def test_density_theme_bounds_require_a_span_feasible_macro_partition():
    records = [{"end": 4795.0}]
    feasible_macros = [
        {"start": float(index * 200), "end": float(index * 200 + 190)}
        for index in range(12)
    ]
    short_macros = [
        {"start": float(index * 100), "end": float(index * 100 + 50)}
        for index in range(12)
    ]

    assert smart_minutes._max_span_feasible_theme_count(feasible_macros) == 12
    assert theme_count_bounds(
        records,
        macro_candidates=feasible_macros,
    ) == (10, 10)
    assert smart_minutes._max_span_feasible_theme_count(short_macros) == 4
    assert theme_count_bounds(
        records,
        macro_candidates=short_macros,
    ) == (3, 4)


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


def test_hierarchical_action_scout_recovers_truncated_json_and_resumes(monkeypatch):
    records = _action_scout_chunk_records()
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    cache = {
        "action_scout_chunks": [],
        "action_scout": None,
        "last_rejected_action_scout_chunk": None,
    }
    checkpoints: list[dict] = []
    requests: list[list[str]] = []

    monkeypatch.setattr(
        smart_minutes,
        "ACTION_SCOUT_TRUNCATION_SPLIT_MIN_RECORDS",
        1,
    )

    def fake_request(*, messages, config, max_tokens=16000):
        transcript = json.loads(messages[1]["content"])["transcript"]
        requests.append([record["segment_id"] for record in transcript])
        if len(transcript) == len(records):
            return None, {
                "status": "invalid_model_json",
                "starts_with_object": True,
                "ends_with_object": False,
            }
        return {"actions": []}, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    actions, status = smart_minutes._run_hierarchical_action_scout(
        records=records,
        chunks=[chunk],
        required_action_groups=[],
        follow_up_hints=[],
        intent_recall_hints=[],
        config=DeepSeekConfig(model="test-model"),
        cache=cache,
        save_checkpoint=lambda: checkpoints.append(copy.deepcopy(cache)),
    )

    assert actions == []
    assert status["status"] == "ok"
    assert len(requests) == 3
    assert requests[0] == [record["segment_id"] for record in records]
    assert not set(requests[1]).intersection(requests[2])
    assert requests[1] + requests[2] == requests[0]
    entry = cache["action_scout_chunks"][0]
    assert entry["split"]["initial_truncation"]["status"] == "invalid_model_json"
    assert entry["status"]["status"] == "recovered_after_truncation"
    assert checkpoints

    def unexpected_request(**kwargs):
        raise AssertionError("recovered action-scout chunk should resume from cache")

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", unexpected_request)
    resumed_actions, resumed_status = smart_minutes._run_hierarchical_action_scout(
        records=records,
        chunks=[chunk],
        required_action_groups=[],
        follow_up_hints=[],
        intent_recall_hints=[],
        config=DeepSeekConfig(model="test-model"),
        cache=cache,
        save_checkpoint=lambda: None,
    )

    assert resumed_actions == []
    assert resumed_status["status"] == "cached"


def test_hierarchical_action_scout_does_not_split_generic_invalid_json(monkeypatch):
    records = _action_scout_chunk_records()
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    cache = {
        "action_scout_chunks": [],
        "action_scout": None,
        "last_rejected_action_scout_chunk": None,
    }
    calls = 0

    def fake_request(*, messages, config, max_tokens=16000):
        nonlocal calls
        calls += 1
        return None, {
            "status": "invalid_model_json",
            "starts_with_object": True,
            "ends_with_object": True,
        }

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    actions, status = smart_minutes._run_hierarchical_action_scout(
        records=records,
        chunks=[chunk],
        required_action_groups=[],
        follow_up_hints=[],
        intent_recall_hints=[],
        config=DeepSeekConfig(model="test-model"),
        cache=cache,
        save_checkpoint=lambda: None,
    )

    assert actions is None
    assert status["status"] == "action_scout_failed"
    assert calls == 1
    assert "split" not in cache["action_scout_chunks"][0]


def test_hierarchical_action_scout_stops_at_bounded_truncation_depth(monkeypatch):
    records = _action_scout_chunk_records()
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    cache = {
        "action_scout_chunks": [],
        "action_scout": None,
        "last_rejected_action_scout_chunk": None,
    }

    monkeypatch.setattr(
        smart_minutes,
        "ACTION_SCOUT_TRUNCATION_SPLIT_MIN_RECORDS",
        1,
    )
    monkeypatch.setattr(
        smart_minutes,
        "ACTION_SCOUT_TRUNCATION_SPLIT_MAX_DEPTH",
        1,
    )

    def fake_request(*, messages, config, max_tokens=16000):
        return None, {
            "status": "invalid_model_json",
            "starts_with_object": True,
            "ends_with_object": False,
        }

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    actions, status = smart_minutes._run_hierarchical_action_scout(
        records=records,
        chunks=[chunk],
        required_action_groups=[],
        follow_up_hints=[],
        intent_recall_hints=[],
        config=DeepSeekConfig(model="test-model"),
        cache=cache,
        save_checkpoint=lambda: None,
    )

    assert actions is None
    assert status["status"] == "action_scout_truncation_unresolved"
    assert status["depth"] == 1


def test_hierarchical_action_scout_rejects_evidence_outside_child_chunk(monkeypatch):
    records = _action_scout_chunk_records()
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    cache = {
        "action_scout_chunks": [],
        "action_scout": None,
        "last_rejected_action_scout_chunk": None,
    }
    calls = 0

    monkeypatch.setattr(
        smart_minutes,
        "ACTION_SCOUT_TRUNCATION_SPLIT_MIN_RECORDS",
        1,
    )

    def fake_request(*, messages, config, max_tokens=16000):
        nonlocal calls
        calls += 1
        if calls == 1:
            return None, {
                "status": "invalid_model_json",
                "starts_with_object": True,
                "ends_with_object": False,
            }
        return {
            "actions": [
                {
                    "owner": "Billy",
                    "item": "在子片段外引用的事项。",
                    "segment_ids": [records[-1]["segment_id"]],
                    "basis": "self_commitment",
                }
            ]
        }, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    actions, status = smart_minutes._run_hierarchical_action_scout(
        records=records,
        chunks=[chunk],
        required_action_groups=[],
        follow_up_hints=[],
        intent_recall_hints=[],
        config=DeepSeekConfig(model="test-model"),
        cache=cache,
        save_checkpoint=lambda: None,
    )

    assert actions is None
    assert status["status"] == "action_scout_invalid"
    assert any("evidence_unknown" in error for error in status["errors"])


def test_hierarchical_action_scout_drops_post_repair_owner_evidence_mismatch(
    monkeypatch,
):
    records = canonical_transcript_records(
        [
            {
                "id": "owner-billy",
                "start": 0.0,
                "end": 10.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "We discussed the deployment plan.",
            },
            {
                "id": "owner-xin",
                "start": 12.0,
                "end": 22.0,
                "speaker": "Speaker 2",
                "name": "Xin",
                "name_confidence": 0.95,
                "text": "I will update the deployment configuration.",
            },
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    cache = {
        "action_scout_chunks": [],
        "action_scout": None,
        "last_rejected_action_scout_chunk": None,
    }
    invalid = {
        "actions": [
            {
                "owner": "Billy",
                "item": "更新部署配置。",
                "segment_ids": [records[1]["segment_id"]],
                "basis": "self_commitment",
            }
        ]
    }
    responses = [invalid, invalid]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    actions, status = smart_minutes._run_hierarchical_action_scout(
        records=records,
        chunks=[chunk],
        required_action_groups=[],
        follow_up_hints=[],
        intent_recall_hints=[],
        config=DeepSeekConfig(model="test-model"),
        cache=cache,
        save_checkpoint=lambda: None,
    )

    assert actions == []
    assert status["status"] == "ok"
    entry = cache["action_scout_chunks"][0]
    drop = entry["status"]["deterministic_owner_evidence_drop"]
    assert drop["dropped_action_count"] == 1
    assert drop["dropped_actions"] == [
        {
            "original_position": 1,
            "owner": "Billy",
            "evidence_speakers": ["Xin"],
            "validation_error": "action_scout:1:owner_evidence_mismatch",
        }
    ]
    assert entry["payload"] == {"actions": []}


def test_action_scout_owner_evidence_drop_is_bounded():
    records = canonical_transcript_records(
        [
            {
                "id": f"bounded-{index}",
                "start": float(index * 10),
                "end": float(index * 10 + 8),
                "speaker": "Speaker 1",
                "name": "Xin",
                "name_confidence": 0.95,
                "text": f"I will update item {index}.",
            }
            for index in range(3)
        ]
    )
    payload = {
        "actions": [
            {
                "owner": "Billy",
                "item": f"错误归属事项 {index}",
                "segment_ids": [records[index]["segment_id"]],
                "basis": "self_commitment",
            }
            for index in range(3)
        ]
    }

    recovered, actions, status = (
        smart_minutes._drop_deterministically_mismatched_action_scout_actions(
            payload=payload,
            validation_errors=[
                f"action_scout:{index}:owner_evidence_mismatch"
                for index in range(1, 4)
            ],
            transcript_records=records,
            required_action_groups=[],
        )
    )

    assert recovered is None
    assert actions is None
    assert status["status"] == "owner_evidence_drop_threshold_exceeded"


def test_hierarchical_action_scout_persists_owner_drop_failure_details(monkeypatch):
    records = canonical_transcript_records(
        [
            {
                "id": "threshold-billy",
                "start": 0.0,
                "end": 8.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "We reviewed the deployment work.",
            },
            *[
                {
                    "id": f"threshold-xin-{index}",
                    "start": float(10 + index * 10),
                    "end": float(18 + index * 10),
                    "speaker": "Speaker 2",
                    "name": "Xin",
                    "name_confidence": 0.95,
                    "text": f"I will update deployment item {index}.",
                }
                for index in range(3)
            ],
        ]
    )
    chunk = transcript_record_chunks(
        records,
        target_chars=100_000,
        hard_chars=120_000,
        overlap_records=0,
    )[0]
    cache = {
        "action_scout_chunks": [],
        "action_scout": None,
        "last_rejected_action_scout_chunk": None,
    }
    invalid = {
        "actions": [
            {
                "owner": "Billy",
                "item": f"错误归属事项 {index}",
                "segment_ids": [records[index + 1]["segment_id"]],
                "basis": "self_commitment",
            }
            for index in range(3)
        ]
    }
    responses = [invalid, invalid]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)
    actions, status = smart_minutes._run_hierarchical_action_scout(
        records=records,
        chunks=[chunk],
        required_action_groups=[],
        follow_up_hints=[],
        intent_recall_hints=[],
        config=DeepSeekConfig(model="test-model"),
        cache=cache,
        save_checkpoint=lambda: None,
    )

    assert actions is None
    assert status["status"] == "action_scout_invalid"
    rejected = cache["action_scout_chunks"][0]["rejected"]
    assert rejected["owner_evidence_drop"]["status"] == (
        "owner_evidence_drop_threshold_exceeded"
    )
    assert rejected["owner_evidence_drop"]["mismatched_action_positions"] == [1, 2, 3]
    assert rejected["payload"] == invalid


def test_action_scout_chunk_merge_is_deterministic_deduplicated_and_capped():
    records = _action_scout_chunk_records(28)
    actions = [
        {
            "owner": "Billy",
            "item": f"行动 {index}",
            "segment_ids": [records[index]["segment_id"]],
            "basis": "self_commitment",
        }
        for index in range(26)
    ]
    actions.append(
        {
            "owner": "Billy",
            "item": "行动 0",
            "segment_ids": [records[26]["segment_id"]],
            "basis": "self_commitment",
        }
    )

    merged = smart_minutes._deduplicate_chunk_actions(
        actions,
        transcript_records=records,
        required_action_groups=[],
    )

    assert len(merged) == MAX_ACTION_SCOUT_ACTIONS
    assert [action["item"] for action in merged] == [
        f"行动 {index}"
        for index in range(MAX_ACTION_SCOUT_ACTIONS)
    ]
    assert merged[0]["segment_ids"] == [
        records[0]["segment_id"],
        records[26]["segment_id"],
    ]


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


def test_partial_implicit_follow_up_validation_preserves_original_hint_indexes():
    records = canonical_transcript_records(_segments())
    hints = _two_follow_up_hints(records)

    actions, errors = validate_implicit_follow_up_judgments(
        {"judgments": [_negative_implicit_judgment(2)]},
        follow_up_hints=hints,
        transcript_records=records,
        hint_indexes=[2],
    )

    assert actions == []
    assert errors == []


def test_implicit_follow_up_prompt_fingerprint_binds_full_hint_content():
    records = canonical_transcript_records(_segments())
    first_hints = _two_follow_up_hints(records)
    second_hints = copy.deepcopy(first_hints)
    second_hints[1]["context"][0]["text"] = "A changed follow-up context."

    first_messages = smart_minutes.build_implicit_follow_up_messages(
        first_hints,
        explicit_actions=[],
    )
    second_messages = smart_minutes.build_implicit_follow_up_messages(
        second_hints,
        explicit_actions=[],
    )

    assert smart_minutes._messages_fingerprint(first_messages) != (
        smart_minutes._messages_fingerprint(second_messages)
    )


def test_implicit_follow_up_coverage_requeries_only_the_missing_hint(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    responses = [
        _action_scout(records),
        {"judgments": [_negative_implicit_judgment(1)]},
        {"judgments": [_negative_implicit_judgment(2)]},
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    calls: list[list[dict[str, str]]] = []

    def fake_request(*, messages, config, max_tokens=16000):
        calls.append(messages)
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    assert len(calls) == 7
    focused_input = json.loads(calls[2][1]["content"])
    assert [
        hint["hint_index"]
        for hint in focused_input["follow_up_context_hints"]
    ] == [2]
    recovery = result.audit["implicit_follow_up_scout"]["status"][
        "coverage_recovery"
    ]
    assert recovery["status"] == "focused_missing_hint_requery"
    assert recovery["missing_hint_indexes"] == [2]
    assert recovery["deterministic_negative_hint_indexes"] == []
    assert result.audit["implicit_follow_up_scout"]["actions"] == []


def test_implicit_follow_up_coverage_fallback_is_auditable_and_non_promoting(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    responses = [
        _action_scout(records),
        {"judgments": [_negative_implicit_judgment(1)]},
        None,
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    checkpoints: list[dict] = []

    def fake_request(*, messages, config, max_tokens=16000):
        response = responses.pop(0)
        if response is None:
            return None, {"status": "temporary_failure"}
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert result is not None, status
    recovery = result.audit["implicit_follow_up_scout"]["status"][
        "coverage_recovery"
    ]
    assert recovery["status"] == "deterministic_negative_coverage_fill"
    assert recovery["deterministic_negative_hint_indexes"] == [2]
    assert result.audit["implicit_follow_up_scout"]["actions"] == []
    cached = checkpoints[-1]["implicit_follow_up_scout"]
    assert cached["status"]["coverage_recovery"][
        "deterministic_negative_hint_indexes"
    ] == [2]
    assert [
        judgment["hint_index"]
        for judgment in cached["payload"]["judgments"]
    ] == [1, 2]

    def unexpected_request(**kwargs):
        raise AssertionError("cached deterministic fill must not call the model")

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", unexpected_request)
    resumed, resumed_status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint=checkpoints[-1],
    )

    assert resumed is not None, resumed_status
    assert resumed_status["implicit_follow_up_scout"]["status"] == "cached"
    assert resumed_status["implicit_follow_up_scout"]["coverage_recovery"][
        "deterministic_negative_hint_indexes"
    ] == [2]


def test_implicit_follow_up_repair_preserves_failed_coverage_recovery_audit(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    ungrounded = {
        "hint_index": 1,
        "qualifies": True,
        "owner": "Billy",
        "item": "复核最新发票并计算开发与支持占比。",
        "segment_ids": [records[0]["segment_id"]],
        "reason": "Billy 承诺复核发票。",
    }
    responses = [
        _action_scout(records),
        {"judgments": [ungrounded]},
        None,
        {
            "judgments": [
                ungrounded,
                _negative_implicit_judgment(2),
            ]
        },
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        response = responses.pop(0)
        if response is None:
            return None, {"status": "temporary_failure"}
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    implicit_status = result.audit["implicit_follow_up_scout"]["status"]
    assert implicit_status["repair_attempted"] is True
    assert implicit_status["coverage_recovery"]["status"] == (
        "coverage_recovery_failed"
    )
    assert implicit_status["deterministic_grounding_downgrade"][
        "downgraded_hint_indexes"
    ] == [1]
    assert any(
        error.endswith("owned_follow_up_not_grounded")
        for error in implicit_status["coverage_recovery"]["fallback_validation_errors"]
    )


def test_implicit_follow_up_repair_failure_downgrades_complete_coverage_candidate(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    ungrounded = {
        "hint_index": 1,
        "qualifies": True,
        "owner": "Billy",
        "item": "复核最新发票并计算开发与支持占比。",
        "segment_ids": [records[0]["segment_id"]],
        "reason": "Billy 承诺复核发票。",
    }
    responses = [
        _action_scout(records),
        {"judgments": [ungrounded]},
        None,
        None,
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        response = responses.pop(0)
        if response is None:
            return None, {"status": "temporary_failure"}
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    implicit_status = result.audit["implicit_follow_up_scout"]["status"]
    assert implicit_status["status"] == (
        "deterministic_grounding_downgrade_after_repair_request_failure"
    )
    assert implicit_status["repair_status"]["status"] == "temporary_failure"
    assert implicit_status["grounding_downgrade_source"] == (
        "coverage_recovery_candidate"
    )
    assert implicit_status["coverage_recovery"]["status"] == (
        "coverage_recovery_failed"
    )
    assert implicit_status["deterministic_grounding_downgrade"][
        "downgraded_hint_indexes"
    ] == [1]
    assert result.audit["implicit_follow_up_scout"]["actions"] == []


def test_implicit_follow_up_repair_failure_downgrades_complete_original_candidate(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    ungrounded = {
        "hint_index": 1,
        "qualifies": True,
        "owner": "Billy",
        "item": "复核最新发票并计算开发与支持占比。",
        "segment_ids": [records[0]["segment_id"]],
        "reason": "Billy 承诺复核发票。",
    }
    responses = [
        _action_scout(records),
        {
            "judgments": [
                ungrounded,
                _negative_implicit_judgment(2),
            ]
        },
        None,
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        response = responses.pop(0)
        if response is None:
            return None, {"status": "temporary_failure"}
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    implicit_status = result.audit["implicit_follow_up_scout"]["status"]
    assert implicit_status["status"] == (
        "deterministic_grounding_downgrade_after_repair_request_failure"
    )
    assert implicit_status["grounding_downgrade_source"] == (
        "original_complete_candidate"
    )
    assert implicit_status["deterministic_grounding_downgrade"][
        "downgraded_hint_indexes"
    ] == [1]


def test_implicit_follow_up_invalid_repair_downgrades_complete_original_candidate(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    original = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": True,
                "owner": "Billy",
                "item": "复核最新发票并计算开发与支持占比。",
                "segment_ids": [records[0]["segment_id"]],
                "reason": "Billy 承诺复核发票。",
            },
            _negative_implicit_judgment(2),
        ]
    }
    incomplete_repair = {
        "judgments": [
            _negative_implicit_judgment(2),
        ]
    }
    responses = [
        _action_scout(records),
        original,
        incomplete_repair,
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    implicit_status = result.audit["implicit_follow_up_scout"]["status"]
    assert implicit_status["status"] == (
        "deterministic_grounding_downgrade_after_invalid_repair_original_candidate"
    )
    assert implicit_status["grounding_downgrade_source"] == (
        "original_complete_candidate_after_invalid_repair"
    )
    assert implicit_status["repair_validation_errors"] == [
        "implicit_follow_up_hint_coverage_invalid"
    ]
    assert implicit_status["repair_payload_deterministic_grounding_downgrade"][
        "status"
    ] == "grounding_downgrade_not_applicable"
    assert implicit_status["deterministic_grounding_downgrade"][
        "downgraded_hint_indexes"
    ] == [1]


def test_implicit_follow_up_invalid_repair_preserves_dual_downgrade_failures(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    original = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": True,
                "owner": "Billy",
                "item": "复核最新发票并计算开发与支持占比。",
                "segment_ids": [records[0]["segment_id"]],
                "reason": "Billy 承诺复核发票。",
            },
            _negative_implicit_judgment(2),
        ]
    }
    incomplete_repair = {
        "judgments": [
            _negative_implicit_judgment(2),
        ]
    }
    responses = [
        _action_scout(records),
        original,
        incomplete_repair,
    ]
    checkpoints: list[dict] = []
    downgrade_payloads: list[object] = []

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    def fake_downgrade(*, payload, validation_errors, follow_up_hints, transcript_records):
        downgrade_payloads.append(payload)
        if len(downgrade_payloads) == 1:
            return None, None, {
                "status": "grounding_downgrade_not_applicable",
                "validation_errors": validation_errors,
            }
        return None, None, {
            "status": "grounding_downgrade_revalidation_failed",
            "validation_errors": validation_errors,
            "revalidation_errors": ["implicit_follow_up_hint_coverage_invalid"],
        }

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes._downgrade_deterministically_rejected_implicit_judgments",
        fake_downgrade,
    )
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert result is None
    assert status["status"] == "implicit_follow_up_scout_invalid"
    assert downgrade_payloads == [incomplete_repair, original]
    assert status["implicit_follow_up_scout"][
        "deterministic_grounding_downgrade"
    ]["status"] == "grounding_downgrade_not_applicable"
    assert status["implicit_follow_up_scout"][
        "original_candidate_deterministic_grounding_downgrade"
    ]["status"] == "grounding_downgrade_revalidation_failed"
    rejected = checkpoints[-1]["last_rejected_implicit_follow_up_scout"]
    assert rejected["payload"] == original
    assert rejected["repair_payload"] == incomplete_repair
    assert rejected["initial_validation_errors"] == [
        "implicit_follow_up:1:owned_follow_up_not_grounded"
    ]
    assert rejected["deterministic_grounding_downgrade"]["status"] == (
        "grounding_downgrade_not_applicable"
    )
    assert rejected["original_candidate_deterministic_grounding_downgrade"][
        "status"
    ] == "grounding_downgrade_revalidation_failed"


def test_implicit_follow_up_post_repair_downgrades_only_deterministic_rejections(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    ungrounded = {
        "hint_index": 1,
        "qualifies": True,
        "owner": "Billy",
        "item": "复核最新发票并计算开发与支持占比。",
        "segment_ids": [records[0]["segment_id"]],
        "reason": "Billy 承诺复核发票。",
    }
    rejected_payload = {
        "judgments": [
            ungrounded,
            _negative_implicit_judgment(2),
        ]
    }
    responses = [
        _action_scout(records),
        rejected_payload,
        rejected_payload,
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _source_payload([record["segment_id"] for record in records], english=True),
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None, status
    implicit_status = result.audit["implicit_follow_up_scout"]["status"]
    assert implicit_status["status"] == (
        "deterministic_grounding_downgrade_after_repair"
    )
    downgrade = implicit_status["deterministic_grounding_downgrade"]
    assert downgrade["downgraded_hint_indexes"] == [1]
    assert any(
        error.endswith("owned_follow_up_not_grounded")
        for error in downgrade["rejected_by_hint"][1]
    )
    assert result.audit["implicit_follow_up_scout"]["actions"] == []
    assert "original_candidate_deterministic_grounding_downgrade" not in (
        implicit_status
    )


def test_implicit_grounding_downgrade_rejects_mixed_validation_errors():
    records = canonical_transcript_records(_segments())
    hints = _two_follow_up_hints(records)
    payload = {
        "judgments": [
            _negative_implicit_judgment(1),
            _negative_implicit_judgment(2),
        ]
    }

    downgraded, actions, status = (
        smart_minutes._downgrade_deterministically_rejected_implicit_judgments(
            payload=payload,
            validation_errors=[
                "implicit_follow_up:1:owned_follow_up_not_grounded",
                "implicit_follow_up:2:owner_not_anchor",
            ],
            follow_up_hints=hints,
            transcript_records=records,
        )
    )

    assert downgraded is None
    assert actions is None
    assert status["status"] == "grounding_downgrade_not_applicable"


def test_implicit_grounding_downgrade_rejects_inconsistent_payload_safely():
    records = canonical_transcript_records(_segments())
    hints = _two_follow_up_hints(records)
    payload = {
        "judgments": [
            _negative_implicit_judgment(1),
            "invalid judgment",
        ]
    }

    downgraded, actions, status = (
        smart_minutes._downgrade_deterministically_rejected_implicit_judgments(
            payload=payload,
            validation_errors=[
                "implicit_follow_up:1:owned_follow_up_not_grounded",
            ],
            follow_up_hints=hints,
            transcript_records=records,
        )
    )

    assert downgraded is None
    assert actions is None
    assert status["status"] == "grounding_downgrade_payload_invalid"


def test_implicit_follow_up_non_deterministic_post_repair_error_fails_closed(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    original = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": True,
                "owner": "Xin",
                "item": "原始候选错误地指定了另一位负责人。",
                "segment_ids": [records[0]["segment_id"]],
                "reason": "原始候选未通过负责人验证。",
            },
            _negative_implicit_judgment(2),
        ]
    }
    repair_invalid = {
        "judgments": [
            {
                "hint_index": 1,
                "qualifies": True,
                "owner": "Xin",
                "item": "复核最新发票并计算开发与支持占比。",
                "segment_ids": [records[0]["segment_id"]],
                "reason": "错误地指定了另一位负责人。",
            },
            _negative_implicit_judgment(2),
        ]
    }
    responses = [
        _action_scout(records),
        original,
        repair_invalid,
    ]
    checkpoints: list[dict] = []

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.follow_up_context_hints",
        _two_follow_up_hints,
    )

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert result is None
    assert status["status"] == "implicit_follow_up_scout_invalid"
    assert status["errors"] == ["implicit_follow_up:1:owner_not_anchor"]
    assert status["implicit_follow_up_scout"][
        "deterministic_grounding_downgrade"
    ]["status"] == "grounding_downgrade_not_applicable"
    rejected = checkpoints[-1]["last_rejected_implicit_follow_up_scout"]
    assert rejected["payload"] == original
    assert rejected["repair_payload"] == repair_invalid
    assert rejected["original_candidate_deterministic_grounding_downgrade"] is None


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


def test_direct_self_commitment_protection_recognizes_clear_variants():
    records = canonical_transcript_records(
        [
            {
                "id": "make-sure",
                "start": 0.0,
                "end": 8.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "name_source": "visual_active_speaker_highlight",
                "text": "I'm going to make sure that we work on the document and present it.",
            },
            {
                "id": "start-work",
                "start": 9.0,
                "end": 15.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "name_source": "visual_active_speaker_highlight",
                "text": "I'll start working on it today.",
            },
            {
                "id": "send-before-thursday",
                "start": 16.0,
                "end": 25.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "name_source": "visual_active_speaker_highlight",
                "text": "I want to send the document before Thursday so the team can review it.",
            },
        ],
        allow_cluster_name_consensus=False,
    )
    candidates = [
        {
            "owner": "John",
            "item": "Prepare the document.",
            "segment_ids": [record["segment_id"]],
            "basis": "self_commitment",
        }
        for record in records
    ]

    smart_minutes._protect_direct_self_commitment_candidates(
        candidates,
        transcript_records=records,
    )

    assert all(
        smart_minutes._positive_self_commitment([record])
        for record in records
    )
    assert all(candidate.get("must_keep") is True for candidate in candidates)


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


def test_final_gate_rejects_collapsed_distinct_must_keep_candidates():
    records, minutes, review, scout = _collapsed_must_keep_fixture()

    errors = validate_publication_gate(
        review,
        minutes,
        transcript_records=records,
        action_scout=scout,
    )

    assert errors == [
        "publication_gate_candidate_disposition:2:"
        "must_keep_candidates_collapsed:1",
        "publication_gate_candidate_disposition:3:"
        "must_keep_candidates_collapsed:1",
    ]


def test_deterministic_final_repair_splits_collapsed_must_keep_candidates():
    records, _minutes, review, scout = _collapsed_must_keep_fixture()
    errors = validate_publication_gate(
        review,
        review["minutes"],
        transcript_records=records,
        action_scout=scout,
    )
    errors.append("action:1:evidence_span_too_wide:492.0>120.0")

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=errors,
        action_scout=scout,
        transcript_records=records,
    )

    assert repaired is not None
    assert "split_must_keep_action:1:3" in changes
    assert repaired["minutes"]["actions"] == [
        {
            "owner": candidate["owner"],
            "item": candidate["item"],
            "segment_ids": candidate["segment_ids"],
        }
        for candidate in scout
    ]
    assert [
        disposition["action_index"]
        for disposition in repaired["candidate_dispositions"]
    ] == [1, 2, 3]
    assert repaired["action_support"] == [
        {
            "action_index": index,
            "segment_ids": candidate["segment_ids"],
            "basis": candidate["basis"],
        }
        for index, candidate in enumerate(scout, start=1)
    ]
    assert validate_publication_gate(
        repaired,
        repaired["minutes"],
        transcript_records=records,
        action_scout=scout,
    ) == []


def test_deterministic_final_repair_restores_rejected_must_keep_candidate():
    records, _minutes, _review, full_scout = _collapsed_must_keep_fixture()
    scout = full_scout[:2]
    minutes = {
        "themes": [],
        "project_updates": [],
        "decisions": [],
        "actions": [
            {
                "owner": scout[0]["owner"],
                "item": scout[0]["item"],
                "segment_ids": scout[0]["segment_ids"],
            }
        ],
    }
    review = _publication_review(
        minutes,
        candidate_dispositions=[
            {
                "candidate_index": 1,
                "disposition": "kept",
                "action_index": 1,
                "reason_code": "supported",
                "reason": "The first commitment is supported.",
            },
            {
                "candidate_index": 2,
                "disposition": "rejected",
                "action_index": None,
                "reason_code": "unsupported_item",
                "reason": "The second commitment is already covered.",
            },
        ],
    )

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "publication_gate_candidate_disposition:2:"
            "must_keep_candidate_rejected"
        ],
        action_scout=scout,
        transcript_records=records,
    )

    assert repaired is not None
    assert "restored_supported_candidate:2" in changes
    assert repaired["minutes"]["actions"][1] == {
        "owner": scout[1]["owner"],
        "item": scout[1]["item"],
        "segment_ids": scout[1]["segment_ids"],
    }
    assert repaired["candidate_dispositions"][1]["action_index"] == 2
    assert validate_publication_gate(
        repaired,
        repaired["minutes"],
        transcript_records=records,
        action_scout=scout,
    ) == []


def test_deterministic_final_repair_neutralizes_nested_theme_and_update_literals():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["current_state"] = "当前方案仍在讨论，使用Kubernetes处理90%的流量。"
    minutes["project_updates"][0]["update"] = "当前状态已记录，处理90%的流量。"
    minutes["actions"] = []
    review = _publication_review(minutes, candidate_dispositions=[])

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "theme:1:current_state:named_entity_ungrounded:Kubernetes",
            "theme:1:current_state:number_ungrounded:90",
            "project_update:1:number_ungrounded:90",
        ],
        action_scout=[],
        transcript_records=records,
    )

    assert repaired is not None
    assert "dropped_ungrounded_theme_entity_clause:1:current_state" in changes
    assert "neutralized_ungrounded_project_update_literals:1" in changes
    assert "Kubernetes" not in repaired["minutes"]["themes"][0]["current_state"]
    assert "90" not in repaired["minutes"]["themes"][0]["current_state"]
    assert "90" not in repaired["minutes"]["project_updates"][0]["update"]


def test_deterministic_final_repair_rebuilds_wide_follow_up_update():
    records = canonical_transcript_records(
        [
            {
                "id": "john-prepare",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "text": "I will prepare the product document.",
            },
            {
                "id": "john-send",
                "start": 300.0,
                "end": 312.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "text": "I will send it before Thursday.",
            },
        ]
    )
    action = {
        "owner": "John",
        "item": "准备产品文档。",
        "segment_ids": [records[0]["segment_id"]],
    }
    minutes = {
        "themes": [],
        "project_updates": [
            {
                "participant": "John",
                "project": "后续跟进",
                "update": "准备产品文档，并在周四前发送。",
                "segment_ids": [
                    records[0]["segment_id"],
                    records[1]["segment_id"],
                ],
            }
        ],
        "decisions": [],
        "actions": [action],
    }
    review = _publication_review(minutes)
    scout = [{**action, "basis": "self_commitment"}]

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["project_update:1:evidence_span_too_wide:312.0>120.0"],
        action_scout=scout,
        transcript_records=records,
    )

    assert repaired is not None
    assert "rebuilt_project_update:1" in changes
    assert repaired["minutes"]["project_updates"][0] == {
        "participant": "John",
        "project": "后续跟进",
        "update": "准备产品文档。",
        "segment_ids": [records[0]["segment_id"]],
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


def test_positive_self_commitment_uses_contiguous_action_evidence():
    assert smart_minutes._positive_self_commitment(
        [
            {"text": "I'll continue to"},
            {"text": "work on the integration."},
        ]
    )
    assert smart_minutes._positive_self_commitment(
        [{"text": "I gotta check why it failed."}]
    )
    assert smart_minutes._positive_self_commitment(
        [{"text": "I need to create the application first."}]
    )
    assert smart_minutes._positive_self_commitment(
        [{"text": "I will wait for the response and talk to John."}]
    )
    assert not smart_minutes._positive_self_commitment(
        [{"text": "Could you say I will check it?"}]
    )


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
    assert "completed, in progress, requested" in messages[0]["content"]
    assert "candidate_dispositions" in messages[0]["content"]
    assert "prior_finding_dispositions" in messages[0]["content"]


def test_prior_review_findings_are_severity_bounded_with_audit_index():
    findings = [
        {
            "severity": "low",
            "category": "wording",
            "description": "Low-priority wording cleanup.",
        },
        {
            "severity": "material",
            "category": "action_recall",
            "description": "A material action might be missing.",
        },
        {
            "severity": "high",
            "category": "speaker_attribution",
            "description": "A speaker attribution requires correction.",
        },
        {
            "severity": "medium",
            "category": "decision_precision",
            "description": "A decision needs tighter wording.",
        },
        {
            "severity": "blocker",
            "category": "unsupported_action",
            "description": "An action lacks direct evidence.",
        },
        {
            "severity": "invalid",
            "category": "bad category",
            "description": "This malformed finding must not reach the final model.",
        },
    ]

    retained, audit = smart_minutes._bounded_prior_review_findings(findings)

    assert [finding["severity"] for finding in retained] == [
        "blocker",
        "high",
        "material",
        "medium",
    ]
    assert audit["retained_source_indexes"] == [5, 3, 2, 4]
    assert audit["discarded_count"] == 2
    assert audit["discarded_findings"] == [
        {
            "source_index": 1,
            "severity": "low",
            "category": "wording",
            "description_sha256": smart_minutes._review_finding_digest(
                "Low-priority wording cleanup."
            ),
            "reason": "over_budget",
        },
        {
            "source_index": 6,
            "severity": "invalid",
            "category": "bad category",
            "description_sha256": smart_minutes._review_finding_digest(
                "This malformed finding must not reach the final model."
            ),
            "reason": "invalid_severity",
        },
    ]


def test_final_review_prompt_includes_prior_finding_budget():
    records = canonical_transcript_records(_segments())
    draft = _source_payload([record["segment_id"] for record in records])
    findings, budget = smart_minutes._bounded_prior_review_findings(
        [
            {
                "severity": "high",
                "category": "speaker_attribution",
                "description": "Verify the named speaker.",
            }
        ]
    )

    messages = build_review_messages(
        records,
        draft,
        required_project_participants=["Billy"],
        prior_findings=findings,
        prior_finding_budget=budget,
        pass_index=2,
    )
    payload = json.loads(messages[1]["content"])

    assert payload["prior_findings"] == findings
    assert payload["prior_finding_budget"] == budget
    assert "omitted advisory findings" in messages[0]["content"]


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


def test_deterministic_final_repair_neutralizes_entities_and_deduplicates_updates():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["actions"][0]["item"] = "继续推进集成工作，处理 GitLab 配置。"
    minutes["project_updates"][0].update(
        {
            "project": "GitLab 集成",
            "update": "继续推进 GitLab 集成。",
        }
    )
    minutes["project_updates"].append(
        {
            "participant": "Billy",
            "project": "重复更新",
            "update": "这是一条重复的项目更新。",
            "segment_ids": [records[1]["segment_id"]],
        }
    )
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "action:1:named_entity_ungrounded:GitLab",
            "project_update:1:named_entity_ungrounded:GitLab",
            "project_update:2:participant_duplicate",
        ],
        action_scout=_action_scout(records)["actions"],
    )

    assert repaired is not None
    assert "dropped_ungrounded_action_entity_clause:1" in changes
    assert "rebuilt_project_update:1" in changes
    assert "dropped_duplicate_project_update:2" in changes
    assert "GitLab" not in repaired["minutes"]["actions"][0]["item"]
    assert repaired["minutes"]["project_updates"] == [
        {
            "participant": "Billy",
            "project": "后续跟进",
            "update": "继续推进集成工作",
            "segment_ids": [records[0]["segment_id"]],
        }
    ]


def test_deterministic_final_repair_drops_ungrounded_decision_and_shortens_reason():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["decisions"] = [
        {
            "text": "BPS will be the core system for Kirill.",
            "segment_ids": [records[1]["segment_id"]],
        }
    ]
    review = _publication_review(minutes)
    review["candidate_dispositions"][0]["reason"] = "x" * 121

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "decision:1:named_entity_ungrounded:BPS",
            "decision:1:named_entity_ungrounded:Kirill",
            "publication_gate_decision_support:1:explicit_agreement_not_grounded",
            "publication_gate_candidate_disposition:1:reason_too_long",
        ],
        action_scout=_action_scout(records)["actions"],
        transcript_records=records,
    )

    assert repaired is not None
    assert "dropped_ungrounded_decision:1" in changes
    assert "shortened_candidate_disposition_reason:1" in changes
    assert repaired["minutes"]["decisions"] == []
    assert repaired["decision_support"] == []
    assert len(repaired["candidate_dispositions"][0]["reason"]) <= (
        smart_minutes.FINAL_REVIEW_REASON_MAX_CHARS
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
        action_scout=_action_scout(records)["actions"],
    ) == []


def test_deterministic_final_repair_reindexes_surviving_decision_support():
    records = canonical_transcript_records(
        [
            {
                "id": "decision-1",
                "start": 0.0,
                "end": 10.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "We should discuss the deployment plan.",
            },
            {
                "id": "decision-2",
                "start": 11.0,
                "end": 21.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.95,
                "text": "We selected the single deployment path.",
            },
        ]
    )
    minutes = {
        "themes": [],
        "project_updates": [],
        "decisions": [
            {
                "text": "BPS is approved.",
                "segment_ids": [records[0]["segment_id"]],
            },
            {
                "text": "The single deployment path was selected.",
                "segment_ids": [records[1]["segment_id"]],
            },
        ],
        "actions": [],
    }
    review = {
        "findings": [],
        "minutes": minutes,
        "prior_finding_dispositions": [],
        "candidate_dispositions": [],
        "action_support": [],
        "decision_support": [
            {
                "decision_index": 1,
                "segment_ids": [records[0]["segment_id"]],
                "basis": "explicit_agreement",
            },
            {
                "decision_index": 2,
                "segment_ids": [records[1]["segment_id"]],
                "basis": "selected_direction",
            },
        ],
        "publishable": True,
    }

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "decision:1:named_entity_ungrounded:BPS",
            "publication_gate_decision_support:1:explicit_agreement_not_grounded",
        ],
        action_scout=[],
        transcript_records=records,
    )

    assert repaired is not None
    assert "dropped_ungrounded_decision:1" in changes
    assert repaired["minutes"]["decisions"] == [minutes["decisions"][1]]
    assert repaired["decision_support"] == [
        {
            "decision_index": 1,
            "segment_ids": [records[1]["segment_id"]],
            "basis": "selected_direction",
        }
    ]


def test_deterministic_final_repair_neutralizes_future_owner_after_action_drop():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["outcome"] = "Billy will build the API gateway."
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["publication_gate_action_support:1:self_commitment_not_grounded"],
        action_scout=_action_scout(records)["actions"],
        transcript_records=records,
    )

    assert repaired is not None
    assert "dropped_unsupported_action:1" in changes
    assert "neutralized_future_owner_theme:1" in changes
    assert "Billy" not in repaired["minutes"]["themes"][0]["outcome"]
    _, source_errors = validate_source_minutes(
        repaired["minutes"],
        transcript_records=records,
        required_project_participants=[],
    )
    assert not [
        error
        for error in source_errors
        if error.startswith("theme:1:future_owner_without_action:")
    ]


def test_deterministic_final_repair_neutralizes_unconfirmed_generic_outcome():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["outcome"] = "团队同意将构建 API 网关并实施数据库复制。"
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["theme:1:outcome_unconfirmed_commitment"],
        action_scout=_action_scout(records)["actions"],
        transcript_records=records,
    )

    assert repaired is not None
    assert changes == ["neutralized_unconfirmed_theme_outcome:1"]
    assert repaired["minutes"]["themes"][0]["outcome"] == (
        "讨论了相关方案与后续安排，尚未形成有明确证据支持的最终决定。"
    )


def test_deterministic_final_repair_neutralizes_ungrounded_consensus_outcome():
    records = canonical_transcript_records(_segments())
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["outcome"] = "讨论形成共识：团队将继续推进该方案。"
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["theme:1:outcome_ungrounded_consensus"],
        action_scout=_action_scout(records)["actions"],
        transcript_records=records,
    )

    assert repaired is not None
    assert changes == ["neutralized_unconfirmed_theme_outcome:1"]
    assert repaired["minutes"]["themes"][0]["outcome"] == (
        "讨论了相关方案与后续安排，尚未形成有明确证据支持的最终决定。"
    )


def test_deterministic_final_repair_restores_outline_evidence_for_short_theme():
    segments = _segments() + [
        {
            "id": "outline-end",
            "start": 190.0,
            "end": 200.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_confidence": 0.95,
            "text": "The closing portion of the same topic.",
        }
    ]
    records = canonical_transcript_records(segments)
    minutes = _source_payload([records[0]["segment_id"], records[1]["segment_id"]])
    review = _publication_review(minutes)
    outline = [
        {
            "start_segment_id": records[0]["segment_id"],
            "end_segment_id": records[2]["segment_id"],
            "anchor_segment_ids": [records[2]["segment_id"]],
        }
    ]

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "theme:1:outline_anchor_missing",
            "theme:1:span_too_short:70.0<180.0",
        ],
        action_scout=_action_scout(records)["actions"],
        transcript_records=records,
        theme_outline=outline,
    )

    assert repaired is not None
    assert changes == ["restored_theme_outline_evidence:1"]
    assert records[2]["segment_id"] in repaired["minutes"]["themes"][0][
        "evidence_segment_ids"
    ]


def test_deterministic_final_repair_drops_anonymous_point_without_guessing_name():
    records = canonical_transcript_records(
        _segments(),
        allow_cluster_name_consensus=False,
    )
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["evidence_segment_ids"] = [records[1]["segment_id"]]
    minutes["themes"][0]["key_points"] = [
        {
            "speaker": "Speaker 1",
            "text": "NPC raised a visibility concern.",
            "segment_ids": [records[1]["segment_id"]],
        }
    ]
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "theme:1:point:1:named_entity_ungrounded:NPC",
            "theme:1:point:1:speaker_anonymous",
        ],
        action_scout=_action_scout(records)["actions"],
        transcript_records=records,
    )

    assert repaired is not None
    assert changes == ["dropped_unattributed_theme_point:1:1"]
    assert repaired["minutes"]["themes"][0]["key_points"] == []


def test_deterministic_final_repair_keeps_entity_named_in_current_action_evidence():
    records = canonical_transcript_records(
        [
            {
                "id": "john-commitment",
                "start": 0.0,
                "end": 15.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.95,
                "name_source": "visual_active_speaker_highlight",
                "text": "I will prepare the document with Billy.",
            },
            {
                "id": "neutral-context",
                "start": 16.0,
                "end": 25.0,
                "speaker": "Speaker 2",
                "name": None,
                "name_confidence": 0.0,
                "text": "The document will describe the current product.",
            },
        ],
        allow_cluster_name_consensus=False,
    )
    minutes = {
        "themes": [
            {
                "title": "文档准备",
                "current_state": "讨论了产品说明文档。",
                "outcome": "尚未形成最终决定。",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [
                    {
                        "speaker": "John",
                        "text": "John将与Billy一起准备文档。",
                        "segment_ids": [records[0]["segment_id"]],
                    }
                ],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [
            {
                "owner": "John",
                "item": "与Billy一起准备文档。",
                "segment_ids": [records[0]["segment_id"]],
            }
        ],
    }
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["action:1:named_entity_ungrounded:Billy"],
        action_scout=[],
        transcript_records=records,
    )

    assert repaired is None
    assert changes == []


def test_deterministic_final_repair_corrects_or_drops_mismatched_point_speakers():
    segments = _segments()
    segments[0]["name_source"] = "visual_active_speaker_highlight"
    records = canonical_transcript_records(
        segments,
        allow_cluster_name_consensus=False,
    )
    minutes = _source_payload([record["segment_id"] for record in records])
    minutes["themes"][0]["key_points"] = [
        {
            "speaker": "John",
            "text": "Billy discussed the invoice review.",
            "segment_ids": [records[0]["segment_id"]],
        },
        {
            "speaker": "Billy",
            "text": "An unresolved participant raised a visibility concern.",
            "segment_ids": [records[1]["segment_id"]],
        },
    ]
    review = _publication_review(minutes)

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=[
            "theme:1:point:1:speaker_evidence_mismatch",
            "theme:1:point:2:speaker_evidence_mismatch",
        ],
        action_scout=_action_scout(records)["actions"],
        transcript_records=records,
    )

    assert repaired is not None
    assert "corrected_theme_point_speaker:1:1" in changes
    assert "dropped_unattributed_theme_point:1:2" in changes
    assert repaired["minutes"]["themes"][0]["key_points"] == [
        {
            "speaker": "Billy",
            "text": "Billy discussed the invoice review.",
            "segment_ids": [records[0]["segment_id"]],
        }
    ]


def test_direct_only_transcript_records_keep_cluster_extensions_anonymous():
    segments = _segments()
    segments[0]["name_source"] = "visual_active_speaker_highlight"
    segments[1].update(
        {
            "name": "Billy",
            "name_confidence": 0.95,
            "name_source": "voice_registry",
        }
    )
    records = canonical_transcript_records(
        segments,
        allow_cluster_name_consensus=False,
    )

    assert records[0]["identity_kind"] == "direct"
    assert records[1]["identity_kind"] == "anonymous"
    assert records[1]["speaker"] == "Speaker 1"


def test_idea_framed_let_me_is_not_a_positive_self_commitment():
    assert not smart_minutes._positive_self_commitment(
        [
            {
                "text": "Let me continue my my idea was we're going to build an API gateway.",
            }
        ]
    )


def test_generate_smart_minutes_rejects_checkpoint_from_another_identity_policy(
    monkeypatch,
):
    segments = _segments()
    for segment in segments:
        segment.update(
            {
                "name": "Billy",
                "name_confidence": 0.95,
                "name_source": "visual_active_speaker_highlight",
            }
        )
    records = canonical_transcript_records(
        segments,
        allow_cluster_name_consensus=False,
    )
    valid = _source_payload([record["segment_id"] for record in records])
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _publication_review(valid),
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    calls: list[list[dict[str, str]]] = []
    checkpoints: list[dict] = []

    def fake_request(*, messages, config, max_tokens=16000):
        calls.append(messages)
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)
    stale_checkpoint = {
        "format": smart_minutes.SMART_MINUTES_CHECKPOINT_FORMAT,
        "prompt_version": smart_minutes.SMART_PROMPT_VERSION,
        "transcript_sha256": smart_minutes.transcript_fingerprint(segments),
        "model": "test-model",
        "identity_policy": "cluster_consensus_allowed",
    }

    result, status = generate_smart_minutes(
        segments=segments,
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint=stale_checkpoint,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
        allow_cluster_name_consensus=False,
    )

    assert result is not None, status
    assert len(calls) == 7
    assert checkpoints[0]["identity_policy"] == "direct_visual_only"
    assert result.audit["identity_policy"] == "direct_visual_only"


def test_generate_smart_minutes_repairs_rejected_checkpoint_before_model_retry(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        _publication_review(valid),
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    checkpoints: list[dict] = []

    def initial_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", initial_request)
    first, first_status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert first is not None, first_status
    checkpoint = copy.deepcopy(checkpoints[-1])
    rejected_payload = _publication_review(copy.deepcopy(valid))
    rejected_payload["minutes"]["themes"][0]["key_points"][0]["speaker"] = "John"
    checkpoint["reviews"] = [copy.deepcopy(checkpoint["reviews"][0])]
    checkpoint["last_rejected_review"] = {
        "pass": 2,
        "input_sha256": "rejected-checkpoint",
        "payload": rejected_payload,
        "status": {"status": "review_schema_invalid"},
        "validation_errors": ["theme:1:point:1:speaker_evidence_mismatch"],
    }

    def unexpected_model_request(*, messages, config, max_tokens=16000):
        raise AssertionError("checkpoint repair should avoid another model request")

    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.request_deepseek_json",
        unexpected_model_request,
    )
    resumed, resumed_status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint=checkpoint,
    )

    assert resumed is not None, resumed_status
    final_status = resumed.audit["reviews"][1]["status"]
    assert final_status["status"] == "deterministic_checkpoint_repair"
    assert final_status["deterministic_repair_phase"] == "checkpoint"
    assert "corrected_theme_point_speaker:1:1" in final_status[
        "deterministic_repair_changes"
    ]


def test_entity_neutralization_removes_empty_related_modifiers_in_both_languages():
    assert smart_minutes._neutralize_ungrounded_entities(
        "排查并修复相关集成中的认证失败问题",
        {"GitLab"},
    ) == "排查并修复集成中的认证失败问题"
    assert smart_minutes._neutralize_ungrounded_entities(
        "Create related applications and configure interconnections.",
        {"Authentik"},
    ) == "Create applications and configure interconnections."


def test_deterministic_repair_expands_theme_local_evidence_for_entity():
    records = canonical_transcript_records(
        [
            {
                "id": "theme-anchor",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "text": "The interface needs another test.",
            },
            {
                "id": "bps-detail",
                "start": 36.0,
                "end": 48.0,
                "speaker": "Speaker 2",
                "text": "Please retest through the BPS interface.",
            },
        ],
        allow_cluster_name_consensus=False,
    )
    minutes = {
        "themes": [
            {
                "title": "接口复测",
                "current_state": "需要通过BPS接口重新测试。",
                "outcome": "尚未形成最终决定。",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }
    review = _publication_review(minutes, candidate_dispositions=[])
    outline = [
        {
            "start": 0.0,
            "end": 60.0,
            "anchor_segment_ids": [records[0]["segment_id"]],
        }
    ]

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["theme:1:current_state:named_entity_ungrounded:BPS"],
        action_scout=[],
        transcript_records=records,
        theme_outline=outline,
        entity_fallback=False,
    )

    assert repaired is not None
    assert "expanded_theme_entity_evidence:1:current_state:1" in changes
    assert repaired["minutes"]["themes"][0]["evidence_segment_ids"] == [
        records[0]["segment_id"],
        records[1]["segment_id"],
    ]
    assert repaired["minutes"]["themes"][0]["current_state"] == "需要通过BPS接口重新测试。"


def test_deterministic_repair_does_not_expand_entity_outside_theme_range():
    records = canonical_transcript_records(
        [
            {
                "id": "theme-anchor",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "text": "The interface needs another test.",
            },
            {
                "id": "out-of-range-bps",
                "start": 600.0,
                "end": 612.0,
                "speaker": "Speaker 2",
                "text": "Please retest through the BPS interface.",
            },
        ],
        allow_cluster_name_consensus=False,
    )
    minutes = {
        "themes": [
            {
                "title": "接口复测",
                "current_state": "需要通过BPS接口重新测试。",
                "outcome": "尚未形成最终决定。",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }
    review = _publication_review(minutes, candidate_dispositions=[])

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["theme:1:current_state:named_entity_ungrounded:BPS"],
        action_scout=[],
        transcript_records=records,
        theme_outline=[
            {
                "start": 0.0,
                "end": 60.0,
                "anchor_segment_ids": [records[0]["segment_id"]],
            }
        ],
        entity_fallback=False,
    )

    assert repaired is None
    assert changes == []


def test_theme_entity_expansion_never_relabels_anonymous_point():
    records = canonical_transcript_records(
        [
            {
                "id": "anonymous-point",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 1",
                "text": "The interface needs another test.",
            },
            {
                "id": "john-bps",
                "start": 20.0,
                "end": 32.0,
                "speaker": "Speaker 2",
                "name": "John",
                "name_confidence": 0.99,
                "name_source": "visual_active_speaker_highlight",
                "text": "Please retest through the BPS interface.",
            },
        ],
        allow_cluster_name_consensus=False,
    )
    minutes = {
        "themes": [
            {
                "title": "接口复测",
                "current_state": "接口需要再次测试。",
                "outcome": "尚未形成最终决定。",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [
                    {
                        "speaker": "Speaker 1",
                        "text": "需要通过BPS接口重新测试。",
                        "segment_ids": [records[0]["segment_id"]],
                    }
                ],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }
    review = _publication_review(minutes, candidate_dispositions=[])

    repaired, changes = _deterministic_final_review_repair(
        review,
        errors=["theme:1:point:1:text:named_entity_ungrounded:BPS"],
        action_scout=[],
        transcript_records=records,
        theme_outline=[
            {
                "start": 0.0,
                "end": 60.0,
                "anchor_segment_ids": [records[0]["segment_id"]],
            }
        ],
        entity_fallback=False,
    )

    assert repaired is None
    assert changes == []
    assert review["minutes"]["themes"][0]["key_points"][0]["speaker"] == "Speaker 1"


def test_targeted_field_repair_removes_masking_without_identity_change():
    records = canonical_transcript_records(
        [
            {
                "id": "john-commitment",
                "start": 0.0,
                "end": 15.0,
                "speaker": "Speaker 1",
                "name": "John",
                "name_confidence": 0.99,
                "name_source": "visual_active_speaker_highlight",
                "text": "I will start writing the document today and finish it with Billy.",
            }
        ],
        allow_cluster_name_consensus=False,
    )
    minutes = {
        "themes": [],
        "project_updates": [],
        "decisions": [],
        "actions": [
            {
                "owner": "John",
                "item": "与Billy一起完成。",
                "segment_ids": [records[0]["segment_id"]],
            }
        ],
    }
    review = _publication_review(minutes)
    errors = ["action:1:item:quality_incomplete_action"]

    targets = smart_minutes._field_repair_targets(
        review,
        errors,
        transcript_records=records,
    )
    repaired, applied = smart_minutes._apply_targeted_field_repairs(
        review,
        {
            "repairs": [
                {
                    "field": "action:1:item",
                    "text": "开始与Billy共同编写文档。",
                }
            ]
        },
        targets=targets,
    )

    assert repaired is not None
    assert applied == ["action:1:item"]
    assert repaired["minutes"]["actions"][0]["owner"] == "John"
    assert repaired["minutes"]["actions"][0]["segment_ids"] == [
        records[0]["segment_id"]
    ]
    assert smart_minutes._publication_language_quality_errors(
        repaired["minutes"]
    ) == []


def test_targeted_field_repair_repairs_status_inversion():
    records = canonical_transcript_records(
        [
            {
                "id": "pending-retest",
                "start": 0.0,
                "end": 12.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "I need your help to retest address generation through BPS.",
            }
        ]
    )
    minutes = {
        "themes": [
            {
                "title": "地址重试",
                "current_state": "地址生成已通过BPS接口重新测试。",
                "outcome": "讨论形成方向，尚未决定。",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }
    review = _publication_review(minutes, candidate_dispositions=[])
    errors = ["theme:1:current_state:status_unsupported_completion:retest"]

    targets = smart_minutes._field_repair_targets(
        review,
        errors,
        transcript_records=records,
    )
    repaired, applied = smart_minutes._apply_targeted_field_repairs(
        review,
        {
            "repairs": [
                {
                    "field": "theme:1:current_state",
                    "text": "地址生成需要通过BPS接口重新测试。",
                }
            ]
        },
        targets=targets,
    )

    assert [target["field"] for target in targets] == ["theme:1:current_state"]
    assert applied == ["theme:1:current_state"]
    assert repaired is not None
    assert repaired["minutes"]["themes"][0]["current_state"] == (
        "地址生成需要通过BPS接口重新测试。"
    )


def test_repair_source_minutes_text_fields_preserves_reviewed_structure(monkeypatch):
    records = canonical_transcript_records(
        [
            {
                "id": "pending-retest",
                "start": 0.0,
                "end": 8.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "I need your help to retest that particular issue.",
            },
            {
                "id": "retest-context",
                "start": 8.0,
                "end": 16.0,
                "speaker": "Speaker 4",
                "name": None,
                "name_confidence": 0.0,
                "text": "Use the BPS interface to generate the address once again.",
            },
        ]
    )
    source = {
        "themes": [
            {
                "title": "地址重试",
                "current_state": "地址生成已通过BPS接口重新测试。",
                "outcome": "讨论形成方向，尚未决定。",
                "evidence_segment_ids": [records[1]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    def fake_request(*, messages, config, max_tokens=16000):
        assert max_tokens == 4_000
        assert "text fields" in messages[0]["content"]
        return {
            "repairs": [
                {
                    "field": "theme:1:current_state",
                    "text": "地址生成需要通过BPS接口重新测试。",
                }
            ]
        }, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(smart_minutes, "request_deepseek_json", fake_request)

    repaired, status = repair_source_minutes_text_fields(
        source,
        transcript_records=records,
        required_project_participants=[],
        config=DeepSeekConfig(model="test-model"),
    )

    assert repaired is not None
    assert status["status"] == "repaired"
    assert status["field_repair"][0]["applied_fields"] == [
        "theme:1:current_state"
    ]
    assert repaired["themes"][0]["current_state"] == (
        "地址生成需要通过BPS接口重新测试。"
    )
    assert repaired["themes"][0]["evidence_segment_ids"] == [
        records[1]["segment_id"]
    ]


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


def test_sanitize_reviewed_minutes_rejects_non_action_repair_that_would_desync_languages():
    segments = _segments()
    records = canonical_transcript_records(segments)
    minutes = _minutes_payload([record["segment_id"] for record in records])
    minutes["decisions"] = [
        {
            "text_zh": "BPS 已获批准。",
            "text_en": "BPS was approved.",
            "segment_ids": [records[0]["segment_id"]],
        }
    ]
    source = {"format": SMART_MINUTES_FORMAT, "minutes": minutes}
    chinese_source = smart_minutes._bilingual_to_source(minutes, language="zh")
    final_review = _publication_review(chinese_source)
    audit = {
        "reviews": [{"findings": []}, final_review],
        "action_scout": {"actions": _action_scout(records)["actions"]},
    }

    result, errors = sanitize_reviewed_smart_minutes(
        source,
        segments=segments,
        source_audit=audit,
    )

    assert result is None
    assert errors == ["reviewed_payload_non_action_repair_requires_regeneration"]


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


def test_validator_rejects_project_update_built_from_distant_statuses():
    segments = _segments() + [
        {
            "id": "late-status",
            "start": 500.0,
            "end": 510.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_confidence": 0.95,
            "text": "I will send the meeting notes after the review.",
        }
    ]
    records = canonical_transcript_records(segments)
    payload = _minutes_payload([record["segment_id"] for record in records[:2]])
    payload["project_updates"][0].update(
        {
            "update_zh": "复核发票并发送会议纪要。",
            "update_en": "Review the invoice and send the meeting notes.",
            "segment_ids": [records[0]["segment_id"], records[2]["segment_id"]],
        }
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert any(
        error.startswith("project_update:1:evidence_span_too_wide:")
        for error in errors
    )


def test_validator_rejects_cjk_adjacent_number_missing_from_project_update_evidence():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["project_updates"][0].update(
        {
            "update_zh": "当前处理90%的流量。",
            "update_en": "Currently handles 90% of traffic.",
            "segment_ids": [records[0]["segment_id"]],
        }
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "project_update:1:number_ungrounded:90" in errors


def test_validator_rejects_cjk_adjacent_number_missing_from_theme_evidence():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["themes"][0]["current_state_zh"] = "当前处理90%的流量。"
    payload["themes"][0]["current_state_en"] = "Currently handles 90% of traffic."

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "theme:1:current_state:number_ungrounded:90" in errors


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


def test_validator_allows_neutral_theme_without_named_evidence_points():
    records = canonical_transcript_records(
        [
            {
                "id": "unresolved-topic",
                "start": 0.0,
                "end": 20.0,
                "speaker": "Speaker 1",
                "name": None,
                "name_confidence": 0.0,
                "text": "An unresolved participant discussed the architecture tradeoff.",
            }
        ]
    )
    payload = {
        "themes": [
            {
                "title_zh": "架构取舍讨论",
                "title_en": "Architecture Trade-off Discussion",
                "current_state_zh": "讨论了架构取舍。",
                "current_state_en": "The architecture trade-off was discussed.",
                "outcome_zh": "尚未形成最终决定。",
                "outcome_en": "No final decision was made.",
                "evidence_segment_ids": [records[0]["segment_id"]],
                "key_points": [],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=[],
    )

    assert errors == []
    assert cleaned is not None
    assert cleaned["themes"][0]["key_points"] == []


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


def test_validator_rejects_unconfirmed_generic_theme_commitment():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["themes"][0]["outcome_zh"] = "团队同意将构建 API 网关并实施数据库复制。"
    payload["themes"][0]["outcome_en"] = (
        "The team agreed to build the API gateway and implement database replication."
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "theme:1:outcome_unconfirmed_commitment" in errors


def test_validator_rejects_ungrounded_theme_consensus():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["themes"][0]["outcome_zh"] = "讨论形成共识：AI将替代开发者。"
    payload["themes"][0]["outcome_en"] = "Consensus: AI will replace developers."

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "theme:1:outcome_ungrounded_consensus" in errors


def test_validator_rejects_slash_joined_theme_title():
    records = canonical_transcript_records(_segments())
    payload = _minutes_payload([record["segment_id"] for record in records])
    payload["themes"][0]["title_zh"] = (
        "API网关、高可用性与数据库扩展 / 团队效率和安全风险"
    )

    cleaned, errors = validate_smart_minutes(
        payload,
        transcript_records=records,
        required_project_participants=["Billy"],
    )

    assert cleaned is None
    assert "theme:1:title_zh_combined_topic" in errors


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


def test_validator_grounds_entities_from_short_asr_context_without_weakening_owner_evidence():
    segments = [
        {
            "id": "commitment",
            "start": 0.0,
            "end": 1.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_confidence": 0.95,
            "text": "I'll continue to work on it.",
        },
        {
            "id": "context",
            "start": 1.0,
            "end": 3.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.95,
            "text": "The authentic and GitLab integration.",
        },
        {
            "id": "distant",
            "start": 30.0,
            "end": 32.0,
            "speaker": "Speaker 2",
            "name": "Xin",
            "name_confidence": 0.95,
            "text": "The unrelated Proxmox migration.",
        },
    ]
    records = canonical_transcript_records(segments)
    errors = smart_minutes._claim_fidelity_errors(
        "继续完成 GitLab 与 Authentik 集成，并推进 Proxmox 迁移。",
        [records[0]["segment_id"]],
        records={record["segment_id"]: record for record in records},
        canonical_names={"Billy", "Xin"},
        field="action:1",
    )

    assert "action:1:named_entity_ungrounded:Proxmox" in errors
    assert not any("GitLab" in error or "Authentik" in error for error in errors)


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


def test_final_review_forces_structural_rewrite_after_theme_count_mismatch(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    invalid_final_review = _publication_review(copy.deepcopy(valid))
    invalid_final_review["minutes"]["themes"].append(
        copy.deepcopy(valid["themes"][0])
    )
    repaired_final_review = _publication_review(copy.deepcopy(valid))
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        invalid_final_review,
        repaired_final_review,
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    calls: list[list[dict[str, str]]] = []

    def fake_request(*, messages, config, max_tokens=16000):
        calls.append(messages)
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

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
    assert len(calls) == 8
    assert "invalid_review" in calls[6][1]["content"]
    final_review_status = result.audit["reviews"][1]["status"]
    assert final_review_status["repair_attempted"] is True
    assert "field_repair" not in final_review_status


def test_structural_review_rewrite_includes_outline_window_failures():
    assert smart_minutes._requires_full_final_review_rewrite(
        ["theme:1:outside_outline_range:seg_3"]
    )
    assert smart_minutes._requires_full_final_review_rewrite(
        ["theme:2:outline_anchor_missing"]
    )
    assert smart_minutes._requires_full_final_review_rewrite(
        ["theme:3:span_too_short:40.0<180.0"]
    )
    assert not smart_minutes._requires_full_final_review_rewrite(
        ["theme:1:point:1:speaker_evidence_mismatch"]
    )


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
    assert result.audit["synthesis_initial_validation_errors"] == [
        "theme:1:keys_invalid"
    ]
    assert result.audit["final_review_validation_errors"] == []


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


def test_final_review_retries_with_smaller_prior_findings_after_json_truncation(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    first_findings = [
        {
            "severity": "low",
            "category": "wording",
            "description": "Low-priority wording cleanup.",
        },
        {
            "severity": "material",
            "category": "action_recall",
            "description": "A material action might be missing.",
        },
        {
            "severity": "high",
            "category": "speaker_attribution",
            "description": "A speaker attribution requires correction.",
        },
        {
            "severity": "medium",
            "category": "decision_precision",
            "description": "A decision needs tighter wording.",
        },
        {
            "severity": "blocker",
            "category": "unsupported_action",
            "description": "An action lacks direct evidence.",
        },
        {
            "severity": "low",
            "category": "coverage",
            "description": "A minor topic could use more detail.",
        },
    ]
    final_review = _publication_review(
        valid,
        prior_finding_dispositions=[
            {
                "finding_index": 1,
                "disposition": "addressed",
                "reason": "The action evidence was checked and retained.",
            },
            {
                "finding_index": 2,
                "disposition": "addressed",
                "reason": "The speaker attribution was checked and retained.",
            },
        ],
    )
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": first_findings, "minutes": valid},
        None,
        final_review,
        _source_payload([record["segment_id"] for record in records], english=True),
    ]
    calls: list[tuple[list[dict[str, str]], int]] = []

    def fake_request(*, messages, config, max_tokens=16000):
        calls.append((messages, max_tokens))
        response = responses.pop(0)
        if response is None:
            return None, {
                "status": "invalid_model_json",
                "starts_with_object": True,
                "ends_with_object": False,
            }
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
    )

    assert result is not None, status
    initial_final_payload = json.loads(calls[5][0][1]["content"])
    retried_final_payload = json.loads(calls[6][0][1]["content"])
    assert len(initial_final_payload["prior_findings"]) == 4
    assert len(retried_final_payload["prior_findings"]) == 2
    assert calls[5][1] == smart_minutes.FINAL_REVIEW_MAX_TOKENS
    assert calls[6][1] == smart_minutes.FINAL_REVIEW_MAX_TOKENS
    assert result.audit["reviews"][0]["finding_budget"]["retained_source_indexes"] == [
        5,
        3,
        2,
        4,
    ]
    final_audit = result.audit["reviews"][1]
    assert final_audit["prior_finding_budget"]["truncation_retry"] == {
        "initial_prior_finding_count": 4,
        "retry_prior_finding_count": 2,
    }
    assert final_audit["prior_finding_budget"]["retained_count"] == 2
    assert final_audit["prior_finding_budget"]["discarded_count"] == 4
    assert (
        final_audit["prior_finding_budget"]["valid_count"]
        == final_audit["prior_finding_budget"]["retained_count"]
        + final_audit["prior_finding_budget"]["discarded_count"]
    )
    assert final_audit["status"]["truncation_retry"][
        "retry_prior_finding_count"
    ] == 2
    assert smart_minutes.FINAL_REVIEW_MAX_TOKENS == 12_000


def test_final_review_blocks_publication_when_truncation_retry_is_exhausted(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    first_findings = [
        {
            "severity": "high",
            "category": "speaker_attribution",
            "description": "A speaker attribution requires correction.",
        },
        {
            "severity": "material",
            "category": "action_recall",
            "description": "A material action might be missing.",
        },
    ]
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": first_findings, "minutes": valid},
        None,
        None,
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        response = responses.pop(0)
        if response is None:
            return None, {
                "status": "invalid_model_json",
                "starts_with_object": True,
                "ends_with_object": False,
            }
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
    )

    assert result is None
    assert status["status"] == "review_truncation_retry_exhausted"
    assert status["review"]["truncation_retry"]["initial_prior_finding_count"] == 2
    assert status["review"]["truncation_retry"]["retry_prior_finding_count"] == 1


def test_final_review_retry_cache_resumes_from_reduced_context(monkeypatch):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    first_findings = [
        {
            "severity": "blocker",
            "category": "unsupported_action",
            "description": "An action lacks direct evidence.",
        },
        {
            "severity": "high",
            "category": "speaker_attribution",
            "description": "A speaker attribution requires correction.",
        },
        {
            "severity": "material",
            "category": "action_recall",
            "description": "A material action might be missing.",
        },
        {
            "severity": "medium",
            "category": "decision_precision",
            "description": "A decision needs tighter wording.",
        },
    ]
    final_review = _publication_review(
        valid,
        prior_finding_dispositions=[
            {
                "finding_index": 1,
                "disposition": "addressed",
                "reason": "The action evidence was checked and retained.",
            },
            {
                "finding_index": 2,
                "disposition": "addressed",
                "reason": "The speaker attribution was checked and retained.",
            },
        ],
    )
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": first_findings, "minutes": valid},
        None,
        final_review,
        None,
    ]
    checkpoints: list[dict] = []

    def first_request(*, messages, config, max_tokens=16000):
        response = responses.pop(0)
        if response is None:
            return None, {
                "status": "invalid_model_json",
                "starts_with_object": True,
                "ends_with_object": False,
            }
        return response, {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", first_request)

    first, first_status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert first is None
    assert first_status["status"] == "translation_failed"
    checkpoint = checkpoints[-1]
    cached_final = checkpoint["reviews"][1]
    assert "truncation_retry" in cached_final["status"]

    resumed_calls: list[list[dict[str, str]]] = []

    def resumed_request(*, messages, config, max_tokens=16000):
        resumed_calls.append(messages)
        return _source_payload(
            [record["segment_id"] for record in records],
            english=True,
        ), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr(
        "meeting_minutes.smart_minutes.request_deepseek_json",
        resumed_request,
    )
    resumed, resumed_status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=2,
        checkpoint=checkpoint,
    )

    assert resumed is not None, resumed_status
    assert len(resumed_calls) == 1
    assert "source_minutes_zh" in resumed_calls[0][1]["content"]
    final_audit = resumed.audit["reviews"][1]
    assert final_audit["status"]["status"] == "cached"
    assert final_audit["prior_finding_budget"]["retained_count"] == 2
    assert final_audit["prior_finding_budget"]["discarded_count"] == 2


def test_final_review_uses_baseline_when_first_review_minutes_are_invalid(
    monkeypatch,
):
    records = canonical_transcript_records(_segments())
    valid = _source_payload([record["segment_id"] for record in records])
    invalid_review = copy.deepcopy(valid)
    invalid_review["themes"][0]["unexpected"] = "invalid"
    final_review = _publication_review(
        valid,
        prior_finding_dispositions=[
            {
                "finding_index": 1,
                "disposition": "addressed",
                "reason": "The final minutes use the valid baseline wording.",
            }
        ],
    )
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {
            "findings": [
                {
                    "severity": "medium",
                    "category": "wording",
                    "description": "The draft needs a concise wording check.",
                }
            ],
            "minutes": invalid_review,
        },
        final_review,
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
        review_passes=2,
    )

    assert result is not None, status
    final_review_payload = json.loads(calls[5][1]["content"])
    assert final_review_payload["draft_minutes"] == valid
    assert "unexpected" not in final_review_payload["draft_minutes"]["themes"][0]


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
    assert result.audit["reviews"][1]["status"]["deterministic_repair_attempted"] is True


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


def test_invalid_translation_payload_is_not_reused_from_checkpoint(monkeypatch):
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    valid = _source_payload(segment_ids)
    invalid_translation = _source_payload(segment_ids, english=True)
    invalid_translation["actions"][0]["owner"] = "Xin"
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        invalid_translation,
        invalid_translation,
    ]
    checkpoints: list[dict] = []

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
        checkpoint_callback=lambda payload: checkpoints.append(copy.deepcopy(payload)),
    )

    assert result is None
    assert status["status"] == "translation_alignment_invalid"
    assert "translation_action:1:owner_mismatch" in status["errors"]
    assert checkpoints[-1]["translation"] is None
    assert (
        "translation_action:1:owner_mismatch"
        in checkpoints[-1]["last_rejected_translation"]["validation_errors"]
    )


def test_translation_alignment_repairs_invalid_target_once(monkeypatch):
    records = canonical_transcript_records(_segments())
    segment_ids = [record["segment_id"] for record in records]
    valid = _source_payload(segment_ids)
    invalid_translation = _source_payload(segment_ids, english=True)
    invalid_translation["actions"][0]["owner"] = "Xin"
    repaired_translation = _source_payload(segment_ids, english=True)
    responses = [
        _action_scout(records),
        _implicit_follow_up_scout(),
        _theme_outline(records),
        valid,
        {"findings": [], "minutes": valid},
        invalid_translation,
        repaired_translation,
    ]

    def fake_request(*, messages, config, max_tokens=16000):
        return responses.pop(0), {"status": "ok", "requested_model": config.model}

    monkeypatch.setattr("meeting_minutes.smart_minutes.request_deepseek_json", fake_request)

    result, status = generate_smart_minutes(
        segments=_segments(),
        config=DeepSeekConfig(model="test-model"),
        review_passes=1,
    )

    assert result is not None
    assert status["status"] == "reviewed_draft"
    assert result.status["translation"]["status"] == "repair"


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


def test_renderer_shows_a_time_range_for_multi_segment_key_points():
    records = canonical_transcript_records(
        [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.94,
                "text": "The first part of the update.",
            },
            {
                "start": 20.0,
                "end": 25.0,
                "speaker": "Speaker 1",
                "name": "Billy",
                "name_confidence": 0.94,
                "text": "The second part of the update.",
            },
        ]
    )
    first, second = [record["segment_id"] for record in records]
    payload = {
        "themes": [
            {
                "title_zh": "状态更新",
                "title_en": "Status Update",
                "current_state_zh": "讨论了状态。",
                "current_state_en": "Status was discussed.",
                "outcome_zh": "尚未形成最终决定。",
                "outcome_en": "No final decision was made.",
                "evidence_segment_ids": [first, second],
                "key_points": [
                    {
                        "speaker": "Billy",
                        "text_zh": "Billy 说明了两个部分。",
                        "text_en": "Billy described both parts.",
                        "segment_ids": [first, second],
                    }
                ],
            }
        ],
        "project_updates": [],
        "decisions": [],
        "actions": [],
    }

    chinese, english = render_smart_minutes(payload, transcript_records=records)

    assert "关键观点（00:00:00-00:00:25，Billy）" in chinese
    assert "Key point (00:00:00-00:00:25, Billy)" in english
