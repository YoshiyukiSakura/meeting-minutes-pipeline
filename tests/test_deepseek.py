import hashlib
import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from meeting_minutes.cli import build_parser, run_pipeline, summarize_existing
from meeting_minutes.deepseek import (
    DeepSeekConfig,
    DeepSeekInputTooLarge,
    DeepSeekRedirectBlocked,
    MAX_API_RESPONSE_BYTES,
    _NoRedirectHandler,
    _chunk_evidence_input,
    _validated_base_url,
    build_deepseek_messages,
    generate_deepseek_review,
    prepare_deepseek_evidence_input,
    resolve_deepseek_api_key,
    validate_deepseek_review,
    write_deepseek_review,
)
from meeting_minutes.jsonio import write_json


def _segments():
    return [
        {
            "start": 10.0,
            "end": 14.0,
            "speaker": "Avery",
            "name": "Avery",
            "text": "We agreed to use cloud hosting for the first release.",
        },
        {
            "start": 18.0,
            "end": 22.0,
            "speaker": "Riley",
            "name": "Riley",
            "text": "The architecture proposal still needs a design review.",
        },
    ]


def _prepared():
    return prepare_deepseek_evidence_input(
        segments=_segments(),
        keyframes=[
            {
                "time": 12.0,
                "path": "/private/meeting/Avery-frame.jpg",
                "reasons": ["opening_frame", "scene_change:0.42"],
            }
        ],
    )


def _payload(prepared):
    first, second = prepared.segments
    return {
        "overview": [
            {
                "text": "讨论了首个版本的云端托管方向。",
                "evidence": [{"segment_id": first["segment_id"]}],
            }
        ],
        "discussion": [
            {
                "text": "架构方案仍处于设计评审阶段。",
                "evidence": [{"segment_id": second["segment_id"]}],
            }
        ],
        "decisions": [],
    }


def test_outbound_prompt_omits_identity_labels_and_frame_paths():
    prepared = _prepared()
    messages = build_deepseek_messages(prepared)
    system = messages[0]["content"]
    prompt = messages[1]["content"]

    assert "Avery" not in prompt
    assert "Riley" not in prompt
    assert "Speaker" not in prompt
    assert "/private/meeting" not in prompt
    assert "Avery-frame.jpg" not in prompt
    assert '"start"' not in prompt
    assert '"end"' not in prompt
    assert '"time"' not in prompt
    assert '"opening_frame"' in prompt
    assert '"scene_change"' in prompt
    assert "Rewrite task-shaped evidence" in system
    assert "讨论了测试环境中的问题和修复路径" in system


def test_outbound_fingerprint_covers_only_the_exact_transmitted_transcript_fields():
    prepared = _prepared()
    outbound_records = [
        {"segment_id": record["segment_id"], "text": record["text"]}
        for record in prepared.segments
    ]
    encoded = json.dumps(outbound_records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    assert prepared.transcript_sha256 == hashlib.sha256(encoded).hexdigest()
    assert prepared.transcript_characters == len(encoded.decode("utf-8"))


def test_discussion_coverage_anchors_are_required_and_validated():
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "Avery", "text": "The first discussion covers the current service status."},
        {"start": 1805.0, "end": 1810.0, "speaker": "Riley", "text": "The later discussion covers the gateway design tradeoff."},
    ]
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])
    assert len(prepared.coverage_anchor_segment_ids) == 2
    prompt = build_deepseek_messages(prepared)[1]["content"]
    for anchor_id in prepared.coverage_anchor_segment_ids:
        assert anchor_id in prompt

    complete = {
        "overview": [],
        "discussion": [
            {"text": "讨论了服务的当前状态。", "evidence": [{"segment_id": prepared.coverage_anchor_segment_ids[0]}]},
            {"text": "讨论了网关设计的方案比较。", "evidence": [{"segment_id": prepared.coverage_anchor_segment_ids[1]}]},
        ],
        "decisions": [],
    }
    review = validate_deepseek_review(
        complete,
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )
    assert review["validation"]["warnings"] == []

    incomplete = {**complete, "discussion": complete["discussion"][:1]}
    review = validate_deepseek_review(
        incomplete,
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )
    assert "missing_discussion_coverage:1" in review["validation"]["warnings"]


def test_evidence_input_chunks_preserve_order_and_coverage_anchors():
    segments = [
        {"start": float(index * 365), "end": float(index * 365 + 1), "speaker": "Avery", "text": f"segment {index} " + "details " * 40}
        for index in range(4)
    ]
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])

    chunks = _chunk_evidence_input(prepared, max_characters=180)

    assert len(chunks) == len(segments)
    assert [record["segment_id"] for chunk in chunks for record in chunk.segments] == [
        record["segment_id"] for record in prepared.segments
    ]
    assert {anchor for chunk in chunks for anchor in chunk.coverage_anchor_segment_ids} == set(
        prepared.coverage_anchor_segment_ids
    )


def test_validator_derives_exact_source_quote_and_time_locally():
    prepared = _prepared()
    review = validate_deepseek_review(
        _payload(prepared),
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
        response_model="deepseek-v4-pro",
    )

    overview = review["sections"]["overview"][0]
    assert overview["start"] == 10.0
    assert overview["end"] == 14.0
    assert overview["evidence"][0]["quote"] == "We agreed to use cloud hosting for the first release."
    assert review["validation"]["accepted"] == 2
    assert review["validation"]["rejected"] == []
    assert review["input"]["speaker_labels_transmitted"] is False
    assert review["input"]["transcript_fields"] == ["segment_id", "text"]


@pytest.mark.parametrize(
    ("item", "expected_error"),
    [
        (
            {"text": "部署方向已经达成一致。", "evidence": [{"segment_id": "missing"}]},
            "unknown_segment_id",
        ),
        (
            {
                "text": "部署方向已经达成一致。",
                "evidence": [{"segment_id": "first"}, {"segment_id": "first"}],
            },
            "duplicate_evidence",
        ),
        (
            {"text": "Avery 将负责后续工作。", "evidence": [{"segment_id": "first"}]},
            "participant_name_not_allowed",
        ),
        (
            {"text": "需要一个两小时的维护窗口。", "evidence": [{"segment_id": "first"}]},
            "duration_not_allowed",
        ),
        (
            {"text": "后续行动项将由团队跟进。", "evidence": [{"segment_id": "first"}]},
            "action_or_owner_not_allowed",
        ),
        (
            {"text": "The deployment direction is agreed.", "evidence": [{"segment_id": "first"}]},
            "summary_language_not_chinese",
        ),
        (
            {"text": "The migration discussion remains open 中文内容", "evidence": [{"segment_id": "first"}]},
            "summary_language_not_chinese",
        ),
        (
            {"text": "部署将在本周完成。", "evidence": [{"segment_id": "first"}]},
            "model_timestamp_not_allowed",
        ),
    ],
)
def test_validator_rejects_unsupported_or_unsafe_claims(item, expected_error):
    prepared = _prepared()
    first = prepared.segments[0]
    for evidence in item["evidence"]:
        if evidence["segment_id"] == "first":
            evidence["segment_id"] = first["segment_id"]
    review = validate_deepseek_review(
        {"overview": [item], "discussion": [], "decisions": []},
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
    )

    assert review["sections"]["overview"] == []
    assert expected_error in review["validation"]["rejected"][0]["errors"]


def test_validator_rejects_unknown_top_level_action_output():
    prepared = _prepared()
    payload = _payload(prepared)
    payload["action_items"] = [{"owner": "Avery"}]
    review = validate_deepseek_review(
        payload,
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
    )

    assert "unexpected_top_level:action_items" in review["validation"]["rejected"][0]["errors"]


def test_validator_rejects_a_response_missing_required_top_level_sections():
    prepared = _prepared()
    review = validate_deepseek_review(
        {"overview": []},
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
    )

    rejected_errors = [error for item in review["validation"]["rejected"] for error in item["errors"]]
    assert "missing_top_level:discussion" in rejected_errors
    assert "missing_top_level:decisions" in rejected_errors


@pytest.mark.parametrize(
    ("item", "expected_error"),
    [
        (
            {
                "text": "首个版本的部署方向已经达成一致。",
                "evidence": [{"segment_id": "first", "quote": "untrusted model text"}],
            },
            "unexpected_evidence_field:quote",
        ),
        (
            {
                "text": "首个版本的部署方向已经达成一致。",
                "owner": "Avery",
                "evidence": [{"segment_id": "first"}],
            },
            "forbidden_field:owner",
        ),
        (
            {
                "text": "首个版本的部署方向已经达成一致。",
                "person": "Avery",
                "evidence": [{"segment_id": "first"}],
            },
            "forbidden_field:person",
        ),
    ],
)
def test_validator_rejects_extra_model_fields(item, expected_error):
    prepared = _prepared()
    item["evidence"][0]["segment_id"] = prepared.segments[0]["segment_id"]
    review = validate_deepseek_review(
        {"overview": [item], "discussion": [], "decisions": []},
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
    )

    assert review["sections"]["overview"] == []
    assert expected_error in review["validation"]["rejected"][0]["errors"]


@pytest.mark.parametrize(
    "text",
    [
        "该问题需要在测试环境复现并修复。",
        "基础设施计划迁移到新集群。",
        "后续继续跟踪系统指标。",
        "团队打算迁移服务。",
        "团队准备修复该问题。",
        "团队要部署补丁。",
        "The team needs to reproduce and fix the issue.",
        "The next step is to migrate the infrastructure.",
        "We're going to migrate the cluster.",
        "The team is going to deploy the fix.",
        "They are going to migrate the cluster.",
        "The team has to migrate.",
        "The team is supposed to fix it.",
        "The fix is to be deployed.",
        "The team intends to investigate the issue.",
        "The team shall deploy the fix.",
        "团队会尽快修复该问题。",
        "团队会逐步迁移服务。",
        "团队要马上处理该问题。",
    ],
)
def test_validator_rejects_implicit_future_tasks(text):
    prepared = _prepared()
    review = validate_deepseek_review(
        {"overview": [{"text": text, "evidence": [{"segment_id": prepared.segments[0]["segment_id"]}]}], "discussion": [], "decisions": []},
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
        output_language="en" if text.isascii() else "zh-CN",
    )

    assert review["sections"]["overview"] == []
    assert "action_or_owner_not_allowed" in review["validation"]["rejected"][0]["errors"]


def test_name_filter_uses_word_boundaries_and_explicit_aliases():
    prepared = _prepared()
    first = prepared.segments[0]
    review = validate_deepseek_review(
        {
            "overview": [
                {"text": "The existing deployment option remains available.", "evidence": [{"segment_id": first["segment_id"]}]},
                {"text": "The Avery alias must not be repeated.", "evidence": [{"segment_id": first["segment_id"]}]},
                {"text": "The Casey alias must not be repeated.", "evidence": [{"segment_id": first["segment_id"]}]},
            ],
            "discussion": [],
            "decisions": [],
        },
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
        output_language="en",
        redacted_names=("Casey",),
    )

    assert review["sections"]["overview"][0]["text"] == "The existing deployment option remains available."
    rejected_errors = [error for item in review["validation"]["rejected"] for error in item["errors"]]
    assert rejected_errors.count("participant_name_not_allowed") == 2


@pytest.mark.parametrize(
    "name_with_invisible_character",
    ["Ave\u200bry", "Ave\u034fry", "Ave\u180bry", "Ave\ufe0fry", "Ave\U000e0100ry"],
)
def test_name_filter_normalizes_invisible_characters(name_with_invisible_character):
    prepared = _prepared()
    first = prepared.segments[0]
    review = validate_deepseek_review(
        {
            "overview": [
                {"text": f"{name_with_invisible_character} was mentioned in the discussion.", "evidence": [{"segment_id": first["segment_id"]}]}
            ],
            "discussion": [],
            "decisions": [],
        },
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
        output_language="en",
    )

    assert "participant_name_not_allowed" in review["validation"]["rejected"][0]["errors"]


@pytest.mark.parametrize(
    ("text", "output_language"),
    [
        ("团队已确认采用云端托管方案。", "zh-CN"),
        ("The deployment direction is agreed.", "en"),
    ],
)
def test_validator_rejects_decision_assertions_outside_the_decisions_section(text, output_language):
    prepared = _prepared()
    review = validate_deepseek_review(
        {
            "overview": [{"text": text, "evidence": [{"segment_id": prepared.segments[0]["segment_id"]}]}],
            "discussion": [],
            "decisions": [],
        },
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
        output_language=output_language,
    )

    assert review["sections"]["overview"] == []
    assert "decision_wording_outside_decisions" in review["validation"]["rejected"][0]["errors"]


def test_validator_allows_an_english_review_only_when_requested():
    prepared = _prepared()
    first = prepared.segments[0]
    review = validate_deepseek_review(
        {
            "overview": [
                {
                    "text": "The discussion covered the deployment direction.",
                    "evidence": [{"segment_id": first["segment_id"]}],
                }
            ],
            "discussion": [],
            "decisions": [],
        },
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
        output_language="en",
    )

    assert review["validation"]["accepted"] == 1
    assert review["output_language"] == "en"


def test_decisions_require_an_already_confirmed_decision():
    prepared = _prepared()
    first = prepared.segments[0]
    action_like = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [{"text": "开始部署云端服务。", "evidence": [{"segment_id": first["segment_id"]}]}],
        },
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
    )
    confirmed = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [{"text": "确认采用云端托管作为首个版本的部署方向。", "evidence": [{"segment_id": first["segment_id"]}]}],
        },
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
    )

    assert "decision_not_confirmed" in action_like["validation"]["rejected"][0]["errors"]
    assert confirmed["validation"]["accepted"] == 1


def test_decision_requires_explicit_decision_language_in_its_evidence():
    prepared = _prepared()
    second = prepared.segments[1]
    review = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [{"text": "确认采用云端托管作为首个版本的部署方向。", "evidence": [{"segment_id": second["segment_id"]}]}],
        },
        evidence_input=prepared,
        source_segments=_segments(),
        requested_model="deepseek-v4-pro",
    )

    assert "decision_evidence_not_explicit" in review["validation"]["rejected"][0]["errors"]


def test_decision_does_not_treat_a_confirmed_problem_fact_as_a_meeting_decision():
    segments = [{"start": 1.0, "end": 3.0, "speaker": "Speaker 1", "text": "该问题已被确认存在于测试环境。"}]
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])
    review = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [
                {"text": "确认采用云端托管作为首个版本的部署方向。", "evidence": [{"segment_id": prepared.segments[0]["segment_id"]}]}
            ],
        },
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )

    assert "decision_evidence_not_explicit" in review["validation"]["rejected"][0]["errors"]


@pytest.mark.parametrize(
    "source_text",
    [
        "如果我们同意，就采用云端托管。",
        "我们同意采用云端托管了吗？",
        "Unless we agreed, the cloud hosting option remains open.",
        "Have we agreed on the cloud option? Nobody objected so far.",
    ],
)
def test_decision_rejects_conditional_or_question_evidence(source_text):
    segments = [{"start": 1.0, "end": 3.0, "speaker": "Speaker 1", "text": source_text}]
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])
    review = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [
                {"text": "确认采用云端托管作为首个版本的部署方向。", "evidence": [{"segment_id": prepared.segments[0]["segment_id"]}]}
            ],
        },
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )

    assert "decision_evidence_not_explicit" in review["validation"]["rejected"][0]["errors"]


def test_decision_cannot_combine_an_unrelated_agreement_and_topic_segment():
    segments = [
        {"start": 1.0, "end": 3.0, "speaker": "Speaker 1", "text": "我们一致同意本次会议先暂停。"},
        {"start": 4.0, "end": 6.0, "speaker": "Speaker 2", "text": "云端托管方案仍在讨论。"},
    ]
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])
    review = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [
                {
                    "text": "确认采用云端托管作为首个版本的部署方向。",
                    "evidence": [{"segment_id": record["segment_id"]} for record in prepared.segments],
                }
            ],
        },
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )

    assert "decision_requires_single_evidence" in review["validation"]["rejected"][0]["errors"]


@pytest.mark.parametrize(
    "source_text",
    [
        "We will use cloud hosting for the first release.",
        "We can use cloud hosting for the first release.",
        "The idea is to use cloud hosting for the first release.",
        "Only change the hosting option after design review.",
    ],
)
def test_decision_rejects_plans_and_proposals_even_with_a_confirmed_title(source_text):
    segments = [{"start": 1.0, "end": 3.0, "speaker": "Speaker 1", "text": source_text}]
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])
    review = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [
                {"text": "确认采用云端托管作为首个版本的部署方向。", "evidence": [{"segment_id": prepared.segments[0]["segment_id"]}]}
            ],
        },
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )

    assert review["sections"]["decisions"] == []
    assert "decision_evidence_not_explicit" in review["validation"]["rejected"][0]["errors"]


def test_decision_accepts_explicit_chinese_agreement_in_its_evidence():
    segments = [{"start": 1.0, "end": 3.0, "speaker": "Speaker 1", "text": "我们已确认首个版本采用云端托管。"}]
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])
    review = validate_deepseek_review(
        {
            "overview": [],
            "discussion": [],
            "decisions": [
                {"text": "确认采用云端托管作为首个版本的部署方向。", "evidence": [{"segment_id": prepared.segments[0]["segment_id"]}]}
            ],
        },
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )

    assert review["validation"]["accepted"] == 1


def test_input_limit_fails_without_truncating():
    with pytest.raises(DeepSeekInputTooLarge):
        prepare_deepseek_evidence_input(segments=_segments(), keyframes=[], max_input_chars=1)


def test_env_file_credential_is_read_without_persisting_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=unit-test-only\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    key, source = resolve_deepseek_api_key(DeepSeekConfig(env_file=env_file))

    assert key == "unit-test-only"
    assert source == "env_file"


def test_credential_discovery_does_not_read_a_parent_directory(tmp_path, monkeypatch):
    parent_env = tmp_path / ".env"
    parent_env.write_text("DEEPSEEK_API_KEY=parent-key\n", encoding="utf-8")
    invocation_dir = tmp_path / "child"
    invocation_dir.mkdir()
    monkeypatch.chdir(invocation_dir)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    key, source = resolve_deepseek_api_key(DeepSeekConfig())

    assert key is None
    assert source == "missing"


def test_keychain_credential_is_used_only_when_explicitly_named(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("meeting_minutes.deepseek._keychain_value", lambda service: "keychain-only" if service == "meeting-review" else None)

    key, source = resolve_deepseek_api_key(
        DeepSeekConfig(env_file=tmp_path / "absent.env", keychain_service="meeting-review")
    )

    assert key == "keychain-only"
    assert source == "keychain"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.com",
        "https://api.deepseek.com?redirect=https://bad.example",
        "https://api.deepseek.com/chat/completions",
        "https://user:secret@api.deepseek.com",
    ],
)
def test_base_url_rejects_unsafe_or_ambiguous_endpoints(base_url):
    with pytest.raises(ValueError):
        _validated_base_url(base_url)


def test_redirect_handler_blocks_redirects_before_following_them():
    request = urllib.request.Request("https://api.deepseek.com/chat/completions")

    with pytest.raises(DeepSeekRedirectBlocked) as exc:
        _NoRedirectHandler().redirect_request(request, None, 302, "Found", {}, "https://bad.example/collect")

    assert exc.value.code == 302


def test_generate_maps_blocked_redirect_to_a_nonretryable_status(tmp_path, monkeypatch):
    def fake_open(request, *, timeout):
        del timeout
        raise DeepSeekRedirectBlocked(request.full_url, 302, "redirect_blocked", {}, None)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-only")
    monkeypatch.setattr("meeting_minutes.deepseek._open_without_redirect", fake_open)
    review, status = generate_deepseek_review(
        segments=_segments(),
        keyframes=[],
        config=DeepSeekConfig(env_file=tmp_path / "absent.env"),
    )

    assert review is None
    assert status["status"] == "redirect_blocked"
    assert status["http_status"] == 302
    assert status["external_processing"] is True


def test_generate_rejects_an_oversized_api_response(tmp_path, monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            del size
            return b"x" * (MAX_API_RESPONSE_BYTES + 1)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-only")
    monkeypatch.setattr("meeting_minutes.deepseek._open_without_redirect", lambda request, *, timeout: FakeResponse())
    review, status = generate_deepseek_review(
        segments=_segments(),
        keyframes=[],
        config=DeepSeekConfig(env_file=tmp_path / "absent.env"),
    )

    assert review is None
    assert status["status"] == "api_response_too_large"
    assert status["external_processing"] is True


def test_loopback_without_explicit_opt_in_does_not_open_network(tmp_path, monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("network must not open for an unauthenticated loopback endpoint")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("meeting_minutes.deepseek._open_without_redirect", fail_open)
    review, status = generate_deepseek_review(
        segments=_segments(),
        keyframes=[],
        config=DeepSeekConfig(base_url="http://127.0.0.1:8080", env_file=tmp_path / "absent.env"),
    )

    assert review is None
    assert status["status"] == "credential_missing"


def test_generate_uses_structured_response_and_never_records_secret(tmp_path, monkeypatch):
    prepared = _prepared()
    response_body = json.dumps(
        {
            "model": "deepseek-v4-pro",
            "choices": [{"message": {"content": json.dumps(_payload(prepared), ensure_ascii=False)}}],
        }
    ).encode("utf-8")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            del size
            return response_body

    def fake_open(request, *, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-only")
    monkeypatch.setattr("meeting_minutes.deepseek._open_without_redirect", fake_open)
    review, status = generate_deepseek_review(
        segments=_segments(),
        keyframes=[],
        config=DeepSeekConfig(env_file=tmp_path / "absent.env"),
    )

    assert review is not None
    assert status["status"] == "draft_only"
    assert status["credential_source"] == "environment"
    assert captured["authorization"] == "Bearer unit-test-only"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert "unit-test-only" not in json.dumps(status)
    assert "Avery" not in captured["body"]["messages"][1]["content"]
    assert '"start"' not in captured["body"]["messages"][1]["content"]
    assert status["request_security"] == {"redirects": "blocked", "loopback": False}


def test_generate_retries_one_empty_json_response_without_thinking(tmp_path, monkeypatch):
    prepared = _prepared()
    response_bodies = [
        json.dumps({"model": "deepseek-v4-pro", "choices": [{"message": {"content": ""}}]}).encode("utf-8"),
        json.dumps(
            {
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": json.dumps(_payload(prepared), ensure_ascii=False)}}],
            }
        ).encode("utf-8"),
    ]
    request_bodies = []

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            del size
            return self.body

    def fake_open(request, *, timeout):
        request_bodies.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(response_bodies.pop(0))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-only")
    monkeypatch.setattr("meeting_minutes.deepseek._open_without_redirect", fake_open)
    review, status = generate_deepseek_review(
        segments=_segments(),
        keyframes=[],
        config=DeepSeekConfig(env_file=tmp_path / "absent.env"),
    )

    assert review is not None
    assert status["attempts"] == 2
    assert len(request_bodies) == 2
    assert request_bodies[0]["thinking"] == {"type": "disabled"}
    assert request_bodies[1]["thinking"] == {"type": "disabled"}
    assert request_bodies[1]["max_tokens"] > request_bodies[0]["max_tokens"]


def test_generate_without_credential_does_not_open_network(tmp_path, monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("network must not be opened without a credential")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("meeting_minutes.deepseek._open_without_redirect", fail_open)
    review, status = generate_deepseek_review(
        segments=_segments(),
        keyframes=[],
        config=DeepSeekConfig(env_file=tmp_path / "absent.env"),
    )

    assert review is None
    assert status["status"] == "credential_missing"


def test_generate_allows_unauthenticated_loopback_only_after_explicit_opt_in(tmp_path, monkeypatch):
    prepared = _prepared()
    response_body = json.dumps(
        {
            "model": "deepseek-v4-pro",
            "choices": [{"message": {"content": json.dumps(_payload(prepared), ensure_ascii=False)}}],
        }
    ).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            del size
            return response_body

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("meeting_minutes.deepseek._open_without_redirect", lambda request, *, timeout: FakeResponse())
    review, status = generate_deepseek_review(
        segments=_segments(),
        keyframes=[],
        config=DeepSeekConfig(
            base_url="http://127.0.0.1:8080",
            env_file=tmp_path / "absent.env",
            allow_unauthenticated_loopback=True,
        ),
    )

    assert review is not None
    assert status["credential_source"] == "unauthenticated_loopback"
    assert status["request_security"] == {"redirects": "blocked", "loopback": True}


def test_markdown_render_escapes_model_text_and_local_source_quote(tmp_path):
    segments = _segments()
    segments[0]["text"] = "Review [link](https://bad.example) <script>alert(1)</script>."
    prepared = prepare_deepseek_evidence_input(segments=segments, keyframes=[])
    payload = {
        "overview": [
            {
                "text": "请审阅这个脚本片段 <script>alert(1)</script>。",
                "evidence": [{"segment_id": prepared.segments[0]["segment_id"]}],
            }
        ],
        "discussion": [],
        "decisions": [],
    }
    review = validate_deepseek_review(
        payload,
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )
    destination = tmp_path / "minutes.deepseek.review.md"
    write_deepseek_review(destination, review)
    rendered = destination.read_text(encoding="utf-8")

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "\\[link\\]\\(https://bad.example\\)" in rendered
    assert "外部 AI 审校草稿，不可直接对外分享" in rendered


def test_deepseek_cli_keeps_canonical_minutes_and_action_ledger_unchanged(tmp_path, monkeypatch):
    segments = _segments()
    write_json(tmp_path / "transcript.json", segments)
    write_json(tmp_path / "keyframes.json", [])
    write_json(tmp_path / "metadata.json", {"input": "/private/recording.mov", "duration": 22.0, "source_offset": 0.0})

    assert summarize_existing(SimpleNamespace(output_dir=tmp_path, summary_engine="extractive")) == 0
    (tmp_path / "minutes.md").write_text("# 已发布纪要\n", encoding="utf-8")
    minutes_before = (tmp_path / "minutes.md").read_bytes()
    actions_before = (tmp_path / "action_items.json").read_bytes()
    prepared = _prepared()
    review = validate_deepseek_review(
        _payload(prepared),
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )
    monkeypatch.setattr(
        "meeting_minutes.cli.generate_deepseek_review",
        lambda **kwargs: (review, {"engine": "deepseek", "status": "draft_only", "external_processing": True}),
    )
    args = build_parser().parse_args(["summarize", "--output-dir", str(tmp_path), "--summary-engine", "deepseek"])

    assert summarize_existing(args) == 0
    assert (tmp_path / "minutes.md").read_bytes() == minutes_before
    assert (tmp_path / "action_items.json").read_bytes() == actions_before
    assert (tmp_path / "minutes.deepseek.review.json").exists()
    assert "不可直接对外分享" in (tmp_path / "minutes.deepseek.review.md").read_text(encoding="utf-8")


def test_deepseek_cli_fails_loudly_without_creating_a_review(tmp_path, monkeypatch, capsys):
    write_json(tmp_path / "transcript.json", _segments())
    write_json(tmp_path / "keyframes.json", [])
    write_json(tmp_path / "metadata.json", {"input": "/private/recording.mov", "duration": 22.0})
    monkeypatch.setattr(
        "meeting_minutes.cli.generate_deepseek_review",
        lambda **kwargs: (None, {"engine": "deepseek", "status": "credential_missing", "external_processing": False}),
    )
    args = build_parser().parse_args(["summarize", "--output-dir", str(tmp_path), "--summary-engine", "deepseek"])

    assert summarize_existing(args) == 2
    captured = capsys.readouterr()
    assert "not generated (credential_missing)" in captured.err
    assert not (tmp_path / "minutes.deepseek.review.json").exists()
    assert not (tmp_path / "minutes.deepseek.review.md").exists()
    status = json.loads((tmp_path / "summary_status.json").read_text(encoding="utf-8"))
    assert status["review_written"] is False


def test_deepseek_cli_archives_an_old_review_when_a_rerun_fails(tmp_path, monkeypatch):
    write_json(tmp_path / "transcript.json", _segments())
    write_json(tmp_path / "keyframes.json", [])
    write_json(tmp_path / "metadata.json", {"input": "/private/recording.mov", "duration": 22.0})
    old_json = tmp_path / "minutes.deepseek.review.json"
    old_markdown = tmp_path / "minutes.deepseek.review.md"
    old_json.write_text('{"old": true}\n', encoding="utf-8")
    old_markdown.write_text("old review\n", encoding="utf-8")
    monkeypatch.setattr(
        "meeting_minutes.cli.generate_deepseek_review",
        lambda **kwargs: (None, {"engine": "deepseek", "status": "network_error", "external_processing": True}),
    )
    args = build_parser().parse_args(["summarize", "--output-dir", str(tmp_path), "--summary-engine", "deepseek"])

    assert summarize_existing(args) == 2
    assert not old_json.exists()
    assert not old_markdown.exists()
    assert (tmp_path / "minutes.deepseek.review.stale.json").read_text(encoding="utf-8") == '{"old": true}\n'
    assert (tmp_path / "minutes.deepseek.review.stale.md").read_text(encoding="utf-8") == "old review\n"
    status = json.loads((tmp_path / "summary_status.json").read_text(encoding="utf-8"))
    assert status["archived_stale_reviews"] == ["minutes.deepseek.review.stale.json", "minutes.deepseek.review.stale.md"]


def test_deepseek_cli_archives_an_old_review_before_an_unexpected_interruption(tmp_path, monkeypatch):
    write_json(tmp_path / "transcript.json", _segments())
    write_json(tmp_path / "keyframes.json", [])
    write_json(tmp_path / "metadata.json", {"input": "/private/recording.mov", "duration": 22.0})
    old_json = tmp_path / "minutes.deepseek.review.json"
    old_markdown = tmp_path / "minutes.deepseek.review.md"
    old_json.write_text('{"old": true}\n', encoding="utf-8")
    old_markdown.write_text("old review\n", encoding="utf-8")

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("meeting_minutes.cli.generate_deepseek_review", interrupt)
    args = build_parser().parse_args(["summarize", "--output-dir", str(tmp_path), "--summary-engine", "deepseek"])

    with pytest.raises(KeyboardInterrupt):
        summarize_existing(args)

    assert not old_json.exists()
    assert not old_markdown.exists()
    assert (tmp_path / "minutes.deepseek.review.stale.json").exists()
    assert (tmp_path / "minutes.deepseek.review.stale.md").exists()


def test_deepseek_cli_does_not_publish_a_partial_review_when_markdown_write_fails(tmp_path, monkeypatch):
    segments = _segments()
    write_json(tmp_path / "transcript.json", segments)
    write_json(tmp_path / "keyframes.json", [])
    write_json(tmp_path / "metadata.json", {"input": "/private/recording.mov", "duration": 22.0})
    old_json = tmp_path / "minutes.deepseek.review.json"
    old_markdown = tmp_path / "minutes.deepseek.review.md"
    old_json.write_text('{"old": true}\n', encoding="utf-8")
    old_markdown.write_text("old review\n", encoding="utf-8")
    prepared = _prepared()
    review = validate_deepseek_review(
        _payload(prepared),
        evidence_input=prepared,
        source_segments=segments,
        requested_model="deepseek-v4-pro",
    )
    monkeypatch.setattr(
        "meeting_minutes.cli.generate_deepseek_review",
        lambda **kwargs: (review, {"engine": "deepseek", "status": "draft_only", "external_processing": True}),
    )
    monkeypatch.setattr("meeting_minutes.cli.write_deepseek_review", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    args = build_parser().parse_args(["summarize", "--output-dir", str(tmp_path), "--summary-engine", "deepseek"])

    with pytest.raises(OSError, match="disk full"):
        summarize_existing(args)

    assert not old_json.exists()
    assert not old_markdown.exists()
    assert (tmp_path / "minutes.deepseek.review.stale.json").exists()
    assert (tmp_path / "minutes.deepseek.review.stale.md").exists()


def test_run_pipeline_returns_nonzero_when_requested_deepseek_review_fails(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "recording.mov"
    input_path.write_bytes(b"not-used-by-mocked-media-stage")
    output_dir = tmp_path / "run-output"
    args = build_parser().parse_args(
        [
            "run",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--expected-speakers",
            "2",
            "--diarization-backend",
            "speechbrain-cluster",
            "--summary-engine",
            "deepseek",
            "--skip-ocr",
        ]
    )
    monkeypatch.setattr("meeting_minutes.cli._effective_media", lambda input_path, output_dir, start, duration: (input_path, 0.0))
    monkeypatch.setattr("meeting_minutes.cli.probe_media", lambda input_path: {"duration": 22.0})
    monkeypatch.setattr("meeting_minutes.cli.extract_audio", lambda input_path, destination: destination)
    monkeypatch.setattr("meeting_minutes.cli.transcribe_audio", lambda audio_path, model, language: (_segments(), {"status": "ok"}))
    monkeypatch.setattr("meeting_minutes.cli.diarize_audio", lambda *args, **kwargs: ([], {"status": "ok"}))
    monkeypatch.setattr("meeting_minutes.cli.attach_speakers", lambda segments, turns: None)
    monkeypatch.setattr("meeting_minutes.cli.regular_times", lambda duration, interval: [])
    monkeypatch.setattr("meeting_minutes.cli.keyword_times", lambda segments: [])
    monkeypatch.setattr("meeting_minutes.cli.extract_frames", lambda *args, **kwargs: [])
    monkeypatch.setattr("meeting_minutes.cli.choose_keyframes", lambda frames, segments: [])
    monkeypatch.setattr("meeting_minutes.cli._write_action_artifacts", lambda output_dir, segments, statuses: {})
    monkeypatch.setattr("meeting_minutes.cli.write_extractive_minutes", lambda *args, **kwargs: None)
    monkeypatch.setattr("meeting_minutes.cli.write_transcript_markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr("meeting_minutes.cli.write_speaker_samples", lambda *args, **kwargs: None)
    monkeypatch.setattr("meeting_minutes.cli.write_quality_report", lambda *args, **kwargs: None)
    monkeypatch.setattr("meeting_minutes.cli.write_review_queue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "meeting_minutes.cli.generate_deepseek_review",
        lambda **kwargs: (None, {"engine": "deepseek", "status": "network_error", "external_processing": True}),
    )

    assert run_pipeline(args) == 2
    captured = capsys.readouterr()
    assert "not generated (network_error)" in captured.err
    summary_status = json.loads((output_dir / "summary_status.json").read_text(encoding="utf-8"))
    assert summary_status["review_written"] is False
    assert not (output_dir / "minutes.deepseek.review.md").exists()


def test_cli_accepts_deepseek_env_file_and_keeps_default_local():
    parser = build_parser()
    run = parser.parse_args(["run", "--input", "/tmp/recording.mov", "--output-dir", "/tmp/output", "--expected-speakers", "2"])
    deepseek = parser.parse_args(
        [
            "summarize",
            "--output-dir",
            "/tmp/output",
            "--summary-engine",
            "deepseek",
            "--deepseek-env-file",
            "/tmp/meeting.env",
            "--deepseek-redact-name",
            "Casey",
        ]
    )

    assert run.summary_engine == "extractive"
    assert deepseek.summary_engine == "deepseek"
    assert deepseek.deepseek_env_file.name == "meeting.env"
    assert deepseek.deepseek_output_language == "zh-CN"
    assert deepseek.deepseek_redact_name == ["Casey"]
    assert deepseek.deepseek_allow_unauthenticated_loopback is False
