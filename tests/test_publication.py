import hashlib

from meeting_minutes.action_items import build_action_ledger, stable_segment_id, transcript_fingerprint
from meeting_minutes.jsonio import read_json, write_json
from meeting_minutes.minutes_contract import ShareableActionRow, ShareableProjectUpdateRow
from meeting_minutes.publication import (
    ACTION_EVIDENCE_FORMAT,
    ACTION_INTENT_REVIEW_FORMAT,
    PROJECT_EVIDENCE_FORMAT,
    PROJECT_UPDATE_COVERAGE_MIN_SECONDS,
    PUBLICATION_FORMAT,
    action_ledger_fingerprint,
    payload_fingerprint,
    recompute_project_update_coverage,
    sync_publication_status,
    validate_reviewed_action_evidence,
    validate_reviewed_action_intent_review,
    validate_reviewed_project_evidence,
)


def _segments():
    return [
        {
            "start": 0.0,
            "end": 10.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "text": "I will deploy MPC today.",
        }
    ]


def _published_status(chinese_path, english_path, segments, ledger):
    return {
        "format": PUBLICATION_FORMAT,
        "status": "published",
        "languages": ["zh", "en"],
        "canonical_sha256": {
            "zh": hashlib.sha256(chinese_path.read_bytes()).hexdigest(),
            "en": hashlib.sha256(english_path.read_bytes()).hexdigest(),
        },
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
    }


def test_sync_publication_archives_canonical_minutes_when_bound_transcript_changes(tmp_path):
    segments = _segments()
    ledger = build_action_ledger(segments)
    minutes = tmp_path / "minutes.md"
    english_minutes = tmp_path / "minutes.en.md"
    minutes.write_text("# 已发布纪要\n", encoding="utf-8")
    english_minutes.write_text("# Published minutes\n", encoding="utf-8")
    write_json(tmp_path / "minutes.publish-status.json", _published_status(minutes, english_minutes, segments, ledger))

    changed_segments = [{**segments[0], "text": "I will deploy the MPC update today."}]
    changed_ledger = build_action_ledger(changed_segments)
    status = sync_publication_status(tmp_path, changed_segments, changed_ledger)

    assert status["status"] == "stale"
    assert "transcript_changed" in status["reasons"]
    assert not minutes.exists()
    assert not english_minutes.exists()
    assert (tmp_path / status["archived_minutes"]["zh"]).exists()
    assert (tmp_path / status["archived_minutes"]["en"]).exists()
    published_status = read_json(tmp_path / "minutes.publish-status.json")
    assert published_status["status"] == "stale"


def test_sync_publication_archives_share_bundle_when_bound_transcript_changes(tmp_path):
    segments = _segments()
    ledger = build_action_ledger(segments)
    minutes = tmp_path / "minutes.md"
    english_minutes = tmp_path / "minutes.en.md"
    minutes.write_text("# 已发布纪要\n", encoding="utf-8")
    english_minutes.write_text("# Published minutes\n", encoding="utf-8")
    share_dir = tmp_path / "share"
    share_dir.mkdir()
    share_paths = {
        "zh": share_dir / "minutes.md",
        "en": share_dir / "minutes.en.md",
        "transcript": share_dir / "transcript.md",
    }
    share_paths["zh"].write_text(minutes.read_text(encoding="utf-8"), encoding="utf-8")
    share_paths["en"].write_text(english_minutes.read_text(encoding="utf-8"), encoding="utf-8")
    share_paths["transcript"].write_text("# Transcript\n", encoding="utf-8")
    published = _published_status(minutes, english_minutes, segments, ledger)
    published["share_sha256"] = {
        artifact: hashlib.sha256(path.read_bytes()).hexdigest()
        for artifact, path in share_paths.items()
    }
    write_json(tmp_path / "minutes.publish-status.json", published)

    changed_segments = [{**segments[0], "text": "I will deploy the MPC update today."}]
    changed_ledger = build_action_ledger(changed_segments)
    status = sync_publication_status(tmp_path, changed_segments, changed_ledger)

    assert status["status"] == "stale"
    assert "transcript_changed" in status["reasons"]
    assert "archived_share_bundle" in status
    assert not share_dir.exists()
    assert (tmp_path / status["archived_share_bundle"]).is_dir()


def test_sync_publication_archives_untracked_legacy_minutes(tmp_path):
    segments = _segments()
    ledger = build_action_ledger(segments)
    minutes = tmp_path / "minutes.md"
    minutes.write_text("# 没有发布状态的旧纪要\n", encoding="utf-8")

    status = sync_publication_status(tmp_path, segments, ledger)

    assert status == {
        "status": "untracked",
        "reason": "missing_publication_status",
        "archived_minutes": {"zh": "minutes.untracked.md"},
    }
    assert not minutes.exists()
    assert (tmp_path / "minutes.untracked.md").exists()


def test_sync_publication_archives_a_published_pair_when_the_english_document_is_missing(tmp_path):
    segments = _segments()
    ledger = build_action_ledger(segments)
    minutes = tmp_path / "minutes.md"
    english_minutes = tmp_path / "minutes.en.md"
    minutes.write_text("# 已发布纪要\n", encoding="utf-8")
    english_minutes.write_text("# Published minutes\n", encoding="utf-8")
    write_json(tmp_path / "minutes.publish-status.json", _published_status(minutes, english_minutes, segments, ledger))
    english_minutes.unlink()

    status = sync_publication_status(tmp_path, segments, ledger)

    assert status["status"] == "stale"
    assert status["reason"] == "missing_canonical_language"
    assert not minutes.exists()
    assert (tmp_path / status["archived_minutes"]["zh"]).exists()


def test_reviewed_context_action_evidence_requires_matching_named_owner_and_source_range():
    segments = _segments()
    ledger = build_action_ledger(segments)
    segment_id = stable_segment_id(segments[0], 0)
    rows = [
        ShareableActionRow(
            index=1,
            time_range="00:00-00:10",
            start=0.0,
            end=10.0,
            item="部署 MPC。",
            owner="Riley",
        )
    ]
    manifest = {
        "format": ACTION_EVIDENCE_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
        "rows": [
            {
                "row": 1,
                "evidence_mode": "reviewed_context",
                "source_segment_ids": [segment_id],
                "owner_evidence_segment_id": segment_id,
                "review_note": "已逐句核对承诺内容与同帧实名映射。",
            }
        ],
    }

    assert (
        validate_reviewed_action_evidence(
            manifest=manifest,
            rows=rows,
            segments=segments,
            action_ledger=ledger,
        )
        == []
    )

    manifest["rows"][0]["owner_evidence_segment_id"] = "seg-not-present"
    errors = validate_reviewed_action_evidence(
        manifest=manifest,
        rows=rows,
        segments=segments,
        action_ledger=ledger,
    )
    assert "action_evidence_row:1:owner_evidence_unknown" in errors


def test_action_intent_review_requires_exact_disposition_and_linked_published_source():
    segments = [
        {
            "start": 0.0,
            "end": 10.0,
            "speaker": "Speaker 1",
            "name": "Armando",
            "name_confidence": 0.95,
            "text": "I would like to create an issue for zero confirmation request-session responses.",
        }
    ]
    ledger = build_action_ledger(segments)
    segment_id = stable_segment_id(segments[0], 0)
    signal_id = ledger["intent_recall"]["signals"][0]["signal_id"]
    rows = [
        ShareableActionRow(
            index=1,
            time_range="00:00-00:10",
            start=0.0,
            end=10.0,
            item="创建 zero confirmation 请求会话响应展示 issue。",
            owner="Armando",
        )
    ]
    action_evidence = {
        "format": ACTION_EVIDENCE_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
        "rows": [
            {
                "row": 1,
                "evidence_mode": "reviewed_context",
                "source_segment_ids": [segment_id],
                "owner_evidence_segment_id": segment_id,
                "review_note": "已核对弱意图原句和 Armando 的实名映射。",
            }
        ],
    }
    manifest = {
        "format": ACTION_INTENT_REVIEW_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
        "items": [
            {
                "signal_id": signal_id,
                "disposition": "published",
                "review_note": "该句被人工确认为明确待办，且已与行动项表逐段关联。",
            }
        ],
    }

    assert (
        validate_reviewed_action_intent_review(
            manifest=manifest,
            rows=rows,
            action_evidence=action_evidence,
            segments=segments,
            action_ledger=ledger,
        )
        == []
    )

    manifest["items"][0]["disposition"] = "rejected"
    manifest["items"][0]["rejection_reason"] = "not_an_action"
    errors = validate_reviewed_action_intent_review(
        manifest=manifest,
        rows=rows,
        action_evidence=action_evidence,
        segments=segments,
        action_ledger=ledger,
    )
    assert f"action_intent_review_signal:{signal_id}:rejected_source_still_published" in errors


def test_action_intent_review_rejects_free_form_disposition_reason_and_stale_signal_set():
    segments = [
        {
            "start": 0.0,
            "end": 10.0,
            "speaker": "Speaker 1",
            "name": "Armando",
            "name_confidence": 0.95,
            "text": "I would like to create an issue for zero confirmation request-session responses.",
        }
    ]
    ledger = build_action_ledger(segments)
    signal_id = ledger["intent_recall"]["signals"][0]["signal_id"]
    manifest = {
        "format": ACTION_INTENT_REVIEW_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
        "items": [
            {
                "signal_id": signal_id,
                "disposition": "rejected",
                "rejection_reason": "free_text_reason",
                "review_note": "已审阅。",
            }
        ],
    }

    errors = validate_reviewed_action_intent_review(
        manifest=manifest,
        rows=[],
        action_evidence=None,
        segments=segments,
        action_ledger=ledger,
    )
    assert f"action_intent_review_signal:{signal_id}:rejection_reason_invalid" in errors

    manifest["items"] = []
    errors = validate_reviewed_action_intent_review(
        manifest=manifest,
        rows=[],
        action_evidence=None,
        segments=segments,
        action_ledger=ledger,
    )
    assert "action_intent_review_items_do_not_match_recall" in errors


def test_project_evidence_requires_current_named_participant_and_coverage():
    segments = [
        {
            "start": 0.0,
            "end": 61.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "name_source": "visual_active_speaker_highlight",
            "text": "I completed the MPC stability work already.",
        }
    ]
    ledger = build_action_ledger(segments)
    segment_id = stable_segment_id(segments[0], 0)
    rows = [
        ShareableProjectUpdateRow(
            index=1,
            time_range="00:00-01:01",
            start=0.0,
            end=61.0,
            participant="Riley",
            project="MPC",
            update="已完成稳定性工作。",
        )
    ]
    manifest = {
        "format": PROJECT_EVIDENCE_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
        "coverage_min_seconds": PROJECT_UPDATE_COVERAGE_MIN_SECONDS,
        "rows": [
            {
                "row": 1,
                "source_segment_ids": [segment_id],
                "participant_evidence_segment_id": segment_id,
                "review_note": "逐句核对已完成的 MPC 稳定性工作与当前实名映射。",
            }
        ],
        "exceptions": [],
    }

    assert recompute_project_update_coverage(segments)["Riley"].covered_seconds == 61.0
    assert (
        validate_reviewed_project_evidence(
            manifest=manifest,
            rows=rows,
            segments=segments,
            action_ledger=ledger,
        )
        == []
    )

    segments[0]["name"] = "Morgan"
    errors = validate_reviewed_project_evidence(
        manifest=manifest,
        rows=rows,
        segments=segments,
        action_ledger=ledger,
    )
    assert "project_evidence_row:1:source_participant_mismatch" in errors


def test_project_evidence_cannot_reuse_an_accepted_action_commitment():
    segments = [
        {
            "start": 0.0,
            "end": 10.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "name_source": "visual_active_speaker_highlight",
            "text": "I will deploy MPC today.",
        },
        {
            "start": 10.0,
            "end": 61.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "name_source": "visual_active_speaker_highlight",
            "text": "The MPC work is being reviewed by the team.",
        },
    ]
    ledger = build_action_ledger(segments)
    commitment_id = stable_segment_id(segments[0], 0)
    rows = [
        ShareableProjectUpdateRow(
            index=1,
            time_range="00:00-00:10",
            start=0.0,
            end=10.0,
            participant="Riley",
            project="MPC",
            update="将部署 MPC。",
        )
    ]
    manifest = {
        "format": PROJECT_EVIDENCE_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
        "coverage_min_seconds": PROJECT_UPDATE_COVERAGE_MIN_SECONDS,
        "rows": [
            {
                "row": 1,
                "source_segment_ids": [commitment_id],
                "participant_evidence_segment_id": commitment_id,
                "review_note": "该片段为行动承诺，不能作为项目进展发布。",
            }
        ],
        "exceptions": [],
    }

    errors = validate_reviewed_project_evidence(
        manifest=manifest,
        rows=rows,
        segments=segments,
        action_ledger=ledger,
    )

    assert "project_evidence_row:1:shadows_accepted_action_item" in errors


def test_project_coverage_can_only_be_bypassed_with_matching_exception_statistics():
    segments = [
        {
            "start": 0.0,
            "end": 61.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "name_source": "visual_active_speaker_highlight",
            "text": "I reviewed the agenda.",
        }
    ]
    ledger = build_action_ledger(segments)
    segment_id = stable_segment_id(segments[0], 0)
    manifest = {
        "format": PROJECT_EVIDENCE_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(ledger),
        "coverage_min_seconds": PROJECT_UPDATE_COVERAGE_MIN_SECONDS,
        "rows": [],
        "exceptions": [
            {
                "participant": "Riley",
                "covered_seconds": 61.0,
                "identity_sources": ["visual_active_speaker_highlight"],
                "identity_evidence_segment_id": segment_id,
                "reason": "本次发言为议程确认，没有可发布的项目进展。",
            }
        ],
    }

    assert (
        validate_reviewed_project_evidence(
            manifest=manifest,
            rows=[],
            segments=segments,
            action_ledger=ledger,
        )
        == []
    )

    manifest["exceptions"][0]["covered_seconds"] = 60.0
    errors = validate_reviewed_project_evidence(
        manifest=manifest,
        rows=[],
        segments=segments,
        action_ledger=ledger,
    )
    assert "project_evidence_exception_stats_mismatch:Riley" in errors


def test_sync_publication_archives_minutes_when_project_evidence_changes(tmp_path):
    segments = _segments()
    ledger = build_action_ledger(segments)
    minutes = tmp_path / "minutes.md"
    english_minutes = tmp_path / "minutes.en.md"
    minutes.write_text("# 已发布纪要\n", encoding="utf-8")
    english_minutes.write_text("# Published minutes\n", encoding="utf-8")
    project_evidence = tmp_path / "minutes.reviewed.projects.json"
    write_json(project_evidence, {"format": PROJECT_EVIDENCE_FORMAT, "rows": []})
    published = _published_status(minutes, english_minutes, segments, ledger)
    published["project_evidence_path"] = str(project_evidence)
    published["project_evidence_sha256"] = payload_fingerprint(read_json(project_evidence))
    write_json(tmp_path / "minutes.publish-status.json", published)
    write_json(project_evidence, {"format": PROJECT_EVIDENCE_FORMAT, "rows": [{"row": 1}]})

    status = sync_publication_status(tmp_path, segments, ledger)

    assert status["status"] == "stale"
    assert "project_evidence_changed" in status["reasons"]


def test_sync_publication_archives_minutes_when_action_intent_review_changes(tmp_path):
    segments = _segments()
    ledger = build_action_ledger(segments)
    minutes = tmp_path / "minutes.md"
    english_minutes = tmp_path / "minutes.en.md"
    minutes.write_text("# 已发布纪要\n", encoding="utf-8")
    english_minutes.write_text("# Published minutes\n", encoding="utf-8")
    intent_review = tmp_path / "minutes.reviewed.action-intents.json"
    write_json(intent_review, {"format": ACTION_INTENT_REVIEW_FORMAT, "items": []})
    published = _published_status(minutes, english_minutes, segments, ledger)
    published["action_intent_review_path"] = str(intent_review)
    published["action_intent_review_sha256"] = payload_fingerprint(read_json(intent_review))
    write_json(tmp_path / "minutes.publish-status.json", published)
    write_json(intent_review, {"format": ACTION_INTENT_REVIEW_FORMAT, "items": [{"signal_id": "intent-new"}]})

    status = sync_publication_status(tmp_path, segments, ledger)

    assert status["status"] == "stale"
    assert "action_intent_review_changed" in status["reasons"]
