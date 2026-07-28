from pathlib import Path
from types import SimpleNamespace

import pytest

from meeting_minutes import cli
from meeting_minutes.action_items import build_action_ledger, stable_segment_id, transcript_fingerprint
from meeting_minutes.cli import (
    _write_ollama_draft,
    apply_voice_registry_existing,
    build_parser,
    publish_minutes_file,
    relabel_existing,
    validate_actions_existing,
    write_voice_template,
)
from meeting_minutes.jsonio import read_json, write_json
from meeting_minutes.publication import ACTION_EVIDENCE_FORMAT, ACTION_INTENT_REVIEW_FORMAT, action_ledger_fingerprint


def _english_minutes(*, action_row: str | None = None) -> str:
    action_lines = ["- No publishable action items were identified."]
    if action_row:
        action_lines = ["| Time | Item | Owner |", "| --- | --- | --- |", action_row]
    return "\n".join(
        [
            "# Meeting Minutes",
            "",
            "## Topics and Outcomes",
            "",
            "### 1. Topic (00:00-01:00)",
            "- Current state: The current status was confirmed.",
            "- Outcome: The current consensus was recorded.",
            "",
            "## Project Updates",
            "",
            "- No publishable project updates were identified.",
            "",
            "## Confirmed Decisions",
            "",
            "- No verified final decision was made.",
            "",
            "## Action Items",
            "",
            *action_lines,
            "",
        ]
    )


def test_voice_template_uses_generic_speaker_count(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, names=None, speaker_count=3)
    assert write_voice_template(args) == 0

    template = read_json(tmp_path / "voice_enrollment.template.json")
    assert list(template["speakers"]) == ["Speaker 1", "Speaker 2", "Speaker 3"]

    guide = (tmp_path / "voice_enrollment_guide.md").read_text(encoding="utf-8")
    assert '"Speaker 3"' in guide
    assert "Alice" not in guide
    assert "Bob" not in guide


def test_voice_template_accepts_multiple_known_names(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, names=["Alice", "Bob", "Carol"], speaker_count=2)
    assert write_voice_template(args) == 0

    template = read_json(tmp_path / "voice_enrollment.template.json")
    assert list(template["speakers"]) == ["Alice", "Bob", "Carol"]


def test_voice_template_requires_names_or_speaker_count(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, names=None, speaker_count=0)
    with pytest.raises(ValueError, match="speaker-count"):
        write_voice_template(args)


def test_failed_smart_summary_quarantines_prior_active_artifacts(
    tmp_path,
    monkeypatch,
):
    write_json(tmp_path / "transcript.json", [])
    review_dir = tmp_path / "work" / "review"
    review_dir.mkdir(parents=True)
    prior_artifacts = {
        "minutes.smart.md": "old chinese minutes\n",
        "minutes.smart.en.md": "old english minutes\n",
        "minutes.smart.json": '{"old": true}\n',
        "minutes.smart.audit.json": '{"old_audit": true}\n',
    }
    for filename, content in prior_artifacts.items():
        (review_dir / filename).write_text(content, encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "generate_smart_minutes",
        lambda **kwargs: (
            None,
            {
                "status": "review_schema_invalid",
                "errors": ["test_failure"],
            },
        ),
    )
    monkeypatch.setattr(cli, "_deepseek_config", lambda args: object())

    result = cli.smart_summarize_existing(
        SimpleNamespace(output_dir=tmp_path, review_passes=2)
    )

    assert result == 2
    status = read_json(tmp_path / "summary_status.json")
    assert status["archived_stale_smart_reviews"] == [
        "minutes.smart.stale.md",
        "minutes.smart.en.stale.md",
        "minutes.smart.stale.json",
        "minutes.smart.audit.stale.json",
    ]
    for filename in prior_artifacts:
        assert not (review_dir / filename).exists()
    assert (
        review_dir / "minutes.smart.stale.md"
    ).read_text(encoding="utf-8") == prior_artifacts["minutes.smart.md"]
    assert (
        review_dir / "minutes.smart.en.stale.md"
    ).read_text(encoding="utf-8") == prior_artifacts["minutes.smart.en.md"]


def test_direct_visual_cluster_stage_refuses_ambiguous_or_mismatched_visual_artifacts(tmp_path):
    write_json(tmp_path / "visual_identity.json", {"frames": []})
    write_json(tmp_path / "dynamic_visual_identity.json", {"frames": []})

    ambiguous = cli._run_direct_visual_cluster_identity(
        output_dir=tmp_path,
        segments=[],
        turns=[],
        expected_recording={"effective_input": "/recording.mov", "duration": 10.0},
    )

    assert ambiguous["status"] == "skipped_ambiguous_visual_identity_artifacts"

    (tmp_path / "dynamic_visual_identity.json").unlink()
    mismatch = cli._run_direct_visual_cluster_identity(
        output_dir=tmp_path,
        segments=[],
        turns=[],
        expected_recording={"effective_input": "/recording.mov", "duration": 10.0},
    )

    assert mismatch["status"] == "skipped_visual_identity_recording_mismatch"


def test_direct_visual_cluster_skip_retracts_prior_stage_labels(tmp_path):
    write_json(
        tmp_path / "visual_identity.json",
        {"recording": {"effective_input": "/other-recording.mov", "duration": 10.0}, "frames": []},
    )
    segments = [
        {
            "start": 1.0,
            "end": 2.0,
            "speaker": "Speaker 3",
            "name": "Billy",
            "name_source": "direct_visual_voice_cluster_consensus",
            "name_confidence": 0.9,
            "direct_visual_cluster_identity_evidence": {"candidate": "Billy"},
        }
    ]

    status = cli._run_direct_visual_cluster_identity(
        output_dir=tmp_path,
        segments=segments,
        turns=[],
        expected_recording={"effective_input": "/recording.mov", "duration": 10.0},
    )

    assert status["status"] == "skipped_visual_identity_recording_mismatch"
    assert status["cleared_prior_cluster_assignments"] == 1
    assert segments[0].get("name") is None
    assert read_json(tmp_path / "direct_visual_cluster_identity.json")["status"] == "skipped_visual_identity_recording_mismatch"


def test_visual_recording_provenance_detects_same_size_replaced_input(tmp_path):
    recording = tmp_path / "recording.mov"
    recording.write_bytes(b"first")
    first = cli._visual_recording_provenance(recording, 10.0)
    recording.write_bytes(b"other")
    second = cli._visual_recording_provenance(recording, 10.0)

    assert first["effective_input"] == second["effective_input"]
    assert first["duration"] == second["duration"]
    assert first["size_bytes"] == second["size_bytes"]
    assert first["content_sha256"] != second["content_sha256"]


def test_visual_voice_skip_retracts_prior_voiceprint_labels_on_visual_mismatch(tmp_path):
    recording = tmp_path / "recording.mov"
    recording.write_bytes(b"recording")
    audio = tmp_path / "work" / "audio_16k_mono.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"unused for provenance mismatch")
    write_json(
        tmp_path / "metadata.json",
        {"effective_input": str(recording), "duration": 10.0, "source_offset": 0.0},
    )
    write_json(
        tmp_path / "transcript.json",
        [
            {
                "start": 1.0,
                "end": 2.0,
                "speaker": "Speaker 3",
                "text": "I will follow up.",
                "name": "Billy",
                "name_source": "same_session_visual_voiceprint",
                "name_confidence": 0.88,
                "visual_voice_identity_evidence": {"profile": "Billy"},
            }
        ],
    )
    write_json(
        tmp_path / "visual_identity.json",
        {"recording": {"effective_input": str(recording), "duration": 10.0}, "frames": []},
    )

    assert cli.visual_voice_identify_existing(
        SimpleNamespace(output_dir=tmp_path, config=None, speechbrain_cache=None, visual_identity_path=None)
    ) == 0

    segment = read_json(tmp_path / "transcript.json")[0]
    assert segment.get("name") is None
    status = read_json(tmp_path / "run_status.json")["statuses"]["visual_voice_identity"]
    assert status["status"] == "skipped_visual_identity_recording_mismatch"
    assert status["cleared_prior_visual_voice_assignments"] == 1
    assert read_json(tmp_path / "same_session_visual_voice_registry.json")["status"] == "skipped_visual_identity_recording_mismatch"


def test_visual_voice_invalid_config_persists_retraction(tmp_path):
    recording = tmp_path / "recording.mov"
    recording.write_bytes(b"recording")
    recording_provenance = cli._visual_recording_provenance(recording, 10.0)
    audio = tmp_path / "work" / "audio_16k_mono.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"unused because config validation fails first")
    write_json(
        tmp_path / "metadata.json",
        {"effective_input": str(recording), "duration": 10.0, "source_offset": 0.0},
    )
    write_json(
        tmp_path / "transcript.json",
        [
            {
                "start": 1.0,
                "end": 2.0,
                "speaker": "Speaker 3",
                "text": "I will follow up.",
                "name": "Billy",
                "name_source": "same_session_visual_voiceprint",
                "name_confidence": 0.88,
                "visual_voice_identity_evidence": {"profile": "Billy"},
            }
        ],
    )
    write_json(
        tmp_path / "visual_identity.json",
        {
            "recording": recording_provenance,
            "frames": [
                {
                    "active": True,
                    "name": "Billy",
                    "name_source": "visual_profile_reviewed_slot",
                    "actualTime": 2.0,
                }
            ],
        },
    )
    invalid_config = tmp_path / "invalid-visual-voice-config.json"
    write_json(invalid_config, {"settings": {"minimum_score": 0}})

    with pytest.raises(ValueError, match="minimum_score"):
        cli.visual_voice_identify_existing(
            SimpleNamespace(
                output_dir=tmp_path,
                config=invalid_config,
                speechbrain_cache=None,
                visual_identity_path=None,
            )
        )

    assert read_json(tmp_path / "transcript.json")[0].get("name") is None
    voice_status = read_json(tmp_path / "run_status.json")["statuses"]["visual_voice_identity"]
    assert voice_status["status"] == "failed_visual_voice_revalidation"
    assert voice_status["cleared_prior_visual_voice_assignments"] == 1
    assert read_json(tmp_path / "same_session_visual_voice_registry.json")["status"] == "failed_visual_voice_revalidation"


def test_visual_voice_missing_explicit_visual_artifact_persists_retraction(tmp_path):
    recording = tmp_path / "recording.mov"
    recording.write_bytes(b"recording")
    audio = tmp_path / "work" / "audio_16k_mono.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"unused because visual artifact lookup fails first")
    write_json(
        tmp_path / "metadata.json",
        {"effective_input": str(recording), "duration": 10.0, "source_offset": 0.0},
    )
    write_json(
        tmp_path / "transcript.json",
        [
            {
                "start": 1.0,
                "end": 2.0,
                "speaker": "Speaker 3",
                "text": "I will follow up.",
                "name": "Billy",
                "name_source": "same_session_visual_voiceprint",
                "name_confidence": 0.88,
                "visual_voice_identity_evidence": {"profile": "Billy"},
            }
        ],
    )
    missing_visual_identity = tmp_path / "missing-visual-identity.json"

    with pytest.raises(FileNotFoundError, match="Expected visual identity artifact"):
        cli.visual_voice_identify_existing(
            SimpleNamespace(
                output_dir=tmp_path,
                config=None,
                speechbrain_cache=None,
                visual_identity_path=missing_visual_identity,
            )
        )

    assert read_json(tmp_path / "transcript.json")[0].get("name") is None
    voice_status = read_json(tmp_path / "run_status.json")["statuses"]["visual_voice_identity"]
    assert voice_status["status"] == "failed_visual_voice_revalidation"
    assert voice_status["cleared_prior_visual_voice_assignments"] == 1
    assert read_json(tmp_path / "same_session_visual_voice_registry.json")["status"] == "failed_visual_voice_revalidation"


def test_visual_refresh_invalidates_prior_cluster_artifacts(tmp_path, monkeypatch):
    recording = tmp_path / "recording.mov"
    recording.write_bytes(b"recording")
    write_json(
        tmp_path / "metadata.json",
        {"effective_input": str(recording), "duration": 10.0, "source_offset": 0.0},
    )
    write_json(
        tmp_path / "transcript.json",
        [
            {
                "start": 1.0,
                "end": 2.0,
                "speaker": "Speaker 3",
                "text": "Current status.",
                "name": "Billy",
                "name_source": "direct_visual_voice_cluster_consensus",
                "name_confidence": 0.9,
                "direct_visual_cluster_identity_evidence": {"candidate": "Billy"},
            }
        ],
    )
    monkeypatch.setattr(cli, "_run_visual_identity", lambda **_kwargs: {"status": "ok"})

    assert cli.visual_identify_existing(
        SimpleNamespace(output_dir=tmp_path, input=None, visual_profile=tmp_path / "profile.json", max_frame_width=1280)
    ) == 0

    assert read_json(tmp_path / "transcript.json")[0].get("name") is None
    cluster_status = read_json(tmp_path / "run_status.json")["statuses"]["direct_visual_cluster_identity"]
    assert cluster_status["status"] == "invalidated_by_visual_identity_refresh"
    assert cluster_status["cleared_prior_cluster_assignments"] == 1
    assert read_json(tmp_path / "direct_visual_cluster_identity.json")["status"] == "invalidated_by_visual_identity_refresh"


def test_direct_visual_cluster_failure_persists_retraction(tmp_path):
    recording = tmp_path / "recording.mov"
    recording.write_bytes(b"recording")
    recording_provenance = cli._visual_recording_provenance(recording, 10.0)
    write_json(
        tmp_path / "metadata.json",
        {"effective_input": str(recording), "duration": 10.0, "source_offset": 0.0},
    )
    write_json(
        tmp_path / "transcript.json",
        [
            {
                "start": 1.0,
                "end": 2.0,
                "speaker": "Speaker 3",
                "text": "Current status.",
                "name": "Billy",
                "name_source": "direct_visual_voice_cluster_consensus",
                "name_confidence": 0.9,
                "direct_visual_cluster_identity_evidence": {"candidate": "Billy"},
            }
        ],
    )
    write_json(tmp_path / "speaker_turns.json", [{"speaker": "Speaker 3", "start": 0.0, "end": 4.0}])
    write_json(
        tmp_path / "visual_identity.json",
        {
            "recording": recording_provenance,
            "frames": [
                {
                    "active": True,
                    "name": "Billy",
                    "name_source": "visual_profile_reviewed_slot",
                    "actualTime": 2.0,
                }
            ],
        },
    )
    invalid_config = tmp_path / "invalid-cluster-config.json"
    write_json(invalid_config, {"settings": {"minimum_turn_frame_votes": 0}})

    with pytest.raises(ValueError, match="minimum_turn_frame_votes"):
        cli.direct_visual_cluster_identify_existing(
            SimpleNamespace(output_dir=tmp_path, config=invalid_config, visual_identity_path=None)
        )

    assert read_json(tmp_path / "transcript.json")[0].get("name") is None
    cluster_status = read_json(tmp_path / "run_status.json")["statuses"]["direct_visual_cluster_identity"]
    assert cluster_status["status"] == "failed_direct_visual_cluster_revalidation"
    assert cluster_status["cleared_prior_cluster_assignments"] == 1
    assert read_json(tmp_path / "direct_visual_cluster_identity.json")["status"] == "failed_direct_visual_cluster_revalidation"


def test_visual_voice_command_accepts_explicit_visual_identity_artifact():
    args = build_parser().parse_args(
        [
            "visual-voice-identify",
            "--output-dir",
            "/tmp/meeting-output",
            "--visual-identity-path",
            "/tmp/meeting-output/visual_identity.json",
        ]
    )

    assert args.visual_identity_path == Path("/tmp/meeting-output/visual_identity.json")


def test_ollama_draft_is_visibly_marked_as_non_publishable(tmp_path):
    path = tmp_path / "minutes.ollama.draft.md"
    _write_ollama_draft(path, "# Draft\n")

    content = path.read_text(encoding="utf-8")
    assert "草稿，仅供核对" in content
    assert content.endswith("# Draft\n")


def test_validate_minutes_command_accepts_path_and_duration():
    args = build_parser().parse_args(["validate-minutes", "--path", "/tmp/minutes.md", "--duration", "120"])

    assert args.command == "validate-minutes"
    assert args.path == Path("/tmp/minutes.md")
    assert args.duration == 120.0


def test_publish_minutes_command_requires_both_reviewed_languages():
    args = build_parser().parse_args(
        [
            "publish-minutes",
            "--output-dir",
            "/tmp/meeting-output",
                "--source",
                "/tmp/meeting-output/minutes.reviewed.md",
                "--english-source",
                "/tmp/meeting-output/minutes.reviewed.en.md",
        ]
    )

    assert args.command == "publish-minutes"
    assert args.output_dir == Path("/tmp/meeting-output")
    assert args.source == Path("/tmp/meeting-output/minutes.reviewed.md")
    assert args.english_source == Path("/tmp/meeting-output/minutes.reviewed.en.md")
    assert args.action_evidence is None
    assert args.action_intent_review is None
    assert args.project_evidence is None


def test_publish_minutes_validates_before_atomically_writing_canonical_document(tmp_path):
    reviewed = tmp_path / "minutes.reviewed.md"
    english_reviewed = tmp_path / "minutes.reviewed.en.md"
    reviewed.write_text(
        "\n".join(
            [
                "# 会议纪要",
                "",
                "## 议题与结论",
                "",
                "### 1. 议题（00:00-01:00）",
                "- 现状：已确认当前状态。",
                "- 讨论结果：形成了当前共识。",
                "",
                "## 项目进展",
                "",
                "- 本次未出现可发布的项目进展。",
                "",
                "## 已确认决定",
                "",
                "- 本次未出现可验证的最终决定。",
                "",
                "## 行动项",
                "",
                "- 本次未出现可发布的明确行动项。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    english_reviewed.write_text(_english_minutes(), encoding="utf-8")
    transcript = [
        {
            "start": 0.0,
            "end": 59.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "text": "We reviewed the current status.",
        }
    ]
    write_json(tmp_path / "transcript.json", transcript)
    write_json(tmp_path / "action_items.json", build_action_ledger(transcript))
    write_json(tmp_path / "metadata.json", {"duration": 60.0})

    args = SimpleNamespace(
        output_dir=tmp_path,
        source=reviewed,
        english_source=english_reviewed,
        duration=0.0,
        action_evidence=None,
    )
    assert publish_minutes_file(args) == 0

    assert (tmp_path / "minutes.md").read_text(encoding="utf-8") == reviewed.read_text(encoding="utf-8")
    assert (tmp_path / "minutes.en.md").read_text(encoding="utf-8") == english_reviewed.read_text(encoding="utf-8")
    publish_status = read_json(tmp_path / "minutes.publish-status.json")
    assert publish_status["status"] == "published"
    assert publish_status["duration_seconds"] == 60.0
    assert publish_status["action_row_count"] == 0
    assert (tmp_path / "share" / "minutes.md").read_text(encoding="utf-8") == reviewed.read_text(encoding="utf-8")
    assert (tmp_path / "share" / "minutes.en.md").read_text(encoding="utf-8") == english_reviewed.read_text(encoding="utf-8")
    assert "Riley" in (tmp_path / "share" / "transcript.md").read_text(encoding="utf-8")
    assert (tmp_path / "work" / "review" / "minutes.coverage.json").is_file()
    assert not (tmp_path / "minutes.coverage.json").exists()


def test_publish_minutes_rejects_action_rows_without_internal_evidence(tmp_path, capsys):
    reviewed = tmp_path / "minutes.reviewed.md"
    english_reviewed = tmp_path / "minutes.reviewed.en.md"
    reviewed.write_text(
        "\n".join(
            [
                "# 会议纪要",
                "",
                "## 议题与结论",
                "",
                "### 1. 议题（00:00-01:00）",
                "- 现状：已确认当前状态。",
                "- 讨论结果：形成了当前共识。",
                "",
                "## 项目进展",
                "",
                "- 本次未出现可发布的项目进展。",
                "",
                "## 已确认决定",
                "",
                "- 本次未出现可验证的最终决定。",
                "",
                "## 行动项",
                "",
                "| 时间点 | 事项 | 负责人 |",
                "| --- | --- | --- |",
                "| 00:00-01:00 | 完成已确认事项。 | Riley |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    english_reviewed.write_text(
        _english_minutes(action_row="| 00:00-01:00 | Complete the confirmed item. | Riley |"),
        encoding="utf-8",
    )
    transcript = [
        {
            "start": 0.0,
            "end": 60.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "text": "I will complete the confirmed task.",
        }
    ]
    write_json(tmp_path / "transcript.json", transcript)
    write_json(tmp_path / "action_items.json", build_action_ledger(transcript))
    write_json(tmp_path / "metadata.json", {"duration": 60.0})

    args = SimpleNamespace(
        output_dir=tmp_path,
        source=reviewed,
        english_source=english_reviewed,
        duration=0.0,
        action_evidence=None,
    )
    assert publish_minutes_file(args) == 1
    assert "action_evidence_required" in capsys.readouterr().err
    assert not (tmp_path / "minutes.md").exists()


def test_publish_minutes_rejects_empty_action_table_when_named_weak_intent_needs_disposition(tmp_path, capsys):
    reviewed = tmp_path / "minutes.reviewed.md"
    english_reviewed = tmp_path / "minutes.reviewed.en.md"
    reviewed.write_text(
        "\n".join(
            [
                "# 会议纪要",
                "",
                "## 议题与结论",
                "",
                "### 1. 零确认请求响应（00:00-01:00）",
                "- 现状：讨论了请求响应无法完整展示的问题。",
                "- 讨论结果：继续核对展示方式。",
                "",
                "## 项目进展",
                "",
                "- 本次未出现可发布的项目进展。",
                "",
                "## 已确认决定",
                "",
                "- 本次未出现可验证的最终决定。",
                "",
                "## 行动项",
                "",
                "- 本次未出现可发布的明确行动项。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    english_reviewed.write_text(_english_minutes(), encoding="utf-8")
    transcript = [
        {
            "start": 0.0,
            "end": 59.0,
            "speaker": "Speaker 1",
            "name": "Armando",
            "name_confidence": 0.95,
            "text": "I would like to create an issue to show zero confirmation request-session responses.",
        }
    ]
    write_json(tmp_path / "transcript.json", transcript)
    write_json(tmp_path / "action_items.json", build_action_ledger(transcript))
    write_json(tmp_path / "metadata.json", {"duration": 60.0})

    args = SimpleNamespace(
        output_dir=tmp_path,
        source=reviewed,
        english_source=english_reviewed,
        duration=0.0,
        action_evidence=None,
        action_intent_review=None,
    )
    assert publish_minutes_file(args) == 1
    assert "action_intent_review_required" in capsys.readouterr().err
    assert not (tmp_path / "minutes.md").exists()


def test_publish_minutes_accepts_a_reviewed_weak_intent_only_when_its_source_is_linked(tmp_path):
    reviewed = tmp_path / "minutes.reviewed.md"
    english_reviewed = tmp_path / "minutes.reviewed.en.md"
    reviewed.write_text(
        "\n".join(
            [
                "# 会议纪要",
                "",
                "## 议题与结论",
                "",
                "### 1. 零确认请求响应（00:00-01:00）",
                "- 现状：讨论了请求响应无法完整展示的问题。",
                "- 讨论结果：通过 issue 记录展示需求。",
                "",
                "## 项目进展",
                "",
                "- 本次未出现可发布的项目进展。",
                "",
                "## 已确认决定",
                "",
                "- 本次未出现可验证的最终决定。",
                "",
                "## 行动项",
                "",
                "| 时间点 | 事项 | 负责人 |",
                "| --- | --- | --- |",
                "| 00:00-00:59 | 创建 issue，展示 zero confirmation 请求会话中的响应。 | Armando |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    english_reviewed.write_text(
        _english_minutes(
            action_row="| 00:00-00:59 | Create an issue to show responses in the zero-confirmation request session. | Armando |"
        ),
        encoding="utf-8",
    )
    transcript = [
        {
            "start": 0.0,
            "end": 59.0,
            "speaker": "Speaker 1",
            "name": "Armando",
            "name_confidence": 0.95,
            "text": "I would like to create an issue to show zero confirmation request-session responses.",
        }
    ]
    ledger = build_action_ledger(transcript)
    segment_id = stable_segment_id(transcript[0], 0)
    signal_id = ledger["intent_recall"]["signals"][0]["signal_id"]
    evidence = tmp_path / "minutes.reviewed.actions.json"
    write_json(
        evidence,
        {
            "format": ACTION_EVIDENCE_FORMAT,
            "transcript_sha256": transcript_fingerprint(transcript),
            "action_ledger_sha256": action_ledger_fingerprint(ledger),
            "rows": [
                {
                    "row": 1,
                    "evidence_mode": "reviewed_context",
                    "source_segment_ids": [segment_id],
                    "owner_evidence_segment_id": segment_id,
                    "review_note": "已逐句核对 Armando 的行动意图与实名映射。",
                }
            ],
        },
    )
    intent_review = tmp_path / "minutes.reviewed.action-intents.json"
    write_json(
        intent_review,
        {
            "format": ACTION_INTENT_REVIEW_FORMAT,
            "transcript_sha256": transcript_fingerprint(transcript),
            "action_ledger_sha256": action_ledger_fingerprint(ledger),
            "items": [
                {
                    "signal_id": signal_id,
                    "disposition": "published",
                    "review_note": "将该意图作为 Armando 的正式行动项发布，原句与表格时间范围已核对。",
                }
            ],
        },
    )
    write_json(tmp_path / "transcript.json", transcript)
    write_json(tmp_path / "action_items.json", ledger)
    write_json(tmp_path / "metadata.json", {"duration": 60.0})

    args = SimpleNamespace(
        output_dir=tmp_path,
        source=reviewed,
        english_source=english_reviewed,
        duration=0.0,
        action_evidence=evidence,
        action_intent_review=intent_review,
    )
    assert publish_minutes_file(args) == 0

    publish_status = read_json(tmp_path / "minutes.publish-status.json")
    assert publish_status["action_intent_signal_count"] == 1
    assert publish_status["action_intent_review_sha256"]


def test_publish_minutes_binds_reviewed_context_evidence_to_current_transcript_and_ledger(tmp_path):
    reviewed = tmp_path / "minutes.reviewed.md"
    english_reviewed = tmp_path / "minutes.reviewed.en.md"
    reviewed.write_text(
        "\n".join(
            [
                "# 会议纪要",
                "",
                "## 议题与结论",
                "",
                "### 1. MPC 部署（00:00-01:00）",
                "- 现状：已确认部署事项。",
                "- 讨论结果：由负责人继续执行。",
                "",
                "## 项目进展",
                "",
                "- 本次未出现可发布的项目进展。",
                "",
                "## 已确认决定",
                "",
                "- 本次未出现可验证的最终决定。",
                "",
                "## 行动项",
                "",
                "| 时间点 | 事项 | 负责人 |",
                "| --- | --- | --- |",
                "| 00:00-01:00 | 部署 MPC。 | Riley |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    english_reviewed.write_text(
        _english_minutes(action_row="| 00:00-01:00 | Deploy MPC. | Riley |"),
        encoding="utf-8",
    )
    transcript = [
        {
            "start": 0.0,
            "end": 59.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "text": "I will deploy MPC.",
        }
    ]
    ledger = build_action_ledger(transcript)
    segment_id = stable_segment_id(transcript[0], 0)
    evidence = tmp_path / "minutes.reviewed.actions.json"
    write_json(
        evidence,
        {
            "format": ACTION_EVIDENCE_FORMAT,
            "transcript_sha256": transcript_fingerprint(transcript),
            "action_ledger_sha256": action_ledger_fingerprint(ledger),
            "rows": [
                {
                    "row": 1,
                    "evidence_mode": "reviewed_context",
                    "source_segment_ids": [segment_id],
                    "owner_evidence_segment_id": segment_id,
                    "review_note": "已逐句核对承诺和实名映射。",
                }
            ],
        },
    )
    write_json(tmp_path / "transcript.json", transcript)
    write_json(tmp_path / "action_items.json", ledger)
    write_json(tmp_path / "metadata.json", {"duration": 60.0})
    write_json(
        tmp_path / "summary_status.json",
        {"status": "reviewed_draft", "actions": 7, "project_updates": 0},
    )

    args = SimpleNamespace(
        output_dir=tmp_path,
        source=reviewed,
        english_source=english_reviewed,
        duration=0.0,
        action_evidence=evidence,
    )
    assert publish_minutes_file(args) == 0

    publish_status = read_json(tmp_path / "minutes.publish-status.json")
    assert publish_status["action_row_count"] == 1
    assert publish_status["transcript_sha256"] == transcript_fingerprint(transcript)
    assert publish_status["action_ledger_sha256"] == action_ledger_fingerprint(ledger)
    assert publish_status["action_evidence_sha256"]
    summary_status = read_json(tmp_path / "summary_status.json")
    assert summary_status["status"] == "published"
    assert summary_status["actions"] == 1
    assert summary_status["project_updates"] == 0
    assert summary_status["publication"]["format"] == "meeting-minutes/shareable-minutes-v4"


def test_publish_minutes_rejects_a_duration_that_disagrees_with_metadata(tmp_path, capsys):
    reviewed = tmp_path / "minutes.reviewed.md"
    english_reviewed = tmp_path / "minutes.reviewed.en.md"
    reviewed.write_text("# 任意内容\n", encoding="utf-8")
    english_reviewed.write_text("# Any content\n", encoding="utf-8")
    write_json(tmp_path / "metadata.json", {"duration": 60.0})

    args = SimpleNamespace(
        output_dir=tmp_path,
        source=reviewed,
        english_source=english_reviewed,
        duration=61.0,
        action_evidence=None,
    )
    assert publish_minutes_file(args) == 1
    assert "minutes_duration_mismatch" in capsys.readouterr().err


def test_visual_identify_command_accepts_profile_and_output_directory(tmp_path):
    args = build_parser().parse_args(
        [
            "visual-identify",
            "--output-dir",
            str(tmp_path / "meeting-output"),
            "--visual-profile",
            str(tmp_path / "visual-profile.json"),
        ]
    )

    assert args.command == "visual-identify"
    assert args.max_frame_width == 1280


def test_relabel_command_accepts_output_and_participant_map(tmp_path):
    args = build_parser().parse_args(
        [
            "relabel",
            "--output-dir",
            str(tmp_path / "meeting-output"),
            "--participant-map",
            str(tmp_path / "participant-map.json"),
        ]
    )

    assert args.command == "relabel"
    assert args.participant_map.name == "participant-map.json"


def test_audit_actions_command_accepts_output_directory(tmp_path):
    args = build_parser().parse_args(
        [
            "audit-actions",
            "--output-dir",
            str(tmp_path / "meeting-output"),
        ]
    )

    assert args.command == "audit-actions"


def test_validate_actions_command_rejects_ungrounded_duration(tmp_path):
    transcript = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "text": "I will figure out when to upgrade MPC.",
        }
    ]
    ledger = build_action_ledger(transcript)
    write_json(tmp_path / "transcript.json", transcript)
    write_json(tmp_path / "action_items.json", ledger)
    candidate = ledger["candidates"][0]
    proposed_items = tmp_path / "proposed-items.json"
    write_json(
        proposed_items,
        [
            {
                "candidate_id": candidate["candidate_id"],
                "owner": "Riley",
                "source_quote": candidate["source_quote"],
                "text": "Coordinate a two-hour MPC maintenance window.",
            }
        ],
    )

    assert validate_actions_existing(SimpleNamespace(output_dir=tmp_path, items=proposed_items)) == 1
    result = read_json(tmp_path / "action_items.validation.json")
    assert result["status"] == "rejected"
    assert "duration_not_grounded" in result["results"][0]["errors"]
    assert "published_text_not_verbatim" in result["results"][0]["errors"]


def test_validate_actions_rejects_a_ledger_after_transcript_changes(tmp_path):
    transcript = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "text": "I will figure out when to upgrade MPC.",
        }
    ]
    ledger = build_action_ledger(transcript)
    write_json(tmp_path / "action_items.json", ledger)
    changed_transcript = [{**transcript[0], "text": "I will set the MPC upgrade timing."}]
    write_json(tmp_path / "transcript.json", changed_transcript)
    candidate = ledger["candidates"][0]
    proposed_items = tmp_path / "proposed-items.json"
    write_json(
        proposed_items,
        [
            {
                "candidate_id": candidate["candidate_id"],
                "owner": "Riley",
                "source_quote": candidate["source_quote"],
                "text": candidate["source_quote"],
            }
        ],
    )

    assert validate_actions_existing(SimpleNamespace(output_dir=tmp_path, items=proposed_items)) == 1
    result = read_json(tmp_path / "action_items.validation.json")
    assert result["ledger_fresh"] is False
    assert "ledger_stale" in result["results"][0]["errors"]
    assert "ledger_tampered" in result["results"][0]["errors"]


def test_validate_actions_rejects_a_tampered_ledger_even_when_transcript_is_unchanged(tmp_path):
    transcript = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Riley",
            "name_confidence": 0.95,
            "text": "I will figure out when to upgrade MPC.",
        }
    ]
    ledger = build_action_ledger(transcript)
    tampered_ledger = {**ledger, "candidates": [{**ledger["candidates"][0], "owner": "Alex Example"}]}
    write_json(tmp_path / "transcript.json", transcript)
    write_json(tmp_path / "action_items.json", tampered_ledger)
    candidate = ledger["candidates"][0]
    proposed_items = tmp_path / "proposed-items.json"
    write_json(
        proposed_items,
        [
            {
                "candidate_id": candidate["candidate_id"],
                "owner": "Riley",
                "source_quote": candidate["source_quote"],
                "text": candidate["source_quote"],
            }
        ],
    )

    assert validate_actions_existing(SimpleNamespace(output_dir=tmp_path, items=proposed_items)) == 1
    result = read_json(tmp_path / "action_items.validation.json")
    assert result["results"][0]["errors"] == ["ledger_tampered"]


def test_relabel_replaces_cross_recording_registry_guesses(tmp_path):
    write_json(
        tmp_path / "transcript.json",
        [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "first speaker",
                "speaker": "Speaker 1",
                "name": "Wrong Registry Name",
                "name_source": "voice_registry",
                "name_confidence": 0.91,
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "second speaker",
                "speaker": "Speaker 2",
                "name": "Another Wrong Name",
                "name_source": "voice_registry",
                "name_confidence": 0.81,
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "unresolved speaker",
                "speaker": "Speaker Unknown",
                "name": "Wrong Registry Name",
                "name_source": "voice_registry",
                "name_confidence": 0.76,
            },
        ],
    )
    participant_map = tmp_path / "participant-map.json"
    write_json(participant_map, {"Speaker 1": "Alice", "Speaker 2": "Bob"})

    assert relabel_existing(SimpleNamespace(output_dir=tmp_path, participant_map=participant_map)) == 0

    transcript = read_json(tmp_path / "transcript.json")
    assert [(item.get("name"), item.get("name_source")) for item in transcript] == [
        ("Alice", "participant_map"),
        ("Bob", "participant_map"),
        (None, None),
    ]
    assert transcript[0]["name_confidence"] == 0.95
    status = read_json(tmp_path / "participant_map_status.json")
    assert status["cleared_voice_registry_segments"] == 3
    assert (tmp_path / "transcript.md").exists()


def test_voice_registry_commands_accept_build_arguments(tmp_path):
    args = build_parser().parse_args(
        [
            "voice-registry",
            "build",
            "--sources",
            str(tmp_path / "sources.json"),
            "--output",
            str(tmp_path / "registry.json"),
        ]
    )

    assert args.command == "voice-registry"
    assert args.voice_registry_command == "build"
    assert args.target_far == 0.01


def test_voice_registry_apply_preserves_existing_speaker_clusters(tmp_path, monkeypatch):
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "audio_16k_mono.wav").write_bytes(b"audio")
    registry_path = tmp_path / "registry.json"
    write_json(registry_path, {"profiles": {"Billy": {}}})
    write_json(
        tmp_path / "transcript.json",
        [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "Speaker 2",
                "text": "I will update the access token.",
                "name": None,
                "name_source": None,
                "name_confidence": 0.0,
            }
            for _ in range(12)
        ],
    )

    def fake_apply(audio_path, segments, voice_registry_path, **kwargs):
        assert audio_path.name == "audio_16k_mono.wav"
        assert voice_registry_path == registry_path
        assert kwargs["threshold"] is None
        assert kwargs["margin"] is None
        for segment in segments:
            segment["name"] = "Billy"
            segment["name_source"] = "voice_registry"
            segment["name_confidence"] = 0.88
            segment["voice_registry_evidence"] = {"accepted_windows": 1}
        return {"status": "ok", "assigned_segments": 12}

    monkeypatch.setattr(cli, "apply_voice_registry", fake_apply)
    args = SimpleNamespace(
        output_dir=tmp_path,
        registry=registry_path,
        registry_threshold=None,
        registry_margin=None,
        speechbrain_cache=None,
    )

    assert apply_voice_registry_existing(args) == 0

    transcript = read_json(tmp_path / "transcript.json")
    assert all(segment["speaker"] == "Speaker 2" for segment in transcript)
    assert all(segment["name"] == "Billy" for segment in transcript)
    assert all(segment["name_source"] == "voice_registry" for segment in transcript)
    assert read_json(tmp_path / "voice_registry_status.json")["assigned_segments"] == 12


def test_voice_registry_commands_accept_apply_arguments(tmp_path):
    args = build_parser().parse_args(
        [
            "voice-registry",
            "apply",
            "--output-dir",
            str(tmp_path / "meeting-output"),
            "--registry",
            str(tmp_path / "registry.json"),
        ]
    )

    assert args.command == "voice-registry"
    assert args.voice_registry_command == "apply"
    assert args.registry.name == "registry.json"


def test_voice_registry_commands_accept_consensus_arguments(tmp_path):
    args = build_parser().parse_args(
        [
            "voice-registry",
            "consensus",
            "--output-dir",
            str(tmp_path / "meeting-output"),
        ]
    )

    assert args.command == "voice-registry"
    assert args.voice_registry_command == "consensus"


def test_diarize_command_accepts_voice_registry(tmp_path):
    args = build_parser().parse_args(
        [
            "diarize",
            "--output-dir",
            str(tmp_path / "meeting-output"),
            "--expected-speakers",
            "3",
            "--voice-registry",
            str(tmp_path / "registry.json"),
        ]
    )

    assert args.voice_registry.name == "registry.json"
