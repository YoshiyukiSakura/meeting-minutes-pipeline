from types import SimpleNamespace

import pytest

from meeting_minutes.action_items import build_action_ledger
from meeting_minutes.cli import _write_ollama_draft, build_parser, relabel_existing, validate_actions_existing, write_voice_template
from meeting_minutes.jsonio import read_json, write_json


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


def test_ollama_draft_is_visibly_marked_as_non_publishable(tmp_path):
    path = tmp_path / "minutes.ollama.draft.md"
    _write_ollama_draft(path, "# Draft\n")

    content = path.read_text(encoding="utf-8")
    assert "草稿，仅供核对" in content
    assert content.endswith("# Draft\n")


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
