from meeting_minutes.action_items import (
    build_action_ledger,
    extract_attributes,
    normalize_text,
    stable_segment_id,
    validate_published_action_item,
)


def _segment(start, end, speaker, text, *, name=None):
    return {
        "start": start,
        "end": end,
        "speaker": speaker,
        "name": name or speaker,
        "name_confidence": 0.95,
        "text": text,
    }


def _counterexample_segments():
    return [
        _segment(179.0, 185.0, "Alex Example", "We need to upgrade MPC. It is not urgent because we are not using it."),
        _segment(186.0, 191.0, "Alex Example", "We also have a new feature in the Operator BPS."),
        _segment(191.0, 197.0, "Alex Example", "It can encrypt the private keys in the database."),
        _segment(207.0, 210.0, "Alex Example", "We need to schedule like a two hour"),
        _segment(211.0, 213.0, "Riley", "window to be able to do that."),
        _segment(213.0, 218.0, "Riley", "I need to talk to Operations about the best time to do the encryption."),
        _segment(224.0, 228.0, "Riley", "I will figure out when to upgrade MPC."),
        _segment(230.0, 232.0, "Alex Example", "That upgrade does not require downtime."),
    ]


def test_counterexample_is_split_into_two_atomic_action_candidates():
    ledger = build_action_ledger(_counterexample_segments())
    accepted = [candidate for candidate in ledger["candidates"] if candidate["status"] == "accepted"]

    assert len(accepted) == 2
    encryption = next(candidate for candidate in accepted if "private_key_encryption" in candidate["attributes"]["topics"])
    mpc = next(candidate for candidate in accepted if "mpc" in candidate["attributes"]["topics"])
    assert encryption["owner"] == "Riley"
    # The two-hour sentence says only "do that". It remains contextual evidence
    # but cannot be attributed to encryption or MPC without an explicit topic.
    assert encryption["attributes"]["duration_minutes"] == []
    assert mpc["owner"] == "Riley"
    assert mpc["attributes"]["duration_minutes"] == []
    assert mpc["attributes"]["topics"] == ["mpc"]
    assert ledger["constraints"] == []


def test_evidence_gate_rejects_two_hour_mpc_merge():
    ledger = build_action_ledger(_counterexample_segments())
    mpc = next(candidate for candidate in ledger["candidates"] if candidate["attributes"]["topics"] == ["mpc"])

    for text in (
        "Coordinate a two-hour maintenance window for the MPC upgrade.",
        "Coordinate a 120-minute maintenance window for the MPC upgrade.",
        "Coordinate a 2h maintenance window for the MPC upgrade.",
        "为 MPC 升级安排两个小时的维护窗口。",
    ):
        errors = validate_published_action_item(
            {
                "candidate_id": mpc["candidate_id"],
                "owner": "Riley",
                "source_quote": mpc["source_quote"],
                "text": text,
            },
            ledger,
        )

        assert "published_text_not_verbatim" in errors
        assert "duration_not_grounded" in errors


def test_evidence_gate_rejects_unparsed_duration_instead_of_silently_allowing_it():
    ledger = build_action_ledger(_counterexample_segments())
    mpc = next(candidate for candidate in ledger["candidates"] if candidate["attributes"]["topics"] == ["mpc"])

    errors = validate_published_action_item(
        {
            "candidate_id": mpc["candidate_id"],
            "owner": "Riley",
            "source_quote": mpc["source_quote"],
            "text": "Coordinate a couple of hours of maintenance for the MPC upgrade.",
        },
        ledger,
    )

    assert "duration_unparsed" in errors


def test_evidence_gate_accepts_encryption_window_when_grounded():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Alex Example", "Private-key encryption needs a two-hour window."),
            _segment(3.0, 5.0, "Riley", "I need to talk to Operations about private-key encryption."),
        ]
    )
    encryption = ledger["candidates"][0]
    assert encryption["attributes"]["duration_minutes"] == [120]

    errors = validate_published_action_item(
        {
            "candidate_id": encryption["candidate_id"],
            "owner": "Riley",
            "source_quote": encryption["source_quote"],
            "text": encryption["source_quote"],
        },
        ledger,
    )

    assert errors == []


def test_unanchored_duration_cannot_be_absorbed_by_a_later_topic():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Alex Example", "It can encrypt the private keys in the database."),
            _segment(3.0, 5.0, "Alex Example", "We need to schedule a two hour window to do that."),
            _segment(6.0, 8.0, "Riley", "I will figure out when to upgrade MPC."),
        ]
    )
    mpc = ledger["candidates"][0]
    assert mpc["attributes"]["topics"] == ["mpc"]
    assert mpc["attributes"]["duration_minutes"] == []

    errors = validate_published_action_item(
        {
            "candidate_id": mpc["candidate_id"],
            "owner": "Riley",
            "source_quote": mpc["source_quote"],
            "text": "Coordinate a two-hour maintenance window for the MPC upgrade.",
        },
        ledger,
    )
    assert "duration_not_grounded" in errors


def test_published_action_must_remain_a_verbatim_commitment_with_named_topic():
    ledger = build_action_ledger(_counterexample_segments())
    mpc = next(candidate for candidate in ledger["candidates"] if candidate["attributes"]["topics"] == ["mpc"])

    paraphrase_errors = validate_published_action_item(
        {
            "candidate_id": mpc["candidate_id"],
            "owner": "Riley",
            "source_quote": mpc["source_quote"],
            "text": "Set the timing for the MPC upgrade.",
        },
        ledger,
    )
    foreign_topic_errors = validate_published_action_item(
        {
            "candidate_id": mpc["candidate_id"],
            "owner": "Riley",
            "source_quote": mpc["source_quote"],
            "text": "Coordinate the database migration window.",
        },
        ledger,
    )

    assert "published_text_not_verbatim" in paraphrase_errors
    assert "published_topic_unresolved" in foreign_topic_errors


def test_evidence_gate_rejects_owner_and_quote_changes():
    ledger = build_action_ledger(_counterexample_segments())
    mpc = next(candidate for candidate in ledger["candidates"] if candidate["attributes"]["topics"] == ["mpc"])

    errors = validate_published_action_item(
        {
            "candidate_id": mpc["candidate_id"],
            "owner": "Alex Example",
            "source_quote": "I will arrange a two-hour MPC maintenance window.",
            "text": "Set the timing for the MPC upgrade.",
        },
        ledger,
    )

    assert "owner_mismatch" in errors
    assert "published_quote_mismatch" in errors


def test_unresolved_commitment_is_preserved_for_review_not_dropped():
    ledger = build_action_ledger(
        [
            {
                "start": 0.0,
                "end": 2.0,
                "speaker": "Speaker Unknown",
                "text": "I will handle it.",
            }
        ]
    )

    assert ledger["summary"] == {"accepted": 0, "review": 1, "constraints": 0}
    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert set(candidate["review_reasons"]) == {"owner_unresolved", "topic_unresolved"}


def test_downtime_constraint_survives_apostrophe_normalization():
    assert extract_attributes("The MPC upgrade doesn't require a downtime.")["downtime"] is False


def test_duration_parser_normalizes_common_equivalent_forms():
    assert extract_attributes("120-minute window")["duration_minutes"] == [120]
    assert extract_attributes("2h window")["duration_minutes"] == [120]
    assert extract_attributes("twelve hours")["duration_minutes"] == [720]
    assert extract_attributes("half an hour")["duration_minutes"] == [30]
    assert extract_attributes("两个小时")["duration_minutes"] == [120]


def test_text_normalization_is_not_semantic_duration_normalization():
    assert normalize_text("two hour") != normalize_text("120-minute")
    assert normalize_text("two hour") != normalize_text("2h")


def test_subjectless_downtime_cannot_inherit_a_nearby_topic():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Riley", "I will set the MPC upgrade timing."),
            _segment(3.0, 4.0, "Alex Example", "It does not require downtime."),
        ]
    )

    assert ledger["constraints"] == []
    assert extract_attributes("We need a two hour window to do that.")["downtime"] is None


def test_ability_or_status_statement_is_not_an_action_candidate():
    ledger = build_action_ledger([_segment(0.0, 2.0, "Riley", "I can hear you clearly.")])

    assert ledger["candidates"] == []


def test_segment_id_is_independent_of_list_position_and_name_relabeling():
    segment = _segment(10.0, 12.0, "Speaker 1", "I will set the MPC upgrade timing.", name="Riley")
    relabeled = {**segment, "name": "Different Display Name"}

    assert stable_segment_id(segment, 0) == stable_segment_id(relabeled, 99)


def test_explicit_named_assignment_is_publishable_when_owner_and_topic_share_a_source():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Alex Example", "Riley will set the MPC upgrade timing."),
            _segment(3.0, 4.0, "Riley", "Okay."),
        ]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "accepted"
    assert candidate["owner"] == "Riley"
    assert candidate["owner_evidence"] == "explicit_name_assignment"
    assert candidate["attributes"]["topics"] == ["mpc"]


def test_cross_turn_unresolved_assignment_is_preserved_for_review():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Alex Example", "Who will set the MPC upgrade timing?"),
            _segment(3.0, 4.0, "Riley", "I will."),
        ]
    )

    assert len(ledger["candidates"]) == 1
    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert candidate["review_reasons"] == ["topic_unresolved"]


def test_conflicting_downtime_constraints_are_rejected():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Riley", "I will set the MPC upgrade timing."),
            _segment(3.0, 4.0, "Alex Example", "The MPC upgrade requires downtime."),
            _segment(5.0, 6.0, "Alex Example", "The MPC upgrade does not require downtime."),
        ]
    )
    candidate = ledger["candidates"][0]

    errors = validate_published_action_item(
        {
            "candidate_id": candidate["candidate_id"],
            "owner": "Riley",
            "source_quote": candidate["source_quote"],
            "text": candidate["source_quote"],
        },
        ledger,
    )

    assert "downtime_constraint_conflict" in errors
