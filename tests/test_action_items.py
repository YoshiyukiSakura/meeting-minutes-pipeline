from meeting_minutes.action_items import (
    build_action_ledger,
    extract_attributes,
    normalize_text,
    stable_segment_id,
    validate_published_action_item,
)
from meeting_minutes.identity_authority import ACTIVE_SPEAKER_HIGHLIGHT_SOURCE


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


def test_active_speaker_name_cannot_own_an_action_without_agent_confirmation():
    segment = _segment(0.0, 4.0, "Speaker 1", "I will prepare the migration plan.", name="John")
    segment["name_source"] = ACTIVE_SPEAKER_HIGHLIGHT_SOURCE

    unconfirmed = build_action_ledger([segment])
    assert unconfirmed["candidates"][0]["owner"] is None

    segment["visual_identity_agent_audit"] = {"status": "confirmed"}
    confirmed = build_action_ledger([segment])
    assert confirmed["candidates"][0]["owner"] == "John"


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
    mpc = next(
        candidate
        for candidate in ledger["candidates"]
        if candidate["attributes"]["topics"] == ["mpc"]
    )
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


def test_incomplete_assurance_clause_is_held_for_review():
    ledger = build_action_ledger([_segment(0.0, 2.0, "Riley", "I need to make sure that MPC one")])

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert "commitment_incomplete" in candidate["review_reasons"]


def test_complete_assurance_clause_remains_publishable():
    ledger = build_action_ledger([_segment(0.0, 2.0, "Riley", "I need to make sure that MPC one is restored.")])

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "accepted"


def test_segment_id_is_independent_of_list_position_and_name_relabeling():
    segment = _segment(10.0, 12.0, "Speaker 1", "I will set the MPC upgrade timing.", name="Riley")
    relabeled = {**segment, "name": "Different Display Name"}

    assert stable_segment_id(segment, 0) == stable_segment_id(relabeled, 99)


def test_explicit_named_assignment_waits_for_verified_owner_acceptance():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Alex Example", "Riley will set the MPC upgrade timing."),
            _segment(3.0, 4.0, "Riley", "Okay."),
        ]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert candidate["owner"] == "Riley"
    assert candidate["owner_evidence"] == "explicit_name_assignment"
    assert "owner_acceptance_unverified" in candidate["review_reasons"]
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


def test_weak_named_intent_is_recalled_but_never_auto_promoted_to_an_action():
    ledger = build_action_ledger(
        [
            _segment(
                4890.0,
                4903.0,
                "Armando",
                "And then I would like to create an issue to show the response in zero confirmation.",
            )
        ]
    )

    candidate = ledger["candidates"][0]
    signal = ledger["intent_recall"]["signals"][0]
    assert candidate["status"] == "review"
    assert candidate["owner"] == "Armando"
    assert candidate["commitment_kind"] == "weak_intent"
    assert "weak_intent_cue" in candidate["review_reasons"]
    assert signal["participant"] == "Armando"
    assert signal["candidate_ids"] == [candidate["candidate_id"]]
    assert signal["candidate_statuses"] == ["review"]


def test_weak_group_intent_never_assigns_the_speaker_as_owner():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                4.0,
                "Armando",
                "I would like us to create an issue for zero confirmation.",
            )
        ]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert candidate["owner"] is None
    assert set(candidate["review_reasons"]) >= {"weak_intent_cue", "owner_unresolved"}


def test_independent_recall_surfaces_broader_named_intent_without_auto_candidate():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                4.0,
                "Armando",
                "I want to create an issue for zero confirmation.",
            )
        ]
    )

    assert ledger["candidates"] == []
    signals = ledger["intent_recall"]["signals"]
    assert len(signals) == 1
    assert signals[0]["candidate_ids"] == []
    assert ledger["intent_recall"]["summary"]["unmatched_signals"] == 1


def test_recall_does_not_treat_discourse_wishes_or_past_counterfactuals_as_actions():
    ledger = build_action_ledger(
        [
            _segment(0.0, 2.0, "Riley", "I would have liked to create an issue for zero confirmation."),
            _segment(3.0, 5.0, "Riley", "I want to add one thing before we continue."),
            _segment(6.0, 8.0, "Riley", "I wish we could create an issue for zero confirmation."),
        ]
    )

    assert ledger["candidates"] == []
    assert ledger["intent_recall"]["signals"] == []


def test_conditional_weak_intent_is_preserved_for_review_with_the_condition_flag():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                5.0,
                "Riley",
                "If the response stays hidden, I would like to create an issue for zero confirmation.",
            )
        ]
    )

    candidate = ledger["candidates"][0]
    signal = ledger["intent_recall"]["signals"][0]
    assert candidate["status"] == "review"
    assert "conditional_or_hypothetical" in candidate["review_reasons"]
    assert signal["cue_kind"] == "conditional_self_intent"


def test_explicit_gonna_commitment_with_business_requirements_is_publishable():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                8.0,
                "John",
                "I'm gonna work on what the business needs and get the requirements ready.",
            )
        ]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "accepted"
    assert candidate["owner"] == "John"
    assert candidate["attributes"]["topics"] == ["business_requirements"]


def test_let_me_invoice_review_is_publishable_for_a_trusted_speaker():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                8.0,
                "Billy",
                "Let me review the last invoice and calculate the development and support percentages.",
            )
        ]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "accepted"
    assert candidate["owner"] == "Billy"
    assert candidate["attributes"]["topics"] == ["invoice_review"]


def test_conversational_let_me_cue_is_not_a_follow_up_commitment():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                8.0,
                "Billy",
                "Let me think, we are going to have a merchant use the API gateway.",
            ),
            _segment(
                9.0,
                15.0,
                "Billy",
                "Let me ask you about the latest invoice.",
            ),
        ]
    )

    assert ledger["candidates"] == []


def test_idea_framed_let_me_is_not_a_follow_up_commitment():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                8.0,
                "Billy",
                "Let me continue my my idea was we're going to build an API gateway.",
            )
        ]
    )

    assert ledger["candidates"] == []


def test_anonymous_self_commitment_never_becomes_an_accepted_owner():
    ledger = build_action_ledger(
        [_segment(0.0, 8.0, "Speaker 4", "I will send the meeting notes.")]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert candidate["owner"] is None
    assert "owner_unresolved" in candidate["review_reasons"]


def test_negated_self_commitment_is_not_an_action_candidate():
    ledger = build_action_ledger(
        [_segment(0.0, 8.0, "Billy", "I will not review the latest invoice.")]
    )

    assert ledger["candidates"] == []


def test_conditional_strong_commitment_stays_in_review():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                8.0,
                "Billy",
                "If Xin agrees, I will review the latest invoice.",
            )
        ]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert "conditional_or_hypothetical" in candidate["review_reasons"]


def test_questioned_named_assignment_never_becomes_accepted():
    ledger = build_action_ledger(
        [
            _segment(
                0.0,
                8.0,
                "Alex",
                "Do you think Billy will review the latest invoice?",
            ),
            _segment(9.0, 12.0, "Billy", "I am still checking the numbers."),
        ]
    )

    candidate = ledger["candidates"][0]
    assert candidate["status"] == "review"
    assert candidate["owner"] == "Billy"
    assert "owner_acceptance_unverified" in candidate["review_reasons"]
