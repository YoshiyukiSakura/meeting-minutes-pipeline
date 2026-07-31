from meeting_minutes.minutes_contract import (
    validate_bilingual_minutes,
    validate_shareable_minutes,
    validate_shareable_minutes_en,
)


def _minutes() -> str:
    topics = []
    for index, time_range in enumerate(("00:00-02:00", "02:00-06:00", "06:00-12:01"), start=1):
        topics += [
            f"### {index}. 议题{index}（{time_range}）",
            "- 现状：已讨论当前状态。",
            "- 讨论结果：记录了本段讨论结果。",
            "",
        ]
    return "\n".join(
        [
            "# 会议纪要",
            "",
            "## 议题与结论",
            "",
            *topics,
            "## 项目进展",
            "",
            "| 时间点 | 参与者 | 项目 | 进展 |",
            "| --- | --- | --- | --- |",
            "| 00:00-02:00 | Riley | MPC | 已完成当前稳定性工作。 |",
            "",
            "## 已确认决定",
            "",
            "- 未出现可验证的最终决定。",
            "",
            "## 行动项",
            "",
            "| 时间点 | 事项 | 负责人 |",
            "| --- | --- | --- |",
            "| 00:00-02:00 | 已接受的行动项 | Riley |",
            "",
        ]
    )


def _english_minutes() -> str:
    topics = []
    for index, time_range in enumerate(("00:00-02:00", "02:00-06:00", "06:00-12:01"), start=1):
        topics += [
            f"### {index}. Topic {index} ({time_range})",
            "- Current state: The current state was discussed.",
            "- Outcome: The outcome of this discussion was recorded.",
            "",
        ]
    return "\n".join(
        [
            "# Meeting Minutes",
            "",
            "## Topics and Outcomes",
            "",
            *topics,
            "## Project Updates",
            "",
            "| Time | Participant | Project | Update |",
            "| --- | --- | --- | --- |",
            "| 00:00-02:00 | Riley | MPC | Completed the current stability work. |",
            "",
            "## Confirmed Decisions",
            "",
            "- No verified final decision was made.",
            "",
            "## Action Items",
            "",
            "| Time | Item | Owner |",
            "| --- | --- | --- |",
            "| 00:00-02:00 | Complete the accepted action item. | Riley |",
            "",
        ]
    )


def test_shareable_minutes_contract_accepts_complete_chronological_minutes():
    assert validate_shareable_minutes(_minutes(), duration=721.0) == []
    assert validate_shareable_minutes_en(_english_minutes(), duration=721.0) == []
    assert validate_bilingual_minutes(_minutes(), _english_minutes(), duration=721.0) == []


def test_shareable_minutes_contract_rejects_sparse_or_forbidden_minutes():
    removed_topic = "### 3. 议题3（06:00-12:01）\n- 现状：已讨论当前状态。\n- 讨论结果：记录了本段讨论结果。\n\n"
    forbidden_section = "## 基本信息\n\n- 不应出现在可分享纪要。\n\n## 已确认决定"
    minutes = _minutes().replace(removed_topic, "")
    minutes = minutes.replace("## 已确认决定", forbidden_section)

    errors = validate_shareable_minutes(minutes, duration=5401.0)

    assert "forbidden_heading:## 基本信息" in errors
    assert "topic_coverage_too_sparse:2<4" in errors


def test_shareable_minutes_contract_accepts_semantic_themes_for_a_long_meeting():
    themes = []
    for index, time_range in enumerate(
        (
            "00:05:00-00:20:00",
            "00:35:00-00:50:00",
            "01:05:00-01:20:00",
            "01:30:00-01:45:00",
            "02:00:00-02:20:00",
        ),
        start=1,
    ):
        themes += [
            f"### {index}. 语义主题{index}（{time_range}）",
            "- 现状：按业务主题归纳当前状态。",
            "- 讨论结果：保留关键观点、说话人与时间点。",
            "",
        ]
    prefix, remainder = _minutes().split("### 1.", maxsplit=1)
    _old_topics, suffix = remainder.split("## 项目进展", maxsplit=1)
    minutes = f"{prefix}{chr(10).join(themes)}## 项目进展{suffix}"

    assert validate_shareable_minutes(minutes, duration=8509.0) == []


def test_shareable_minutes_contract_rejects_unresolved_owner_or_schedule_in_owner_column():
    minutes = _minutes().replace(
        "| 00:00-02:00 | 已接受的行动项 | Riley |",
        "| 00:00-02:00 | 已接受的行动项 | Speaker 1（本周末） |",
    )

    errors = validate_shareable_minutes(minutes, duration=721.0)

    assert "action_items_owner_unresolved:1" in errors
    assert "action_items_owner_contains_schedule:1" in errors


def test_shareable_minutes_contract_rejects_invalid_or_out_of_bounds_action_times():
    original_row = "| 00:00-02:00 | 已接受的行动项 | Riley |"
    invalid_range = _minutes().replace(original_row, "| 02:00-01:00 | 已接受的行动项 | Riley |")
    out_of_bounds = _minutes().replace(original_row, "| 00:00-12:30 | 已接受的行动项 | Riley |")

    invalid_errors = validate_shareable_minutes(invalid_range, duration=721.0)
    out_of_bounds_errors = validate_shareable_minutes(out_of_bounds, duration=721.0)

    assert "action_items_time_range_invalid:1" in invalid_errors
    assert "action_items_time_range_out_of_bounds:1" in out_of_bounds_errors


def test_shareable_minutes_contract_rejects_an_unformatted_topic_heading():
    minutes = _minutes().replace("### 2. 议题2（02:00-06:00）", "### 没有时间范围的标题")

    errors = validate_shareable_minutes(minutes, duration=721.0)

    assert any(error.startswith("topic_heading_invalid:") for error in errors)


def test_bilingual_contract_rejects_an_english_action_row_with_a_different_owner():
    english = _english_minutes().replace("| 00:00-02:00 | Complete the accepted action item. | Riley |", "| 00:00-02:00 | Complete the accepted action item. | Morgan |")

    errors = validate_bilingual_minutes(_minutes(), english, duration=721.0)

    assert "bilingual_action_owner_mismatch:1" in errors


def test_bilingual_contract_allows_an_explicit_unassigned_action_owner():
    chinese = _minutes().replace(
        "| 00:00-02:00 | 已接受的行动项 | Riley |",
        "| 00:00-02:00 | 已接受的行动项 | 待确认 |",
    )
    english = _english_minutes().replace(
        "| 00:00-02:00 | Complete the accepted action item. | Riley |",
        "| 00:00-02:00 | Complete the accepted action item. | To confirm |",
    )

    assert validate_bilingual_minutes(chinese, english, duration=721.0) == []


def test_bilingual_contract_rejects_a_project_update_with_a_different_participant():
    english = _english_minutes().replace(
        "| 00:00-02:00 | Riley | MPC | Completed the current stability work. |",
        "| 00:00-02:00 | Morgan | MPC | Completed the current stability work. |",
    )

    errors = validate_bilingual_minutes(_minutes(), english, duration=721.0)

    assert "bilingual_project_update_participant_mismatch:1" in errors


def test_shareable_minutes_contract_requires_the_v4_project_updates_section():
    minutes = _minutes().replace(
        "## 项目进展\n\n| 时间点 | 参与者 | 项目 | 进展 |\n| --- | --- | --- | --- |\n| 00:00-02:00 | Riley | MPC | 已完成当前稳定性工作。 |\n\n",
        "",
    )

    errors = validate_shareable_minutes(minutes, duration=721.0)

    assert "contract_version_mismatch:project_updates_v4_required" in errors
