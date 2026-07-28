from __future__ import annotations

from dataclasses import dataclass
import math
import re


# Public minutes are a semantic digest, not a fixed-interval transcript index.
# The private transcript and evidence manifests retain the complete timeline.
TOPIC_COVERAGE_WINDOW_SECONDS = 1800.0
MAX_UNCOVERED_GAP_SECONDS = 2700.0

REQUIRED_HEADINGS = (
    "# 会议纪要",
    "## 议题与结论",
    "## 项目进展",
    "## 已确认决定",
    "## 行动项",
)

EN_REQUIRED_HEADINGS = (
    "# Meeting Minutes",
    "## Topics and Outcomes",
    "## Project Updates",
    "## Confirmed Decisions",
    "## Action Items",
)

PROJECT_UPDATE_TABLE_HEADER = "| 时间点 | 参与者 | 项目 | 进展 |"
PROJECT_UPDATE_TABLE_SEPARATOR = "| --- | --- | --- | --- |"
NO_PUBLISHABLE_PROJECT_UPDATES = "- 本次未出现可发布的项目进展。"
EN_PROJECT_UPDATE_TABLE_HEADER = "| Time | Participant | Project | Update |"
EN_PROJECT_UPDATE_TABLE_SEPARATOR = "| --- | --- | --- | --- |"
EN_NO_PUBLISHABLE_PROJECT_UPDATES = "- No publishable project updates were identified."
ACTION_TABLE_HEADER = "| 时间点 | 事项 | 负责人 |"
ACTION_TABLE_SEPARATOR = "| --- | --- | --- |"
NO_PUBLISHABLE_ACTIONS = "- 本次未出现可发布的明确行动项。"
EN_ACTION_TABLE_HEADER = "| Time | Item | Owner |"
EN_ACTION_TABLE_SEPARATOR = "| --- | --- | --- |"
EN_NO_PUBLISHABLE_ACTIONS = "- No publishable action items were identified."

FORBIDDEN_HEADINGS = (
    "## 基本信息",
    "## 身份边界",
    "## 风险与待确认",
    "## 证据",
    "## 关键帧",
    "## Source",
    "## Speaker Coverage",
    "## Key Frames",
    "## Basic Information",
    "## Identity Boundaries",
    "## Risks and Open Questions",
    "## Evidence",
)

_TOPIC_HEADING = re.compile(
    r"^###\s+\d+\.\s+.+（(?P<start>\d{2}:\d{2}(?::\d{2})?)-(?P<end>\d{2}:\d{2}(?::\d{2})?)）\s*$"
)
_EN_TOPIC_HEADING = re.compile(
    r"^(?P<prefix>###\s+\d+\.\s+.+)\s+\((?P<start>\d{2}:\d{2}(?::\d{2})?)-(?P<end>\d{2}:\d{2}(?::\d{2})?)\)\s*$"
)
_UNKNOWN_ACTION_OWNER = re.compile(
    r"(?:^|[\s（(])(?:speaker(?:[_\s-]*\d+)?|说话人(?:[_\s-]*\d+)?|unknown|unresolved|unassigned|"
    r"未确认|未知|未实名)(?:$|[\s）)])|未知|未实名|未确认",
    re.IGNORECASE,
)
_ACTION_TIME_RANGE = re.compile(r"^\d{2}:\d{2}(?::\d{2})?-\d{2}:\d{2}(?::\d{2})?$")
_OWNER_SCHEDULE_QUALIFIER = re.compile(
    r"[（(][^）)]*(?:今天|明天|当天|本周|下周|周末|月底|本月|day|week|month|today|tomorrow)[^）)]*[）)]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ShareableActionRow:
    """A validated action row from the fixed shareable minutes table."""

    index: int
    time_range: str
    start: float
    end: float
    item: str
    owner: str


@dataclass(frozen=True)
class ShareableProjectUpdateRow:
    """A validated project-update row from the fixed shareable minutes table."""

    index: int
    time_range: str
    start: float
    end: float
    participant: str
    project: str
    update: str


def _timestamp_seconds(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        if seconds >= 60:
            raise ValueError(f"Invalid timestamp: {value}")
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        if minutes >= 60 or seconds >= 60:
            raise ValueError(f"Invalid timestamp: {value}")
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Unsupported timestamp: {value}")


def _section_lines(lines: list[str], positions: dict[str, int], heading: str) -> list[str]:
    start = positions[heading] + 1
    following_positions = [position for position in positions.values() if position > positions[heading]]
    end = min(following_positions) if following_positions else len(lines)
    return lines[start:end]


def validate_shareable_minutes(markdown: str, *, duration: float = 0.0) -> list[str]:
    """Validate the fixed, shareable meeting-minutes format.

    The contract checks structure and timeline coverage only. It does not try to
    prove semantic claims, which remains the responsibility of cited transcript
    evidence and the deterministic action ledger.
    """

    lines = markdown.splitlines()
    errors: list[str] = []
    positions: dict[str, int] = {}
    for heading in REQUIRED_HEADINGS:
        matching = [index for index, line in enumerate(lines) if line.strip() == heading]
        if len(matching) != 1:
            errors.append(f"required_heading_count:{heading}:{len(matching)}")
            if heading == "## 项目进展" and not matching:
                errors.append("contract_version_mismatch:project_updates_v4_required")
        elif matching:
            positions[heading] = matching[0]
    if len(positions) == len(REQUIRED_HEADINGS):
        ordered = [positions[heading] for heading in REQUIRED_HEADINGS]
        if ordered != sorted(ordered):
            errors.append("required_headings_out_of_order")
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,2}\s+", stripped) and stripped not in REQUIRED_HEADINGS:
            errors.append(f"unexpected_top_level_heading:{stripped}")

    for heading in FORBIDDEN_HEADINGS:
        if any(line.strip() == heading for line in lines):
            errors.append(f"forbidden_heading:{heading}")

    if "## 议题与结论" not in positions:
        return errors
    if len(positions) != len(REQUIRED_HEADINGS):
        return errors

    topic_lines = _section_lines(lines, positions, "## 议题与结论")
    for index, line in enumerate(topic_lines, start=1):
        stripped = line.strip()
        if stripped.startswith("###") and not _TOPIC_HEADING.match(stripped):
            errors.append(f"topic_heading_invalid:{index}")
    headings = [(index, _TOPIC_HEADING.match(line.strip())) for index, line in enumerate(topic_lines)]
    headings = [(index, match) for index, match in headings if match]
    if not headings:
        errors.append("no_timed_topics")
        return errors

    topic_ranges: list[tuple[float, float]] = []
    for offset, (index, match) in enumerate(headings):
        assert match is not None
        end_index = headings[offset + 1][0] if offset + 1 < len(headings) else len(topic_lines)
        body = "\n".join(topic_lines[index + 1 : end_index])
        if "- 现状：" not in body:
            errors.append(f"topic_missing_current_state:{offset + 1}")
        if "- 讨论结果：" not in body:
            errors.append(f"topic_missing_discussion_result:{offset + 1}")
        try:
            start = _timestamp_seconds(match.group("start"))
            end = _timestamp_seconds(match.group("end"))
        except ValueError:
            errors.append(f"topic_timestamp_invalid:{offset + 1}")
            continue
        if end <= start:
            errors.append(f"topic_invalid_range:{offset + 1}")
        if duration > 0 and (start >= duration or end > duration + 0.5):
            errors.append(f"topic_range_out_of_bounds:{offset + 1}")
        topic_ranges.append((start, end))

    if not topic_ranges:
        return errors

    for index in range(1, len(topic_ranges)):
        previous_start, previous_end = topic_ranges[index - 1]
        start, _end = topic_ranges[index]
        if start < previous_start:
            errors.append(f"topic_not_chronological:{index + 1}")
        if start - previous_end > MAX_UNCOVERED_GAP_SECONDS:
            errors.append(f"topic_gap_too_large:{index + 1}")

    if duration > 0:
        minimum_topics = max(1, math.ceil(duration / TOPIC_COVERAGE_WINDOW_SECONDS))
        if len(topic_ranges) < minimum_topics:
            errors.append(f"topic_coverage_too_sparse:{len(topic_ranges)}<{minimum_topics}")
        if topic_ranges[0][0] > TOPIC_COVERAGE_WINDOW_SECONDS:
            errors.append("opening_coverage_missing")
        if topic_ranges[-1][1] < duration - TOPIC_COVERAGE_WINDOW_SECONDS:
            errors.append("closing_coverage_missing")

    project_update_lines = [
        line.strip() for line in _section_lines(lines, positions, "## 项目进展") if line.strip()
    ]
    if not project_update_lines:
        errors.append("project_updates_section_empty")
    elif project_update_lines[0] == NO_PUBLISHABLE_PROJECT_UPDATES:
        if len(project_update_lines) != 1:
            errors.append("no_project_updates_marker_mixed_with_rows")
    else:
        if len(project_update_lines) < 3:
            errors.append("project_updates_table_incomplete")
        else:
            if project_update_lines[0] != PROJECT_UPDATE_TABLE_HEADER:
                errors.append("project_updates_table_header_invalid")
            if project_update_lines[1] != PROJECT_UPDATE_TABLE_SEPARATOR:
                errors.append("project_updates_table_separator_invalid")
            previous_start: float | None = None
            rows = project_update_lines[2:]
            if not rows:
                errors.append("project_updates_table_empty")
            for row_index, row in enumerate(rows, start=1):
                if not row.startswith("|") or not row.endswith("|"):
                    errors.append(f"project_updates_row_not_table:{row_index}")
                    continue
                cells = [cell.strip() for cell in row[1:-1].split("|")]
                if len(cells) != 4:
                    errors.append(f"project_updates_row_column_count:{row_index}:{len(cells)}")
                    continue
                if any(not cell for cell in cells):
                    errors.append(f"project_updates_row_empty_cell:{row_index}")
                if _UNKNOWN_ACTION_OWNER.search(cells[1]):
                    errors.append(f"project_updates_participant_unresolved:{row_index}")
                if not _ACTION_TIME_RANGE.match(cells[0]):
                    errors.append(f"project_updates_time_range_invalid:{row_index}")
                    continue
                start_text, end_text = cells[0].split("-", maxsplit=1)
                try:
                    start = _timestamp_seconds(start_text)
                    end = _timestamp_seconds(end_text)
                except ValueError:
                    errors.append(f"project_updates_time_range_invalid:{row_index}")
                    continue
                if end <= start:
                    errors.append(f"project_updates_time_range_invalid:{row_index}")
                if duration > 0 and (start >= duration or end > duration + 0.5):
                    errors.append(f"project_updates_time_range_out_of_bounds:{row_index}")
                if previous_start is not None and start < previous_start:
                    errors.append(f"project_updates_not_chronological:{row_index}")
                previous_start = start

    decision_lines = [line.strip() for line in _section_lines(lines, positions, "## 已确认决定") if line.strip()]
    if not decision_lines:
        errors.append("decisions_section_empty")

    action_lines = [line.strip() for line in _section_lines(lines, positions, "## 行动项") if line.strip()]
    if not action_lines:
        errors.append("action_items_section_empty")
    elif action_lines[0] == NO_PUBLISHABLE_ACTIONS:
        if len(action_lines) != 1:
            errors.append("no_actions_marker_mixed_with_rows")
    else:
        if len(action_lines) < 3:
            errors.append("action_items_table_incomplete")
        else:
            if action_lines[0] != ACTION_TABLE_HEADER:
                errors.append("action_items_table_header_invalid")
            if action_lines[1] != ACTION_TABLE_SEPARATOR:
                errors.append("action_items_table_separator_invalid")
            rows = action_lines[2:]
            if not rows:
                errors.append("action_items_table_empty")
            for row_index, row in enumerate(rows, start=1):
                if not row.startswith("|") or not row.endswith("|"):
                    errors.append(f"action_items_row_not_table:{row_index}")
                    continue
                cells = [cell.strip() for cell in row[1:-1].split("|")]
                if len(cells) != 3:
                    errors.append(f"action_items_row_column_count:{row_index}:{len(cells)}")
                    continue
                if any(not cell for cell in cells):
                    errors.append(f"action_items_row_empty_cell:{row_index}")
                if not _ACTION_TIME_RANGE.match(cells[0]):
                    errors.append(f"action_items_time_range_invalid:{row_index}")
                else:
                    start_text, end_text = cells[0].split("-", maxsplit=1)
                    try:
                        start = _timestamp_seconds(start_text)
                        end = _timestamp_seconds(end_text)
                    except ValueError:
                        errors.append(f"action_items_time_range_invalid:{row_index}")
                    else:
                        if end <= start:
                            errors.append(f"action_items_time_range_invalid:{row_index}")
                        if duration > 0 and (start >= duration or end > duration + 0.5):
                            errors.append(f"action_items_time_range_out_of_bounds:{row_index}")
                if _UNKNOWN_ACTION_OWNER.search(cells[2]):
                    errors.append(f"action_items_owner_unresolved:{row_index}")
                if _OWNER_SCHEDULE_QUALIFIER.search(cells[2]):
                    errors.append(f"action_items_owner_contains_schedule:{row_index}")
    return errors


def parse_shareable_action_rows(markdown: str) -> list[ShareableActionRow]:
    """Return action rows after the shareable minutes contract has passed."""

    lines = markdown.splitlines()
    try:
        action_start = next(index for index, line in enumerate(lines) if line.strip() == "## 行动项")
    except StopIteration as exc:
        raise ValueError("action_items_section_missing") from exc
    action_lines = [line.strip() for line in lines[action_start + 1 :] if line.strip()]
    if not action_lines or action_lines[0] == NO_PUBLISHABLE_ACTIONS:
        return []
    if action_lines[:2] != [ACTION_TABLE_HEADER, ACTION_TABLE_SEPARATOR]:
        raise ValueError("action_items_table_invalid")

    rows: list[ShareableActionRow] = []
    for row_index, row in enumerate(action_lines[2:], start=1):
        if not row.startswith("|") or not row.endswith("|"):
            raise ValueError(f"action_items_row_not_table:{row_index}")
        cells = [cell.strip() for cell in row[1:-1].split("|")]
        if len(cells) != 3:
            raise ValueError(f"action_items_row_column_count:{row_index}:{len(cells)}")
        start_text, end_text = cells[0].split("-", maxsplit=1)
        rows.append(
            ShareableActionRow(
                index=row_index,
                time_range=cells[0],
                start=_timestamp_seconds(start_text),
                end=_timestamp_seconds(end_text),
                item=cells[1],
                owner=cells[2],
            )
        )
    return rows


def parse_shareable_project_update_rows(markdown: str) -> list[ShareableProjectUpdateRow]:
    """Return project-update rows after the shareable minutes contract has passed."""

    lines = markdown.splitlines()
    try:
        project_updates_start = next(
            index for index, line in enumerate(lines) if line.strip() == "## 项目进展"
        )
    except StopIteration as exc:
        raise ValueError("project_updates_section_missing") from exc
    try:
        following_start = next(
            index
            for index, line in enumerate(lines[project_updates_start + 1 :], start=project_updates_start + 1)
            if line.strip().startswith("## ")
        )
    except StopIteration:
        following_start = len(lines)
    project_update_lines = [
        line.strip() for line in lines[project_updates_start + 1 : following_start] if line.strip()
    ]
    if not project_update_lines or project_update_lines[0] == NO_PUBLISHABLE_PROJECT_UPDATES:
        return []
    if project_update_lines[:2] != [PROJECT_UPDATE_TABLE_HEADER, PROJECT_UPDATE_TABLE_SEPARATOR]:
        raise ValueError("project_updates_table_invalid")

    rows: list[ShareableProjectUpdateRow] = []
    for row_index, row in enumerate(project_update_lines[2:], start=1):
        if not row.startswith("|") or not row.endswith("|"):
            raise ValueError(f"project_updates_row_not_table:{row_index}")
        cells = [cell.strip() for cell in row[1:-1].split("|")]
        if len(cells) != 4:
            raise ValueError(f"project_updates_row_column_count:{row_index}:{len(cells)}")
        start_text, end_text = cells[0].split("-", maxsplit=1)
        rows.append(
            ShareableProjectUpdateRow(
                index=row_index,
                time_range=cells[0],
                start=_timestamp_seconds(start_text),
                end=_timestamp_seconds(end_text),
                participant=cells[1],
                project=cells[2],
                update=cells[3],
            )
        )
    return rows


def _normalize_english_minutes(markdown: str) -> str:
    """Translate fixed English structure into the internal contract vocabulary."""

    exact_lines = {
        "# Meeting Minutes": "# 会议纪要",
        "## Topics and Outcomes": "## 议题与结论",
        "## Project Updates": "## 项目进展",
        "## Confirmed Decisions": "## 已确认决定",
        "## Action Items": "## 行动项",
        EN_PROJECT_UPDATE_TABLE_HEADER: PROJECT_UPDATE_TABLE_HEADER,
        EN_PROJECT_UPDATE_TABLE_SEPARATOR: PROJECT_UPDATE_TABLE_SEPARATOR,
        EN_NO_PUBLISHABLE_PROJECT_UPDATES: NO_PUBLISHABLE_PROJECT_UPDATES,
        EN_ACTION_TABLE_HEADER: ACTION_TABLE_HEADER,
        EN_ACTION_TABLE_SEPARATOR: ACTION_TABLE_SEPARATOR,
        EN_NO_PUBLISHABLE_ACTIONS: NO_PUBLISHABLE_ACTIONS,
    }
    prefixes = {
        "- Current state:": "- 现状：",
        "- Outcome:": "- 讨论结果：",
        "- Next step:": "- 下一步：",
    }
    normalized_lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped in exact_lines:
            normalized_lines.append(exact_lines[stripped])
            continue
        for english_prefix, chinese_prefix in prefixes.items():
            if stripped.startswith(english_prefix):
                normalized_lines.append(chinese_prefix + stripped[len(english_prefix) :].lstrip())
                break
        else:
            match = _EN_TOPIC_HEADING.match(stripped)
            if match:
                normalized_lines.append(f"{match.group('prefix')}（{match.group('start')}-{match.group('end')}）")
            else:
                normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _english_heading_errors(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    positions: dict[str, int] = {}
    errors: list[str] = []
    for heading in EN_REQUIRED_HEADINGS:
        matches = [index for index, line in enumerate(lines) if line.strip() == heading]
        if len(matches) != 1:
            errors.append(f"english_required_heading_count:{heading}:{len(matches)}")
            if heading == "## Project Updates" and not matches:
                errors.append("english_contract_version_mismatch:project_updates_v4_required")
        elif matches:
            positions[heading] = matches[0]
    if len(positions) == len(EN_REQUIRED_HEADINGS):
        ordered = [positions[heading] for heading in EN_REQUIRED_HEADINGS]
        if ordered != sorted(ordered):
            errors.append("english_required_headings_out_of_order")
    return errors


def validate_shareable_minutes_en(markdown: str, *, duration: float = 0.0) -> list[str]:
    """Validate the fixed English companion document for shareable minutes."""

    errors = _english_heading_errors(markdown)
    errors.extend(validate_shareable_minutes(_normalize_english_minutes(markdown), duration=duration))
    return sorted(set(errors))


def parse_shareable_action_rows_en(markdown: str) -> list[ShareableActionRow]:
    """Return English action rows after ``validate_shareable_minutes_en`` passes."""

    return parse_shareable_action_rows(_normalize_english_minutes(markdown))


def parse_shareable_project_update_rows_en(markdown: str) -> list[ShareableProjectUpdateRow]:
    """Return English project-update rows after ``validate_shareable_minutes_en`` passes."""

    return parse_shareable_project_update_rows(_normalize_english_minutes(markdown))


def _topic_time_ranges(markdown: str) -> list[tuple[str, str]]:
    return [
        (match.group("start"), match.group("end"))
        for line in markdown.splitlines()
        if (match := _TOPIC_HEADING.match(line.strip()))
    ]


def validate_bilingual_minutes(
    chinese_markdown: str,
    english_markdown: str,
    *,
    duration: float = 0.0,
) -> list[str]:
    """Check both publication languages share the same non-negotiable timeline facts."""

    errors = [f"zh:{error}" for error in validate_shareable_minutes(chinese_markdown, duration=duration)]
    errors.extend(f"en:{error}" for error in validate_shareable_minutes_en(english_markdown, duration=duration))
    if errors:
        return sorted(set(errors))

    normalized_english = _normalize_english_minutes(english_markdown)
    if _topic_time_ranges(chinese_markdown) != _topic_time_ranges(normalized_english):
        errors.append("bilingual_topic_timeline_mismatch")

    chinese_rows = parse_shareable_action_rows(chinese_markdown)
    english_rows = parse_shareable_action_rows(normalized_english)
    if len(chinese_rows) != len(english_rows):
        errors.append("bilingual_action_row_count_mismatch")
        return errors
    for chinese_row, english_row in zip(chinese_rows, english_rows, strict=True):
        if chinese_row.time_range != english_row.time_range:
            errors.append(f"bilingual_action_time_mismatch:{chinese_row.index}")
        if chinese_row.owner != english_row.owner:
            errors.append(f"bilingual_action_owner_mismatch:{chinese_row.index}")

    chinese_project_rows = parse_shareable_project_update_rows(chinese_markdown)
    english_project_rows = parse_shareable_project_update_rows(normalized_english)
    if len(chinese_project_rows) != len(english_project_rows):
        errors.append("bilingual_project_update_row_count_mismatch")
        return sorted(set(errors))
    for chinese_row, english_row in zip(chinese_project_rows, english_project_rows, strict=True):
        if chinese_row.time_range != english_row.time_range:
            errors.append(f"bilingual_project_update_time_mismatch:{chinese_row.index}")
        if chinese_row.participant != english_row.participant:
            errors.append(f"bilingual_project_update_participant_mismatch:{chinese_row.index}")
    return sorted(set(errors))
