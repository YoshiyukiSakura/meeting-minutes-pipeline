from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from .time_utils import format_ts


LEDGER_FORMAT = "meeting-minutes/action-ledger-v2"
MAX_EVIDENCE_SEGMENTS = 8
MAX_EVIDENCE_SECONDS = 45.0

_SELF_COMMITMENT = re.compile(
    r"\b(?:i\s+(?:will|need\s+to|plan\s+to|am\s+going\s+to)|i['’]ll|i['’]m\s+going\s+to)\b|"
    r"(?:我会|我将|我要|我需要|我来|我负责)",
    re.IGNORECASE,
)
_TIME_UNIT = r"(?:hours?|hrs?|hr|h|minutes?|mins?|min)"
_ENGLISH_NUMBER_TOKEN = (
    r"(?:a|an|half|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|and)"
)
_DURATION = re.compile(
    rf"\b(?P<numeric_amount>\d+(?:\.\d+)?)\s*(?P<numeric_unit>{_TIME_UNIT})\b|"
    rf"\b(?P<word_amount>(?:{_ENGLISH_NUMBER_TOKEN}\s+)*{_ENGLISH_NUMBER_TOKEN})\s*"
    rf"(?P<word_unit>{_TIME_UNIT})\b|"
    r"(?P<chinese_half>半)\s*(?:个)?(?P<chinese_half_unit>小时|钟头|分钟|分)|"
    r"(?P<chinese_amount>[零〇一二两三四五六七八九十百千万\d]+)\s*(?:个|多)?\s*"
    r"(?P<chinese_unit>小时|钟头|分钟|分)",
    re.IGNORECASE,
)
_DURATION_SIGNAL = re.compile(
    rf"\b(?:\d+(?:\.\d+)?|(?:{_ENGLISH_NUMBER_TOKEN}|couple|few|several|dozen)"
    rf"(?:\s+(?:{_ENGLISH_NUMBER_TOKEN}|couple|few|several|dozen|of))*)\s*{_TIME_UNIT}\b|"
    r"(?:半|[零〇一二两三四五六七八九十百千万\d]+)\s*(?:个|多)?\s*(?:小时|钟头|分钟|分)",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def normalize_text(value: str) -> str:
    """Normalize transcript text for deterministic evidence comparisons."""

    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = normalized.replace("’", "'").replace("-", " ")
    normalized = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def stable_segment_id(segment: dict[str, Any], _index: int) -> str:
    """Create an ID stable across name relabeling but sensitive to transcript changes."""

    payload = "\x1f".join(
        [
            f"{float(segment.get('start', 0.0)):.3f}",
            f"{float(segment.get('end', 0.0)):.3f}",
            str(segment.get("speaker") or "Speaker Unknown"),
            str(segment.get("text") or ""),
        ]
    )
    return f"seg-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def transcript_fingerprint(segments: list[dict[str, Any]]) -> str:
    """Fingerprint the exact transcript state used to create an action ledger."""

    encoded = json.dumps(segments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _speaker_label(segment: dict[str, Any]) -> str | None:
    name = str(segment.get("name") or "").strip()
    if name and float(segment.get("name_confidence", 0.0)) >= 0.6:
        return name
    speaker = str(segment.get("speaker") or "Speaker Unknown")
    return None if speaker == "Speaker Unknown" else speaker


def _topic_groups(text: str) -> list[str]:
    normalized = normalize_text(text)
    topics: list[str] = []
    if re.search(r"(?<![a-z0-9])mpc(?![a-z0-9])", normalized):
        topics.append("mpc")
    if any(token in normalized for token in ("private key", "private keys", "encrypt", "encryption", "私钥", "加密")):
        topics.append("private_key_encryption")
    elif re.search(r"(?<![a-z0-9])bps(?![a-z0-9])", normalized):
        topics.append("bps")
    if "zero confirmation" in normalized or "零确认" in normalized:
        topics.append("zero_confirmation")
    return topics


def _english_amount(value: str) -> float | None:
    tokens = [token for token in normalize_text(value).split() if token not in {"and", "a", "an"}]
    if not tokens:
        return 1.0
    if tokens == ["half"]:
        return 0.5
    if "half" in tokens:
        return None
    total = 0.0
    current = 0.0
    for token in tokens:
        if token == "hundred":
            current = max(current, 1.0) * 100
            continue
        number = _NUMBER_WORDS.get(token)
        if number is None:
            return None
        current += number
    return total + current


def _chinese_amount(value: str) -> float | None:
    normalized = normalize_text(value)
    if normalized.isdigit():
        return float(normalized)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    current = 0
    for char in normalized:
        if char in digits:
            current = digits[char]
        elif char in units:
            if current == 0:
                current = 1
            total += current * units[char]
            current = 0
        else:
            return None
    return float(total + current)


def _duration_unit_minutes(unit: str) -> float:
    normalized = normalize_text(unit)
    return 60.0 if normalized in {"hour", "hours", "hr", "hrs", "h", "小时", "钟头"} else 1.0


def _duration_attributes(text: str) -> tuple[list[int], list[str]]:
    normalized = normalize_text(text)
    values: list[int] = []
    parsed_spans: list[tuple[int, int]] = []
    for match in _DURATION.finditer(normalized):
        amount: float | None
        unit: str | None
        if match.group("numeric_amount"):
            amount = float(match.group("numeric_amount"))
            unit = match.group("numeric_unit")
        elif match.group("word_amount"):
            amount = _english_amount(match.group("word_amount"))
            unit = match.group("word_unit")
        elif match.group("chinese_half"):
            amount = 0.5
            unit = match.group("chinese_half_unit")
        else:
            amount = _chinese_amount(match.group("chinese_amount") or "")
            unit = match.group("chinese_unit")
        if amount is None or unit is None or amount <= 0:
            continue
        values.append(round(amount * _duration_unit_minutes(unit)))
        parsed_spans.append(match.span())

    unparsed: list[str] = []
    for match in _DURATION_SIGNAL.finditer(normalized):
        start, end = match.span()
        if not any(parsed_start <= start and end <= parsed_end for parsed_start, parsed_end in parsed_spans):
            unparsed.append(match.group(0))
    return sorted(set(values)), sorted(set(unparsed))


def _downtime_value(text: str) -> bool | None:
    normalized = normalize_text(text)
    if re.search(
        r"(?:does not|doesnt|doesn t|do not|dont|don t) require (?:a )?downtime|"
        r"no downtime|without downtime|不需要停机|无需停机",
        normalized,
    ):
        return False
    if re.search(r"(?:requires?|needs?) (?:a )?downtime|需要停机|需停机", normalized):
        return True
    return None


def extract_attributes(text: str) -> dict[str, Any]:
    duration_minutes, unparsed_duration_mentions = _duration_attributes(text)
    return {
        "topics": _topic_groups(text),
        "duration_minutes": duration_minutes,
        "unparsed_duration_mentions": unparsed_duration_mentions,
        "downtime": _downtime_value(text),
    }


def _deduplicated_commitments(text: str) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in _SELF_COMMITMENT.finditer(text):
        cue = normalize_text(match.group(0))
        if matches:
            previous = matches[-1]
            previous_cue = normalize_text(previous.group(0))
            if cue == previous_cue and match.start() - previous.start() <= 16:
                continue
        matches.append(match)
    return matches


def _direct_assignment_matches(text: str, known_owners: list[str]) -> list[tuple[re.Match[str], str]]:
    if not known_owners:
        return []
    canonical_names = {normalize_text(name): name for name in known_owners}
    alternatives = "|".join(re.escape(name) for name in sorted(known_owners, key=len, reverse=True))
    pattern = re.compile(
        rf"\b(?P<owner>{alternatives})\s+(?:will|should|needs?\s+to|is\s+going\s+to)\b",
        re.IGNORECASE,
    )
    matches: list[tuple[re.Match[str], str]] = []
    for match in pattern.finditer(text):
        owner = canonical_names.get(normalize_text(match.group("owner")))
        if owner:
            matches.append((match, owner))
    return matches


def _action_specs(text: str, speaker: str | None, known_owners: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    specs.extend(
        {
            "start": match.start(),
            "match": match,
            "owner": speaker,
            "owner_evidence": "speaker_self_commitment" if speaker else "unresolved_speaker",
            "commitment_kind": "self",
        }
        for match in _deduplicated_commitments(text)
    )
    for match, owner in _direct_assignment_matches(text, known_owners):
        specs.append(
            {
                "start": match.start(),
                "match": match,
                "owner": owner,
                "owner_evidence": "explicit_name_assignment",
                "commitment_kind": "direct_assignment",
            }
        )
    return sorted(specs, key=lambda item: (item["start"], item["owner_evidence"]))


def _segment_record(segment: dict[str, Any], index: int) -> dict[str, Any]:
    text = str(segment.get("text") or "")
    segment_id = stable_segment_id(segment, index)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "segment_id": segment_id,
        "content_hash": content_hash,
        "index": index,
        "start": float(segment.get("start", 0.0)),
        "end": float(segment.get("end", 0.0)),
        "speaker": _speaker_label(segment),
        "text": text,
        "attributes": extract_attributes(text),
    }


def _evidence_window(records: list[dict[str, Any]], index: int, topics: list[str]) -> list[dict[str, Any]]:
    """Collect a bounded same-topic context without crossing a topic switch."""

    evidence = [records[index]]
    if not topics:
        return evidence
    expected_topics = set(topics)
    for current_index in range(index - 1, max(-1, index - MAX_EVIDENCE_SEGMENTS), -1):
        record = records[current_index]
        if evidence[0]["end"] - record["start"] > MAX_EVIDENCE_SECONDS:
            break
        previous_topics = set(record["attributes"]["topics"])
        if previous_topics and previous_topics.isdisjoint(expected_topics):
            break
        if _deduplicated_commitments(record["text"]):
            break
        evidence.append(record)
    return sorted(evidence, key=lambda item: item["index"])


def _candidate_id(commitment_segment_id: str, quote: str) -> str:
    digest = hashlib.sha256(f"{commitment_segment_id}\x1f{quote}".encode("utf-8")).hexdigest()
    return f"action-{digest[:16]}"


def _candidate_attributes(
    evidence: Iterable[dict[str, Any]],
    source_attributes: dict[str, Any],
    commitment_segment_id: str,
) -> dict[str, Any]:
    """Keep properties only when their source names the action's exact topic.

    Adjacent speech such as "two hours to do that" has no stable subject. It can
    remain evidence context, but it must never add duration or downtime to a
    differently named commitment.
    """

    expected_topics = set(source_attributes["topics"])
    topics = set(expected_topics)
    durations = set(source_attributes["duration_minutes"])
    unparsed_durations = set(source_attributes["unparsed_duration_mentions"])
    downtime = source_attributes["downtime"]
    for record in evidence:
        if record["segment_id"] == commitment_segment_id:
            continue
        attributes = record["attributes"]
        record_topics = set(attributes["topics"])
        # Attribute inheritance requires a full explicit topic match, not mere
        # temporal adjacency or an ambiguous pronoun.
        if not expected_topics or record_topics != expected_topics:
            continue
        durations.update(attributes["duration_minutes"])
        unparsed_durations.update(attributes["unparsed_duration_mentions"])
        if attributes["downtime"] is not None:
            downtime = bool(attributes["downtime"])
    return {
        "topics": sorted(topics),
        "duration_minutes": sorted(durations),
        "unparsed_duration_mentions": sorted(unparsed_durations),
        "downtime": downtime,
    }


def validate_action_candidate(candidate: dict[str, Any]) -> list[str]:
    """Return deterministic errors for a candidate before it can be rendered."""

    errors: list[str] = []
    evidence = list(candidate.get("evidence") or [])
    commitment_id = candidate.get("commitment_segment_id")
    commitment = next((item for item in evidence if item["segment_id"] == commitment_id), None)
    if commitment is None:
        return ["missing_commitment_segment"]
    quote = str(candidate.get("source_quote") or "")
    if not quote or normalize_text(quote) not in normalize_text(commitment["text"]):
        errors.append("quote_not_grounded")
    if len(evidence) > MAX_EVIDENCE_SEGMENTS:
        errors.append("evidence_span_too_wide")
    if evidence and evidence[-1]["end"] - evidence[0]["start"] > MAX_EVIDENCE_SECONDS:
        errors.append("evidence_span_too_long")
    commitment_kind = candidate.get("commitment_kind")
    commitment_count = sum(len(_deduplicated_commitments(item["text"])) for item in evidence)
    if commitment_kind == "self" and commitment_count != 1:
        errors.append("multiple_commitments_in_evidence")
    if commitment_kind == "direct_assignment":
        owner = str(candidate.get("owner") or "")
        assignment = re.compile(
            rf"\b{re.escape(owner)}\s+(?:will|should|needs?\s+to|is\s+going\s+to)\b",
            re.IGNORECASE,
        )
        if not owner or not assignment.search(commitment["text"]):
            errors.append("owner_not_explicit_in_commitment")
    source_topics = set(candidate.get("commitment_attributes", candidate.get("attributes", {})).get("topics") or [])
    for item in evidence:
        item_topics = set(item["attributes"]["topics"])
        if item_topics and source_topics and item_topics.isdisjoint(source_topics):
            errors.append("cross_topic_evidence")
            break
    owner = candidate.get("owner")
    if candidate.get("owner_evidence") == "speaker_self_commitment" and owner != commitment.get("speaker"):
        errors.append("owner_not_explicit_in_commitment")
    return errors


def _constraint_facts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for record in records:
        downtime = record["attributes"]["downtime"]
        if downtime is None:
            continue
        topics = set(record["attributes"]["topics"])
        topic_evidence_id = record["segment_id"]
        # A subjectless statement such as "which does not require downtime" is
        # semantically ambiguous even when it follows a named action. Keep it out
        # of automatic facts; it belongs in human review instead of a constraint.
        if len(topics) != 1:
            continue
        for topic in topics:
            fact_id = hashlib.sha256(f"{record['segment_id']}\x1f{topic}\x1f{downtime}".encode("utf-8")).hexdigest()[:16]
            facts.append(
                {
                    "fact_id": f"constraint-{fact_id}",
                    "topic": topic,
                    "attribute": "downtime",
                    "value": downtime,
                    "source_segment_id": record["segment_id"],
                    "topic_evidence_segment_id": topic_evidence_id,
                    "source_quote": record["text"],
                    "content_hash": record["content_hash"],
                }
            )
    return facts


def build_action_ledger(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract atomic, source-grounded action candidates from a transcript."""

    records = [_segment_record(segment, index) for index, segment in enumerate(segments)]
    known_owners = sorted({str(record["speaker"]) for record in records if record.get("speaker")})
    candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        specs = _action_specs(record["text"], record["speaker"], known_owners)
        for spec_index, spec in enumerate(specs):
            quote_end = specs[spec_index + 1]["start"] if spec_index + 1 < len(specs) else len(record["text"])
            quote = record["text"][spec["start"] : quote_end].strip(" \t,;:.-")
            if not quote:
                continue
            source_attributes = extract_attributes(quote)
            evidence = _evidence_window(records, index, source_attributes["topics"])
            attributes = _candidate_attributes(evidence, source_attributes, record["segment_id"])
            candidate = {
                "candidate_id": _candidate_id(record["segment_id"], quote),
                "status": "accepted",
                "commitment_segment_id": record["segment_id"],
                "start": record["start"],
                "end": record["end"],
                "owner": spec["owner"],
                "owner_evidence": spec["owner_evidence"],
                "commitment_kind": spec["commitment_kind"],
                "source_quote": quote,
                "commitment_attributes": source_attributes,
                "attributes": attributes,
                "evidence": evidence,
            }
            errors = validate_action_candidate(candidate)
            review_reasons: list[str] = []
            if not spec["owner"]:
                review_reasons.append("owner_unresolved")
            if not source_attributes["topics"]:
                review_reasons.append("topic_unresolved")
            elif len(source_attributes["topics"]) != 1:
                review_reasons.append("multiple_topics_in_commitment")
            if errors:
                review_reasons.extend(errors)
            if review_reasons:
                candidate["status"] = "review"
                candidate["review_reasons"] = sorted(set(review_reasons))
            candidates.append(candidate)
    constraints = _constraint_facts(records)
    return {
        "format": LEDGER_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "candidates": candidates,
        "constraints": constraints,
        "summary": {
            "accepted": sum(item["status"] == "accepted" for item in candidates),
            "review": sum(item["status"] == "review" for item in candidates),
            "constraints": len(constraints),
        },
    }


def validate_published_action_item(item: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    """Reject a rendered action row when it adds facts beyond its atomic evidence."""

    candidates = {candidate["candidate_id"]: candidate for candidate in ledger.get("candidates", [])}
    candidate = candidates.get(str(item.get("candidate_id") or ""))
    if candidate is None:
        return ["unknown_candidate"]
    errors = list(validate_action_candidate(candidate))
    if candidate.get("status") != "accepted":
        errors.append("candidate_not_publishable")
    if item.get("owner") != candidate.get("owner"):
        errors.append("owner_mismatch")
    quote = str(item.get("source_quote") or "")
    if normalize_text(quote) != normalize_text(str(candidate.get("source_quote") or "")):
        errors.append("published_quote_mismatch")

    published_text = str(item.get("text") or "")
    # This gate validates publishable records, not creative summaries. A rewrite
    # or translation needs human review because deterministic parsing cannot
    # prove that every semantic relation survived the rewrite.
    if normalize_text(published_text) != normalize_text(str(candidate.get("source_quote") or "")):
        errors.append("published_text_not_verbatim")

    published_attributes = extract_attributes(published_text)
    source_attributes = candidate.get("attributes") or {}
    commitment_attributes = candidate.get("commitment_attributes") or source_attributes
    source_topics = set(commitment_attributes.get("topics") or [])
    published_topics = set(published_attributes["topics"])
    # A curated row must state the same named subject as its commitment. This
    # prevents an unrecognised replacement subject from bypassing subset checks.
    if not published_topics:
        errors.append("published_topic_unresolved")
    elif not published_topics.issubset(source_topics):
        errors.append("topic_not_grounded")
    if published_attributes["unparsed_duration_mentions"]:
        errors.append("duration_unparsed")
    if not set(published_attributes["duration_minutes"]).issubset(set(source_attributes.get("duration_minutes") or [])):
        errors.append("duration_not_grounded")

    allowed_downtime_values = {
        value
        for value in [source_attributes.get("downtime")]
        if value is not None
    }
    allowed_downtime_values.update(
        fact["value"]
        for fact in ledger.get("constraints", [])
        if fact.get("attribute") == "downtime" and fact.get("topic") in source_topics and fact.get("value") is not None
    )
    if len(allowed_downtime_values) > 1:
        errors.append("downtime_constraint_conflict")
    elif published_attributes["downtime"] is not None:
        if not allowed_downtime_values or published_attributes["downtime"] not in allowed_downtime_values:
            errors.append("downtime_not_grounded")
    return sorted(set(errors))


def action_item_lines(ledger: dict[str, Any], *, heading: str = "## Action Items") -> list[str]:
    lines = [heading, ""]
    candidates = [item for item in ledger.get("candidates", []) if item.get("status") == "accepted"]
    for candidate in candidates:
        owner = candidate.get("owner") or "Unassigned"
        timestamp = f"{format_ts(float(candidate['start']))}-{format_ts(float(candidate['end']))}"
        lines.append(f"- `{timestamp}` **{owner}**: {candidate['source_quote']}")
    if not candidates:
        lines.append("- No action items passed the evidence gate.")
    return lines


def action_ledger_lines(ledger: dict[str, Any]) -> list[str]:
    lines = ["# Action Item Ledger", "", "## Publishable", ""]
    publishable = [item for item in ledger.get("candidates", []) if item.get("status") == "accepted"]
    for candidate in publishable:
        timestamp = f"{format_ts(float(candidate['start']))}-{format_ts(float(candidate['end']))}"
        topics = ", ".join(candidate.get("attributes", {}).get("topics", [])) or "unresolved"
        lines.append(f"- `{candidate['candidate_id']}` `{timestamp}` **{candidate.get('owner') or 'Unassigned'}** [{topics}]: {candidate['source_quote']}")
    if not publishable:
        lines.append("- Empty.")

    lines += ["", "## Needs Review", ""]
    pending = [item for item in ledger.get("candidates", []) if item.get("status") == "review"]
    for candidate in pending:
        timestamp = f"{format_ts(float(candidate['start']))}-{format_ts(float(candidate['end']))}"
        reasons = ", ".join(candidate.get("review_reasons", [])) or "review_required"
        lines.append(f"- `{candidate['candidate_id']}` `{timestamp}` [{reasons}]: {candidate['source_quote']}")
    if not pending:
        lines.append("- Empty.")
    return lines
