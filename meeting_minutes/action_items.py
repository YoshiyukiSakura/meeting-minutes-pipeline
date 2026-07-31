from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from .identity_authority import ACTIVE_SPEAKER_HIGHLIGHT_SOURCE
from .time_utils import format_ts

LEDGER_FORMAT = "meeting-minutes/action-ledger-v2"
ACTION_INTENT_AUDIT_FORMAT = "meeting-minutes/action-intent-audit-v1"
MAX_EVIDENCE_SEGMENTS = 8
MAX_EVIDENCE_SECONDS = 45.0

_SELF_COMMITMENT = re.compile(
    r"\b(?:i\s+(?:will|need\s+to|plan\s+to|am\s+going\s+to)|i['’]ll|"
    r"i['’]m\s+(?:going\s+to|gonna)|let\s+me)\b|"
    r"(?:我会|我将|我要|我需要|我来|我负责)",
    re.IGNORECASE,
)
_CONVERSATIONAL_LET_ME = re.compile(
    r"^\s*let\s+me\s+(?:ask|explain|look|say|see|show|think)\b",
    re.IGNORECASE,
)
_IDEA_FRAMED_LET_ME = re.compile(
    r"^\s*let\s+me\s+(?:continue\s+)?(?:my\s+){0,2}idea\s+(?:was|is)\b|"
    r"^\s*my\s+(?:main\s+)?idea\s+(?:was|is)\b",
    re.IGNORECASE,
)
_NEGATED_SELF_COMMITMENT = re.compile(
    r"^\s*(?:"
    r"i\s+will\s+(?:not|never)|i\s+won['’]?t|"
    r"i['’]ll\s+(?:not|never)|i\s+am\s+not\s+going\s+to|"
    r"i['’]m\s+not\s+(?:going\s+to|gonna)|我(?:将|会)\s*不"
    r")\b",
    re.IGNORECASE,
)
_ACTION_VERB = (
    r"(?:create|file|open|prepare|update|deploy|investigate|review|fix|implement|"
    r"document|coordinate|schedule|follow\s+up(?:\s+on)?|work\s+on|move|migrate|"
    r"onboard|send|provide|calculate|test)"
)
# These cues are deliberately held for review. They express a concrete intent,
# but lack the strength of an explicit "I will" commitment.
_WEAK_SELF_ACTION_INTENT = re.compile(
    rf"\b(?:i\s+(?:would\s+like\s+to|intend\s+to)|i['’]d\s+like\s+to)\s+(?P<verb>{_ACTION_VERB})\b",
    re.IGNORECASE,
)
_WEAK_GROUP_ACTION_INTENT = re.compile(
    rf"\b(?:i\s+would\s+like\s+us\s+to|i['’]d\s+like\s+us\s+to|"
    rf"we\s+(?:need|plan|want)\s+to|we\s+want\s+them\s+to)\s+(?P<verb>{_ACTION_VERB})\b",
    re.IGNORECASE,
)
# This is intentionally broader than _WEAK_*_ACTION_INTENT. It is an
# independent recall audit, not the same matcher used to make candidates.
_ACTION_INTENT_RECALL = re.compile(
    rf"\b(?P<cue>i\s+(?:would\s+like\s+(?:us\s+)?to|intend\s+to|want\s+to)|"
    rf"i['’]d\s+like\s+(?:us\s+)?to)\s+(?P<verb>{_ACTION_VERB})\b",
    re.IGNORECASE,
)
_ASSURANCE_CLAUSE = re.compile(
    r"\b(?:make\s+sure|ensure|confirm|verify)\s+that\s+(?P<clause>.+)$",
    re.IGNORECASE,
)
_ASSURANCE_CLAUSE_PREDICATE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|will|would|"
    r"can|could|should|must|needs?|works?|runs?|passes?|fails?|completes?|"
    r"starts?|stops?|remains?|gets?|becomes?|[a-z]+(?:ed|ing))\b",
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
    if name and _trusted_named_segment(segment):
        return name
    speaker = str(segment.get("speaker") or "Speaker Unknown")
    return None if speaker == "Speaker Unknown" else speaker


def _trusted_named_segment(segment: dict[str, Any]) -> bool:
    """Return whether a segment has a real, evidence-backed display name."""

    name = str(segment.get("name") or "").strip()
    if not name or name.casefold().startswith("speaker ") or name.casefold() in {"unknown", "unresolved"}:
        return False
    if float(segment.get("name_confidence", 0.0) or 0.0) < 0.6:
        return False
    if str(segment.get("name_source") or "") == ACTIVE_SPEAKER_HIGHLIGHT_SOURCE:
        audit = segment.get("visual_identity_agent_audit")
        return isinstance(audit, dict) and audit.get("status") == "confirmed"
    return True


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
    if "gateway" in normalized or "网关" in normalized:
        topics.append("api_gateway")
    if re.search(r"\b(?:onboard|onboarding)\b", normalized) or "商户接入" in normalized:
        topics.append("merchant_onboarding")
    if re.search(r"\b(?:meeting notes?|minutes)\b", normalized) or "会议纪要" in normalized:
        topics.append("meeting_notes")
    if re.search(r"\bbusiness (?:needs?|requirements?)\b", normalized) or "业务需求" in normalized:
        topics.append("business_requirements")
    if "github" in normalized or "gitlab" in normalized or "git lab" in normalized:
        topics.append("source_control_migration")
    if "invoice" in normalized or "发票" in normalized or "账单" in normalized:
        topics.append("invoice_review")
    if "contract" in normalized or "合同" in normalized:
        topics.append("contract")
    if any(token in normalized for token in ("swagger", "postman", "documentation", "api docs", "文档")):
        topics.append("api_documentation")
    if "manual payout" in normalized or "手动付款" in normalized or "手动出款" in normalized:
        topics.append("manual_payout")
    if any(token in normalized for token in ("api key", "authentication", "merchant id", "认证", "商户标识")):
        topics.append("merchant_authentication")
    if any(token in normalized for token in (" kpi", "metrics", "code lines", "merge requests", "绩效", "度量")):
        topics.append("team_metrics")
    return list(dict.fromkeys(topics))


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
        suffix = text[match.start() :]
        if (
            _CONVERSATIONAL_LET_ME.match(suffix)
            or _IDEA_FRAMED_LET_ME.match(suffix)
            or _NEGATED_SELF_COMMITMENT.match(suffix)
        ):
            continue
        cue = normalize_text(match.group(0))
        if matches:
            previous = matches[-1]
            previous_cue = normalize_text(previous.group(0))
            if cue == previous_cue and match.start() - previous.start() <= 16:
                continue
        matches.append(match)
    return matches


def _weak_action_intent_matches(text: str) -> list[tuple[re.Match[str], str]]:
    """Return low-certainty action intent cues that must stay in review."""

    matches: list[tuple[re.Match[str], str]] = []
    for pattern, cue_kind in (
        (_WEAK_GROUP_ACTION_INTENT, "weak_group_intent"),
        (_WEAK_SELF_ACTION_INTENT, "weak_self_intent"),
    ):
        matches.extend((match, cue_kind) for match in pattern.finditer(text))
    return sorted(matches, key=lambda item: (item[0].start(), item[1]))


def _conditional_intent(text: str, cue_start: int) -> bool:
    """Treat an intent inside a recent if-clause as review-only context."""

    prefix = text[max(0, cue_start - 96) : cue_start]
    return bool(re.search(r"\bif\b[^.!?]{0,96}$", prefix, re.IGNORECASE))


def _questioned_or_reported_intent(text: str, cue_start: int) -> bool:
    """Detect a commitment cue embedded in another person's question or report."""

    prefix = text[max(0, cue_start - 120) : cue_start]
    return bool(
        re.search(
            r"\b(?:do|did|would|could|can)\s+you\s+"
            r"(?:think|expect|say|confirm)\b[^.!?]{0,100}$",
            prefix,
            re.IGNORECASE,
        )
    )


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


def _action_specs(
    text: str,
    speaker: str | None,
    known_owners: list[str],
    *,
    speaker_identity_trusted: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for match in _deduplicated_commitments(text):
        review_reasons: list[str] = []
        if _conditional_intent(text, match.start()):
            review_reasons.append("conditional_or_hypothetical")
        if _questioned_or_reported_intent(text, match.start()):
            review_reasons.append("questioned_or_reported_commitment")
        specs.append(
            {
                "start": match.start(),
                "match": match,
                "owner": speaker if speaker_identity_trusted else None,
                "owner_evidence": (
                    "speaker_self_commitment"
                    if speaker_identity_trusted
                    else "unresolved_speaker"
                ),
                "commitment_kind": "self",
                "force_review_reasons": review_reasons,
            }
        )
    for match, owner in _direct_assignment_matches(text, known_owners):
        specs.append(
            {
                "start": match.start(),
                "match": match,
                "owner": owner,
                "owner_evidence": "explicit_name_assignment",
                "commitment_kind": "direct_assignment",
                "force_review_reasons": ["owner_acceptance_unverified"],
            }
        )
    for match, cue_kind in _weak_action_intent_matches(text):
        group_intent = cue_kind == "weak_group_intent"
        owner = speaker if speaker_identity_trusted and not group_intent else None
        review_reasons = ["weak_intent_cue"]
        if group_intent:
            review_reasons.append("owner_unresolved")
        if _conditional_intent(text, match.start()):
            review_reasons.append("conditional_or_hypothetical")
        specs.append(
            {
                "start": match.start(),
                "match": match,
                "owner": owner,
                "owner_evidence": "speaker_weak_intent" if owner else "unresolved_weak_intent",
                "commitment_kind": "weak_intent",
                "intent_cue_kind": cue_kind,
                "force_review_reasons": review_reasons,
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
        "speaker_identity_trusted": _trusted_named_segment(segment),
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
        if _deduplicated_commitments(record["text"]) or _weak_action_intent_matches(record["text"]):
            break
        evidence.append(record)
    return sorted(evidence, key=lambda item: item["index"])


def _candidate_id(commitment_segment_id: str, quote: str) -> str:
    digest = hashlib.sha256(f"{commitment_segment_id}\x1f{quote}".encode()).hexdigest()
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


def _commitment_completeness_error(quote: str) -> str | None:
    """Hold incomplete ASR clauses for review instead of publishing an invented task."""

    assurance = _ASSURANCE_CLAUSE.search(normalize_text(quote))
    if assurance and not _ASSURANCE_CLAUSE_PREDICATE.search(assurance.group("clause")):
        return "commitment_incomplete"
    return None


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
    completeness_error = _commitment_completeness_error(quote)
    if completeness_error:
        errors.append(completeness_error)
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
    if commitment_kind == "weak_intent":
        cue_start = candidate.get("intent_cue_start")
        if not isinstance(cue_start, int) or not any(
            match.start() == cue_start for match, _cue_kind in _weak_action_intent_matches(commitment["text"])
        ):
            errors.append("weak_intent_cue_not_grounded")
    source_topics = set(candidate.get("commitment_attributes", candidate.get("attributes", {})).get("topics") or [])
    for item in evidence:
        item_topics = set(item["attributes"]["topics"])
        if item_topics and source_topics and item_topics.isdisjoint(source_topics):
            errors.append("cross_topic_evidence")
            break
    owner = candidate.get("owner")
    if candidate.get("owner_evidence") in {"speaker_self_commitment", "speaker_weak_intent"} and owner != commitment.get("speaker"):
        errors.append("owner_not_explicit_in_commitment")
    return errors


def _intent_signal_id(segment_id: str, cue_start: int, cue_kind: str) -> str:
    digest = hashlib.sha256(f"{segment_id}\x1f{cue_start}\x1f{cue_kind}".encode()).hexdigest()
    return f"intent-{digest[:16]}"


def build_action_intent_recall(
    segments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit high-confidence named speech for weak action intent independently.

    The recall expression is broader than the candidate matcher by design. A
    signal with no matching candidate is a release-blocking review item rather
    than proof that it should become an action automatically.
    """

    candidates_by_cue: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for candidate in candidates:
        cue_start = candidate.get("intent_cue_start")
        if candidate.get("commitment_kind") != "weak_intent" or not isinstance(cue_start, int):
            continue
        key = (str(candidate.get("commitment_segment_id") or ""), cue_start)
        candidates_by_cue.setdefault(key, []).append(candidate)

    signals: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not _trusted_named_segment(segment):
            continue
        text = str(segment.get("text") or "")
        segment_id = stable_segment_id(segment, index)
        participant = str(segment.get("name") or "").strip()
        for match in _ACTION_INTENT_RECALL.finditer(text):
            cue = normalize_text(match.group("cue"))
            cue_kind = "group_intent" if " us " in f" {cue} " else "self_intent"
            if _conditional_intent(text, match.start()):
                cue_kind = f"conditional_{cue_kind}"
            matching_candidates = sorted(
                candidates_by_cue.get((segment_id, match.start()), []),
                key=lambda candidate: str(candidate.get("candidate_id") or ""),
            )
            signals.append(
                {
                    "signal_id": _intent_signal_id(segment_id, match.start(), cue_kind),
                    "segment_id": segment_id,
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "participant": participant,
                    "cue_kind": cue_kind,
                    "cue": match.group(0),
                    "source_quote": text[match.start() :].strip(),
                    "candidate_ids": [str(candidate.get("candidate_id")) for candidate in matching_candidates],
                    "candidate_statuses": [str(candidate.get("status")) for candidate in matching_candidates],
                }
            )
    matched = sum(bool(signal["candidate_ids"]) for signal in signals)
    return {
        "format": ACTION_INTENT_AUDIT_FORMAT,
        "signals": signals,
        "summary": {
            "signals": len(signals),
            "matched_candidates": matched,
            "unmatched_signals": len(signals) - matched,
        },
    }


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
            fact_id = hashlib.sha256(f"{record['segment_id']}\x1f{topic}\x1f{downtime}".encode()).hexdigest()[:16]
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
        specs = _action_specs(
            record["text"],
            record["speaker"],
            known_owners,
            speaker_identity_trusted=bool(record["speaker_identity_trusted"]),
        )
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
            if spec.get("intent_cue_kind"):
                candidate["intent_cue_kind"] = spec["intent_cue_kind"]
                candidate["intent_cue_start"] = spec["start"]
            errors = validate_action_candidate(candidate)
            review_reasons: list[str] = []
            if not spec["owner"]:
                review_reasons.append("owner_unresolved")
            if not source_attributes["topics"]:
                review_reasons.append("topic_unresolved")
            elif len(source_attributes["topics"]) != 1:
                review_reasons.append("multiple_topics_in_commitment")
            review_reasons.extend(spec.get("force_review_reasons", []))
            if errors:
                review_reasons.extend(errors)
            if review_reasons:
                candidate["status"] = "review"
                candidate["review_reasons"] = sorted(set(review_reasons))
            candidates.append(candidate)
    constraints = _constraint_facts(records)
    intent_recall = build_action_intent_recall(segments, candidates)
    return {
        "format": LEDGER_FORMAT,
        "transcript_sha256": transcript_fingerprint(segments),
        "candidates": candidates,
        "constraints": constraints,
        "intent_recall": intent_recall,
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
    elif published_attributes["downtime"] is not None and (
        not allowed_downtime_values or published_attributes["downtime"] not in allowed_downtime_values
    ):
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

    intent_recall = ledger.get("intent_recall") or {}
    signals = intent_recall.get("signals") if isinstance(intent_recall, dict) else []
    lines += ["", "## Intent Recall Audit", ""]
    if isinstance(signals, list) and signals:
        for signal in signals:
            timestamp = f"{format_ts(float(signal['start']))}-{format_ts(float(signal['end']))}"
            candidate_ids = ", ".join(signal.get("candidate_ids", [])) or "unmatched"
            lines.append(
                f"- `{signal['signal_id']}` `{timestamp}` **{signal['participant']}** "
                f"[{signal['cue_kind']}; {candidate_ids}]: {signal['source_quote']}"
            )
    else:
        lines.append("- Empty.")
    return lines
