from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .identity_authority import (
    USER_CONFIRMED_SPEAKER_VOLUME_SOURCE,
    VOICE_ENROLLMENT_SOURCE,
    has_operator_locked_identity,
    is_protected_from_ocr_overwrite,
    name_source,
)
from .jsonio import read_json

UI_STOPWORDS = {
    "gmt",
    "random",
    "mute",
    "unmute",
    "share",
    "screen",
    "recording",
    "zoom",
    "meet",
    "teams",
    "participants",
    "chat",
    "更多",
    "共享",
    "录制",
    "会议",
    "消息",
    "消息列",
    "首页",
    "评论",
    "本视",
    "专业",
    "团队",
    "应用中心",
    "审批",
    "搜索",
    "静音",
    "取消静音",
    "聊天",
    "参会者",
    "退出",
    "结束",
    "举手",
}


def load_participant_map(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("participant map must be a JSON object")
    return {str(k): str(v) for k, v in payload.items()}


def extract_candidate_names(text: str) -> list[str]:
    candidates: list[str] = []
    for raw in re.split(r"[\n\r|•·]+", text):
        value = re.sub(r"\s+", " ", raw).strip(" :-_[]()（）")
        if not value:
            continue
        lowered = value.lower()
        if lowered in UI_STOPWORDS:
            continue
        if any(word in lowered for word in UI_STOPWORDS if re.fullmatch(r"[a-z]+", word)):
            continue
        if any(word in value for word in UI_STOPWORDS if re.search(r"[\u4e00-\u9fff]", word)):
            continue
        if re.search(r"\d{1,2}:\d{2}|\d{4}|https?://|@|%", value):
            continue
        if len(value) > 28:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]{2,5}", value):
            candidates.append(value)
            continue
        if re.fullmatch(r"[A-Z][a-zA-Z.'-]+ [A-Z][a-zA-Z.'-]+(?: [A-Z][a-zA-Z.'-]+)?", value):
            candidates.append(value)
    return candidates


def build_ocr_name_index(ocr_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for record in ocr_records:
        text = "\n".join(item.get("text", "") for item in record.get("texts", []))
        names = extract_candidate_names(text)
        if names:
            index.append(
                {
                    "time": float(record.get("time", 0.0)),
                    "path": record.get("path"),
                    "names": names,
                    "text": text,
                }
            )
    return index


def attach_names(
    segments: list[dict[str, Any]],
    ocr_records: list[dict[str, Any]],
    participant_map: dict[str, str] | None = None,
    *,
    allow_ocr_names: bool = False,
    window_seconds: float = 6.0,
) -> list[dict[str, Any]]:
    participant_map = participant_map or {}
    name_index = build_ocr_name_index(ocr_records)
    for segment in segments:
        existing_source = name_source(segment)
        if (
            has_operator_locked_identity(segment)
            and existing_source in {VOICE_ENROLLMENT_SOURCE, USER_CONFIRMED_SPEAKER_VOLUME_SOURCE}
            and float(segment.get("name_confidence", 0.0)) >= 0.6
        ):
            continue
        if has_operator_locked_identity(segment) and existing_source not in {
            VOICE_ENROLLMENT_SOURCE,
            USER_CONFIRMED_SPEAKER_VOLUME_SOURCE,
        }:
            continue

        mapped = participant_map.get(str(segment.get("speaker")))
        if mapped:
            segment["name"] = mapped
            segment["name_source"] = "participant_map"
            segment["name_confidence"] = 0.95
            continue

        if is_protected_from_ocr_overwrite(segment):
            continue

        midpoint = (float(segment["start"]) + float(segment["end"])) / 2
        nearby = [
            record
            for record in name_index
            if abs(float(record["time"]) - midpoint) <= window_seconds
        ]
        counter: Counter[str] = Counter()
        frame_refs: list[str] = []
        for record in nearby:
            counter.update(record["names"])
            if record.get("path"):
                frame_refs.append(str(record["path"]))
        if not counter:
            continue

        segment["name_candidates"] = [
            {"name": name, "count": count}
            for name, count in counter.most_common(5)
        ]
        [(name, count)] = counter.most_common(1)
        total = sum(counter.values())
        if allow_ocr_names and (count >= 2 or (total == 1 and len(counter) == 1)):
            segment["name"] = name
            segment["name_source"] = "ocr_nearby_name"
            segment["name_confidence"] = 0.65 if count < 2 else min(0.9, 0.55 + 0.1 * count)
            segment["frame_refs"] = sorted(set(frame_refs))[:3]
        else:
            segment["name_source"] = "ocr_candidates_only"
            segment["name_confidence"] = 0.0
            segment["frame_refs"] = sorted(set(frame_refs))[:3]
    return segments
