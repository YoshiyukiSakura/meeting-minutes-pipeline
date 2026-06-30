from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .time_utils import format_ts


def _segment_lines(segments: list[dict[str, Any]], max_chars: int = 24000) -> str:
    lines: list[str] = []
    total = 0
    for segment in segments:
        line = f"[{format_ts(float(segment['start']))}-{format_ts(float(segment['end']))}] {segment.get('speaker') or 'Speaker Unknown'}: {segment.get('text', '')}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _keyframe_lines(keyframes: list[dict[str, Any]], max_items: int = 40) -> str:
    lines = []
    for frame in keyframes[:max_items]:
        reasons = ", ".join(frame.get("reasons", []))
        lines.append(f"[{format_ts(float(frame['time']))}] {Path(frame['path']).name} {reasons}")
    return "\n".join(lines)


def build_minutes_prompt(
    *,
    segments: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    return f"""你是会议纪要整理助手。请只根据下面 transcript 和关键帧信息写中文会议纪要，不要编造未出现的事实或人名。

硬性要求：
- 每条关键结论、风险、行动项都必须带时间戳。
- 说话人未知时写“未知说话人”，不要猜实名。
- 如果 transcript 不足以确定负责人或截止时间，写“未明确”。
- 输出 Markdown，包含：会议主题、背景、关键讨论、决议/结论、行动项、风险/待确认、证据索引。

输入文件：{metadata.get('input')}
时长：{format_ts(float(metadata.get('duration', 0.0)))}

Transcript:
{_segment_lines(segments)}

关键帧:
{_keyframe_lines(keyframes)}
"""


def generate_ollama_minutes(
    *,
    segments: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    metadata: dict[str, Any],
    model: str,
    timeout: int = 240,
) -> tuple[str | None, dict[str, Any]]:
    prompt = build_minutes_prompt(segments=segments, keyframes=keyframes, metadata=metadata)
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 32768,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, {
            "engine": "ollama",
            "model": model,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    text = str(payload.get("response", "")).strip()
    if not text:
        return None, {
            "engine": "ollama",
            "model": model,
            "status": "empty",
        }
    return text + "\n", {
        "engine": "ollama",
        "model": model,
        "status": "ok",
    }

