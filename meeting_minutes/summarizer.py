from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .time_utils import format_ts


DEFAULT_TRANSCRIPT_CHUNK_CHARS = 18000


def _segment_line(segment: dict[str, Any]) -> str:
    return (
        f"[{format_ts(float(segment['start']))}-{format_ts(float(segment['end']))}] "
        f"{segment.get('speaker') or 'Speaker Unknown'}: {segment.get('text', '')}"
    )


def _segment_lines(segments: list[dict[str, Any]]) -> str:
    return "\n".join(_segment_line(segment) for segment in segments)


def _segment_chunks(
    segments: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_TRANSCRIPT_CHUNK_CHARS,
) -> list[list[dict[str, Any]]]:
    """Keep every transcript segment while bounding each local-model request."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for segment in segments:
        line_size = len(_segment_line(segment)) + 1
        if current and current_size + line_size > max_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += line_size
    if current:
        chunks.append(current)
    return chunks


def _keyframes_for_chunk(keyframes: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    start = float(segments[0]["start"])
    end = float(segments[-1]["end"])
    return [frame for frame in keyframes if start <= float(frame.get("time", -1.0)) <= end]


def _keyframe_lines(keyframes: list[dict[str, Any]], max_items: int = 40) -> str:
    lines: list[str] = []
    for frame in keyframes[:max_items]:
        reasons = ", ".join(frame.get("reasons", []))
        lines.append(f"[{format_ts(float(frame['time']))}] {Path(frame['path']).name} {reasons}")
    return "\n".join(lines)


def build_minutes_prompt(
    *,
    segments: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    metadata: dict[str, Any],
    chunk_index: int = 1,
    chunk_count: int = 1,
) -> str:
    return f"""你是会议纪要整理助手。请只根据下面 transcript 和关键帧信息写中文会议纪要片段，不要编造未出现的事实或人名。

硬性要求：
- 这是整场会议的第 {chunk_index}/{chunk_count} 个连续时间片段。覆盖本片段的全部实质性讨论，不要只总结开头。
- 说话人未知时写“未知说话人”，不要猜实名。
- 输出 Markdown 片段，只能包含按时间顺序排列的三级议题标题和其下的两项内容，格式固定为：
  ### 简洁议题名称（开始时间-结束时间）
  - 现状：已经发生或已经确认的事实。
  - 讨论结果：本段实际讨论出的结论、方案比较或当前状态。
- 不得输出一级或二级标题、行动项、负责人、截止时间、维护窗口、证据文件或来源栏目。行动项由独立的确定性证据账本生成。
- 未明确拍板的方案必须写为讨论或未形成最终决定，不能写成已确认决定。

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
    chunks = _segment_chunks(segments)
    if not chunks:
        return None, {"engine": "ollama", "model": model, "status": "empty_input"}
    rendered_chunks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = build_minutes_prompt(
            segments=chunk,
            keyframes=_keyframes_for_chunk(keyframes, chunk),
            metadata=metadata,
            chunk_index=index,
            chunk_count=len(chunks),
        )
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
                "chunk": index,
                "chunks": len(chunks),
                "error": f"{type(exc).__name__}: {exc}",
            }
        text = str(payload.get("response", "")).strip()
        if not text:
            return None, {
                "engine": "ollama",
                "model": model,
                "status": "empty",
                "chunk": index,
                "chunks": len(chunks),
            }
        rendered_chunks.append(text)
    return "# 会议纪要\n\n## 议题与结论\n\n" + "\n\n".join(rendered_chunks) + "\n", {
        "engine": "ollama",
        "model": model,
        "status": "ok",
        "chunks": len(chunks),
    }
