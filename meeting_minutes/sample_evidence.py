"""Stable, per-segment frame selection for identity evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def unique_frames_for_requests(
    requests: list[dict[str, Any]],
    frame_by_time: Mapping[float, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return at most one frame for each rounded requested video timestamp."""

    result: list[dict[str, Any]] = []
    seen_times: set[float] = set()
    for request in requests:
        try:
            video_time = round(float(request["video_time"]), 3)
        except (KeyError, TypeError, ValueError):
            continue
        if video_time in seen_times:
            continue
        seen_times.add(video_time)
        frame = frame_by_time.get(video_time)
        if frame:
            result.append(frame)
    return result
