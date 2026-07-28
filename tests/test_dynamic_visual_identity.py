from __future__ import annotations

import json

from PIL import Image, ImageDraw

from meeting_minutes.dynamic_visual_identity import (
    attach_dynamic_ocr,
    attach_dynamic_visual_identity,
    detect_active_tiles,
    load_dynamic_visual_profile,
)


def _profile(tmp_path):
    path = tmp_path / "dynamic-profile.json"
    path.write_text(
        json.dumps(
            {
                "participants": ["Billy", "Xin", "Nick Burin"],
                "settings": {
                    "min_active_score": 0.75,
                    "minimum_tile_width": 0.08,
                    "minimum_tile_height": 0.08,
                    "horizontal_run_min_pixels": 30,
                    "search_region": [0.10, 0.10, 0.98, 0.98],
                    "nameplate_lower_fraction": 0.65,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_dynamic_visual_profile(path)


def _image_with_active_tiles(boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    image = Image.new("RGB", (800, 500), "black")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=(48, 198, 90), width=3)
    return image


def _vision_nameplate(name: str, *, x: float, top: float) -> dict:
    height = 0.03
    return {
        "text": name,
        "confidence": 0.95,
        "bbox": {"x": x, "y": 1.0 - top - height, "width": 0.10, "height": height},
    }


def test_detect_active_tiles_without_static_person_coordinates(tmp_path):
    profile = _profile(tmp_path)
    image = _image_with_active_tiles([(170, 100, 410, 300), (460, 100, 700, 300)])

    tiles = detect_active_tiles(image, profile)

    assert len(tiles) == 2
    assert all(tile["score"] >= 0.75 for tile in tiles)


def test_same_person_can_move_to_a_new_tile_when_ocr_evidence_moves(tmp_path):
    profile = _profile(tmp_path)
    detected = [
        {
            "time": 1.0,
            "actualTime": 1.0,
            "path": "/tmp/left.jpg",
            "active_tiles": [{"tile": [0.20, 0.20, 0.50, 0.60], "score": 0.99}],
            "reason": "single_active_tile",
        },
        {
            "time": 2.0,
            "actualTime": 2.0,
            "path": "/tmp/right.jpg",
            "active_tiles": [{"tile": [0.55, 0.20, 0.85, 0.60], "score": 0.99}],
            "reason": "single_active_tile",
        },
    ]
    ocr = [
        {"path": "/tmp/left.jpg", "texts": [_vision_nameplate("Billy", x=0.25, top=0.54)]},
        {"path": "/tmp/right.jpg", "texts": [_vision_nameplate("Billy", x=0.60, top=0.54)]},
    ]

    scored = attach_dynamic_ocr(detected, ocr, profile)

    assert [frame["name"] for frame in scored] == ["Billy", "Billy"]
    segments = [
        {"start": 0.8, "end": 1.2, "speaker": "Speaker 1"},
        {"start": 1.8, "end": 2.2, "speaker": "Speaker 1"},
    ]
    summary = attach_dynamic_visual_identity(
        segments,
        [
            {"segment_index": 0, "video_time": 1.0},
            {"segment_index": 1, "video_time": 2.0},
        ],
        scored,
        profile,
    )

    assert summary["assigned"] == 2
    assert [segment["name"] for segment in segments] == ["Billy", "Billy"]
    assert all(segment["name_source"] == "dynamic_visual_in_tile_nameplate_ocr" for segment in segments)


def test_multiple_active_tiles_stay_unresolved_even_when_one_nameplate_is_read(tmp_path):
    profile = _profile(tmp_path)
    frame = {
        "time": 1.0,
        "actualTime": 1.0,
        "path": "/tmp/concurrent.jpg",
        "active_tiles": [
            {"tile": [0.20, 0.20, 0.50, 0.60], "score": 0.99},
            {"tile": [0.55, 0.20, 0.85, 0.60], "score": 0.99},
        ],
        "reason": "multiple_active_tiles",
    }
    segments = [{"start": 0.8, "end": 1.2, "speaker": "Speaker 1"}]

    summary = attach_dynamic_visual_identity(
        segments,
        [{"segment_index": 0, "video_time": 1.0}],
        [frame],
        profile,
    )

    assert summary["assigned"] == 0
    assert summary["conflicts"] == 1
    assert segments[0]["name"] is None
    assert segments[0]["name_source"] == "dynamic_visual_identity_conflict"


def test_static_visual_name_is_cleared_when_dynamic_evidence_is_absent(tmp_path):
    profile = _profile(tmp_path)
    segments = [
        {
            "start": 0.8,
            "end": 1.2,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": "visual_active_speaker_highlight",
            "name_confidence": 0.91,
        }
    ]
    frame = {"time": 1.0, "actualTime": 1.0, "path": "/tmp/no-active.jpg", "active_tiles": [], "reason": "no_active_tile"}

    attach_dynamic_visual_identity(segments, [{"segment_index": 0, "video_time": 1.0}], [frame], profile)

    assert segments[0]["name"] is None
    assert segments[0]["name_source"] == "dynamic_visual_identity_unresolved"
