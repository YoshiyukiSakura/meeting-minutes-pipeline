from pathlib import Path

from PIL import Image, ImageDraw

from meeting_minutes.jsonio import write_json
from meeting_minutes.visual_highlight import (
    load_boxes,
    score_frame,
    score_green_highlight_border,
    score_green_speaker_cue,
    score_highlight_border,
    write_highlight_scores,
)


def test_score_highlight_border_detects_colored_edge():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), outline=(0, 255, 120), width=5)

    active = score_highlight_border(image, (0.05, 0.083, 0.45, 0.75), edge_pixels=5)
    inactive = score_highlight_border(image, (0.55, 0.083, 0.95, 0.75), edge_pixels=5)

    assert active > 0.25
    assert inactive == 0.0


def test_score_frame_reports_best_tile():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((110, 10, 190, 90), outline=(0, 255, 120), width=5)

    records = score_frame(
        image,
        {
            "grid": {
                "left": (0.05, 0.083, 0.45, 0.75),
                "right": (0.55, 0.083, 0.95, 0.75),
            }
        },
        edge_pixels=5,
    )

    assert records[0]["mode"] == "grid"
    assert records[0]["best"] == "right"
    assert records[0]["scores"]["right"] > records[0]["scores"]["left"]


def test_score_highlight_border_rejects_uniform_colored_tile():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), fill=(30, 180, 120))

    score = score_highlight_border(image, (0.05, 0.083, 0.45, 0.75), edge_pixels=5)

    assert score < 0.05


def test_score_green_speaker_cue_requires_bright_green_waveform():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 80, 40, 110), radius=12, fill=(250, 250, 250))
    draw.rectangle((17, 93, 19, 99), fill=(65, 210, 105))
    draw.rectangle((22, 89, 24, 103), fill=(65, 210, 105))
    draw.rectangle((27, 92, 29, 100), fill=(65, 210, 105))

    active = score_green_speaker_cue(image, (0.05, 0.66, 0.21, 0.93))
    inactive = score_green_speaker_cue(image, (0.55, 0.66, 0.71, 0.93))

    assert active > 0.9
    assert inactive == 0.0


def test_score_green_speaker_cue_rejects_a_green_border_without_a_white_pill():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 80, 40, 110), outline=(65, 210, 105), width=4)

    score = score_green_speaker_cue(image, (0.05, 0.66, 0.21, 0.93))

    assert score == 0.0


def test_score_green_highlight_border_ignores_non_green_tile_edges():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), fill=(35, 110, 225), outline=(65, 210, 105), width=3)
    draw.rectangle((110, 10, 190, 90), fill=(230, 45, 95), outline=(60, 130, 240), width=3)

    active = score_green_highlight_border(image, (0.05, 0.083, 0.45, 0.75), edge_pixels=5)
    inactive = score_green_highlight_border(image, (0.55, 0.083, 0.95, 0.75), edge_pixels=5)

    assert active > 0.9
    assert inactive == 0.0


def test_score_green_highlight_border_rejects_a_green_corner_avatar():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), fill=(25, 25, 25))
    draw.ellipse((10, 10, 58, 58), fill=(65, 210, 105))

    score = score_green_highlight_border(image, (0.05, 0.083, 0.45, 0.75), edge_pixels=5)

    assert score == 0.0


def test_score_green_highlight_border_rejects_a_green_filled_tile():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), fill=(65, 210, 105))

    score = score_green_highlight_border(image, (0.05, 0.083, 0.45, 0.75), edge_pixels=5)

    assert score == 0.0


def test_score_green_highlight_border_accepts_a_bottom_occluded_outline():
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), outline=(65, 210, 105), width=3)
    draw.rectangle((10, 80, 90, 90), fill=(20, 20, 20))

    score = score_green_highlight_border(image, (0.05, 0.083, 0.45, 0.75), edge_pixels=5)

    assert score > 0.55


def test_score_green_highlight_border_accepts_discord_green_hues():
    for color in ((35, 165, 90), (67, 181, 129)):
        image = Image.new("RGB", (200, 120), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 90, 90), outline=color, width=3)

        score = score_green_highlight_border(image, (0.05, 0.083, 0.45, 0.75), edge_pixels=5)

        assert score > 0.9


def test_write_highlight_scores_from_keyframes(tmp_path: Path):
    frame = tmp_path / "frame.jpg"
    image = Image.new("RGB", (200, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 90), outline=(0, 255, 120), width=5)
    image.save(frame)

    keyframes = tmp_path / "keyframes.json"
    boxes = tmp_path / "boxes.json"
    output = tmp_path / "scores.json"
    write_json(keyframes, [{"time": 12.3, "actualTime": 12.25, "path": str(frame), "reasons": ["test"]}])
    write_json(boxes, {"grid": {"left": [0.05, 0.083, 0.45, 0.75], "right": [0.55, 0.083, 0.95, 0.75]}})

    assert load_boxes(boxes)["grid"]["left"] == (0.05, 0.083, 0.45, 0.75)
    records = write_highlight_scores(keyframes, boxes, output)

    assert output.exists()
    assert records[0]["time"] == 12.3
    assert records[0]["best"] == "left"
    assert records[0]["best_score"] > 0.25
