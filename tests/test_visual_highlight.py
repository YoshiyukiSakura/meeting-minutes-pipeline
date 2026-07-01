from pathlib import Path

from PIL import Image, ImageDraw

from meeting_minutes.jsonio import write_json
from meeting_minutes.visual_highlight import load_boxes, score_frame, score_highlight_border, write_highlight_scores


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
