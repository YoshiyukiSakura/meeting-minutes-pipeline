from PIL import Image

from meeting_minutes.keyframes import choose_keyframes


def test_choose_keyframes_skips_failed_frame_records(tmp_path):
    valid = tmp_path / "valid.jpg"
    Image.new("RGB", (32, 32), color="white").save(valid)
    frames = [
        {"time": 0.0, "error": "Cannot Open"},
        {"time": 1.0, "path": str(valid)},
    ]

    selected = choose_keyframes(frames, [])

    assert selected == [{"time": 1.0, "path": str(valid), "reasons": ["opening_frame"]}]
