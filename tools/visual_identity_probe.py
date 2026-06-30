from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


DEFAULT_TILES = {
    "top_left": (0.006, 0.150, 0.500, 0.510),
    "top_right": (0.503, 0.150, 0.993, 0.510),
    "bottom_left": (0.006, 0.510, 0.500, 0.868),
    "bottom_right": (0.503, 0.510, 0.993, 0.868),
}


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in value).strip("_")


def extract_frame(video: Path, output: Path, seconds: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{seconds:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        "scale=1600:-1",
        "-q:v",
        "2",
        str(output),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_samples(path: Path) -> dict[str, list[tuple[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: dict[str, list[tuple[str, float]]] = {}
    for speaker, entries in payload.items():
        parsed = []
        for entry in entries:
            if isinstance(entry, dict):
                parsed.append((str(entry.get("range", entry.get("label", ""))), float(entry["time"])))
            else:
                label, seconds = entry
                parsed.append((str(label), float(seconds)))
        samples[str(speaker)] = parsed
    return samples


def load_tiles(path: Path | None) -> dict[str, tuple[float, float, float, float]]:
    if not path:
        return DEFAULT_TILES
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(name): tuple(float(value) for value in box) for name, box in payload.items()}


def crop_tiles(frame_path: Path, crop_dir: Path, tiles: dict[str, tuple[float, float, float, float]]) -> dict[str, Path]:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    result = {}
    crop_dir.mkdir(parents=True, exist_ok=True)
    for name, (x1, y1, x2, y2) in tiles.items():
        box = (
            math.floor(x1 * width),
            math.floor(y1 * height),
            math.ceil(x2 * width),
            math.ceil(y2 * height),
        )
        crop = image.crop(box)
        path = crop_dir / f"{safe_name(name)}.jpg"
        crop.save(path, quality=92)
        result[name] = path
    return result


def label_image(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    pad = 10
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle(
        [0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2],
        fill=(0, 0, 0),
    )
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return image


def make_contact_sheet(speaker: str, frames: list[tuple[str, Path]], output: Path) -> None:
    font = ImageFont.load_default()
    thumbs = []
    for timerange, path in frames:
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 300))
        thumbs.append(label_image(image, f"{speaker} {timerange}", font))

    if not thumbs:
        return

    cols = 2
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (22, 22, 22))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(thumb, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def make_tile_contact_sheet(
    speaker: str,
    tile_frames: list[tuple[str, dict[str, Path]]],
    output: Path,
    tiles: dict[str, tuple[float, float, float, float]],
) -> None:
    font = ImageFont.load_default()
    tile_names = list(tiles)
    rendered_rows = []
    for timerange, tile_paths in tile_frames:
        thumbs = []
        for tile_name in tile_names:
            image = Image.open(tile_paths[tile_name]).convert("RGB")
            image.thumbnail((360, 220))
            thumbs.append(label_image(image, f"{timerange} {tile_name}", font))
        cell_w = max(t.width for t in thumbs)
        cell_h = max(t.height for t in thumbs)
        row = Image.new("RGB", (len(thumbs) * cell_w, cell_h), (22, 22, 22))
        for idx, thumb in enumerate(thumbs):
            row.paste(thumb, (idx * cell_w, 0))
        rendered_rows.append(row)

    if not rendered_rows:
        return

    sheet_w = max(row.width for row in rendered_rows)
    sheet_h = sum(row.height for row in rendered_rows)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (22, 22, 22))
    y = 0
    for row in rendered_rows:
        sheet.paste(row, (0, y))
        y += row.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-json", type=Path, required=True, help="JSON mapping speaker labels to time samples.")
    parser.add_argument("--tiles-json", type=Path, help="Optional normalized tile boxes. Defaults to a Proton Meet 2x2 grid.")
    args = parser.parse_args()

    samples_by_speaker = load_samples(args.samples_json)
    tiles = load_tiles(args.tiles_json)
    frame_root = args.output / "frames"
    crop_root = args.output / "tile_crops"
    sheet_root = args.output / "contact_sheets"
    for speaker, samples in samples_by_speaker.items():
        speaker_frames = []
        speaker_tile_frames = []
        for timerange, seconds in samples:
            stem = f"{safe_name(speaker)}_{int(seconds * 1000):07d}_{timerange.replace(':', '').replace('-', '_')}"
            frame_path = frame_root / f"{stem}.jpg"
            extract_frame(args.video, frame_path, seconds)
            speaker_frames.append((timerange, frame_path))
            tile_paths = crop_tiles(frame_path, crop_root / safe_name(speaker) / stem, tiles)
            speaker_tile_frames.append((timerange, tile_paths))
        make_contact_sheet(speaker, speaker_frames, sheet_root / f"{safe_name(speaker)}_full.jpg")
        make_tile_contact_sheet(speaker, speaker_tile_frames, sheet_root / f"{safe_name(speaker)}_tiles.jpg", tiles)

    print(f"Wrote visual identity probe to {args.output}")


if __name__ == "__main__":
    main()
