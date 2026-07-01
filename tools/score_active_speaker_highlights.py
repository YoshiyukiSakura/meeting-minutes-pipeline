from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from meeting_minutes.visual_highlight import write_highlight_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge-pixels", type=int, default=5)
    parser.add_argument("--saturation-threshold", type=float, default=0.25)
    parser.add_argument("--value-threshold", type=float, default=0.48)
    args = parser.parse_args()

    records = write_highlight_scores(
        args.keyframes.expanduser().resolve(),
        args.boxes_json.expanduser().resolve(),
        args.output.expanduser().resolve(),
        edge_pixels=args.edge_pixels,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
    )
    active = [record for record in records if float(record.get("best_score", 0.0)) >= 0.25]
    by_mode = Counter(str(record.get("mode")) for record in active)
    by_label = Counter(str(record.get("best")) for record in active)
    print(
        json.dumps(
            {
                "records": len(records),
                "active_frames_at_0_25": len(active),
                "active_by_mode": dict(by_mode),
                "active_by_label": dict(by_label),
                "output": str(args.output.expanduser().resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
