# Dynamic Visual Identity

Use dynamic visual identity for video-call recordings whose participant tiles can reorder during the call. It does not bind a person to a fixed screen coordinate.

For each sampled transcript timestamp, the pipeline:

1. Detects green active-speaker outlines directly from the frame.
2. Rejects the frame when zero or multiple active outlines are found.
3. Runs OCR only on a frame with exactly one active outline.
4. Accepts a name only when a participant-whitelisted nameplate is inside the lower portion of that same active tile.
5. Requires segment-level agreement across samples before writing the name.

Static layout profiles and sidebar names are not fallback sources for a direct name. They may be retained as audit artifacts, but they cannot assign a real name in this mode.

Example profile:

```json
{
  "participants": ["Billy", "Xin", "Nick Burin", "Aleksei", "Kirillb"],
  "settings": {
    "samples_per_segment": 3,
    "min_active_score": 0.75,
    "minimum_tile_width": 0.06,
    "minimum_tile_height": 0.06,
    "search_region": [0.20, 0.10, 0.98, 1.0],
    "nameplate_lower_fraction": 0.65,
    "minimum_segment_vote_share": 0.67
  }
}
```

Apply it to an existing run:

```bash
meeting-minutes dynamic-visual-identify \
  --output-dir /absolute/path/to/output \
  --dynamic-visual-profile /absolute/path/to/dynamic-profile.json
```

Artifacts:

- `dynamic_visual_identity.json`: frame-by-frame active tile, nameplate, and score evidence.
- `dynamic_visual_identity_detection.json`: detected active-tile boxes before OCR.
- `dynamic_visual_identity_report.md`: concise calibration and result report.
- `transcript.json`: direct names only where same-frame evidence and segment consensus passed.
