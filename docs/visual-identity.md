# Visual Identity Workflow

This project treats visual identity as evidence, not decoration.

## Required Inputs

- `transcript.json`: ASR segments with diarized speaker labels.
- keyframes or extracted still frames from the recording.
- calibrated tile boxes for the meeting UI.
- a reviewed UI-label-to-person map, for example `{"Tile A": "Alice"}`.

## Calibration Steps

1. Identify the meeting UI layout.

   Common layouts:

   - full 2x2 grid before screen sharing,
   - right-side participant rail during screen sharing,
   - floating speaker thumbnails.

2. Define normalized tile boxes.

   Coordinates are `[x1, y1, x2, y2]`, normalized to image width and height.

3. Pick high-confidence speech samples per diarized speaker.

   Use `speaker_samples.md` and choose clear, non-overlapping samples.

4. Generate contact sheets.

   ```bash
   uv run python tools/visual_identity_probe.py \
     --video "/path/to/recording.mov" \
     --output "/path/to/identity_probe" \
     --samples-json examples/visual_samples.json \
     --tiles-json examples/proton_meet_tiles.json
   ```

5. Score or review active-speaker borders.

   A border can mean different things depending on the UI. For example, a presenter border during screen sharing may not mean active speaker.

6. Apply reviewed mapping.

   ```bash
   uv run python tools/apply_visual_identity.py \
     --output-dir "/path/to/output" \
     --scores "/path/to/identity_probe/highlight_scores.json" \
     --ui-map examples/ui_name_map.json \
     --cluster-fallback examples/cluster_fallback.json \
     --mixed-clusters "Speaker 3"
   ```

## Reporting Rules

- Count segment-level visual assignments separately from cluster fallback assignments.
- Mark mixed clusters explicitly.
- Keep unverified segments in `review_queue.md`.
- Do not attribute speech to a participant who is merely visible in the UI.
