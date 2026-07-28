# Visual Identity Workflow

`meeting-minutes visual-identify` converts an active-speaker UI signal into segment-level identity evidence. It is local-first and does not send video frames or names to a SaaS. It uses nameplate OCR and UI activity indicators; it does not perform facial recognition.

## Profile Contract

A profile has non-overlapping time-bounded layouts. Every coordinate is normalized as `[x1, y1, x2, y2]` against the extracted frame width and height, so it is independent of the recording resolution.

```json
{
  "participants": ["Alice", "Bob"],
  "settings": {
    "samples_per_segment": 3,
    "min_active_score": 0.14,
    "min_active_margin": 0.05,
    "minimum_nameplate_observations": 2,
    "minimum_nameplate_share": 0.67,
    "minimum_segment_vote_share": 0.66,
    "allow_direct_assignment": false
  },
  "layouts": [
    {
      "name": "screen_share_side_rail",
      "start": 90,
      "slots": {
        "slot_1": {
          "tile": [0.633, 0.12, 0.844, 0.262],
          "active_signal": "green_highlight_border",
          "nameplate": [0.638, 0.212, 0.81, 0.261]
        }
      }
    }
  ]
}
```

- `participants` is strongly recommended. It is a local whitelist for resolving small OCR mistakes such as `Wiliam` to `William`. Without it, OCR output is retained only as a candidate.
- A slot may use `person` instead of `nameplate` only after a reviewer has calibrated that slot to a named participant.
- `active_signal` is one of `green_highlight_border`, `green_speaker_cue`, or `highlight_border`. When direct visual assignment is enabled, every slot must declare it explicitly; the pipeline rejects a profile that would otherwise silently fall back to a generic scorer.
- A layout is never applied before `start`, after `end`, or across an overlapping layout window.
- `allow_direct_assignment` defaults to `false`. Enable it only after a reviewed sample shows that the UI cue tracks active speech rather than presenter, focus, or another tile state.

## Command

```bash
uv run meeting-minutes visual-identify \
  --output-dir "$HOME/Documents/meeting-output" \
  --visual-profile "$HOME/Documents/meeting-output/visual-profile.json"
```

The same profile can be supplied to a new full run with `--visual-profile`.

## Assignment Rules

1. Short ASR segments receive one visual sample; longer segments receive two or three samples.
2. A frame is active only when the highest tile-border score clears the profile threshold and leads the runner-up score by the configured margin.
3. A slot name must come from a reviewed `person` field or multiple OCR observations that resolve to the participant whitelist.
4. A segment needs one strong sample when it has one sample, or at least two consistent samples with a two-thirds vote share when it has multiple samples.
5. Different active names in the same segment, unresolved nameplates, no calibrated layout, or weak evidence leave the segment anonymous.
6. Existing voice-enrollment and reviewed participant-map names are preserved. A disagreement with visual evidence is recorded as a conflict.
7. With direct assignment disabled, visual names are retained only as audit candidates and cannot enter the transcript or minutes.

The direct visual identity confidence is `min(0.94, 0.62 + 0.16 * vote_share + 0.22 * average_active_border_score)`. It is an evidence-quality score, not a statistical probability or an accuracy claim.

## Outputs

- `visual_identity.json`: all scored frames, resolved slot names, and assignment summary.
- `visual_identity_report.md`: calibration report with profile path, slot OCR evidence, assignment counts, and limits.
- `visual_identity_nameplate_ocr.json`: cropped local OCR evidence.
- `visual_identity_frames.json` and `work/visual_identity_frames/`: timestamped visual samples.

These artifacts are private meeting data and are excluded by `.gitignore`.
