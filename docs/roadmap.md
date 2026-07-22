# Roadmap

Status: technical backlog. The canonical project sequence and current status live in [project-plan.md](project-plan.md).

The roadmap is ordered by evidence quality and local practicality.

## P0: Reproducibility And Safety

- Add `doctor --profile apple-silicon-local`.
- Persist `doctor.json` in output directories.
- Add `run_manifest.json` with CLI args, model names, dependency versions, and runtime durations.
- Add sample fixture generation using synthetic or public-domain media only.

## P1: Better ASR / Diarization Alignment

- Optional forced alignment stage for word-level timestamps.
- VAD-assisted segment cleanup before speaker attachment.
- Overlap detection for multi-speaker interruptions.
- Confidence decomposition: ASR confidence, diarization overlap confidence, identity confidence.

## P2: Audio Preprocessing Profiles

- `--preprocess-level 0..4` with explicit metadata.
- Level 0: raw extraction.
- Level 1: mono 16k PCM sanitization.
- Level 2: filtering.
- Level 3: noise reduction.
- Level 4: normalization.
- Channel-aware mode for stereo call-center style recordings.

## P3: Visual Identity Automation

- Convert manual highlight scoring into a first-class CLI command. Done for frame-record based scoring via `tools/score_active_speaker_highlights.py`.
- Support per-layout tile boxes and layout switch points. Basic scoring supports per-layout boxes; identity application supports a manual layout switch point.
- Detect presenter-only borders separately from active-speaker borders.
- Produce an identity calibration report with contact sheets and score distributions.

## P4: Output Formats

- Export SRT/VTT transcript.
- Export `minutes.json` with decisions, action items, risks, owners, and evidence refs.
- Export `review_queue.csv`.
- Add HTML report with transcript, screenshots, and click-to-time evidence.

## P5: Deployment Surfaces

- Keep local CLI as the source of truth.
- Add optional REST API wrapper.
- Add optional Dockerfile for Linux/GPU users.
- Add cloud/SaaS comparison hooks only as quality baselines, not authoritative identity sources.
