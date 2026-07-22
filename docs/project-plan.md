# Project Plan

This document is the canonical implementation plan. `docs/roadmap.md` is retained as a technical backlog and must not be treated as the project status source.

## Current Engineering Contract

- Local-first CLI remains the source of truth.
- Apple Silicon macOS is the full runtime target.
- Linux CI runs lightweight unit tests only.
- Private recordings and generated meeting artifacts stay out of Git.
- Identity claims must cite voice enrollment, reviewed maps, or visual evidence.
- Every release-worthy change needs tests or a documented manual media validation path.

## P0: Project Governance And Safety

Status: implemented.

Deliverables:

- `LICENSE`
- `CONTRIBUTING.md`
- `Makefile`
- `.github/workflows/test.yml`
- `docs/product-requirements.zh.md`
- `docs/architecture.md`
- `docs/adr/0001-local-first-evidence-pipeline.md`
- `docs/acceptance-matrix.md`
- `docs/peer-review/claude-projectization-2026-07-08.md`
- Generic test fixtures without private participant names.

Exit criteria:

- `make test-light` passes.
- `make test` passes on the local project environment.
- `make lint` passes.
- `git status --short` contains no private media artifacts.

## P1: Reproducible Runs

Goal: make each private run auditable without exposing the recording.

Work packages:

- Add `run_manifest.json` with CLI args, model names, source media hash, dependency versions, stage durations, and output paths.
- Persist `doctor.json` beside generated reports.
- Add a command that validates required artifacts in an output directory.
- Add synthetic or public-domain media fixtures that do not contain private meeting content.

Exit criteria:

- A private run can be audited from manifest and quality report alone.
- Public tests cover manifest generation without private fixtures.

## P2: ASR And Diarization Quality

Goal: improve transcript-to-speaker alignment and confidence reporting.

Work packages:

- Add VAD-assisted cleanup before speaker attachment.
- Add optional forced alignment for word-level timestamps.
- Separate ASR confidence, diarization overlap confidence, identity confidence, and summary confidence.
- Add overlap detection for interruptions.
- Add stereo or channel-aware preprocessing where recordings contain separate speaker channels.

Exit criteria:

- Mixed segments are visible in `quality_report.md`.
- Speaker attachment confidence can be explained from measurable inputs.

## P3: Visual Identity Automation

Goal: move visual active-speaker evidence from helper scripts into a first-class workflow.

Status: implemented for calibrated profiles. Generic UI layout discovery and presenter-border semantics remain deferred.

Local media validation: a 16-frame private recording slice produced three active frames and one frame-backed segment assignment; five segments correctly stayed unresolved. This is a runtime smoke test, not a general accuracy benchmark.

Delivered work packages:

- `meeting-minutes visual-identify` and optional `run --visual-profile` integration.
- Normalized, non-overlapping layout windows with per-slot tile and nameplate boxes.
- Segment-aware sampling, active-score plus runner-up margin gates, cropped Apple Vision OCR, and participant-whitelist correction for minor OCR errors.
- `visual_identity.json` and `visual_identity_report.md` with frame-level evidence and explicit unresolved states.
- No automatic diarization-cluster fallback. Mixed or insufficient visual evidence stays anonymous.

Exit criteria:

- A reviewer can audit why a name was attached to each visually mapped segment.
- Mixed clusters remain marked instead of being silently promoted.

## P4: Output Products

Goal: make outputs easier to review and integrate.

Work packages:

- Add `minutes.json` with decisions, action items, risks, owners, and evidence refs.
- Add `review_queue.csv`.
- Add SRT and VTT transcript exports.
- Add an HTML report with transcript, screenshots, and click-to-time evidence.

Exit criteria:

- Markdown remains human-readable.
- JSON and CSV are stable enough for downstream tools.

## P5: Optional Service Surfaces

Goal: expose the pipeline without weakening the local-first source of truth.

Work packages:

- Add an optional REST wrapper.
- Add a local desktop or web review UI.
- Add optional SaaS comparison hooks as quality baselines only.
- Add Docker or Linux GPU support only after dependency strategy is explicit.

Exit criteria:

- CLI output remains the canonical artifact contract.
- SaaS output cannot overwrite evidence-bound local identity decisions.
