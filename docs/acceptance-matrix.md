# Acceptance Matrix

This matrix is the verification source of truth. It separates automated evidence from manual private-media evidence because the repository does not contain public meeting recordings.

| ID | Acceptance item | Automated evidence | Manual evidence | Status |
| --- | --- | --- | --- | --- |
| A1 | The only required input is a local recording file. | `tests/test_doctor.py::test_doctor_report_lists_required_result` protects environment reporting. | Run `meeting-minutes run` on a local `.mov` or `.mp4` without platform exports. | Manual media validation required. |
| A2 | Transcript segments retain timestamps. | `tests/test_report.py::test_quality_report_counts_anonymous_speaker_labels` verifies segment accounting. | Inspect `transcript.json` and `transcript.md` from a private run. | Automated partial coverage. |
| A3 | Voice clusters do not become real names without evidence. | `tests/test_identity.py::test_voice_enrollment_name_has_priority_over_participant_map`, `tests/test_identity.py::test_user_confirmed_name_has_priority_over_participant_map`, and `tests/test_diarization.py::test_attach_speakers_lowers_confidence_for_mixed_speaker_segments`. | Inspect `quality_report.md` and `review_queue.md` for unknown or low-confidence speakers. | Automated partial coverage. |
| A4 | Voice enrollment can attach known names and preserve confidence. | `tests/test_diarization.py::test_load_voice_enrollment_original_file_offsets_ranges` and `tests/test_diarization.py::test_attach_speakers_propagates_voice_enrollment_name`. | Review enrollment ranges against the local recording before trusting names. | Automated partial coverage. |
| A5 | Mixed speaker segments lower confidence instead of hiding uncertainty. | `tests/test_diarization.py::test_attach_speakers_lowers_confidence_for_mixed_speaker_segments`. | Review mixed or overlapping speech in `review_queue.md`. | Covered for attachment logic. |
| A6 | Active-speaker visual evidence is calibrated before use. | `tests/test_visual_highlight.py` verifies border contrast; `tests/test_visual_identity.py` verifies profile windows, participant-whitelisted OCR correction, temporal consensus, conflicts, and no cluster fallback. | Review `visual_identity_report.md`, slot-name evidence, score distributions, and frame refs for each UI layout. | Automated logic plus local-media validation required. |
| A7 | Meeting minutes contain timestamp-backed evidence. | `tests/test_report.py::test_review_queue_skips_unknown_speaker_when_visual_name_is_resolved` protects review behavior. | Check each decision, risk, and action candidate in `minutes.md` against `transcript.md` and frame refs. | Manual media validation required. |
| A8 | Private artifacts are not committed. | `.gitignore` excludes generated recordings, audio, transcripts, OCR, keyframes, reports, and Claude review logs. | Run `git status --short` and inspect staged paths before commit. | Process gate. |
| A9 | Linux CI does not install Apple Silicon-only runtime dependencies. | `.github/workflows/test.yml` runs `uv run --no-project --with pytest --with numpy --with pillow pytest -q` with `PYTHONPATH` set. | Review workflow logs after push or pull request. | Covered by workflow design. |
| A10 | Local target Mac can run the full project environment checks. | `make doctor` executes `uv run meeting-minutes doctor`. | Run `make doctor` on the target machine before long recordings. | Local validation required. |
| A11 | An action item cannot combine adjacent but distinct topics, owners, durations, or downtime claims. | `tests/test_action_items.py` rejects the MPC/two-hour merge across English and Chinese duration forms, blocks unanchored adjacent duration absorption, refuses subjectless downtime inheritance, requires exact source text, rejects conflicting downtime facts, and preserves ambiguous items for review. `tests/test_cli.py` rejects a stale or hand-edited ledger. | Run `meeting-minutes audit-actions` followed by `meeting-minutes validate-actions` on a local recording before sharing a canonical minutes document. | Covered by deterministic unit tests plus local-media validation. |

## Required Commands

Lightweight CI-equivalent test:

```bash
make test-light
```

Full local test:

```bash
make test
```

Lint:

```bash
make lint
```

Runtime doctor:

```bash
make doctor
```
