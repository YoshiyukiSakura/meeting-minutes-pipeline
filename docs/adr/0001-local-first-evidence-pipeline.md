# ADR 0001: Local-First Evidence Pipeline

## Status

Accepted.

## Context

The input is a flattened video-call recording. It does not include meeting platform metadata, server-side captions, participant logs, chat exports, or reliable active-speaker labels. The user also needs private recordings to remain local by default.

Meeting-minutes systems commonly optimize for clean transcript summaries. This project optimizes for evidence: every important claim should be traceable to transcript timestamps and, when available, visual frames.

## Decision

Build a local-first CLI pipeline with these rules:

- Treat audio, video frames, OCR, diarization, visual highlights, and summaries as separate evidence streams on a shared timeline.
- Use local ASR and local diarization by default.
- Keep real-name identity assignment evidence-bound.
- Produce `quality_report.md` and `review_queue.md` as required artifacts, not optional debug files.
- Keep generated meeting artifacts outside the repository.
- Use lightweight CI only for code paths that can run without Apple Silicon-specific runtime dependencies.

## Consequences

Benefits:

- Private recordings do not leave the machine by default.
- Identity claims remain reviewable.
- Reports can explain what is known, what is inferred, and what still needs review.
- Linux CI can still protect pure logic and reporting behavior without pretending to validate the full media stack.

Costs:

- Full ASR and diarization quality require local Mac validation.
- Without public media fixtures, end-to-end quality cannot be fully automated.
- Visual identity requires layout calibration for each meeting UI.
- The project must maintain clear docs separating automated tests from manual media acceptance.

## Alternatives Considered

- SaaS-first transcription: rejected as the default because it breaks local-first privacy and may introduce unverifiable identity assumptions.
- Platform API integration: rejected because the core use case has only a recording file.
- Voice-cluster-to-name guessing: rejected because voice clusters do not contain real identity.
