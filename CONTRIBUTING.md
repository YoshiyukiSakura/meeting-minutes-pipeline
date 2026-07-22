# Contributing

## Development Setup

The full project environment targets Apple Silicon macOS:

```bash
uv sync
```

Check runtime readiness:

```bash
make doctor
```

Run the full local test suite:

```bash
make test
```

## Lightweight Test Contract

Linux CI intentionally avoids `uv sync` because the default project dependencies include Apple Silicon-specific `mlx` packages.

Use the lightweight command for cross-platform logic checks:

```bash
make test-light
```

This command installs only `pytest`, `numpy`, and `pillow`, then imports the source tree through `PYTHONPATH`.

## Lint

```bash
make lint
```

The lint contract is intentionally small: keep imports clean, avoid unused symbols, and keep lines within the configured length.

## Privacy Gate

Do not commit private meeting recordings, extracted audio, keyframes, OCR output, transcripts, generated minutes, speaker identity reports, or real participant maps.

Before committing, run:

```bash
git status --short
```

Review every staged file. Generated meeting artifacts should be outside the repository or under ignored output directories.

## Identity Rules

Do not add code, tests, docs, or examples that imply the system can infer a real name from an anonymous voice cluster alone.

Real names require one of these evidence sources:

- voice enrollment,
- reviewed participant map,
- segment-level visual evidence,
- reviewed cluster fallback marked as cluster-level evidence.

Low-confidence identity must remain visible in `quality_report.md` or `review_queue.md`.

## Pull Request Checklist

- `make test-light` passes.
- `make test` passes on the local target machine when the change touches runtime behavior.
- `make lint` passes.
- No private media artifact appears in `git status --short`.
- Identity behavior is covered by tests or listed in `docs/acceptance-matrix.md` as manual evidence.
- User-facing behavior updates README or the relevant docs.
