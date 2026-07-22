# Claude Projectization Review: 2026-07-08

## Scope

Claude Code CLI performed a read-only review of the repository before projectization changes. The review focused on whether the repository could become a maintainable engineering project rather than a pile of scripts and notes.

## Findings

| Finding | Response |
| --- | --- |
| Default Linux CI would fail if it ran `uv sync`, because `mlx` is Apple Silicon-specific. | Added a lightweight CI workflow that does not run `uv sync` and installs only `pytest`, `numpy`, and `pillow`. |
| The repository had no license. | Added MIT license. |
| Documentation risked multiple drifting truth sources. | Added README project contract, `docs/project-plan.md`, and updated the old roadmap to point to the canonical plan. |
| Acceptance criteria needed direct links to tests and manual evidence. | Added `docs/acceptance-matrix.md`. |
| Tests contained private participant-style names. | Replaced them with generic names. |
| Generated Claude review logs should not become repository artifacts. | Added `claude_*_review.jsonl` to `.gitignore` and removed the local raw log from the worktree. |

## Residual Risks

- Automated tests do not prove ASR quality, diarization quality, or visual identity accuracy on real recordings.
- Full validation still requires a local Apple Silicon machine and private-media review.
- Git history may still contain old private-style fixture names from earlier commits; this change only fixes the current tree.
