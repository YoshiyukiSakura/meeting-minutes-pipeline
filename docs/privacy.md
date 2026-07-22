# Privacy Policy for Local Meeting Artifacts

This repository should contain code, generic examples, and documentation only.

Do not commit:

- original meeting recordings,
- extracted audio,
- screenshots or keyframes from private meetings,
- OCR output from private meetings,
- transcripts,
- generated minutes,
- speaker identity reports from private meetings,
- participant names from private meetings unless they are generic examples.

Generated outputs belong outside the repository, or under an ignored directory such as `outputs/`.

The `.gitignore` is intentionally conservative because meeting artifacts often contain private names, screen content, URLs, credentials, issue titles, customer information, or internal planning details.

## Optional DeepSeek Processing

DeepSeek processing is disabled by default. It runs only when the user explicitly selects `--summary-engine deepseek`.

When enabled, the pipeline sends only transcript `segment_id` and text, plus generic keyframe selection reasons, to the configured DeepSeek endpoint. It does not send timestamps, recordings, extracted audio, screenshots, frame files or paths, OCR, diarization labels, participant mappings, or visual identity data. A spoken name or sensitive detail may still exist in the transcript itself, so the operator is responsible for obtaining appropriate consent before enabling it. Repeat `--deepseek-redact-name` for every known participant name, alias, and mentioned non-participant name that must not appear in model-written draft text. The option does not redact locally derived exact source quotes, which can retain those names; the review draft is therefore never a shareable redacted artifact.

The pipeline reads only the configured DeepSeek key variable from the process environment, a selected `.env` file, `.env` in the invocation directory when no file was selected, or an explicitly named macOS Keychain service. It never searches parent directories for credentials and never writes the credential to generated artifacts, status files, logs, or this repository. Do not commit `.env` files.

Configured remote endpoints must use HTTPS, and HTTP redirects are blocked. Loopback endpoints also require a credential by default. `--deepseek-allow-unauthenticated-loopback` is an explicit local-process trust exception and still transmits transcript text to the listening process.

The local name filter normalizes Unicode NFKC form, removes zero-width format characters, and strips an explicit set of invisible code points, including variation selectors, before matching. It cannot reliably detect every visually confusable Unicode spelling, transliteration, or unknown alias. The operator must review the draft and include known variants explicitly.

The local validator checks the schema, allowed evidence IDs, decision wording, and selected unsafe text forms. A decision source containing a conditional or question anywhere in the cited segment is rejected. It does not establish full semantic entailment between a model paraphrase and the cited source quote. Treat every external review as draft material, and use the canonical local minutes and action ledger for shareable records.

Every DeepSeek rerun archives earlier active review artifacts to collision-safe `.stale.json` and `.stale.md` files before remote processing starts. The new JSON and Markdown files are prepared before either active path is replaced; an interrupted or failed rerun therefore leaves no older draft at `minutes.deepseek.review.json` or `minutes.deepseek.review.md` to be mistaken for the current run.
