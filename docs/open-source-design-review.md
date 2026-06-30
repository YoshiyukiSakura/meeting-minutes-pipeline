# Open Source Design Review

This note records good design ideas observed from related open-source projects and how this repository should adapt them without copying implementation code.

## Reviewed Projects

- [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)
- [dacchi-s/meeting-minutes-whisper-pyannote-chatgpt](https://github.com/dacchi-s/meeting-minutes-whisper-pyannote-chatgpt)
- [rafaelgalle/whisper-diarization-advanced](https://github.com/rafaelgalle/whisper-diarization-advanced)
- [surajchandrann/meeting-minutes-ai](https://github.com/surajchandrann/meeting-minutes-ai)

## Good Designs To Adopt

### 1. Alignment Before Speaker Attribution

`whisper-diarization` emphasizes VAD, speaker embeddings, timestamp correction, and forced alignment. The core lesson is that diarization quality depends heavily on timestamp quality.

Adopt:

- add an optional forced-alignment stage after ASR,
- store word-level timestamps when available,
- report ASR/diarization time-shift risk in `quality_report.md`,
- keep speaker assignment confidence separate from text confidence.

Do not adopt blindly:

- source separation by default. It can improve embeddings, but it can also damage voice characteristics and is expensive on local machines.

### 2. Explicit Runtime Diagnostics

The Docker-heavy meeting-minutes project documents CUDA, ffmpeg, pyannote, OpenAI API keys, and environment setup in detail. The lesson is not that this repo needs Docker first; it needs clear preflight checks.

Adopted now:

- `meeting-minutes doctor` checks local executables, Python modules, optional `HF_TOKEN`, and optional Ollama.

Future:

- add `doctor --profile apple-silicon-local`,
- add remediation hints for missing dependencies,
- write `doctor.json` into each output directory.

### 3. Configurable Audio Preprocessing

`whisper-diarization-advanced` exposes preprocessing levels, filtering, noise reduction, normalization, and stereo support. This is useful because recordings vary widely.

Adopt:

- add named preprocessing profiles,
- track preprocessing in `metadata.json`,
- support channel-aware diarization for recordings with separate participants per channel,
- avoid aggressive denoising by default because it may hurt speaker embeddings.

### 4. Multiple Execution Surfaces

`whisper-diarization-advanced` separates local, API, Cog/Replicate, and cloud deployment modes. This repo should keep local-first as the default, but the architecture should not prevent later service use.

Adopt:

- keep core logic importable by CLI and future API wrappers,
- define stable JSON artifact contracts,
- avoid hard-coding local paths in reports or code,
- document SaaS/API use as optional baseline comparison, not source of truth.

### 5. Prompt / Hotword Support

Advanced diarization projects expose prompt or vocabulary fields. For meetings with product names, acronyms, and participant names, this can materially improve ASR.

Adopt:

- add an ASR prompt/hotwords option where supported by the backend,
- store the provided vocabulary in `metadata.json`,
- never use prompt names as identity proof.

### 6. Structured Outputs

Meeting-minutes generators often produce transcript and minutes only. This repo should keep its richer artifact set but add easier export formats.

Adopt:

- `transcript.srt` / `transcript.vtt`,
- `minutes.json` with decisions, actions, risks, evidence,
- `review_queue.csv` for spreadsheet review.

## Designs To Avoid

- Cloud-first defaults for private recordings.
- Real-name attribution from speaker labels alone.
- One-shot LLM summaries without timestamp evidence.
- GPU/CUDA-only assumptions on Apple Silicon.
- Generated meeting artifacts committed into the repository.

## Immediate Changes Made

- Added `meeting-minutes doctor`.
- Added this design review and roadmap.
- Kept visual identity as a separate evidence layer rather than merging it into voice clustering.
