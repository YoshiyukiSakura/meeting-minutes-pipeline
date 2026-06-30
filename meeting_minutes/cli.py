from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .asr import transcribe_audio
from .diarization import attach_speakers, diarize_audio
from .identity import attach_names, load_participant_map
from .jsonio import read_json, write_json
from .keyframes import choose_keyframes, keyword_times, regular_times
from .media import extract_audio, extract_frames, make_clip, ocr_frames, probe_media
from .report import write_minutes, write_quality_report, write_review_queue, write_speaker_samples, write_transcript_markdown
from .summarizer import generate_ollama_minutes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first meeting minutes from a video-call recording.")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="Probe media metadata with AVFoundation.")
    probe.add_argument("input", type=Path)

    run = sub.add_parser("run", help="Run the full local pipeline.")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--start", type=float, default=0.0, help="Optional clip start in seconds.")
    run.add_argument("--duration", type=float, default=0.0, help="Optional clip duration in seconds; 0 means full input.")
    run.add_argument("--language", default="auto", help="ASR language hint; use auto unless the recording is known monolingual.")
    run.add_argument("--asr-model", default="mlx-community/whisper-large-v3-turbo")
    run.add_argument("--frame-interval", type=float, default=5.0)
    run.add_argument("--max-frame-width", type=int, default=1280)
    run.add_argument("--expected-speakers", type=int, default=0)
    run.add_argument("--diarization-backend", choices=["auto", "local-cluster", "speechbrain-cluster", "pyannote", "none"], default="auto")
    run.add_argument("--enrollment", type=Path, help="Voice enrollment JSON. When set, explicit voiceprint names override clustering labels.")
    run.add_argument("--similarity-threshold", type=float, default=0.4, help="Minimum cosine similarity for voice enrollment naming.")
    run.add_argument("--similarity-margin", type=float, default=0.06, help="Minimum top-vs-runner-up cosine margin for voice enrollment naming.")
    run.add_argument("--speechbrain-cache", type=Path, help="Optional SpeechBrain model cache directory. Defaults under the output work directory.")
    run.add_argument("--participant-map", type=Path)
    run.add_argument("--auto-ocr-names", action="store_true", help="Allow OCR-only nearby names to be written as speaker names. Off by default to avoid false real-name attribution.")
    run.add_argument("--summary-engine", choices=["extractive", "ollama"], default="extractive")
    run.add_argument("--ollama-model", default="qwen2.5:1.5b")
    run.add_argument("--skip-asr", action="store_true")
    run.add_argument("--skip-ocr", action="store_true")

    summarize = sub.add_parser("summarize", help="Regenerate minutes from an existing output directory.")
    summarize.add_argument("--output-dir", required=True, type=Path)
    summarize.add_argument("--summary-engine", choices=["extractive", "ollama"], default="ollama")
    summarize.add_argument("--ollama-model", default="qwen2.5:1.5b")

    diarize = sub.add_parser("diarize", help="Run or rerun speaker diarization on an existing output directory.")
    diarize.add_argument("--output-dir", required=True, type=Path)
    diarize.add_argument("--expected-speakers", type=int, default=0, help="Required for anonymous clustering; ignored when --enrollment is set.")
    diarize.add_argument("--diarization-backend", choices=["auto", "local-cluster", "speechbrain-cluster", "pyannote", "none"], default="speechbrain-cluster")
    diarize.add_argument("--enrollment", type=Path, help="Voice enrollment JSON. When set, explicit voiceprint names override clustering labels.")
    diarize.add_argument("--similarity-threshold", type=float, default=0.4)
    diarize.add_argument("--similarity-margin", type=float, default=0.06)
    diarize.add_argument("--speechbrain-cache", type=Path)
    diarize.add_argument("--participant-map", type=Path)

    voice_template = sub.add_parser("voice-template", help="Create a voice enrollment JSON template for an existing output directory.")
    voice_template.add_argument("--output-dir", required=True, type=Path)
    voice_template.add_argument("--names", nargs="+", help="Known participant names. Omit for generic Speaker 1..N placeholders.")
    voice_template.add_argument("--speaker-count", type=int, default=0, help="Generic speaker count when --names is omitted.")
    return parser


def _effective_media(input_path: Path, output_dir: Path, start: float, duration: float) -> tuple[Path, float]:
    if duration <= 0:
        return input_path, 0.0
    clip_path = output_dir / "work" / f"clip_{start:.0f}_{duration:.0f}.mov"
    return make_clip(input_path, clip_path, start=start, duration=duration), start


def run_pipeline(args: argparse.Namespace) -> int:
    started = time.time()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    frames_dir = output_dir / "keyframes"
    work_dir.mkdir(parents=True, exist_ok=True)

    effective_input, source_offset = _effective_media(input_path, output_dir, args.start, args.duration)
    probe = probe_media(effective_input)
    duration = float(probe.get("duration", 0.0))
    write_json(output_dir / "metadata.json", {"input": str(input_path), "effective_input": str(effective_input), "source_offset": source_offset, **probe})

    audio_path = extract_audio(effective_input, work_dir / "audio_16k_mono.wav")
    speechbrain_cache = args.speechbrain_cache.expanduser().resolve() if args.speechbrain_cache else work_dir / "speechbrain_models"
    enrollment_path = args.enrollment.expanduser().resolve() if args.enrollment else None
    if (
        not enrollment_path
        and args.diarization_backend in {"local-cluster", "speechbrain-cluster"}
        and args.expected_speakers < 2
    ):
        raise ValueError("--expected-speakers N (N >= 2) is required for anonymous speaker clustering")

    statuses: dict[str, Any] = {}
    if args.skip_asr:
        segments: list[dict[str, Any]] = []
        statuses["asr"] = {"status": "skipped"}
    else:
        segments, statuses["asr"] = transcribe_audio(audio_path, model=args.asr_model, language=args.language)
    write_json(output_dir / "transcript.raw.json", segments)

    turns, statuses["diarization"] = diarize_audio(
        audio_path,
        expected_speakers=args.expected_speakers or None,
        backend=args.diarization_backend,
        enrollment_path=enrollment_path,
        source_offset=source_offset,
        speechbrain_cache=speechbrain_cache,
        similarity_threshold=args.similarity_threshold,
        similarity_margin=args.similarity_margin,
    )
    attach_speakers(segments, turns)
    write_json(output_dir / "speaker_turns.json", turns)

    times = regular_times(duration, args.frame_interval) + keyword_times(segments)
    frames = extract_frames(effective_input, times, frames_dir, max_width=args.max_frame_width)

    if args.skip_ocr:
        ocr_records: list[dict[str, Any]] = []
        statuses["ocr"] = {"status": "skipped"}
    else:
        try:
            ocr_records = ocr_frames(frames, work_dir)
            statuses["ocr"] = {"status": "ok", "frames": len(ocr_records)}
        except Exception as exc:
            ocr_records = []
            statuses["ocr"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(output_dir / "ocr.json", ocr_records)

    participant_map = load_participant_map(args.participant_map)
    attach_names(segments, ocr_records, participant_map, allow_ocr_names=args.auto_ocr_names)

    keyframes = choose_keyframes(frames, segments)
    write_json(output_dir / "keyframes.json", keyframes)
    write_json(output_dir / "transcript.json", segments)

    report_meta = {"input": str(input_path), "duration": duration, "source_offset": source_offset}
    write_minutes(output_dir / "minutes.extractive.md", segments=segments, keyframes=keyframes, metadata=report_meta)
    if args.summary_engine == "ollama":
        ollama_minutes, statuses["summary"] = generate_ollama_minutes(
            segments=segments,
            keyframes=keyframes,
            metadata=report_meta,
            model=args.ollama_model,
        )
        if ollama_minutes:
            (output_dir / "minutes.ollama.md").write_text(ollama_minutes, encoding="utf-8")
            (output_dir / "minutes.md").write_text(ollama_minutes, encoding="utf-8")
        else:
            write_minutes(output_dir / "minutes.md", segments=segments, keyframes=keyframes, metadata=report_meta)
    else:
        statuses["summary"] = {"engine": "extractive", "status": "ok"}
        write_minutes(output_dir / "minutes.md", segments=segments, keyframes=keyframes, metadata=report_meta)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments)
    write_json(output_dir / "run_status.json", {"elapsed_seconds": round(time.time() - started, 3), "statuses": statuses})
    print(str(output_dir))
    return 0


def summarize_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    segments = read_json(output_dir / "transcript.json")
    keyframes = read_json(output_dir / "keyframes.json")
    metadata = read_json(output_dir / "metadata.json")
    report_meta = {
        "input": metadata.get("input"),
        "duration": metadata.get("duration", 0.0),
        "source_offset": metadata.get("source_offset", 0.0),
    }
    write_minutes(output_dir / "minutes.extractive.md", segments=segments, keyframes=keyframes, metadata=report_meta)
    if args.summary_engine == "ollama":
        minutes, status = generate_ollama_minutes(
            segments=segments,
            keyframes=keyframes,
            metadata=report_meta,
            model=args.ollama_model,
        )
        write_json(output_dir / "summary_status.json", status)
        if minutes:
            (output_dir / "minutes.ollama.md").write_text(minutes, encoding="utf-8")
            (output_dir / "minutes.md").write_text(minutes, encoding="utf-8")
            print(str(output_dir / "minutes.md"))
            return 0
    write_minutes(output_dir / "minutes.md", segments=segments, keyframes=keyframes, metadata=report_meta)
    print(str(output_dir / "minutes.md"))
    return 0


def diarize_existing(args: argparse.Namespace) -> int:
    started = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    audio_path = output_dir / "work" / "audio_16k_mono.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Expected extracted audio at {audio_path}")
    metadata_path = output_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    source_offset = float(metadata.get("source_offset", 0.0))
    speechbrain_cache = args.speechbrain_cache.expanduser().resolve() if args.speechbrain_cache else output_dir / "work" / "speechbrain_models"
    enrollment_path = args.enrollment.expanduser().resolve() if args.enrollment else None
    if (
        not enrollment_path
        and args.diarization_backend in {"local-cluster", "speechbrain-cluster"}
        and args.expected_speakers < 2
    ):
        raise ValueError("--expected-speakers N (N >= 2) is required for anonymous speaker clustering")
    transcript_path = output_dir / "transcript.json"
    raw_path = output_dir / "transcript.raw.json"
    segments = read_json(raw_path if raw_path.exists() else transcript_path)
    turns, diarization_status = diarize_audio(
        audio_path,
        expected_speakers=args.expected_speakers,
        backend=args.diarization_backend,
        enrollment_path=enrollment_path,
        source_offset=source_offset,
        speechbrain_cache=speechbrain_cache,
        similarity_threshold=args.similarity_threshold,
        similarity_margin=args.similarity_margin,
    )
    attach_speakers(segments, turns)
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    participant_map = load_participant_map(args.participant_map)
    attach_names(segments, ocr_records, participant_map, allow_ocr_names=False)
    write_json(output_dir / "speaker_turns.json", turns)
    write_json(output_dir / "transcript.json", segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    run_status_path = output_dir / "run_status.json"
    existing_status = read_json(run_status_path) if run_status_path.exists() else {}
    statuses = dict(existing_status.get("statuses", {}))
    statuses["diarization"] = diarization_status
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments)
    write_json(
        output_dir / "diarization_status.json",
        {
            "elapsed_seconds": round(time.time() - started, 3),
            "status": diarization_status,
            "expected_speakers": args.expected_speakers,
            "backend": "voiceprint" if enrollment_path else args.diarization_backend,
        },
    )
    if diarization_status.get("engine") == "speechbrain-ecapa-voiceprint":
        write_json(output_dir / "voiceprint_status.json", diarization_status)
    print(str(output_dir / "speaker_samples.md"))
    return 0


def _voice_template_names(args: argparse.Namespace) -> list[str]:
    if args.names:
        names = [str(name).strip() for name in args.names if str(name).strip()]
    else:
        if args.speaker_count < 2:
            raise ValueError("--speaker-count must be at least 2")
        names = [f"Speaker {index}" for index in range(1, args.speaker_count + 1)]
    if len(names) < 2:
        raise ValueError("voice enrollment template needs at least two speakers")
    if len(set(names)) != len(names):
        raise ValueError("voice enrollment speaker names must be unique")
    return names


def _voice_template_example(names: list[str]) -> list[str]:
    lines = [
        "{",
        '  "enrollment_audio_reference": "effective_clip",',
        '  "speakers": {',
    ]
    for index, name in enumerate(names):
        start = 60.0 + index * 10.0
        end = start + 6.0
        comma = "," if index < len(names) - 1 else ""
        lines.append(f'    "{name}": [{{"start": {start:.1f}, "end": {end:.1f}}}]{comma}')
    lines += [
        "  }",
        "}",
    ]
    return lines


def write_voice_template(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "voice_enrollment.template.json"
    guide_path = output_dir / "voice_enrollment_guide.md"
    names = _voice_template_names(args)
    write_json(
        template_path,
        {
            "enrollment_audio_reference": "effective_clip",
            "speakers": {name: [] for name in names},
        },
    )
    lines = [
        "# Voice Enrollment Guide",
        "",
        "Fill `voice_enrollment.template.json` with non-overlapping timestamp ranges for each known speaker.",
        "For this recording, use the effective clip timeline unless you change `enrollment_audio_reference` to `original_file`.",
        "Each speaker should have at least 3 seconds of clear active speech after silence filtering.",
        "",
        "Example:",
        "",
        "```json",
        *_voice_template_example(names),
        "```",
        "",
        "Then run:",
        "",
        "```bash",
        f'uv run meeting-minutes diarize --output-dir "{output_dir}" --enrollment "{template_path}"',
        "```",
    ]
    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(template_path))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "probe":
        payload = probe_media(args.input.expanduser().resolve())
        import json

        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        return run_pipeline(args)
    if args.command == "summarize":
        return summarize_existing(args)
    if args.command == "diarize":
        return diarize_existing(args)
    if args.command == "voice-template":
        return write_voice_template(args)
    parser.error("unknown command")
    return 2
