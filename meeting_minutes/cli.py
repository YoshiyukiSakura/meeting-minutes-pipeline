from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from meeting_minutes.action_items import build_action_ledger, transcript_fingerprint, validate_published_action_item
from .asr import transcribe_audio
from .diarization import attach_speakers, diarize_audio
from .deepseek import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MAX_INPUT_CHARS,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    DeepSeekConfig,
    generate_deepseek_review,
    write_deepseek_review,
)
from .doctor import collect_environment_checks, doctor_exit_code, render_doctor_report
from .identity import attach_names, load_participant_map
from .jsonio import read_json, write_json
from .keyframes import choose_keyframes, keyword_times, regular_times
from .media import extract_audio, extract_frames, make_clip, ocr_frames, ocr_regions, probe_media
from .report import (
    write_action_item_ledger,
    write_minutes,
    write_quality_report,
    write_review_queue,
    write_speaker_samples,
    write_transcript_markdown,
)
from .summarizer import generate_ollama_minutes
from .visual_identity import (
    attach_visual_identity,
    build_nameplate_manifest,
    build_segment_sample_requests,
    load_visual_profile,
    resolve_slot_names,
    score_visual_frames,
    unique_video_times,
    write_visual_identity_report,
)
from .voice_registry import apply_voice_registry, build_voice_registry


OLLAMA_DRAFT_NOTICE = (
    "> **草稿，仅供核对。** 此文件是模型草稿，不可作为最终会议纪要分享；"
    "请以 `minutes.md` 和经校验的行动项账本为准。\n\n"
)


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
    run.add_argument("--voice-registry", type=Path, help="Cross-recording voice registry JSON. Names are attached after anonymous clustering.")
    run.add_argument("--registry-threshold", type=float, help="Override the cross-recording voice registry score threshold.")
    run.add_argument("--registry-margin", type=float, help="Override the cross-recording voice registry best-vs-runner-up margin.")
    run.add_argument("--similarity-threshold", type=float, default=0.4, help="Minimum cosine similarity for voice enrollment naming.")
    run.add_argument("--similarity-margin", type=float, default=0.06, help="Minimum top-vs-runner-up cosine margin for voice enrollment naming.")
    run.add_argument("--speechbrain-cache", type=Path, help="Optional SpeechBrain model cache directory. Defaults under the output work directory.")
    run.add_argument("--participant-map", type=Path)
    run.add_argument("--auto-ocr-names", action="store_true", help="Allow OCR-only nearby names to be written as speaker names. Off by default to avoid false real-name attribution.")
    run.add_argument("--summary-engine", choices=["extractive", "ollama", "deepseek"], default="extractive")
    run.add_argument("--ollama-model", default="qwen2.5:1.5b")
    run.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    run.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    run.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    run.add_argument("--deepseek-env-file", type=Path, help="Optional .env file; only the variable named by --deepseek-api-key-env is read.")
    run.add_argument("--deepseek-keychain-service", help="Optional macOS Keychain generic-password service name.")
    run.add_argument("--deepseek-redact-name", action="append", default=[], help="Known name or alias to reject from model-written external review text; repeat for multiple names.")
    run.add_argument("--deepseek-allow-unauthenticated-loopback", action="store_true", help="Allow a local loopback endpoint without an API key. Disabled by default.")
    run.add_argument("--deepseek-timeout", type=int, default=DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)
    run.add_argument("--deepseek-max-input-chars", type=int, default=DEFAULT_DEEPSEEK_MAX_INPUT_CHARS)
    run.add_argument("--deepseek-output-language", choices=["zh-CN", "en"], default="zh-CN")
    run.add_argument("--skip-asr", action="store_true")
    run.add_argument("--skip-ocr", action="store_true")
    run.add_argument("--visual-profile", type=Path, help="Calibrated visual identity JSON profile. Enables active-speaker evidence mapping.")
    run.add_argument("--visual-max-frame-width", type=int, default=1280)

    summarize = sub.add_parser("summarize", help="Regenerate minutes from an existing output directory.")
    summarize.add_argument("--output-dir", required=True, type=Path)
    summarize.add_argument("--summary-engine", choices=["extractive", "ollama", "deepseek"], default="ollama")
    summarize.add_argument("--ollama-model", default="qwen2.5:1.5b")
    summarize.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    summarize.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    summarize.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    summarize.add_argument("--deepseek-env-file", type=Path, help="Optional .env file; only the variable named by --deepseek-api-key-env is read.")
    summarize.add_argument("--deepseek-keychain-service", help="Optional macOS Keychain generic-password service name.")
    summarize.add_argument("--deepseek-redact-name", action="append", default=[], help="Known name or alias to reject from model-written external review text; repeat for multiple names.")
    summarize.add_argument("--deepseek-allow-unauthenticated-loopback", action="store_true", help="Allow a local loopback endpoint without an API key. Disabled by default.")
    summarize.add_argument("--deepseek-timeout", type=int, default=DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)
    summarize.add_argument("--deepseek-max-input-chars", type=int, default=DEFAULT_DEEPSEEK_MAX_INPUT_CHARS)
    summarize.add_argument("--deepseek-output-language", choices=["zh-CN", "en"], default="zh-CN")

    diarize = sub.add_parser("diarize", help="Run or rerun speaker diarization on an existing output directory.")
    diarize.add_argument("--output-dir", required=True, type=Path)
    diarize.add_argument("--expected-speakers", type=int, default=0, help="Required for anonymous clustering; ignored when --enrollment is set.")
    diarize.add_argument("--diarization-backend", choices=["auto", "local-cluster", "speechbrain-cluster", "pyannote", "none"], default="speechbrain-cluster")
    diarize.add_argument("--enrollment", type=Path, help="Voice enrollment JSON. When set, explicit voiceprint names override clustering labels.")
    diarize.add_argument("--voice-registry", type=Path, help="Cross-recording voice registry JSON. Names are attached after anonymous clustering.")
    diarize.add_argument("--registry-threshold", type=float, help="Override the cross-recording voice registry score threshold.")
    diarize.add_argument("--registry-margin", type=float, help="Override the cross-recording voice registry best-vs-runner-up margin.")
    diarize.add_argument("--similarity-threshold", type=float, default=0.4)
    diarize.add_argument("--similarity-margin", type=float, default=0.06)
    diarize.add_argument("--speechbrain-cache", type=Path)
    diarize.add_argument("--participant-map", type=Path)

    visual_identify = sub.add_parser("visual-identify", help="Apply calibrated active-speaker visual evidence to an existing run.")
    visual_identify.add_argument("--output-dir", required=True, type=Path)
    visual_identify.add_argument("--visual-profile", required=True, type=Path)
    visual_identify.add_argument("--input", type=Path, help="Override the effective input recorded in metadata.json.")
    visual_identify.add_argument("--max-frame-width", type=int, default=1280)

    relabel = sub.add_parser(
        "relabel",
        help="Apply a reviewed diarization speaker map without rerunning ASR, diarization, OCR, or the voice registry.",
    )
    relabel.add_argument("--output-dir", required=True, type=Path)
    relabel.add_argument("--participant-map", required=True, type=Path)

    audit_actions = sub.add_parser(
        "audit-actions",
        help="Build a source-grounded action ledger for an existing transcript without rerunning models.",
    )
    audit_actions.add_argument("--output-dir", required=True, type=Path)

    validate_actions = sub.add_parser(
        "validate-actions",
        help="Validate proposed action rows against the output directory's action ledger.",
    )
    validate_actions.add_argument("--output-dir", required=True, type=Path)
    validate_actions.add_argument("--items", required=True, type=Path, help="JSON array or object with an items array.")

    voice_template = sub.add_parser("voice-template", help="Create a voice enrollment JSON template for an existing output directory.")
    voice_template.add_argument("--output-dir", required=True, type=Path)
    voice_template.add_argument("--names", nargs="+", help="Known participant names. Omit for generic Speaker 1..N placeholders.")
    voice_template.add_argument("--speaker-count", type=int, default=0, help="Generic speaker count when --names is omitted.")

    voice_registry = sub.add_parser("voice-registry", help="Build a reusable cross-recording voice registry.")
    voice_registry_sub = voice_registry.add_subparsers(dest="voice_registry_command", required=True)
    voice_registry_build = voice_registry_sub.add_parser("build", help="Build and calibrate a registry from reviewed source recordings.")
    voice_registry_build.add_argument("--sources", required=True, type=Path, help="Voice registry source-manifest JSON.")
    voice_registry_build.add_argument("--output", required=True, type=Path, help="Output voice registry JSON.")
    voice_registry_build.add_argument("--work-dir", type=Path, help="Cache directory for extracted source audio and SpeechBrain files.")
    voice_registry_build.add_argument("--speechbrain-cache", type=Path)
    voice_registry_build.add_argument("--target-far", type=float, default=0.01, help="Maximum false-accept rate for calibration.")

    doctor = sub.add_parser("doctor", help="Check local runtime dependencies before processing a recording.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of Markdown.")
    doctor.add_argument("--strict", action="store_true", help="Exit non-zero when required checks are missing.")
    return parser


def _effective_media(input_path: Path, output_dir: Path, start: float, duration: float) -> tuple[Path, float]:
    if duration <= 0:
        return input_path, 0.0
    clip_path = output_dir / "work" / f"clip_{start:.0f}_{duration:.0f}.mov"
    return make_clip(input_path, clip_path, start=start, duration=duration), start


def _write_action_artifacts(
    output_dir: Path,
    segments: list[dict[str, Any]],
    statuses: dict[str, Any],
) -> dict[str, Any]:
    ledger = build_action_ledger(segments)
    write_json(output_dir / "action_items.json", ledger)
    write_action_item_ledger(output_dir / "action_items.md", ledger)
    summary = ledger["summary"]
    statuses["action_items"] = {
        "status": "ok" if not summary["review"] else "review_required",
        "accepted": summary["accepted"],
        "review": summary["review"],
        "constraints": summary["constraints"],
    }
    return ledger


def _write_ollama_draft(path: Path, content: str) -> None:
    path.write_text(f"{OLLAMA_DRAFT_NOTICE}{content.lstrip()}", encoding="utf-8")


def _deepseek_config(args: argparse.Namespace) -> DeepSeekConfig:
    env_file = args.deepseek_env_file.expanduser().resolve() if args.deepseek_env_file else None
    return DeepSeekConfig(
        model=args.deepseek_model,
        base_url=args.deepseek_base_url,
        api_key_env=args.deepseek_api_key_env,
        env_file=env_file,
        keychain_service=args.deepseek_keychain_service,
        timeout=args.deepseek_timeout,
        max_input_chars=args.deepseek_max_input_chars,
        output_language=args.deepseek_output_language,
        redacted_names=tuple(args.deepseek_redact_name),
        allow_unauthenticated_loopback=args.deepseek_allow_unauthenticated_loopback,
    )


def _next_stale_review_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.stem}.stale{path.suffix}")
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.stale-{index}{path.suffix}")
        index += 1
    return candidate


def _archive_stale_deepseek_reviews(output_dir: Path) -> list[str]:
    archived: list[str] = []
    for filename in ("minutes.deepseek.review.json", "minutes.deepseek.review.md"):
        source = output_dir / filename
        if source.exists():
            destination = _next_stale_review_path(source)
            source.replace(destination)
            archived.append(destination.name)
    return archived


def _write_deepseek_review_pair(output_dir: Path, review: dict[str, Any]) -> None:
    json_path = output_dir / "minutes.deepseek.review.json"
    markdown_path = output_dir / "minutes.deepseek.review.md"
    temporary_json = json_path.with_name(f".{json_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_markdown = markdown_path.with_name(f".{markdown_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_json(temporary_json, review)
        write_deepseek_review(temporary_markdown, review)
        temporary_json.replace(json_path)
        temporary_markdown.replace(markdown_path)
    except BaseException:
        for temporary_path in (temporary_json, temporary_markdown):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        _archive_stale_deepseek_reviews(output_dir)
        raise


def _write_deepseek_draft(
    output_dir: Path,
    *,
    segments: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    archived = _archive_stale_deepseek_reviews(output_dir)
    review, status = generate_deepseek_review(
        segments=segments,
        keyframes=keyframes,
        config=_deepseek_config(args),
    )
    if review:
        _write_deepseek_review_pair(output_dir, review)
        status["review_json"] = "minutes.deepseek.review.json"
        status["review_markdown"] = "minutes.deepseek.review.md"
        status["review_written"] = True
    else:
        status["review_written"] = False
    if archived:
        status["archived_stale_reviews"] = archived
    return status


def _deepseek_review_succeeded(status: dict[str, Any]) -> bool:
    return status.get("status") == "draft_only" and status.get("review_written") is True


def _report_deepseek_failure(status: dict[str, Any], output_dir: Path) -> None:
    status_name = str(status.get("status") or "unknown_failure")
    print(
        f"DeepSeek review was not generated ({status_name}). Local canonical artifacts remain available; see {output_dir / 'summary_status.json'}.",
        file=sys.stderr,
    )


def _run_visual_identity(
    *,
    input_path: Path,
    output_dir: Path,
    duration: float,
    segments: list[dict[str, Any]],
    profile_path: Path,
    max_frame_width: int,
) -> dict[str, Any]:
    profile_path = profile_path.expanduser().resolve()
    profile = load_visual_profile(profile_path)
    visual_dir = output_dir / "work" / "visual_identity_frames"
    sample_requests = build_segment_sample_requests(segments, profile, duration=duration)
    frames = extract_frames(input_path, unique_video_times(sample_requests), visual_dir, max_width=max_frame_width)
    write_json(output_dir / "visual_identity_samples.json", sample_requests)
    write_json(output_dir / "visual_identity_frames.json", frames)

    nameplate_manifest = build_nameplate_manifest(frames, profile)
    write_json(output_dir / "visual_identity_nameplate_manifest.json", nameplate_manifest)
    ocr_status: dict[str, Any]
    try:
        nameplate_ocr = ocr_regions(nameplate_manifest, output_dir / "work") if nameplate_manifest else []
        ocr_status = {"status": "ok", "frames": len(nameplate_ocr), "regions": sum(len(item["regions"]) for item in nameplate_manifest)}
    except Exception as exc:
        nameplate_ocr = []
        ocr_status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(output_dir / "visual_identity_nameplate_ocr.json", nameplate_ocr)

    slot_names = resolve_slot_names(profile, nameplate_ocr)
    scored_frames = score_visual_frames(frames, profile, slot_names)
    summary = attach_visual_identity(segments, sample_requests, scored_frames, profile)
    visual_payload = {
        "profile": str(profile_path),
        "settings": profile["settings"],
        "slot_names": slot_names,
        "summary": summary,
        "frames": scored_frames,
    }
    write_json(output_dir / "visual_identity.json", visual_payload)
    write_visual_identity_report(
        output_dir / "visual_identity_report.md",
        profile_path=profile_path,
        profile=profile,
        slot_names=slot_names,
        scored_frames=scored_frames,
        summary=summary,
    )
    return {
        "status": "audit_only" if summary["assignment_mode"] == "audit_only" else "ok" if summary["assigned"] else "partial",
        "assignments": summary["assigned"],
        "unresolved": summary["unresolved"],
        "conflicts": summary["conflicts"],
        "unvalidated_candidates": summary["unvalidated_candidates"],
        "visual_frames": len(scored_frames),
        "nameplate_ocr": ocr_status,
    }


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
    voice_registry_path = args.voice_registry.expanduser().resolve() if args.voice_registry else None
    if enrollment_path and voice_registry_path:
        raise ValueError("--enrollment and --voice-registry cannot be used together")
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
    if voice_registry_path:
        statuses["voice_registry"] = apply_voice_registry(
            audio_path,
            segments,
            voice_registry_path,
            speechbrain_cache=speechbrain_cache,
            threshold=args.registry_threshold,
            margin=args.registry_margin,
        )
        write_json(output_dir / "voice_registry_status.json", statuses["voice_registry"])
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

    if args.visual_profile:
        statuses["visual_identity"] = _run_visual_identity(
            input_path=effective_input,
            output_dir=output_dir,
            duration=duration,
            segments=segments,
            profile_path=args.visual_profile,
            max_frame_width=args.visual_max_frame_width,
        )

    keyframes = choose_keyframes(frames, segments)
    write_json(output_dir / "keyframes.json", keyframes)
    write_json(output_dir / "transcript.json", segments)
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)

    report_meta = {"input": str(input_path), "duration": duration, "source_offset": source_offset}
    write_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    if args.summary_engine == "ollama":
        ollama_minutes, statuses["summary"] = generate_ollama_minutes(
            segments=segments,
            keyframes=keyframes,
            metadata=report_meta,
            model=args.ollama_model,
        )
        if ollama_minutes:
            _write_ollama_draft(output_dir / "minutes.ollama.draft.md", ollama_minutes)
            statuses["summary"]["status"] = "draft_only"
    elif args.summary_engine == "deepseek":
        statuses["summary"] = _write_deepseek_draft(
            output_dir,
            segments=segments,
            keyframes=keyframes,
            args=args,
        )
        write_json(output_dir / "summary_status.json", statuses["summary"])
    else:
        statuses["summary"] = {"engine": "extractive", "status": "ok"}
    write_minutes(
        output_dir / "minutes.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
    write_json(output_dir / "run_status.json", {"elapsed_seconds": round(time.time() - started, 3), "statuses": statuses})
    if args.summary_engine == "deepseek" and not _deepseek_review_succeeded(statuses["summary"]):
        _report_deepseek_failure(statuses["summary"], output_dir)
        print(str(output_dir))
        return 2
    print(str(output_dir))
    return 0


def summarize_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    segments = read_json(output_dir / "transcript.json")
    keyframes = read_json(output_dir / "keyframes.json")
    metadata = read_json(output_dir / "metadata.json")
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    if args.summary_engine == "deepseek":
        status = _write_deepseek_draft(
            output_dir,
            segments=segments,
            keyframes=keyframes,
            args=args,
        )
        statuses["summary"] = status
        write_json(output_dir / "summary_status.json", status)
        write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
        if not _deepseek_review_succeeded(status):
            _report_deepseek_failure(status, output_dir)
            print(str(output_dir / "summary_status.json"))
            return 2
        destination = output_dir / ("minutes.deepseek.review.md" if status.get("review_markdown") else "summary_status.json")
        print(str(destination))
        return 0
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)
    report_meta = {
        "input": metadata.get("input"),
        "duration": metadata.get("duration", 0.0),
        "source_offset": metadata.get("source_offset", 0.0),
    }
    write_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    if args.summary_engine == "ollama":
        minutes, status = generate_ollama_minutes(
            segments=segments,
            keyframes=keyframes,
            metadata=report_meta,
            model=args.ollama_model,
        )
        if minutes:
            _write_ollama_draft(output_dir / "minutes.ollama.draft.md", minutes)
            status["status"] = "draft_only"
        write_json(output_dir / "summary_status.json", status)
        statuses["summary"] = status
    else:
        statuses["summary"] = {"engine": "extractive", "status": "ok"}
    write_minutes(
        output_dir / "minutes.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    write_quality_report(
        output_dir / "quality_report.md",
        segments=segments,
        ocr_records=ocr_records,
        keyframes=keyframes,
        statuses=statuses,
    )
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
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
    voice_registry_path = args.voice_registry.expanduser().resolve() if args.voice_registry else None
    if enrollment_path and voice_registry_path:
        raise ValueError("--enrollment and --voice-registry cannot be used together")
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
    if voice_registry_path:
        voice_registry_status = apply_voice_registry(
            audio_path,
            segments,
            voice_registry_path,
            speechbrain_cache=speechbrain_cache,
            threshold=args.registry_threshold,
            margin=args.registry_margin,
        )
        write_json(output_dir / "voice_registry_status.json", voice_registry_status)
    else:
        voice_registry_status = None
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
    if voice_registry_status:
        statuses["voice_registry"] = voice_registry_status
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
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


def build_voice_registry_existing(args: argparse.Namespace) -> int:
    payload = build_voice_registry(
        args.sources,
        args.output,
        work_dir=args.work_dir,
        speechbrain_cache=args.speechbrain_cache,
        target_far=args.target_far,
    )
    print(str(args.output.expanduser().resolve()))
    return 0 if payload.get("calibration") else 1


def visual_identify_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    metadata = read_json(output_dir / "metadata.json")
    input_path = args.input.expanduser().resolve() if args.input else Path(str(metadata.get("effective_input") or metadata.get("input"))).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Visual identity input does not exist: {input_path}")
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    segments = read_json(transcript_path)
    status = _run_visual_identity(
        input_path=input_path,
        output_dir=output_dir,
        duration=float(metadata.get("duration", 0.0)),
        segments=segments,
        profile_path=args.visual_profile,
        max_frame_width=args.max_frame_width,
    )
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["visual_identity"] = status
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)
    write_json(output_dir / "transcript.json", segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
    report_meta = {
        "input": metadata.get("input"),
        "duration": metadata.get("duration", 0.0),
        "source_offset": metadata.get("source_offset", 0.0),
    }
    write_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
    print(str(output_dir / "visual_identity_report.md"))
    return 0


def relabel_existing(args: argparse.Namespace) -> int:
    """Apply a reviewed speaker map to an already processed recording.

    This intentionally works from ``transcript.json`` rather than the raw
    transcript so a reviewed map can correct a bad cross-recording registry
    result without repeating expensive model inference.
    """

    output_dir = args.output_dir.expanduser().resolve()
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    participant_map = load_participant_map(args.participant_map)
    if not participant_map:
        raise ValueError("participant map must contain at least one speaker mapping")
    if "Speaker Unknown" in participant_map:
        raise ValueError("participant map must not assign Speaker Unknown")
    mapped_names = {name.strip() for name in participant_map.values() if name.strip()}
    if len(mapped_names) != len(participant_map):
        raise ValueError("participant map values must be non-empty unique names")

    segments = read_json(transcript_path)
    known_speakers = {str(segment.get("speaker") or "Speaker Unknown") for segment in segments}
    unknown_speakers = sorted(set(participant_map) - known_speakers)
    if unknown_speakers:
        raise ValueError(f"participant map contains speaker labels absent from transcript: {', '.join(unknown_speakers)}")

    # A reviewed map is recording-specific evidence. Clear weaker registry
    # guesses first, then attach only the reviewed labels.
    cleared_registry_segments = 0
    for segment in segments:
        if segment.get("name_source") == "voice_registry":
            segment["name"] = None
            segment["name_source"] = None
            segment["name_confidence"] = 0.0
            cleared_registry_segments += 1
    attach_names(segments, [], participant_map, allow_ocr_names=False)

    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    metadata = read_json(output_dir / "metadata.json") if (output_dir / "metadata.json").exists() else {}
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["identity_mapping"] = {
        "status": "reviewed_participant_map",
        "mapped_speakers": sorted(participant_map),
        "cleared_voice_registry_segments": cleared_registry_segments,
    }
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)

    write_json(output_dir / "transcript.json", segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
    write_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata={
            "input": metadata.get("input"),
            "duration": metadata.get("duration", 0.0),
            "source_offset": metadata.get("source_offset", 0.0),
        },
        action_ledger=action_ledger,
    )
    write_json(
        output_dir / "participant_map_status.json",
        {
            "status": "reviewed_participant_map",
            "participant_map": participant_map,
            "cleared_voice_registry_segments": cleared_registry_segments,
        },
    )
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
    print(str(output_dir / "transcript.md"))
    return 0


def audit_actions_existing(args: argparse.Namespace) -> int:
    """Regenerate the deterministic action ledger for an existing transcript."""

    output_dir = args.output_dir.expanduser().resolve()
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    segments = read_json(transcript_path)
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    write_quality_report(
        output_dir / "quality_report.md",
        segments=segments,
        ocr_records=ocr_records,
        keyframes=keyframes,
        statuses=statuses,
    )
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
    print(str(output_dir / "action_items.md"))
    return 0


def validate_actions_existing(args: argparse.Namespace) -> int:
    """Apply the deterministic publication gate to proposed action rows."""

    output_dir = args.output_dir.expanduser().resolve()
    ledger_path = output_dir / "action_items.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Expected action ledger at {ledger_path}; run audit-actions first")
    ledger = read_json(ledger_path)
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    transcript = read_json(transcript_path)
    current_transcript_hash = transcript_fingerprint(transcript)
    regenerated_ledger = build_action_ledger(transcript)
    ledger_transcript_hash = ledger.get("transcript_sha256")
    freshness_errors = []
    if not ledger_transcript_hash:
        freshness_errors.append("ledger_missing_transcript_hash")
    elif ledger_transcript_hash != current_transcript_hash:
        freshness_errors.append("ledger_stale")
    if ledger != regenerated_ledger:
        freshness_errors.append("ledger_tampered")
    payload = read_json(args.items)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("--items must be a JSON array or an object containing an items array")
    results = [
        {"index": index, "errors": sorted(set(freshness_errors + validate_published_action_item(item, regenerated_ledger)))}
        for index, item in enumerate(items)
    ]
    rejected = [result for result in results if result["errors"]]
    report = {
        "status": "ok" if not rejected else "rejected",
        "accepted": len(results) - len(rejected),
        "rejected": len(rejected),
        "ledger_fresh": not freshness_errors,
        "results": results,
    }
    report_path = output_dir / "action_items.validation.json"
    write_json(report_path, report)
    print(str(report_path))
    return 0 if not rejected else 1


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
    if args.command == "visual-identify":
        return visual_identify_existing(args)
    if args.command == "relabel":
        return relabel_existing(args)
    if args.command == "audit-actions":
        return audit_actions_existing(args)
    if args.command == "validate-actions":
        return validate_actions_existing(args)
    if args.command == "voice-template":
        return write_voice_template(args)
    if args.command == "voice-registry":
        if args.voice_registry_command == "build":
            return build_voice_registry_existing(args)
        parser.error("unknown voice registry command")
    if args.command == "doctor":
        checks = collect_environment_checks()
        if args.json:
            import json

            print(json.dumps({"checks": checks}, ensure_ascii=False, indent=2))
        else:
            print(render_doctor_report(checks), end="")
        return doctor_exit_code(checks, strict=args.strict)
    parser.error("unknown command")
    return 2
