from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from meeting_minutes.action_items import (
    build_action_ledger,
    stable_segment_id,
    transcript_fingerprint,
    validate_published_action_item,
)

from .agent_visual_audit import (
    AGENT_VISUAL_AUDIT_FORMAT,
    apply_agent_visual_audit_veto,
    build_agent_visual_audit_veto,
    parse_agent_visual_audit_response,
    render_agent_visual_audit_prompt,
    restore_direct_visual_candidates_from_manifest,
    summarize_agent_visual_audits,
    validate_agent_visual_audit_response,
    validate_agent_visual_audit_manifest_content,
    write_agent_visual_audit_bundle,
    write_agent_visual_audit_report,
)
from .asr import transcribe_audio
from .avatar_template_identity import (
    attach_avatar_template_identity,
    build_avatar_templates,
    calibrate_avatar_templates,
    load_avatar_template_profile,
    score_avatar_template_frames,
    serializable_template_library,
    write_avatar_template_identity_report,
)
from .deepseek import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MAX_INPUT_CHARS,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    DeepSeekConfig,
    generate_deepseek_review,
    write_deepseek_review,
)
from .diarization import attach_speakers, diarize_audio, split_segments_by_turns
from .direct_visual_cluster_identity import (
    apply_direct_visual_cluster_identity,
    build_direct_visual_cluster_identity,
    clear_direct_visual_cluster_identity,
    direct_visual_active_frame_count,
    load_direct_visual_cluster_config,
    write_direct_visual_cluster_report,
)
from .doctor import collect_environment_checks, doctor_exit_code, render_doctor_report
from .dynamic_visual_identity import (
    attach_dynamic_ocr,
    attach_dynamic_visual_identity,
    build_dynamic_ocr_manifest,
    build_dynamic_sample_requests,
    detect_dynamic_visual_frames,
    load_dynamic_visual_profile,
    unique_dynamic_video_times,
    write_dynamic_visual_identity_report,
)
from .identity import attach_names, load_participant_map
from .jsonio import read_json, write_json
from .keyframes import choose_keyframes, keyword_times, regular_times
from .media import (
    extract_audio,
    extract_frames,
    make_clip,
    ocr_frames,
    ocr_manifest_diagnostics,
    ocr_regions,
    probe_media,
)
from .minutes_contract import (
    parse_shareable_action_rows,
    parse_shareable_project_update_rows,
    validate_bilingual_minutes,
    validate_shareable_minutes,
    validate_shareable_minutes_en,
)
from .publication import (
    ACTION_EVIDENCE_FORMAT,
    ACTION_INTENT_REVIEW_FORMAT,
    PROJECT_EVIDENCE_FORMAT,
    PROJECT_UPDATE_COVERAGE_MIN_SECONDS,
    PUBLICATION_FORMAT,
    action_intent_recall_signals,
    action_ledger_fingerprint,
    build_reviewed_evidence_manifests,
    canonical_minutes_paths,
    payload_fingerprint,
    project_update_coverage_snapshot,
    recompute_project_update_coverage,
    share_bundle_paths,
    sync_publication_status,
    validate_reviewed_action_evidence,
    validate_reviewed_action_intent_review,
    validate_reviewed_project_evidence,
)
from .report import (
    write_action_item_ledger,
    write_extractive_minutes,
    write_quality_report,
    write_review_queue,
    write_speaker_samples,
    write_transcript_markdown,
)
from .roster_avatar_identity import (
    attach_roster_avatar_identity,
    build_reviewed_anchor_requests,
    build_roster_ocr_manifest,
    build_roster_sample_requests,
    calibrate_roster_avatar_identity,
    detect_roster_active_frames,
    load_roster_avatar_profile,
    score_roster_avatar_frames,
    unique_roster_video_times,
    write_roster_avatar_identity_report,
)
from .smart_minutes import (
    SMART_MINUTES_AUDIT_FORMAT,
    generate_smart_minutes,
    repair_reviewed_minutes_text_fields,
    sanitize_reviewed_smart_minutes,
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
from .visual_voice_identity import (
    apply_visual_voice_registry,
    build_visual_voice_registry,
    clear_visual_voice_identity,
    direct_visual_enrollment_frame_count,
    load_visual_voice_config,
    restrict_visual_enrollment_to_agent_confirmed_segments,
    write_visual_voice_report,
)
from .voice_registry import apply_voice_registry, build_voice_registry, enforce_registry_cluster_consensus

OLLAMA_DRAFT_NOTICE = (
    "> **草稿，仅供核对。** 此文件是模型草稿，不可作为最终会议纪要分享；"
    "请以 `minutes.md`、`minutes.en.md` 和经校验的行动项账本为准。\n\n"
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
    run.add_argument(
        "--dynamic-visual-profile",
        type=Path,
        help="Coordinate-free dynamic visual profile. Uses same-frame active border plus nameplate OCR only.",
    )

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

    smart_summarize = sub.add_parser(
        "smart-summarize",
        help="Generate bilingual minutes from the named transcript, then run one or two full-transcript AI review passes.",
    )
    smart_summarize.add_argument("--output-dir", required=True, type=Path)
    smart_summarize.add_argument("--review-passes", type=int, choices=[1, 2], default=2)
    smart_summarize.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    smart_summarize.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    smart_summarize.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    smart_summarize.add_argument(
        "--deepseek-env-file",
        type=Path,
        help="Optional .env file; only the variable named by --deepseek-api-key-env is read.",
    )
    smart_summarize.add_argument("--deepseek-keychain-service")
    smart_summarize.add_argument("--deepseek-redact-name", action="append", default=[])
    smart_summarize.add_argument("--deepseek-allow-unauthenticated-loopback", action="store_true")
    smart_summarize.add_argument("--deepseek-timeout", type=int, default=DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)
    smart_summarize.add_argument(
        "--deepseek-max-input-chars",
        type=int,
        default=DEFAULT_DEEPSEEK_MAX_INPUT_CHARS,
    )
    smart_summarize.set_defaults(deepseek_output_language="zh-CN")

    smart_sanitize = sub.add_parser(
        "smart-repair-reviewed",
        aliases=["smart-sanitize-reviewed"],
        help="Deterministically revalidate a reviewed bilingual artifact, neutralize anonymous labels, and remove disproven action ownership.",
    )
    smart_sanitize.add_argument("--output-dir", required=True, type=Path)
    smart_sanitize.add_argument(
        "--source-smart-json",
        required=True,
        type=Path,
        help="Previously reviewed minutes.smart.json for the same transcript.",
    )
    smart_sanitize.add_argument(
        "--source-audit-json",
        required=True,
        type=Path,
        help="Previously reviewed minutes.smart.audit.json with a publishable final review for the same transcript.",
    )
    smart_sanitize.add_argument(
        "--deepseek-repair",
        action="store_true",
        help="Use a bounded text-only DeepSeek repair when current validation rejects an otherwise reviewed artifact.",
    )
    smart_sanitize.add_argument("--deepseek-model", default=DEFAULT_DEEPSEEK_MODEL)
    smart_sanitize.add_argument("--deepseek-base-url", default=DEFAULT_DEEPSEEK_BASE_URL)
    smart_sanitize.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    smart_sanitize.add_argument(
        "--deepseek-env-file",
        type=Path,
        help="Optional .env file; only the variable named by --deepseek-api-key-env is read.",
    )
    smart_sanitize.add_argument("--deepseek-keychain-service")
    smart_sanitize.add_argument("--deepseek-redact-name", action="append", default=[])
    smart_sanitize.add_argument("--deepseek-allow-unauthenticated-loopback", action="store_true")
    smart_sanitize.add_argument("--deepseek-timeout", type=int, default=DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)
    smart_sanitize.add_argument(
        "--deepseek-max-input-chars",
        type=int,
        default=DEFAULT_DEEPSEEK_MAX_INPUT_CHARS,
    )
    smart_sanitize.set_defaults(deepseek_output_language="zh-CN")

    validate_minutes = sub.add_parser("validate-minutes", help="Validate a shareable Chinese minutes document against the fixed format.")
    validate_minutes.add_argument("--path", required=True, type=Path)
    validate_minutes.add_argument("--duration", type=float, default=0.0, help="Optional recording duration in seconds for coverage validation.")
    validate_minutes.add_argument("--language", choices=["zh", "en"], default="zh")

    publish_minutes = sub.add_parser(
        "publish-minutes",
        help="Publish a reviewed minutes document only after it passes the fixed shareable format.",
    )
    publish_minutes.add_argument("--output-dir", required=True, type=Path)
    publish_minutes.add_argument("--source", required=True, type=Path, help="Reviewed Chinese source document.")
    publish_minutes.add_argument("--english-source", required=True, type=Path, help="Reviewed English companion source document.")
    publish_minutes.add_argument(
        "--action-evidence",
        type=Path,
        help="Internal reviewed action-evidence JSON. Required only when the shareable minutes contain action rows.",
    )
    publish_minutes.add_argument(
        "--action-intent-review",
        type=Path,
        help="Internal disposition JSON required for independently recalled weak action-intent speech.",
    )
    publish_minutes.add_argument(
        "--project-evidence",
        type=Path,
        help="Internal reviewed project-update evidence JSON. Required when project rows or covered participants exist.",
    )
    publish_minutes.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds. Defaults to metadata.json in the output directory.",
    )

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

    agent_visual_audit = sub.add_parser(
        "visual-agent-audit",
        help="Package or run read-only Codex/Cursor checks over direct active-speaker highlight frames.",
    )
    agent_visual_audit.add_argument("--output-dir", required=True, type=Path)
    agent_visual_audit.add_argument(
        "--visual-identity-path",
        type=Path,
        help="Explicit direct visual identity artifact when both static and dynamic artifacts exist.",
    )
    agent_visual_audit.add_argument("--samples-per-identity", type=int, default=2)
    agent_visual_audit.add_argument("--max-samples", type=int, default=24)
    agent_visual_audit.add_argument(
        "--full-coverage",
        action="store_true",
        help="Audit one direct same-frame nameplate crop for every active-speaker named segment. Required before named identities can be published.",
    )
    agent_visual_audit.add_argument(
        "--calibration-frame",
        action="append",
        type=Path,
        default=[],
        help="Same-recording full frame that visibly maps names to fixed grid slots; repeat for multiple references.",
    )
    agent_visual_audit.add_argument(
        "--calibration-layout",
        help="Static layout name for --calibration-frame. Inferred only when the audit has one layout.",
    )
    agent_visual_audit.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum frames sent to one agent request. Smaller batches isolate stalled agents.",
    )
    agent_visual_audit.add_argument(
        "--retry-unconfirmed",
        action="store_true",
        help="Reuse confirmed samples from the exact content-bound manifest and rerun only uncertain or rejected samples.",
    )
    agent_visual_audit.add_argument(
        "--run-agent",
        action="append",
        choices=["codex", "cursor"],
        default=[],
        help="Run one local read-only coding agent. Repeat to require independent reviews.",
    )
    agent_visual_audit.add_argument("--workspace", type=Path, help="Workspace supplied to the local coding agents.")
    agent_visual_audit.add_argument("--codex-bin", default="codex")
    agent_visual_audit.add_argument("--cursor-bin", default="cursor-agent")
    agent_visual_audit.add_argument("--cursor-model", default="cursor-grok-4.5-high")
    agent_visual_audit.add_argument("--timeout", type=int, default=600)
    agent_visual_audit.add_argument(
        "--require-consensus",
        action="store_true",
        help="Return a nonzero exit code unless every sampled frame is confirmed by every requested agent.",
    )

    dynamic_visual_identify = sub.add_parser(
        "dynamic-visual-identify",
        help="Apply coordinate-free same-frame active-speaker and nameplate evidence to an existing run.",
    )
    dynamic_visual_identify.add_argument("--output-dir", required=True, type=Path)
    dynamic_visual_identify.add_argument("--dynamic-visual-profile", required=True, type=Path)
    dynamic_visual_identify.add_argument("--input", type=Path, help="Override the effective input recorded in metadata.json.")
    dynamic_visual_identify.add_argument("--max-frame-width", type=int, default=1280)

    avatar_template_identify = sub.add_parser(
        "avatar-template-identify",
        help="Apply evidence-gated avatar templates from reviewed in-tile nameplate reference frames.",
    )
    avatar_template_identify.add_argument("--output-dir", required=True, type=Path)
    avatar_template_identify.add_argument("--avatar-template-profile", required=True, type=Path)

    roster_avatar_identify = sub.add_parser(
        "roster-avatar-identify",
        help="Apply reviewed same-frame Discord roster-avatar identity evidence to an existing run.",
    )
    roster_avatar_identify.add_argument("--output-dir", required=True, type=Path)
    roster_avatar_identify.add_argument("--roster-avatar-profile", required=True, type=Path)
    roster_avatar_identify.add_argument("--input", type=Path, help="Override the effective input recorded in metadata.json.")
    roster_avatar_identify.add_argument("--max-frame-width", type=int, default=1280)

    visual_voice_identify = sub.add_parser(
        "visual-voice-identify",
        help="Build a same-session voiceprint registry from direct in-tile nameplates or reviewed visual slots, then apply its held-out precision gate.",
    )
    visual_voice_identify.add_argument("--output-dir", required=True, type=Path)
    visual_voice_identify.add_argument("--config", type=Path, help="Optional same-session visual voice settings JSON.")
    visual_voice_identify.add_argument("--speechbrain-cache", type=Path)
    visual_voice_identify.add_argument(
        "--visual-identity-path",
        type=Path,
        help="Explicit current visual identity JSON when both static and dynamic artifacts exist.",
    )

    visual_cluster_identify = sub.add_parser(
        "visual-cluster-identify",
        help="Propagate direct visual identities through an independently validated diarization cluster.",
    )
    visual_cluster_identify.add_argument("--output-dir", required=True, type=Path)
    visual_cluster_identify.add_argument("--config", type=Path, help="Optional direct visual cluster settings JSON.")
    visual_cluster_identify.add_argument(
        "--visual-identity-path",
        type=Path,
        help="Explicit current visual identity JSON when both static and dynamic artifacts exist.",
    )

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

    voice_registry_apply = voice_registry_sub.add_parser(
        "apply",
        help="Apply a calibrated registry to an existing diarized run without rerunning diarization.",
    )
    voice_registry_apply.add_argument("--output-dir", required=True, type=Path)
    voice_registry_apply.add_argument("--registry", required=True, type=Path, help="Cross-recording voice registry JSON.")
    voice_registry_apply.add_argument("--registry-threshold", type=float, help="Override the registry score threshold.")
    voice_registry_apply.add_argument("--registry-margin", type=float, help="Override the best-vs-runner-up score margin.")
    voice_registry_apply.add_argument("--speechbrain-cache", type=Path)

    voice_registry_consensus = voice_registry_sub.add_parser(
        "consensus",
        help="Enforce cluster-level consistency on existing cross-recording registry identities.",
    )
    voice_registry_consensus.add_argument("--output-dir", required=True, type=Path)

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
        "intent_signals": int((ledger.get("intent_recall") or {}).get("summary", {}).get("signals", 0) or 0),
        "intent_unmatched": int((ledger.get("intent_recall") or {}).get("summary", {}).get("unmatched_signals", 0) or 0),
    }
    statuses["publication"] = sync_publication_status(output_dir, segments, ledger)
    return ledger


def _action_ledger_freshness_errors(segments: list[dict[str, Any]], ledger: dict[str, Any]) -> list[str]:
    current_transcript_hash = transcript_fingerprint(segments)
    regenerated_ledger = build_action_ledger(segments)
    errors: list[str] = []
    if not ledger.get("transcript_sha256"):
        errors.append("ledger_missing_transcript_hash")
    elif ledger.get("transcript_sha256") != current_transcript_hash:
        errors.append("ledger_stale")
    if ledger != regenerated_ledger:
        errors.append("ledger_tampered")
    return sorted(set(errors))


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


def _archive_stale_smart_reviews(review_dir: Path) -> list[str]:
    archived: list[str] = []
    for filename in (
        "minutes.smart.md",
        "minutes.smart.en.md",
        "minutes.smart.json",
        "minutes.smart.audit.json",
    ):
        source = review_dir / filename
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


def _recording_content_sha256(input_path: Path) -> str:
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Visual identity input does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visual_recording_provenance(input_path: Path, duration: float) -> dict[str, Any]:
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Visual identity input does not exist: {path}")
    return {
        "effective_input": str(path),
        "duration": round(float(duration), 3),
        "size_bytes": path.stat().st_size,
        "content_sha256": _recording_content_sha256(path),
    }


def _select_visual_identity_artifact(
    *,
    output_dir: Path,
    visual_identity_path: Path | None,
    expected_recording: dict[str, Any] | None,
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Choose one verified direct-visual artifact or return a safe skip status."""

    if visual_identity_path is not None:
        visual_path = visual_identity_path.expanduser().resolve()
        if not visual_path.is_file():
            raise FileNotFoundError(f"Expected visual identity artifact at {visual_path}")
        existing_visual = [(visual_path, read_json(visual_path))]
    else:
        visual_candidates = [
            output_dir / "dynamic_visual_identity.json",
            output_dir / "visual_identity.json",
        ]
        existing_visual = [(path, read_json(path)) for path in visual_candidates if path.exists()]
    if not existing_visual:
        return None, None, {"status": "skipped_no_direct_visual_identity"}
    if len(existing_visual) > 1:
        return None, None, {
            "status": "skipped_ambiguous_visual_identity_artifacts",
            "visual_candidates": [str(path) for path, _payload in existing_visual],
        }
    visual_path, visual_payload = existing_visual[0]
    if expected_recording is not None and visual_payload.get("recording") != expected_recording:
        return None, None, {
            "status": "skipped_visual_identity_recording_mismatch",
            "visual_source": str(visual_path),
            "expected_recording": expected_recording,
            "observed_recording": visual_payload.get("recording"),
        }
    return visual_path, visual_payload, None


def _write_direct_visual_cluster_skip_artifact(output_dir: Path, status: dict[str, Any]) -> None:
    payload = {
        "format": "direct-visual-voice-cluster-identity/v2",
        "status": status.get("status"),
        "clusters": {},
        "accepted_clusters": {},
        "note": "No cluster propagation was applied because direct visual evidence was unavailable, ambiguous, or did not match the current recording.",
    }
    write_json(output_dir / "direct_visual_cluster_identity.json", payload)
    lines = [
        "# Direct Visual Voice Cluster Identity Report",
        "",
        "## Application",
        f"- Status: {status.get('status', 'unknown')}",
        f"- Cleared prior cluster-propagated segments before rerun: {status.get('cleared_prior_cluster_assignments', 0)}",
        "- No cluster propagation was applied.",
    ]
    (output_dir / "direct_visual_cluster_identity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_visual_voice_skip_artifacts(output_dir: Path, status: dict[str, Any]) -> None:
    registry = {
        "format": "same-session-visual-voice-registry/v2",
        "status": status.get("status"),
        "profiles": {},
        "rejected_visual_names": {},
        "calibration": {"status": status.get("status")},
        "note": "No same-session voiceprint assignment is retained until current direct visual enrollment evidence passes selection and calibration.",
    }
    write_json(output_dir / "same_session_visual_voice_registry.json", registry)
    write_visual_voice_report(output_dir / "same_session_visual_voice_report.md", registry=registry, status=status)


def _run_visual_identity(
    *,
    input_path: Path,
    output_dir: Path,
    duration: float,
    segments: list[dict[str, Any]],
    profile_path: Path,
    max_frame_width: int,
    recording_provenance: dict[str, Any] | None = None,
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
        "recording": recording_provenance or _visual_recording_provenance(input_path, duration),
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


def _run_dynamic_visual_identity(
    *,
    input_path: Path,
    output_dir: Path,
    duration: float,
    segments: list[dict[str, Any]],
    profile_path: Path,
    max_frame_width: int,
    recording_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_path = profile_path.expanduser().resolve()
    profile = load_dynamic_visual_profile(profile_path)
    visual_dir = output_dir / "work" / "dynamic_visual_identity_frames"
    sample_requests = build_dynamic_sample_requests(segments, profile, duration=duration)
    frames = extract_frames(input_path, unique_dynamic_video_times(sample_requests), visual_dir, max_width=max_frame_width)
    write_json(output_dir / "dynamic_visual_identity_samples.json", sample_requests)
    write_json(output_dir / "dynamic_visual_identity_frames.json", frames)

    detected_frames = detect_dynamic_visual_frames(frames, profile)
    write_json(output_dir / "dynamic_visual_identity_detection.json", detected_frames)
    ocr_manifest = build_dynamic_ocr_manifest(detected_frames)
    write_json(output_dir / "dynamic_visual_identity_ocr_manifest.json", ocr_manifest)
    try:
        ocr_records = ocr_frames(ocr_manifest, output_dir / "work") if ocr_manifest else []
        ocr_status: dict[str, Any] = {
            "status": "ok",
            "frames": len(ocr_records),
            "manifest_diagnostics": ocr_manifest_diagnostics(output_dir / "work"),
        }
    except Exception as exc:
        ocr_records = []
        ocr_status = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(output_dir / "dynamic_visual_identity_ocr.json", ocr_records)

    scored_frames = attach_dynamic_ocr(detected_frames, ocr_records, profile)
    summary = attach_dynamic_visual_identity(segments, sample_requests, scored_frames, profile)
    visual_payload = {
        "recording": recording_provenance or _visual_recording_provenance(input_path, duration),
        "profile": str(profile_path),
        "settings": profile["settings"],
        "summary": summary,
        "frames": scored_frames,
    }
    write_json(output_dir / "dynamic_visual_identity.json", visual_payload)
    write_dynamic_visual_identity_report(
        output_dir / "dynamic_visual_identity_report.md",
        profile_path=profile_path,
        profile=profile,
        scored_frames=scored_frames,
        summary=summary,
    )
    # The generic report name must represent the authoritative identity stage, not an older static profile run.
    write_dynamic_visual_identity_report(
        output_dir / "visual_identity_report.md",
        profile_path=profile_path,
        profile=profile,
        scored_frames=scored_frames,
        summary=summary,
    )
    return {
        "status": "ok" if summary["assigned"] else "partial",
        "assignments": summary["assigned"],
        "unresolved": summary["unresolved"],
        "conflicts": summary["conflicts"],
        "visual_frames": len(scored_frames),
        "named_active_frames": summary["named_active_frames"],
        "nameplate_ocr": ocr_status,
        "mode": "same_frame_dynamic_active_tile_plus_nameplate",
    }


def _run_direct_visual_cluster_identity(
    *,
    output_dir: Path,
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    config_path: Path | None = None,
    visual_identity_path: Path | None = None,
    expected_recording: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Retract first. A skipped validation run must never retain this stage's
    # previous name propagation as if the current visual evidence approved it.
    cleared = clear_direct_visual_cluster_identity(segments)
    visual_path, visual_payload, skip_status = _select_visual_identity_artifact(
        output_dir=output_dir,
        visual_identity_path=visual_identity_path,
        expected_recording=expected_recording,
    )
    if skip_status is not None:
        status = {**skip_status, "cleared_prior_cluster_assignments": cleared}
        _write_direct_visual_cluster_skip_artifact(output_dir, status)
        return status
    assert visual_path is not None and visual_payload is not None
    visual_payload = restrict_visual_enrollment_to_agent_confirmed_segments(
        visual_payload,
        segments,
    )
    if direct_visual_active_frame_count(visual_payload) == 0:
        status = {
            "status": "skipped_no_direct_visual_active_frames",
            "visual_source": str(visual_path),
            "cleared_prior_cluster_assignments": cleared,
            "agent_visual_audit_filter": visual_payload.get(
                "agent_visual_audit_filter",
                {},
            ),
        }
        _write_direct_visual_cluster_skip_artifact(output_dir, status)
        return status
    settings = load_direct_visual_cluster_config(config_path)
    payload = build_direct_visual_cluster_identity(visual_payload, turns, settings=settings)
    status = {
        **apply_direct_visual_cluster_identity(segments, payload),
        "visual_source": str(visual_path),
        "cleared_prior_cluster_assignments": cleared,
    }
    write_json(output_dir / "direct_visual_cluster_identity.json", payload)
    write_direct_visual_cluster_report(output_dir / "direct_visual_cluster_identity_report.md", payload=payload, status=status)
    return status


def run_pipeline(args: argparse.Namespace) -> int:
    started = time.time()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    frames_dir = output_dir / "keyframes"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.visual_profile and args.dynamic_visual_profile:
        raise ValueError("--visual-profile and --dynamic-visual-profile cannot be used together; dynamic mode has no static-coordinate fallback")

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
    raw_segment_count = len(segments)
    segments = split_segments_by_turns(segments, turns)
    statuses["word_timing_split"] = {
        "status": "ok" if turns else "skipped",
        "input_segments": raw_segment_count,
        "output_segments": len(segments),
        "split_source_segments": len({segment["split_from"] for segment in segments if segment.get("split_from")}),
    }
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
            statuses["ocr"] = {
                "status": "ok",
                "frames": len(ocr_records),
                "manifest_diagnostics": ocr_manifest_diagnostics(work_dir),
            }
        except Exception as exc:
            ocr_records = []
            statuses["ocr"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(output_dir / "ocr.json", ocr_records)

    participant_map = load_participant_map(args.participant_map)
    attach_names(segments, ocr_records, participant_map, allow_ocr_names=args.auto_ocr_names)

    visual_identity_path: Path | None = None
    visual_recording_provenance = (
        _visual_recording_provenance(effective_input, duration)
        if args.visual_profile or args.dynamic_visual_profile
        else None
    )
    if args.visual_profile:
        statuses["visual_identity"] = _run_visual_identity(
            input_path=effective_input,
            output_dir=output_dir,
            duration=duration,
            segments=segments,
            profile_path=args.visual_profile,
            max_frame_width=args.visual_max_frame_width,
            recording_provenance=visual_recording_provenance,
        )
        visual_identity_path = output_dir / "visual_identity.json"
    elif args.dynamic_visual_profile:
        statuses["visual_identity"] = _run_dynamic_visual_identity(
            input_path=effective_input,
            output_dir=output_dir,
            duration=duration,
            segments=segments,
            profile_path=args.dynamic_visual_profile,
            max_frame_width=args.visual_max_frame_width,
            recording_provenance=visual_recording_provenance,
        )
        visual_identity_path = output_dir / "dynamic_visual_identity.json"
    if args.visual_profile or args.dynamic_visual_profile:
        statuses["direct_visual_cluster_identity"] = _run_direct_visual_cluster_identity(
            output_dir=output_dir,
            segments=segments,
            turns=turns,
            visual_identity_path=visual_identity_path,
            expected_recording=visual_recording_provenance,
        )

    keyframes = choose_keyframes(frames, segments)
    write_json(output_dir / "keyframes.json", keyframes)
    write_json(output_dir / "transcript.json", segments)
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)

    report_meta = {"input": str(input_path), "duration": duration, "source_offset": source_offset}
    write_extractive_minutes(
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
    write_extractive_minutes(
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
    destination = output_dir / "minutes.extractive.md"
    if args.summary_engine == "ollama" and (output_dir / "minutes.ollama.draft.md").exists():
        destination = output_dir / "minutes.ollama.draft.md"
    print(str(destination))
    return 0


def smart_summarize_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.is_file():
        print("smart_summary_transcript_missing", file=sys.stderr)
        return 1
    segments, identity_audit_status = _load_identity_audited_segments(output_dir)
    if not isinstance(segments, list):
        print("smart_summary_transcript_invalid", file=sys.stderr)
        return 1

    if identity_audit_status.get("publishable") is not True:
        review_dir = output_dir / "work" / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        archived = _archive_stale_smart_reviews(review_dir)
        status = {
            "status": "identity_audit_not_publishable",
            "identity_audit_gate": identity_audit_status,
            "archived_stale_smart_reviews": archived,
        }
        run_status_path = output_dir / "run_status.json"
        run_status = read_json(run_status_path) if run_status_path.is_file() else {}
        statuses = dict(run_status.get("statuses", {}))
        statuses["smart_summary"] = status
        write_json(output_dir / "summary_status.json", status)
        write_json(run_status_path, {**run_status, "statuses": statuses})
        print("smart_summary_identity_audit_not_publishable", file=sys.stderr)
        return 2

    run_status_path = output_dir / "run_status.json"
    run_status = read_json(run_status_path) if run_status_path.is_file() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["identity_audit_gate"] = identity_audit_status
    _write_action_artifacts(output_dir, segments, statuses)
    review_dir = output_dir / "work" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    archived = _archive_stale_smart_reviews(review_dir)
    checkpoint_path = review_dir / "minutes.smart.checkpoint.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else None

    result, status = generate_smart_minutes(
        segments=segments,
        config=_deepseek_config(args),
        review_passes=args.review_passes,
        checkpoint=checkpoint if isinstance(checkpoint, dict) else None,
        checkpoint_callback=lambda payload: write_json(checkpoint_path, payload),
        allow_cluster_name_consensus=False,
    )
    status = {
        **status,
        "archived_stale_smart_reviews": archived,
        "identity_audit_gate": identity_audit_status,
    }
    statuses["smart_summary"] = status
    write_json(output_dir / "summary_status.json", status)
    write_json(run_status_path, {**run_status, "statuses": statuses})
    if result is None:
        print(f"smart_summary_failed:{status.get('status', 'unknown')}", file=sys.stderr)
        for error in status.get("errors", []):
            print(str(error), file=sys.stderr)
        return 2

    chinese_path = review_dir / "minutes.smart.md"
    english_path = review_dir / "minutes.smart.en.md"
    _atomic_write_markdown(chinese_path, result.chinese_markdown)
    _atomic_write_markdown(english_path, result.english_markdown)
    write_json(review_dir / "minutes.smart.json", result.payload)
    write_json(review_dir / "minutes.smart.audit.json", result.audit)
    action_ledger = read_json(output_dir / "action_items.json")
    evidence_manifests, evidence_errors = build_reviewed_evidence_manifests(
        smart_minutes=result.payload,
        action_rows=parse_shareable_action_rows(result.chinese_markdown),
        project_rows=parse_shareable_project_update_rows(
            result.chinese_markdown
        ),
        segments=segments,
        action_ledger=action_ledger,
    )
    if evidence_manifests is None:
        failed_status = {
            **status,
            "status": "review_evidence_invalid",
            "errors": evidence_errors,
        }
        statuses["smart_summary"] = failed_status
        write_json(output_dir / "summary_status.json", failed_status)
        write_json(run_status_path, {**run_status, "statuses": statuses})
        for error in evidence_errors:
            print(error, file=sys.stderr)
        return 2
    evidence_paths = {
        "action_evidence": review_dir / "minutes.smart.action-evidence.json",
        "action_intent_review": (
            review_dir / "minutes.smart.action-intent-review.json"
        ),
        "project_evidence": review_dir / "minutes.smart.project-evidence.json",
    }
    for evidence_kind, evidence_path in evidence_paths.items():
        write_json(evidence_path, evidence_manifests[evidence_kind])
    completed_status = {
        **status,
        "review_evidence": {
            "status": "validated",
            "action_evidence": str(evidence_paths["action_evidence"]),
            "action_intent_review": str(
                evidence_paths["action_intent_review"]
            ),
            "project_evidence": str(evidence_paths["project_evidence"]),
        },
    }
    statuses["smart_summary"] = completed_status
    write_json(output_dir / "summary_status.json", completed_status)
    write_json(run_status_path, {**run_status, "statuses": statuses})
    print(str(chinese_path))
    print(str(english_path))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_identity_audited_segments(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load direct visual names only through a current, complete agent audit."""

    transcript_path = output_dir / "transcript.json"
    segments = read_json(transcript_path)
    if not isinstance(segments, list):
        return [], {"status": "raw_transcript_invalid", "cleared_segments": 0}
    requires_direct_audit = any(
        str(segment.get("name_source") or "") == "visual_active_speaker_highlight"
        and str(segment.get("name") or "").strip()
        for segment in segments
        if isinstance(segment, dict)
    )
    if not requires_direct_audit:
        return segments, {
            "status": "not_required",
            "cleared_segments": 0,
            "publishable": True,
            "requires_direct_audit": False,
        }

    audit_path = output_dir / "agent_visual_audit.json"
    if not audit_path.is_file():
        gated_segments, status = apply_agent_visual_audit_veto(segments, None)
        return gated_segments, {
            **status,
            "requires_direct_audit": True,
            "agent_visual_audit_path": str(audit_path),
        }
    audit = read_json(audit_path)
    if not isinstance(audit, dict) or audit.get("status") != "passed":
        gated_segments, status = apply_agent_visual_audit_veto(segments, None)
        return gated_segments, {
            **status,
            "status": "audit_not_passed_fail_closed",
            "requires_direct_audit": True,
            "agent_visual_audit_path": str(audit_path),
        }
    summary = audit.get("summary")
    manifest_path_raw = str(audit.get("manifest") or "").strip()
    if not isinstance(summary, dict) or summary.get("coverage_complete") is not True or not manifest_path_raw:
        gated_segments, status = apply_agent_visual_audit_veto(segments, None)
        return gated_segments, {
            **status,
            "status": "audit_coverage_incomplete_fail_closed",
            "requires_direct_audit": True,
            "agent_visual_audit_path": str(audit_path),
        }
    manifest_path = Path(manifest_path_raw).expanduser().resolve()
    if not manifest_path.is_file():
        gated_segments, status = apply_agent_visual_audit_veto(segments, None)
        return gated_segments, {
            **status,
            "status": "audit_manifest_missing_fail_closed",
            "requires_direct_audit": True,
            "agent_visual_audit_path": str(audit_path),
            "manifest_path": str(manifest_path),
        }
    manifest = read_json(manifest_path)
    manifest_errors = (
        validate_agent_visual_audit_manifest_content(manifest)
        if isinstance(manifest, dict)
        else ["manifest_invalid"]
    )
    expected_manifest_sha256 = str(summary.get("manifest_sha256") or "").strip()
    manifest_matches = (
        isinstance(manifest, dict)
        and str(manifest.get("manifest_sha256") or "").strip() == expected_manifest_sha256
    )
    if manifest_errors or not manifest_matches:
        gated_segments, status = apply_agent_visual_audit_veto(segments, None)
        return gated_segments, {
            **status,
            "status": "audit_manifest_stale_fail_closed",
            "requires_direct_audit": True,
            "agent_visual_audit_path": str(audit_path),
            "manifest_path": str(manifest_path),
            "manifest_errors": manifest_errors,
        }
    veto_path = output_dir / "identity_audit_veto.json"
    if not veto_path.is_file():
        gated_segments, status = apply_agent_visual_audit_veto(segments, None)
        return gated_segments, {
            **status,
            "status": "audit_veto_missing_fail_closed",
            "requires_direct_audit": True,
            "agent_visual_audit_path": str(audit_path),
        }
    veto = read_json(veto_path)
    if (
        not isinstance(veto, dict)
        or str(veto.get("manifest_sha256") or "").strip() != expected_manifest_sha256
    ):
        gated_segments, status = apply_agent_visual_audit_veto(segments, None)
        return gated_segments, {
            **status,
            "status": "audit_veto_stale_fail_closed",
            "requires_direct_audit": True,
            "agent_visual_audit_path": str(audit_path),
            "veto_path": str(veto_path),
        }
    gated_segments, status = apply_agent_visual_audit_veto(segments, veto)
    return gated_segments, {
        **status,
        "requires_direct_audit": True,
        "agent_visual_audit_path": str(audit_path),
        "veto_path": str(veto_path),
    }


def smart_sanitize_reviewed_existing(args: argparse.Namespace) -> int:
    """Reissue a reviewed artifact after deterministic evidence-preserving repair."""

    output_dir = args.output_dir.expanduser().resolve()
    transcript_path = output_dir / "transcript.json"
    source_smart_path = args.source_smart_json.expanduser().resolve()
    source_audit_path = args.source_audit_json.expanduser().resolve()
    missing_paths = [
        path
        for path in (transcript_path, source_smart_path, source_audit_path)
        if not path.is_file()
    ]
    if missing_paths:
        for path in missing_paths:
            print(f"smart_repair_missing:{path}", file=sys.stderr)
        return 1

    segments, identity_audit_status = _load_identity_audited_segments(output_dir)
    source_payload = read_json(source_smart_path)
    source_audit = read_json(source_audit_path)
    if not isinstance(segments, list):
        print("smart_repair_transcript_invalid", file=sys.stderr)
        return 1
    if not isinstance(source_audit, dict):
        print("smart_repair_source_audit_invalid", file=sys.stderr)
        return 1
    if source_audit.get("format") != SMART_MINUTES_AUDIT_FORMAT:
        print("smart_repair_source_audit_format_invalid", file=sys.stderr)
        return 1
    transcript_sha256 = transcript_fingerprint(segments)
    if source_audit.get("transcript_sha256") != transcript_sha256:
        print("smart_repair_source_audit_transcript_mismatch", file=sys.stderr)
        return 1
    source_reviews = source_audit.get("reviews")
    if (
        not isinstance(source_reviews, list)
        or not source_reviews
        or not isinstance(source_reviews[-1], dict)
        or source_reviews[-1].get("publishable") is not True
    ):
        print("smart_repair_source_audit_not_publishable", file=sys.stderr)
        return 1

    result, errors = sanitize_reviewed_smart_minutes(
        source_payload,
        segments=segments,
        source_audit=source_audit,
        allow_cluster_name_consensus=False,
    )
    bounded_repair_status: dict[str, Any] | None = None
    if result is None and args.deepseek_repair:
        repaired_payload, bounded_repair_status = repair_reviewed_minutes_text_fields(
            source_payload,
            segments=segments,
            config=_deepseek_config(args),
            allow_cluster_name_consensus=False,
        )
        if repaired_payload is not None:
            result, errors = sanitize_reviewed_smart_minutes(
                repaired_payload,
                segments=segments,
                source_audit=source_audit,
                allow_cluster_name_consensus=False,
            )
        else:
            print("smart_repair_bounded_model_failed", file=sys.stderr)
            print(
                str(bounded_repair_status.get("status", "unknown")),
                file=sys.stderr,
            )
            for error in bounded_repair_status.get("errors", []):
                print(str(error), file=sys.stderr)
            return 2
    if result is None:
        print("smart_repair_validation_failed", file=sys.stderr)
        for error in errors:
            print(str(error), file=sys.stderr)
        return 2

    review_dir = output_dir / "work" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    archived = _archive_stale_smart_reviews(review_dir)
    audit = deepcopy(source_audit)
    if "synthesis_validation_errors" in audit:
        audit["synthesis_initial_validation_errors"] = audit.pop(
            "synthesis_validation_errors"
        )
    audit["final_review_validation_errors"] = []
    audit["derivation"] = {
        "kind": (
            "bounded_ai_reviewed_minutes_field_repair"
            if bounded_repair_status is not None
            else "deterministic_reviewed_minutes_repair"
        ),
        "source_smart_json": source_smart_path.name,
        "source_smart_json_sha256": _file_sha256(source_smart_path),
        "source_audit_json": source_audit_path.name,
        "source_audit_json_sha256": _file_sha256(source_audit_path),
        "source_final_review_publishable": True,
        "source_review_passes": len(source_reviews),
        "changes": result.changes,
        "invariants": [
            "named_speaker_attribution_preserved",
            "retained_entry_segment_ids_preserved",
            "retained_entry_relative_order_preserved",
        ],
        "validation": {
            "smart_minutes": "passed",
            "bilingual_render_contract": "passed",
            "final_publication_gate": "passed",
        },
        **(
            {"bounded_field_repair": bounded_repair_status}
            if bounded_repair_status is not None
            else {}
        ),
    }
    if result.final_review is not None:
        audit_reviews = audit.get("reviews")
        if not isinstance(audit_reviews, list) or not audit_reviews:
            print("smart_repair_source_audit_invalid", file=sys.stderr)
            return 1
        audit_reviews[-1] = result.final_review
    audit["transcript_sha256"] = result.transcript_sha256
    audit["required_project_participants"] = result.required_project_participants
    audit["status"] = "reviewed_draft"

    status = {
        "status": "reviewed_draft",
        "engine": (
            "bounded-ai-reviewed-repair"
            if bounded_repair_status is not None
            else "deterministic-reviewed-repair"
        ),
        "source_final_review_publishable": True,
        "source_review_passes": len(source_reviews),
        "transcript_sha256": result.transcript_sha256,
        "deterministic_changes": result.changes,
        **(
            {"bounded_field_repair": bounded_repair_status}
            if bounded_repair_status is not None
            else {}
        ),
        "archived_stale_smart_reviews": archived,
        "identity_audit_gate": identity_audit_status,
    }
    chinese_path = review_dir / "minutes.smart.md"
    english_path = review_dir / "minutes.smart.en.md"
    _atomic_write_markdown(chinese_path, result.chinese_markdown)
    _atomic_write_markdown(english_path, result.english_markdown)
    write_json(review_dir / "minutes.smart.json", result.payload)
    write_json(review_dir / "minutes.smart.audit.json", audit)

    run_status_path = output_dir / "run_status.json"
    run_status = read_json(run_status_path) if run_status_path.is_file() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["smart_summary"] = status
    write_json(output_dir / "summary_status.json", status)
    write_json(run_status_path, {**run_status, "statuses": statuses})
    print(str(chinese_path))
    print(str(english_path))
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
    raw_segment_count = len(segments)
    segments = split_segments_by_turns(segments, turns)
    split_status = {
        "status": "ok" if turns else "skipped",
        "input_segments": raw_segment_count,
        "output_segments": len(segments),
        "split_source_segments": len({segment["split_from"] for segment in segments if segment.get("split_from")}),
    }
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
    statuses["word_timing_split"] = split_status
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


def apply_voice_registry_existing(args: argparse.Namespace) -> int:
    """Attach registry names to a previously diarized transcript without discarding speaker clusters."""

    started = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    audio_path = output_dir / "work" / "audio_16k_mono.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Expected extracted audio at {audio_path}")
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected diarized transcript at {transcript_path}")
    registry_path = args.registry.expanduser().resolve()
    if not registry_path.exists():
        raise FileNotFoundError(f"Voice registry does not exist: {registry_path}")
    speechbrain_cache = args.speechbrain_cache.expanduser().resolve() if args.speechbrain_cache else output_dir / "work" / "speechbrain_models"
    segments = read_json(transcript_path)
    if not any(segment.get("speaker") and segment.get("speaker") != "Speaker Unknown" for segment in segments):
        raise ValueError("Apply a diarization backend before applying a cross-recording voice registry")

    voice_registry_status = apply_voice_registry(
        audio_path,
        segments,
        registry_path,
        speechbrain_cache=speechbrain_cache,
        threshold=args.registry_threshold,
        margin=args.registry_margin,
    )
    voice_registry_status["cluster_consensus"] = enforce_registry_cluster_consensus(segments)
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    attach_names(segments, ocr_records, allow_ocr_names=False)
    write_json(output_dir / "transcript.json", segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_json(output_dir / "voice_registry_status.json", voice_registry_status)

    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    run_status_path = output_dir / "run_status.json"
    existing_status = read_json(run_status_path) if run_status_path.exists() else {}
    statuses = dict(existing_status.get("statuses", {}))
    diarization_status_path = output_dir / "diarization_status.json"
    if diarization_status_path.exists():
        existing_diarization = read_json(diarization_status_path).get("status")
        if isinstance(existing_diarization, dict):
            statuses["diarization"] = existing_diarization
    statuses["voice_registry"] = voice_registry_status
    _write_action_artifacts(output_dir, segments, statuses)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=read_json(output_dir / "action_items.json"))
    write_json(
        output_dir / "voice_registry_apply_status.json",
        {
            "elapsed_seconds": round(time.time() - started, 3),
            "registry": str(registry_path),
            "status": voice_registry_status,
        },
    )
    print(str(output_dir / "voice_registry_status.json"))
    return 0


def enforce_voice_registry_consensus_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    transcript_path = output_dir / "transcript.json"
    status_path = output_dir / "voice_registry_status.json"
    if not transcript_path.exists() or not status_path.exists():
        raise FileNotFoundError("Apply a cross-recording voice registry before enforcing cluster consensus")
    segments = read_json(transcript_path)
    consensus = enforce_registry_cluster_consensus(segments)
    status = read_json(status_path)
    status["cluster_consensus"] = consensus
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    attach_names(segments, ocr_records, allow_ocr_names=False)
    write_json(transcript_path, segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_json(status_path, status)
    run_status_path = output_dir / "run_status.json"
    existing_status = read_json(run_status_path) if run_status_path.exists() else {}
    statuses = dict(existing_status.get("statuses", {}))
    diarization_status_path = output_dir / "diarization_status.json"
    if diarization_status_path.exists():
        existing_diarization = read_json(diarization_status_path).get("status")
        if isinstance(existing_diarization, dict):
            statuses["diarization"] = existing_diarization
    statuses["voice_registry"] = status
    _write_action_artifacts(output_dir, segments, statuses)
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=read_json(output_dir / "action_items.json"))
    print(str(status_path))
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
    cleared_cluster_assignments = clear_direct_visual_cluster_identity(segments)
    cleared_visual_voice_assignments = clear_visual_voice_identity(segments)
    recording_provenance = _visual_recording_provenance(input_path, float(metadata.get("duration", 0.0)))
    status = _run_visual_identity(
        input_path=input_path,
        output_dir=output_dir,
        duration=float(metadata.get("duration", 0.0)),
        segments=segments,
        profile_path=args.visual_profile,
        max_frame_width=args.max_frame_width,
        recording_provenance=recording_provenance,
    )
    status["cleared_prior_cluster_assignments"] = cleared_cluster_assignments
    status["cleared_prior_visual_voice_assignments"] = cleared_visual_voice_assignments
    cluster_invalidation_status = {
        "status": "invalidated_by_visual_identity_refresh",
        "cleared_prior_cluster_assignments": cleared_cluster_assignments,
    }
    _write_direct_visual_cluster_skip_artifact(output_dir, cluster_invalidation_status)
    _write_visual_voice_skip_artifacts(
        output_dir,
        {
            "status": "invalidated_by_visual_identity_refresh",
            "cleared_prior_visual_voice_assignments": cleared_visual_voice_assignments,
        },
    )
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["visual_identity"] = status
    statuses["direct_visual_cluster_identity"] = cluster_invalidation_status
    statuses["visual_voice_identity"] = {
        "status": "invalidated_by_visual_identity_refresh",
        "cleared_prior_visual_voice_assignments": cleared_visual_voice_assignments,
    }
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
    write_extractive_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
    print(str(output_dir / "visual_identity_report.md"))
    return 0


def _agent_visual_audit_command(
    *,
    agent: str,
    args: argparse.Namespace,
    workspace: Path,
    output_dir: Path,
    bundle: dict[str, Path],
    manifest: dict[str, Any],
    response_path: Path,
    prompt: str,
) -> list[str]:
    frames: list[str] = []
    for calibration in manifest.get("calibrations", []):
        if isinstance(calibration, dict) and isinstance(calibration.get("frame"), str):
            frames.append(str(calibration["frame"]))
        if isinstance(calibration, dict) and isinstance(calibration.get("inspection_frame"), str):
            frames.append(str(calibration["inspection_frame"]))
    for sample in manifest.get("samples", []):
        if isinstance(sample, dict) and isinstance(sample.get("frame"), str):
            frames.append(str(sample["frame"]))
        if isinstance(sample, dict) and isinstance(sample.get("inspection_frame"), str):
            frames.append(str(sample["inspection_frame"]))
        if isinstance(sample, dict) and isinstance(sample.get("roster_inspection_frame"), str):
            frames.append(str(sample["roster_inspection_frame"]))
    frames = list(dict.fromkeys(frames))
    if agent == "codex":
        command = [
            args.codex_bin,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "-C",
            str(workspace),
            "--add-dir",
            str(output_dir),
            "--output-schema",
            str(bundle["schema"]),
            "--output-last-message",
            str(response_path),
        ]
        for frame in frames:
            command.extend(["--image", frame])
        # Codex treats --image as a variadic option. The explicit separator
        # prevents the JSON prompt from being consumed as another image path.
        command.extend(["--", prompt])
        return command
    if agent == "cursor":
        return [
            args.cursor_bin,
            "--trust",
            "--model",
            args.cursor_model,
            "--workspace",
            str(workspace),
            "--add-dir",
            str(output_dir),
            "--print",
            "--output-format",
            "text",
            "--mode",
            "plan",
            prompt,
        ]
    raise ValueError(f"Unsupported visual audit agent: {agent}")


def _run_agent_visual_audit(
    *,
    agent: str,
    args: argparse.Namespace,
    workspace: Path,
    output_dir: Path,
    bundle: dict[str, Path],
    manifest: dict[str, Any],
    seed_response: dict[str, Any] | None = None,
    response_tag: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit_dir = bundle["directory"]
    response_dir = audit_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = response_dir / f"{agent}.normalized.json"
    all_samples = [sample for sample in manifest.get("samples", []) if isinstance(sample, dict)]
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    normalized_samples: list[dict[str, Any]] = []
    normalized_calibrations: list[dict[str, Any]] | None = None
    if seed_response is not None:
        validated_seed, seed_errors = validate_agent_visual_audit_response(
            seed_response,
            manifest,
            expected_agent=agent,
        )
        if validated_seed is None or seed_errors:
            status = {
                "status": "failed",
                "error": "retry_seed_invalid",
                "agent": agent,
                "validation_errors": seed_errors,
            }
            write_json(normalized_path, status)
            return None, status
        normalized_calibrations = validated_seed["calibrations"]
        normalized_samples = [
            sample
            for sample in validated_seed["samples"]
            if sample.get("confirmed") is True
        ]
    confirmed_sample_ids = {
        str(sample.get("sample_id") or "")
        for sample in normalized_samples
        if str(sample.get("sample_id") or "")
    }
    samples = [
        sample
        for sample in all_samples
        if str(sample.get("sample_id") or "") not in confirmed_sample_ids
    ]
    batches = [samples[index : index + args.batch_size] for index in range(0, len(samples), args.batch_size)]
    if not batches:
        if normalized_samples and len(normalized_samples) == len(all_samples):
            normalized, validation_errors = validate_agent_visual_audit_response(
                seed_response,
                manifest,
                expected_agent=agent,
            )
            if normalized is not None and not validation_errors:
                status = {
                    "status": "ok",
                    "agent": agent,
                    "batches": [],
                    "reused_confirmed_samples": len(normalized_samples),
                }
                write_json(normalized_path, {**status, "response": normalized})
                return normalized, status
        status = {"status": "failed", "error": "no_auditable_frames", "agent": agent}
        write_json(normalized_path, status)
        return None, status

    batch_statuses: list[dict[str, Any]] = []
    failed_batches: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches, start=1):
        batch_manifest = {**manifest, "samples": batch}
        prompt = render_agent_visual_audit_prompt(batch_manifest, agent=agent)
        batch_sample_ids = [str(sample.get("sample_id") or "") for sample in batch]
        normalized_batch: dict[str, Any] | None = None
        validation_errors: list[str] = []
        batch_attempts: list[dict[str, Any]] = []
        failed_status: dict[str, Any] | None = None
        stop_after_failure = False
        for attempt in range(1, 3):
            suffix = "" if attempt == 1 else f".retry-{attempt}"
            response_stem = (
                f"{agent}.{response_tag}.batch-{batch_index:03d}"
                if response_tag
                else f"{agent}.batch-{batch_index:03d}"
            )
            response_path = response_dir / f"{response_stem}{suffix}.response.txt"
            stdout_path = response_dir / f"{response_stem}{suffix}.stdout.txt"
            stderr_path = response_dir / f"{response_stem}{suffix}.stderr.txt"
            command = _agent_visual_audit_command(
                agent=agent,
                args=args,
                workspace=workspace,
                output_dir=output_dir,
                bundle=bundle,
                manifest=batch_manifest,
                response_path=response_path,
                prompt=prompt,
            )
            response_path.unlink(missing_ok=True)
            try:
                completed = subprocess.run(
                    command,
                    cwd=workspace,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=args.timeout,
                    check=False,
                )
            except FileNotFoundError:
                failed_status = {
                    "status": "failed",
                    "error": "executable_not_found",
                    "agent": agent,
                    "batch": batch_index,
                    "sample_ids": batch_sample_ids,
                }
                stop_after_failure = True
                break
            except subprocess.TimeoutExpired:
                failed_status = {
                    "status": "failed",
                    "error": "timeout",
                    "agent": agent,
                    "batch": batch_index,
                    "timeout_seconds": args.timeout,
                    "sample_ids": batch_sample_ids,
                }
                stop_after_failure = True
                break

            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            response_text = (
                response_path.read_text(encoding="utf-8")
                if response_path.exists()
                else completed.stdout or ""
            )
            parsed = parse_agent_visual_audit_response(response_text)
            normalized_batch, validation_errors = validate_agent_visual_audit_response(
                parsed,
                batch_manifest,
                expected_agent=agent,
            )
            attempt_status = {
                "attempt": attempt,
                "exit_code": completed.returncode,
                "validation_errors": validation_errors,
            }
            batch_attempts.append(attempt_status)
            if completed.returncode == 0 and normalized_batch is not None:
                break
            retryable_empty_response = (
                attempt == 1
                and completed.returncode == 0
                and validation_errors == ["response_not_json_object"]
            )
            if retryable_empty_response:
                continue
            failed_status = {
                "status": "failed",
                "agent": agent,
                "batch": batch_index,
                "exit_code": completed.returncode,
                "validation_errors": validation_errors,
                "sample_ids": batch_sample_ids,
            }
            break
        if failed_status is not None:
            if batch_attempts:
                failed_status["attempts"] = batch_attempts
            batch_statuses.append(failed_status)
            failed_batches.append(failed_status)
            if stop_after_failure:
                # A timed-out local agent cannot make this audit pass. Preserve
                # completed batches, then stop instead of spending the full
                # timeout budget again on later batches from the same invocation.
                break
            continue
        if normalized_batch is None:
            failed_status = {
                "status": "failed",
                "agent": agent,
                "batch": batch_index,
                "error": "normalization_missing",
                "sample_ids": batch_sample_ids,
                "attempts": batch_attempts,
            }
            batch_statuses.append(failed_status)
            failed_batches.append(failed_status)
            continue
        if (
            normalized_calibrations is not None
            and normalized_batch["calibrations"] != normalized_calibrations
        ):
            failed_status = {
                "status": "failed",
                "agent": agent,
                "batch": batch_index,
                "error": "calibration_verdict_inconsistent_across_batches",
                "sample_ids": batch_sample_ids,
                "attempts": batch_attempts,
            }
            batch_statuses.append(failed_status)
            failed_batches.append(failed_status)
            continue
        batch_statuses.append(
            {
                "batch": batch_index,
                "sample_ids": batch_sample_ids,
                "exit_code": batch_attempts[-1]["exit_code"],
                "validation_errors": batch_attempts[-1]["validation_errors"],
                "attempts": batch_attempts,
                "status": "ok",
            }
        )
        if normalized_calibrations is None:
            normalized_calibrations = normalized_batch["calibrations"]
        normalized_samples.extend(normalized_batch["samples"])

    if failed_batches:
        partial = {
            "format": AGENT_VISUAL_AUDIT_FORMAT,
            "agent": agent,
            "overall_verdict": "needs_review",
            "manifest_sha256": str(manifest.get("manifest_sha256") or ""),
            "calibrations": normalized_calibrations or [],
            "samples": normalized_samples,
        }
        status = {
            "status": "partial",
            "agent": agent,
            "batches": batch_statuses,
            "failed_batches": failed_batches,
        }
        write_json(normalized_path, {**status, "response": partial})
        return partial, status

    assembled = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "agent": agent,
        "overall_verdict": (
            "pass"
            if all(sample.get("confirmed") is True for sample in normalized_samples)
            else "needs_review"
        ),
        "calibrations": normalized_calibrations or [],
        "samples": normalized_samples,
    }
    normalized, validation_errors = validate_agent_visual_audit_response(
        assembled,
        manifest,
        expected_agent=agent,
    )
    if normalized is None:
        status = {
            "status": "failed",
            "agent": agent,
            "validation_errors": validation_errors,
        }
        write_json(normalized_path, {**status, "batches": batch_statuses})
        return None, status
    status = {
        "status": "ok",
        "agent": agent,
        "batches": batch_statuses,
        "reused_confirmed_samples": len(confirmed_sample_ids),
    }
    write_json(normalized_path, {**status, "response": normalized})
    return normalized, status


def _review_action_priority_segment_ids(
    output_dir: Path,
    segments: list[dict[str, Any]],
) -> set[str]:
    """Prioritize visual audit samples that back already drafted action owners."""

    review_path = output_dir / "work" / "review" / "minutes.smart.json"
    if not review_path.is_file():
        return set()
    payload = read_json(review_path)
    minutes = payload.get("minutes") if isinstance(payload, dict) else None
    actions = minutes.get("actions") if isinstance(minutes, dict) else None
    if not isinstance(actions, list):
        return set()
    stable_to_raw = {
        stable_segment_id(segment, index): str(segment.get("id") or "").strip()
        for index, segment in enumerate(segments)
    }
    priority: set[str] = set()
    for action in actions:
        if not isinstance(action, dict):
            continue
        raw_ids = action.get("segment_ids")
        if not isinstance(raw_ids, list):
            continue
        for segment_id in raw_ids:
            raw_id = stable_to_raw.get(str(segment_id or "").strip())
            if raw_id:
                priority.add(raw_id)
    return priority


def _reuse_matching_agent_visual_audit(
    *,
    audit_dir: Path,
    agent: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Reuse only a fully validated result for this exact frame manifest."""

    if validate_agent_visual_audit_manifest_content(manifest):
        return None, None
    normalized_path = audit_dir / "responses" / f"{agent}.normalized.json"
    if not normalized_path.is_file():
        return None, None
    prior = read_json(normalized_path)
    if not isinstance(prior, dict) or prior.get("status") != "ok":
        return None, None
    response = prior.get("response")
    if not isinstance(response, dict):
        return None, None
    if response.get("manifest_sha256") != manifest.get("manifest_sha256"):
        return None, None
    normalized, errors = validate_agent_visual_audit_response(
        response,
        manifest,
        expected_agent=agent,
    )
    if normalized is None or errors:
        return None, None
    prior_batches = prior.get("batches")
    return normalized, {
        "status": "ok",
        "agent": agent,
        "batches": prior_batches if isinstance(prior_batches, list) else [],
        "reused_matching_manifest": True,
    }


def visual_agent_audit_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    metadata_path = output_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    expected_recording: dict[str, Any] | None = None
    effective_input = str(metadata.get("effective_input") or metadata.get("input") or "").strip()
    if effective_input:
        input_path = Path(effective_input).expanduser().resolve()
        if input_path.is_file():
            expected_recording = _visual_recording_provenance(
                input_path,
                float(metadata.get("duration", 0.0) or 0.0),
            )
    visual_path, visual_identity, skip_status = _select_visual_identity_artifact(
        output_dir=output_dir,
        visual_identity_path=args.visual_identity_path,
        expected_recording=expected_recording,
    )
    if visual_identity is None:
        payload = {
            "format": AGENT_VISUAL_AUDIT_FORMAT,
            "status": (skip_status or {}).get("status", "skipped_no_direct_visual_identity"),
            "visual_identity": str(visual_path) if visual_path else None,
            "detail": skip_status or {},
        }
        write_json(output_dir / "agent_visual_audit.json", payload)
        print(str(output_dir / "agent_visual_audit.json"))
        return 2

    current_segments = read_json(transcript_path)
    audit_source_segments = current_segments
    audit_dir = output_dir / "work" / "agent_visual_audit"
    existing_manifest_path = audit_dir / "request.json"
    existing_schema_path = audit_dir / "response.schema.json"
    bundle: dict[str, Path] | None = None
    manifest: dict[str, Any] | None = None
    if (
        bool(getattr(args, "full_coverage", False))
        and existing_manifest_path.is_file()
        and existing_schema_path.is_file()
    ):
        candidate_manifest = read_json(existing_manifest_path)
        if (
            isinstance(candidate_manifest, dict)
            and not validate_agent_visual_audit_manifest_content(candidate_manifest)
            and candidate_manifest.get("coverage", {}).get("selection_mode")
            == "full_coverage"
        ):
            restored_segments, restoration_errors = (
                restore_direct_visual_candidates_from_manifest(
                    current_segments,
                    candidate_manifest,
                )
            )
            if not restoration_errors:
                audit_source_segments = restored_segments
                manifest = candidate_manifest
                bundle = {
                    "directory": audit_dir,
                    "manifest": existing_manifest_path,
                    "schema": existing_schema_path,
                }
    if bundle is None or manifest is None:
        priority_segment_ids = _review_action_priority_segment_ids(
            output_dir,
            current_segments,
        )
        bundle = write_agent_visual_audit_bundle(
            output_dir,
            current_segments,
            visual_identity,
            samples_per_identity=args.samples_per_identity,
            max_samples=args.max_samples,
            full_coverage=bool(getattr(args, "full_coverage", False)),
            priority_segment_ids=priority_segment_ids,
            calibration_frames=[
                frame.expanduser().resolve()
                for frame in getattr(args, "calibration_frame", [])
            ],
            calibration_layout=getattr(args, "calibration_layout", None),
        )
        manifest = read_json(bundle["manifest"])
    requested_agents = list(dict.fromkeys(args.run_agent))
    workspace = (args.workspace or Path.cwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"Agent workspace does not exist: {workspace}")
    results: dict[str, dict[str, Any] | None] = {}
    agent_statuses: dict[str, dict[str, Any]] = {}
    for agent in requested_agents:
        result, status = _reuse_matching_agent_visual_audit(
            audit_dir=bundle["directory"],
            agent=agent,
            manifest=manifest,
        )
        retry_unconfirmed = bool(getattr(args, "retry_unconfirmed", False))
        has_unconfirmed = bool(
            result
            and any(
                sample.get("confirmed") is not True
                for sample in result.get("samples", [])
                if isinstance(sample, dict)
            )
        )
        if result is not None and status is not None and retry_unconfirmed and has_unconfirmed:
            result, status = _run_agent_visual_audit(
                agent=agent,
                args=args,
                workspace=workspace,
                output_dir=output_dir,
                bundle=bundle,
                manifest=manifest,
                seed_response=result,
                response_tag="retry-unconfirmed",
            )
        elif result is None or status is None:
            result, status = _run_agent_visual_audit(
                agent=agent,
                args=args,
                workspace=workspace,
                output_dir=output_dir,
                bundle=bundle,
                manifest=manifest,
            )
        results[agent] = result
        agent_statuses[agent] = status
    summary = summarize_agent_visual_audits(manifest, results)
    status_name = (
        summary["status"]
        if requested_agents
        else "pack_ready"
    )
    veto = build_agent_visual_audit_veto(
        manifest,
        results,
        audit_source_segments,
    )
    gated_segments, gate_status = apply_agent_visual_audit_veto(
        current_segments,
        veto,
    )
    veto_path = output_dir / "identity_audit_veto.json"
    gated_transcript_path = output_dir / "transcript.identity_audited.json"
    gated_transcript_markdown_path = output_dir / "transcript.identity_audited.md"
    write_json(veto_path, veto)
    write_json(gated_transcript_path, gated_segments)
    write_transcript_markdown(gated_transcript_markdown_path, gated_segments)
    payload = {
        "format": AGENT_VISUAL_AUDIT_FORMAT,
        "status": status_name,
        "visual_identity_path": str(visual_path),
        "manifest": str(bundle["manifest"]),
        "schema": str(bundle["schema"]),
        "agent_statuses": agent_statuses,
        "summary": summary,
        "veto": {
            "path": str(veto_path),
            "derived_transcript": str(gated_transcript_path),
            **gate_status,
        },
        "guardrail": "Agent results can only clear an existing direct active-speaker highlight identity; they never create or propagate a real-name assignment.",
    }
    write_json(output_dir / "agent_visual_audit.json", payload)
    write_agent_visual_audit_report(output_dir / "agent_visual_audit_report.md", manifest, summary)
    run_status_path = output_dir / "run_status.json"
    if run_status_path.exists():
        run_status = read_json(run_status_path)
        statuses = dict(run_status.get("statuses", {}))
        statuses["agent_visual_audit"] = {
            "status": status_name,
            "requested_agents": requested_agents,
            "sampled_frames": manifest.get("coverage", {}).get("selected_frames", 0),
            "veto": gate_status,
        }
        write_json(run_status_path, {**run_status, "statuses": statuses})
    print(str(output_dir / "agent_visual_audit_report.md"))
    if any(status.get("status") != "ok" for status in agent_statuses.values()):
        return 2
    if args.require_consensus and summary["status"] != "passed":
        return 3
    return 0


def dynamic_visual_identify_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    metadata = read_json(output_dir / "metadata.json")
    input_path = args.input.expanduser().resolve() if args.input else Path(str(metadata.get("effective_input") or metadata.get("input"))).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dynamic visual identity input does not exist: {input_path}")
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    segments = read_json(transcript_path)
    cleared_cluster_assignments = clear_direct_visual_cluster_identity(segments)
    cleared_visual_voice_assignments = clear_visual_voice_identity(segments)
    recording_provenance = _visual_recording_provenance(input_path, float(metadata.get("duration", 0.0)))
    status = _run_dynamic_visual_identity(
        input_path=input_path,
        output_dir=output_dir,
        duration=float(metadata.get("duration", 0.0)),
        segments=segments,
        profile_path=args.dynamic_visual_profile,
        max_frame_width=args.max_frame_width,
        recording_provenance=recording_provenance,
    )
    status["cleared_prior_cluster_assignments"] = cleared_cluster_assignments
    status["cleared_prior_visual_voice_assignments"] = cleared_visual_voice_assignments
    cluster_invalidation_status = {
        "status": "invalidated_by_dynamic_visual_identity_refresh",
        "cleared_prior_cluster_assignments": cleared_cluster_assignments,
    }
    _write_direct_visual_cluster_skip_artifact(output_dir, cluster_invalidation_status)
    _write_visual_voice_skip_artifacts(
        output_dir,
        {
            "status": "invalidated_by_dynamic_visual_identity_refresh",
            "cleared_prior_visual_voice_assignments": cleared_visual_voice_assignments,
        },
    )
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["dynamic_visual_identity"] = status
    statuses["direct_visual_cluster_identity"] = cluster_invalidation_status
    statuses["visual_voice_identity"] = {
        "status": "invalidated_by_dynamic_visual_identity_refresh",
        "cleared_prior_visual_voice_assignments": cleared_visual_voice_assignments,
    }
    if "visual_identity" in statuses:
        statuses["visual_identity"] = {
            "status": "superseded_by_dynamic_visual_identity",
            "reason": "dynamic stage uses same-frame active-tile and nameplate evidence without coordinate fallback",
        }
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
    write_extractive_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
    print(str(output_dir / "dynamic_visual_identity_report.md"))
    return 0


def _finalize_direct_visual_cluster_identify_existing(
    *,
    output_dir: Path,
    segments: list[dict[str, Any]],
    metadata: dict[str, Any],
    run_status: dict[str, Any],
    status: dict[str, Any],
) -> None:
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    statuses = dict(run_status.get("statuses", {}))
    statuses["direct_visual_cluster_identity"] = status
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
    write_extractive_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})


def direct_visual_cluster_identify_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    transcript_path = output_dir / "transcript.json"
    turns_path = output_dir / "speaker_turns.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    if not turns_path.exists():
        raise FileNotFoundError(f"Expected speaker turns at {turns_path}")
    segments, identity_audit_status = _load_identity_audited_segments(output_dir)
    turns = read_json(turns_path)
    metadata = read_json(output_dir / "metadata.json")
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    prior_cluster_assignments = sum(
        1
        for segment in segments
        if str(segment.get("name_source") or "") == "direct_visual_voice_cluster_consensus"
    )
    if identity_audit_status.get("publishable") is not True:
        clear_direct_visual_cluster_identity(segments)
        status = {
            "status": "skipped_identity_audit_not_publishable",
            "identity_audit_status": identity_audit_status,
            "cleared_prior_cluster_assignments": prior_cluster_assignments,
        }
        _write_direct_visual_cluster_skip_artifact(output_dir, status)
        _finalize_direct_visual_cluster_identify_existing(
            output_dir=output_dir,
            segments=segments,
            metadata=metadata,
            run_status=run_status,
            status=status,
        )
        print(str(output_dir / "direct_visual_cluster_identity_report.md"))
        return 0
    try:
        config_path = args.config.expanduser().resolve() if args.config else None
        visual_identity_path = args.visual_identity_path.expanduser().resolve() if args.visual_identity_path else None
        effective_input_value = metadata.get("effective_input") or metadata.get("input")
        if not effective_input_value:
            raise ValueError("metadata.json does not contain an input path for visual identity provenance")
        effective_input = Path(str(effective_input_value)).expanduser().resolve()
        status = _run_direct_visual_cluster_identity(
            output_dir=output_dir,
            segments=segments,
            turns=turns,
            config_path=config_path,
            visual_identity_path=visual_identity_path,
            expected_recording=_visual_recording_provenance(effective_input, float(metadata.get("duration", 0.0))),
        )
    except Exception as exc:
        clear_direct_visual_cluster_identity(segments)
        remaining_cluster_assignments = sum(
            1
            for segment in segments
            if str(segment.get("name_source") or "") == "direct_visual_voice_cluster_consensus"
        )
        failure_status = {
            "status": "failed_direct_visual_cluster_revalidation",
            "error": f"{type(exc).__name__}: {exc}",
            "cleared_prior_cluster_assignments": prior_cluster_assignments - remaining_cluster_assignments,
        }
        _write_direct_visual_cluster_skip_artifact(output_dir, failure_status)
        _finalize_direct_visual_cluster_identify_existing(
            output_dir=output_dir,
            segments=segments,
            metadata=metadata,
            run_status=run_status,
            status=failure_status,
        )
        raise
    _finalize_direct_visual_cluster_identify_existing(
        output_dir=output_dir,
        segments=segments,
        metadata=metadata,
        run_status=run_status,
        status=status,
    )
    print(str(output_dir / "direct_visual_cluster_identity_report.md"))
    return 0


def visual_voice_identify_existing(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    audio_path = output_dir / "work" / "audio_16k_mono.wav"
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    segments, identity_audit_status = _load_identity_audited_segments(output_dir)
    cleared = clear_visual_voice_identity(segments)
    metadata_path = output_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}

    def finalize(stage_status: dict[str, Any]) -> None:
        keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
        ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
        statuses = dict(run_status.get("statuses", {}))
        statuses["visual_voice_identity"] = stage_status
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
        write_extractive_minutes(
            output_dir / "minutes.extractive.md",
            segments=segments,
            keyframes=keyframes,
            metadata=report_meta,
            action_ledger=action_ledger,
        )
        write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})

    def finalize_skip(status: dict[str, Any]) -> int:
        _write_visual_voice_skip_artifacts(output_dir, status)
        finalize(status)
        print(str(output_dir / "same_session_visual_voice_report.md"))
        return 0

    if identity_audit_status.get("publishable") is not True:
        return finalize_skip(
            {
                "status": "skipped_identity_audit_not_publishable",
                "identity_audit_status": identity_audit_status,
                "cleared_prior_visual_voice_assignments": cleared,
            }
        )
    if not audio_path.exists():
        return finalize_skip(
            {
                "status": "skipped_audio_missing",
                "audio_path": str(audio_path),
                "cleared_prior_visual_voice_assignments": cleared,
            }
        )
    effective_input_value = metadata.get("effective_input") or metadata.get("input")
    if not effective_input_value:
        return finalize_skip(
            {
                "status": "skipped_recording_metadata_missing",
                "cleared_prior_visual_voice_assignments": cleared,
            }
        )
    effective_input = Path(str(effective_input_value)).expanduser().resolve()
    if not effective_input.is_file():
        return finalize_skip(
            {
                "status": "skipped_recording_input_missing",
                "effective_input": str(effective_input),
                "cleared_prior_visual_voice_assignments": cleared,
            }
        )
    try:
        visual_identity_path_arg = getattr(args, "visual_identity_path", None)
        visual_identity_path = visual_identity_path_arg.expanduser().resolve() if visual_identity_path_arg else None
        visual_path, visual_payload, skip_status = _select_visual_identity_artifact(
            output_dir=output_dir,
            visual_identity_path=visual_identity_path,
            expected_recording=_visual_recording_provenance(effective_input, float(metadata.get("duration", 0.0))),
        )
        if skip_status is not None:
            return finalize_skip({**skip_status, "cleared_prior_visual_voice_assignments": cleared})
        assert visual_path is not None and visual_payload is not None
        visual_payload = restrict_visual_enrollment_to_agent_confirmed_segments(
            visual_payload,
            segments,
        )
        if direct_visual_enrollment_frame_count(visual_payload) == 0:
            return finalize_skip(
                {
                    "status": "skipped_no_direct_visual_enrollment_frames",
                    "visual_source": str(visual_path),
                    "cleared_prior_visual_voice_assignments": cleared,
                }
            )
        # Configuration is part of revalidation.  Keep it inside the guarded
        # path so a bad visual input or config cannot leave prior voice-derived
        # names published.
        settings = load_visual_voice_config(args.config.expanduser().resolve() if args.config else None)
        speechbrain_cache = args.speechbrain_cache.expanduser().resolve() if args.speechbrain_cache else output_dir / "work" / "speechbrain_models"
        registry_path = output_dir / "same_session_visual_voice_registry.json"
        registry = build_visual_voice_registry(
            audio_path,
            visual_payload,
            registry_path,
            settings=settings,
            speechbrain_cache=speechbrain_cache,
        )
        status = apply_visual_voice_registry(
            audio_path,
            segments,
            registry_path,
            speechbrain_cache=speechbrain_cache,
        )
    except Exception as exc:
        failure_status = {
            "status": "failed_visual_voice_revalidation",
            "error": f"{type(exc).__name__}: {exc}",
            "cleared_prior_visual_voice_assignments": cleared,
        }
        _write_visual_voice_skip_artifacts(output_dir, failure_status)
        finalize(failure_status)
        raise
    status = {
        **status,
        "visual_source": str(visual_path),
        "cleared_prior_visual_voice_assignments": cleared,
    }
    write_visual_voice_report(output_dir / "same_session_visual_voice_report.md", registry=registry, status=status)
    finalize(status)
    print(str(output_dir / "same_session_visual_voice_report.md"))
    return 0


def avatar_template_identify_existing(args: argparse.Namespace) -> int:
    """Apply a calibrated visual-avatar stage to an existing dynamic visual run."""

    output_dir = args.output_dir.expanduser().resolve()
    visual_path = output_dir / "dynamic_visual_identity.json"
    samples_path = output_dir / "dynamic_visual_identity_samples.json"
    transcript_path = output_dir / "transcript.json"
    for required in (visual_path, samples_path, transcript_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Expected {required}; run dynamic-visual-identify before avatar-template-identify"
            )
    profile_path = args.avatar_template_profile.expanduser().resolve()
    profile = load_avatar_template_profile(profile_path)
    templates = build_avatar_templates(profile)
    visual_payload = read_json(visual_path)
    frames = list(visual_payload.get("frames", []))
    calibration = calibrate_avatar_templates(frames, profile, templates)
    scored_frames = score_avatar_template_frames(frames, profile, templates, calibration)
    segments = read_json(transcript_path)
    attachment_summary = attach_avatar_template_identity(
        segments,
        read_json(samples_path),
        scored_frames,
        profile,
    )
    artifact = {
        "profile": str(profile_path),
        "settings": profile["settings"],
        "templates": serializable_template_library(templates),
        "calibration": calibration,
        "summary": attachment_summary,
        "frames": scored_frames,
    }
    write_json(output_dir / "avatar_template_identity.json", artifact)
    write_avatar_template_identity_report(
        output_dir / "avatar_template_identity_report.md",
        profile_path=profile_path,
        calibration=calibration,
        attachment_summary=attachment_summary,
    )
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["avatar_template_identity"] = {
        "gate": calibration["gate"],
        "eligible_names": calibration["eligible_names"],
        "audit_only_names": calibration["audit_only_names"],
        **attachment_summary,
    }
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)
    write_json(transcript_path, segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
    metadata = read_json(output_dir / "metadata.json")
    report_meta = {
        "input": metadata.get("input"),
        "duration": metadata.get("duration", 0.0),
        "source_offset": metadata.get("source_offset", 0.0),
    }
    write_extractive_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
    print(str(output_dir / "avatar_template_identity_report.md"))
    return 0


def roster_avatar_identify_existing(args: argparse.Namespace) -> int:
    """Apply calibrated same-frame roster-avatar evidence to an existing run."""

    output_dir = args.output_dir.expanduser().resolve()
    metadata = read_json(output_dir / "metadata.json")
    input_path = (
        args.input.expanduser().resolve()
        if args.input
        else Path(str(metadata.get("effective_input") or metadata.get("input"))).expanduser().resolve()
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Roster avatar identity input does not exist: {input_path}")
    transcript_path = output_dir / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Expected transcript at {transcript_path}")
    duration = float(metadata.get("duration", 0.0))
    if duration <= 0.0:
        raise ValueError("Roster avatar identity requires a positive recording duration in metadata.json")
    profile_path = args.roster_avatar_profile.expanduser().resolve()
    profile = load_roster_avatar_profile(profile_path)
    segments = read_json(transcript_path)
    had_active_roster_identity = any(
        segment.get("name") and str(segment.get("name_source") or "") == "visual_roster_avatar_match"
        for segment in segments
    )
    sample_requests = build_roster_sample_requests(segments, profile, duration=duration)
    anchor_requests = build_reviewed_anchor_requests(profile, duration=duration)
    all_requests = [*sample_requests, *anchor_requests]
    if not all_requests:
        raise ValueError("Roster avatar identity has no transcript samples or reviewed anchors to inspect")
    frame_dir = output_dir / "work" / "roster_avatar_identity_frames"
    frames = extract_frames(
        input_path,
        unique_roster_video_times(all_requests),
        frame_dir,
        max_width=args.max_frame_width,
    )
    detected_frames = detect_roster_active_frames(frames, profile)
    ocr_manifest = build_roster_ocr_manifest(frames, profile)
    ocr_dir = output_dir / "work" / "roster_avatar_identity_ocr"
    roster_ocr = ocr_regions(ocr_manifest, ocr_dir) if ocr_manifest else []
    anchor_names_by_time: dict[float, list[str]] = {}
    for request in anchor_requests:
        time = round(float(request["video_time"]), 3)
        anchor_names_by_time.setdefault(time, []).append(str(request["expected_name"]))
    scored_frames = score_roster_avatar_frames(
        detected_frames,
        roster_ocr,
        profile,
        anchor_names_by_time=anchor_names_by_time,
    )
    calibration = calibrate_roster_avatar_identity(scored_frames, profile)
    attachment_summary = attach_roster_avatar_identity(
        segments,
        sample_requests,
        scored_frames,
        profile,
        calibration,
    )
    blocked_preserving_active_identity = (
        calibration.get("gate", {}).get("status") != "passed" and had_active_roster_identity
    )
    serializable_layouts = [
        {
            **layout,
            "end": None if str(layout["end"]) == "inf" else layout["end"],
        }
        for layout in profile["layouts"]
    ]
    artifact = {
        "format": "meeting-minutes.roster-avatar-identity.v1",
        "profile": str(profile_path),
        "recording": _visual_recording_provenance(input_path, duration),
        "participants": profile["participants"],
        "settings": profile["settings"],
        "layouts": serializable_layouts,
        "reviewed_anchors": profile["reviewed_anchors"],
        "calibration": calibration,
        "summary": attachment_summary,
        "frames": scored_frames,
        "application": {
            "status": "blocked_preserved_active_identity"
            if blocked_preserving_active_identity
            else "applied",
            "active_roster_segments_preserved": attachment_summary["preserved_prior_roster_identity"],
        },
    }
    artifact_suffix = ".attempt" if blocked_preserving_active_identity else ""
    artifact_path = output_dir / f"roster_avatar_identity{artifact_suffix}.json"
    samples_path = output_dir / f"roster_avatar_identity_samples{artifact_suffix}.json"
    anchors_path = output_dir / f"roster_avatar_identity_anchors{artifact_suffix}.json"
    ocr_manifest_path = output_dir / f"roster_avatar_identity_ocr_manifest{artifact_suffix}.json"
    ocr_path = output_dir / f"roster_avatar_identity_ocr{artifact_suffix}.json"
    report_path = (
        output_dir / "roster_avatar_identity_attempt_report.md"
        if blocked_preserving_active_identity
        else output_dir / "roster_avatar_identity_report.md"
    )
    write_json(artifact_path, artifact)
    write_json(samples_path, sample_requests)
    write_json(anchors_path, anchor_requests)
    write_json(ocr_manifest_path, ocr_manifest)
    write_json(ocr_path, roster_ocr)
    write_roster_avatar_identity_report(
        report_path,
        profile_path=profile_path,
        calibration=calibration,
        attachment_summary=attachment_summary,
    )
    keyframes = read_json(output_dir / "keyframes.json") if (output_dir / "keyframes.json").exists() else []
    ocr_records = read_json(output_dir / "ocr.json") if (output_dir / "ocr.json").exists() else []
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    if blocked_preserving_active_identity:
        statuses["roster_avatar_identity_attempt"] = {
            "status": "blocked_preserved_active_identity",
            "gate": calibration["gate"],
            "reviewed_anchors": len(calibration["anchors"]),
            "distinct_anchor_identities": calibration["distinct_anchor_identities"],
            **attachment_summary,
        }
        write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
        print(str(report_path))
        return 0
    statuses["roster_avatar_identity"] = {
        "gate": calibration["gate"],
        "reviewed_anchors": len(calibration["anchors"]),
        "distinct_anchor_identities": calibration["distinct_anchor_identities"],
        **attachment_summary,
    }
    action_ledger = _write_action_artifacts(output_dir, segments, statuses)
    write_json(transcript_path, segments)
    write_transcript_markdown(output_dir / "transcript.md", segments)
    write_speaker_samples(output_dir / "speaker_samples.md", segments)
    write_quality_report(output_dir / "quality_report.md", segments=segments, ocr_records=ocr_records, keyframes=keyframes, statuses=statuses)
    write_review_queue(output_dir / "review_queue.md", segments, action_ledger=action_ledger)
    report_meta = {
        "input": metadata.get("input"),
        "duration": duration,
        "source_offset": metadata.get("source_offset", 0.0),
    }
    write_extractive_minutes(
        output_dir / "minutes.extractive.md",
        segments=segments,
        keyframes=keyframes,
        metadata=report_meta,
        action_ledger=action_ledger,
    )
    write_json(output_dir / "run_status.json", {**run_status, "statuses": statuses})
    print(str(report_path))
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
    write_extractive_minutes(
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
    segments, identity_audit_status = _load_identity_audited_segments(output_dir)
    run_status = read_json(output_dir / "run_status.json") if (output_dir / "run_status.json").exists() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["identity_audit_gate"] = identity_audit_status
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
    transcript, _identity_audit_status = _load_identity_audited_segments(output_dir)
    regenerated_ledger = build_action_ledger(transcript)
    freshness_errors = _action_ledger_freshness_errors(transcript, ledger)
    payload = read_json(args.items)
    items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("--items must be a JSON array or an object containing an items array")
    results = [
        {"index": index, "errors": sorted(set(freshness_errors + validate_published_action_item(item, regenerated_ledger)))}
        for index, item in enumerate(items)
    ]
    rejected = [result for result in results if result["errors"]]
    intent_recall = regenerated_ledger.get("intent_recall") or {}
    intent_signals = intent_recall.get("signals") if isinstance(intent_recall, dict) else []
    recall_errors = [
        f"action_intent_recall_unmatched:{signal['signal_id']}"
        for signal in intent_signals
        if isinstance(signal, dict) and not signal.get("candidate_ids")
    ]
    report = {
        "status": "ok" if not rejected and not recall_errors else "rejected",
        "accepted": len(results) - len(rejected),
        "rejected": len(rejected),
        "ledger_fresh": not freshness_errors,
        "intent_recall": intent_recall,
        "recall_errors": recall_errors,
        "results": results,
    }
    report_path = output_dir / "action_items.validation.json"
    write_json(report_path, report)
    print(str(report_path))
    return 0 if not rejected and not recall_errors else 1


def validate_minutes_file(args: argparse.Namespace) -> int:
    path = args.path.expanduser().resolve()
    validator = validate_shareable_minutes_en if getattr(args, "language", "zh") == "en" else validate_shareable_minutes
    errors = validator(path.read_text(encoding="utf-8"), duration=float(args.duration or 0.0))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(str(path))
    return 0


def _atomic_write_markdown(path: Path, markdown: str) -> str:
    content = markdown if markdown.endswith("\n") else markdown + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return content


def publish_minutes_file(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    chinese_source = args.source.expanduser().resolve()
    english_source_arg = getattr(args, "english_source", None)
    if english_source_arg is None:
        print("english_minutes_source_required", file=sys.stderr)
        return 1
    english_source = english_source_arg.expanduser().resolve()
    if not chinese_source.is_file():
        print(f"minutes_source_missing:{chinese_source}", file=sys.stderr)
        return 1
    if not english_source.is_file():
        print(f"english_minutes_source_missing:{english_source}", file=sys.stderr)
        return 1
    metadata_path = output_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    metadata_duration = float(metadata.get("duration", 0.0) or 0.0)
    requested_duration = float(args.duration or 0.0)
    if requested_duration > 0.0 and metadata_duration > 0.0 and abs(requested_duration - metadata_duration) > 0.5:
        print("minutes_duration_mismatch", file=sys.stderr)
        return 1
    duration = requested_duration or metadata_duration
    if duration <= 0.0:
        print("minutes_duration_required", file=sys.stderr)
        return 1
    chinese_markdown = chinese_source.read_text(encoding="utf-8")
    english_markdown = english_source.read_text(encoding="utf-8")
    errors = validate_bilingual_minutes(chinese_markdown, english_markdown, duration=duration)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    transcript_path = output_dir / "transcript.json"
    ledger_path = output_dir / "action_items.json"
    if not transcript_path.is_file():
        print("publication_transcript_missing", file=sys.stderr)
        return 1
    if not ledger_path.is_file():
        print("publication_action_ledger_missing", file=sys.stderr)
        return 1
    segments, identity_audit_status = _load_identity_audited_segments(output_dir)
    if identity_audit_status.get("publishable") is not True:
        print(
            f"publication_identity_audit_not_publishable:{identity_audit_status.get('status', 'unknown')}",
            file=sys.stderr,
        )
        return 1
    action_ledger = read_json(ledger_path)
    ledger_errors = _action_ledger_freshness_errors(segments, action_ledger)
    if ledger_errors:
        for error in ledger_errors:
            print(error, file=sys.stderr)
        return 1
    rows = parse_shareable_action_rows(chinese_markdown)
    action_evidence_arg = getattr(args, "action_evidence", None)
    action_evidence_path = action_evidence_arg.expanduser().resolve() if action_evidence_arg else None
    action_evidence_sha256: str | None = None
    action_evidence: dict[str, Any] | None = None
    if rows:
        if action_evidence_path is None:
            print("action_evidence_required", file=sys.stderr)
            return 1
        if not action_evidence_path.is_file():
            print(f"action_evidence_missing:{action_evidence_path}", file=sys.stderr)
            return 1
        action_evidence = read_json(action_evidence_path)
        if not isinstance(action_evidence, dict):
            print("action_evidence_payload_invalid", file=sys.stderr)
            return 1
        evidence_errors = validate_reviewed_action_evidence(
            manifest=action_evidence,
            rows=rows,
            segments=segments,
            action_ledger=action_ledger,
        )
        if evidence_errors:
            for error in evidence_errors:
                print(error, file=sys.stderr)
            return 1
        action_evidence_sha256 = payload_fingerprint(action_evidence)
    elif action_evidence_path is not None:
        print("action_evidence_unexpected_without_action_rows", file=sys.stderr)
        return 1

    action_intent_signals = action_intent_recall_signals(segments=segments, action_ledger=action_ledger)
    action_intent_review_arg = getattr(args, "action_intent_review", None)
    action_intent_review_path = action_intent_review_arg.expanduser().resolve() if action_intent_review_arg else None
    action_intent_review_sha256: str | None = None
    if action_intent_signals:
        if action_intent_review_path is None:
            print("action_intent_review_required", file=sys.stderr)
            return 1
        if not action_intent_review_path.is_file():
            print(f"action_intent_review_missing:{action_intent_review_path}", file=sys.stderr)
            return 1
        action_intent_review = read_json(action_intent_review_path)
        if not isinstance(action_intent_review, dict):
            print("action_intent_review_payload_invalid", file=sys.stderr)
            return 1
        intent_review_errors = validate_reviewed_action_intent_review(
            manifest=action_intent_review,
            rows=rows,
            action_evidence=action_evidence,
            segments=segments,
            action_ledger=action_ledger,
        )
        if intent_review_errors:
            for error in intent_review_errors:
                print(error, file=sys.stderr)
            return 1
        action_intent_review_sha256 = payload_fingerprint(action_intent_review)
    elif action_intent_review_path is not None:
        print("action_intent_review_unexpected_without_signals", file=sys.stderr)
        return 1

    project_rows = parse_shareable_project_update_rows(chinese_markdown)
    required_project_coverage = recompute_project_update_coverage(segments)
    project_evidence_arg = getattr(args, "project_evidence", None)
    project_evidence_path = project_evidence_arg.expanduser().resolve() if project_evidence_arg else None
    project_evidence_sha256: str | None = None
    project_evidence: dict[str, Any] | None = None
    if project_rows or required_project_coverage:
        if project_evidence_path is None:
            print("project_evidence_required", file=sys.stderr)
            return 1
        if not project_evidence_path.is_file():
            print(f"project_evidence_missing:{project_evidence_path}", file=sys.stderr)
            return 1
        project_evidence = read_json(project_evidence_path)
        if not isinstance(project_evidence, dict):
            print("project_evidence_payload_invalid", file=sys.stderr)
            return 1
        project_evidence_errors = validate_reviewed_project_evidence(
            manifest=project_evidence,
            rows=project_rows,
            segments=segments,
            action_ledger=action_ledger,
        )
        if project_evidence_errors:
            for error in project_evidence_errors:
                print(error, file=sys.stderr)
            return 1
        project_evidence_sha256 = payload_fingerprint(project_evidence)
    elif project_evidence_path is not None:
        print("project_evidence_unexpected_without_project_rows_or_coverage", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "work" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    destinations = canonical_minutes_paths(output_dir)
    chinese_content = _atomic_write_markdown(destinations["zh"], chinese_markdown)
    english_content = _atomic_write_markdown(destinations["en"], english_markdown)
    share_paths = share_bundle_paths(output_dir)
    share_paths["zh"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_markdown(share_paths["zh"], chinese_content)
    _atomic_write_markdown(share_paths["en"], english_content)
    write_transcript_markdown(share_paths["transcript"], segments)
    source_sha256 = {
        "zh": hashlib.sha256(chinese_content.encode("utf-8")).hexdigest(),
        "en": hashlib.sha256(english_content.encode("utf-8")).hexdigest(),
    }
    canonical_sha256 = {
        language: hashlib.sha256(path.read_bytes()).hexdigest()
        for language, path in destinations.items()
    }
    share_sha256 = {
        artifact: hashlib.sha256(path.read_bytes()).hexdigest()
        for artifact, path in share_paths.items()
    }
    publish_status = {
        "format": PUBLICATION_FORMAT,
        "status": "published",
        "languages": ["zh", "en"],
        "sources": {"zh": str(chinese_source), "en": str(english_source)},
        "source_sha256": source_sha256,
        "canonical_sha256": canonical_sha256,
        "share_destinations": {artifact: str(path) for artifact, path in share_paths.items()},
        "share_sha256": share_sha256,
        "transcript_sha256": transcript_fingerprint(segments),
        "action_ledger_sha256": action_ledger_fingerprint(action_ledger),
        "action_evidence_format": ACTION_EVIDENCE_FORMAT if action_evidence_sha256 else None,
        "action_evidence_path": str(action_evidence_path) if action_evidence_path else None,
        "action_evidence_sha256": action_evidence_sha256,
        "action_row_count": len(rows),
        "action_intent_review_format": ACTION_INTENT_REVIEW_FORMAT if action_intent_review_sha256 else None,
        "action_intent_review_path": str(action_intent_review_path) if action_intent_review_path else None,
        "action_intent_review_sha256": action_intent_review_sha256,
        "action_intent_signal_count": len(action_intent_signals),
        "project_evidence_format": PROJECT_EVIDENCE_FORMAT if project_evidence_sha256 else None,
        "project_evidence_path": str(project_evidence_path) if project_evidence_path else None,
        "project_evidence_sha256": project_evidence_sha256,
        "project_update_row_count": len(project_rows),
        "project_update_coverage_min_seconds": PROJECT_UPDATE_COVERAGE_MIN_SECONDS,
        "required_project_update_participants": project_update_coverage_snapshot(segments),
        "project_update_exception_participants": sorted(
            str(exception.get("participant"))
            for exception in (project_evidence or {}).get("exceptions", [])
            if isinstance(exception, dict) and str(exception.get("participant") or "").strip()
        ),
        "duration_seconds": duration,
        "destinations": {language: str(path) for language, path in destinations.items()},
    }
    write_json(output_dir / "minutes.publish-status.json", publish_status)
    write_json(
        review_dir / "minutes.coverage.json",
        {
            "format": "meeting-minutes/project-update-coverage-v1",
            "transcript_sha256": publish_status["transcript_sha256"],
            "coverage_min_seconds": publish_status["project_update_coverage_min_seconds"],
            "required_participants": publish_status["required_project_update_participants"],
            "project_update_participants": [row.participant for row in project_rows],
            "exceptions": (project_evidence or {}).get("exceptions", []),
        },
    )
    run_status_path = output_dir / "run_status.json"
    run_status = read_json(run_status_path) if run_status_path.is_file() else {}
    statuses = dict(run_status.get("statuses", {}))
    statuses["publication"] = {
        "status": "published",
        "format": publish_status["format"],
        "languages": publish_status["languages"],
        "action_row_count": len(rows),
        "action_evidence_bound": bool(action_evidence_sha256),
        "action_intent_review_bound": bool(action_intent_review_sha256),
        "action_intent_signal_count": len(action_intent_signals),
        "project_update_row_count": len(project_rows),
        "project_evidence_bound": bool(project_evidence_sha256),
    }
    write_json(run_status_path, {**run_status, "statuses": statuses})
    summary_status_path = output_dir / "summary_status.json"
    if summary_status_path.is_file():
        try:
            summary_status = read_json(summary_status_path)
        except (OSError, ValueError):
            summary_status = None
        if isinstance(summary_status, dict):
            write_json(
                summary_status_path,
                {
                    **summary_status,
                    "status": "published",
                    "actions": len(rows),
                    "project_updates": len(project_rows),
                    "publication": {
                        "format": publish_status["format"],
                        "languages": publish_status["languages"],
                    },
                },
            )
    print(str(destinations["zh"]))
    print(str(destinations["en"]))
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
    if args.command == "smart-summarize":
        return smart_summarize_existing(args)
    if args.command in {"smart-repair-reviewed", "smart-sanitize-reviewed"}:
        return smart_sanitize_reviewed_existing(args)
    if args.command == "diarize":
        return diarize_existing(args)
    if args.command == "visual-identify":
        return visual_identify_existing(args)
    if args.command == "visual-agent-audit":
        return visual_agent_audit_existing(args)
    if args.command == "dynamic-visual-identify":
        return dynamic_visual_identify_existing(args)
    if args.command == "visual-voice-identify":
        return visual_voice_identify_existing(args)
    if args.command == "visual-cluster-identify":
        return direct_visual_cluster_identify_existing(args)
    if args.command == "avatar-template-identify":
        return avatar_template_identify_existing(args)
    if args.command == "roster-avatar-identify":
        return roster_avatar_identify_existing(args)
    if args.command == "relabel":
        return relabel_existing(args)
    if args.command == "audit-actions":
        return audit_actions_existing(args)
    if args.command == "validate-actions":
        return validate_actions_existing(args)
    if args.command == "validate-minutes":
        return validate_minutes_file(args)
    if args.command == "publish-minutes":
        return publish_minutes_file(args)
    if args.command == "voice-template":
        return write_voice_template(args)
    if args.command == "voice-registry":
        if args.voice_registry_command == "build":
            return build_voice_registry_existing(args)
        if args.voice_registry_command == "apply":
            return apply_voice_registry_existing(args)
        if args.voice_registry_command == "consensus":
            return enforce_voice_registry_consensus_existing(args)
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
