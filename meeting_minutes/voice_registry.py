from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .diarization import (
    SPEECHBRAIN_ECAPA_MODEL,
    _encode_windows_with_classifier,
    _load_speechbrain_classifier,
    _normalize_rows,
    _read_wav,
    _speech_windows,
    _speechbrain_cache,
    _windows_from_ranges,
    load_voice_enrollment,
)
from .media import extract_audio

VOICE_REGISTRY_FORMAT = "meeting-minutes/voice-registry-v1"
VOICE_REGISTRY_SOURCES_FORMAT = "meeting-minutes/voice-registry-sources-v1"
_TRUSTED_NAME_SOURCES = {"voice_enrollment", "voice_registry", "user_confirmed_speaker_volume_mapping"}


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_voice_registry_sources(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    payload = _read_json_object(manifest_path)
    if payload.get("format") != VOICE_REGISTRY_SOURCES_FORMAT:
        raise ValueError(f"Unsupported voice registry sources format in {manifest_path}")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("voice registry sources need a non-empty sources list")

    sources: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(raw_sources, start=1):
        if not isinstance(item, dict):
            raise ValueError("voice registry source entries must be JSON objects")
        source_id = str(item.get("id") or f"source-{index}").strip()
        if not source_id or source_id in ids:
            raise ValueError("voice registry source ids must be unique and non-empty")
        input_path = _resolve_path(item.get("input"), base_dir=manifest_path.parent)
        enrollment_path = _resolve_path(item.get("enrollment"), base_dir=manifest_path.parent)
        if not input_path.exists():
            raise FileNotFoundError(f"voice registry source media does not exist: {input_path}")
        if not enrollment_path.exists():
            raise FileNotFoundError(f"voice registry enrollment does not exist: {enrollment_path}")
        source_offset = float(item.get("source_offset", 0.0))
        sources.append(
            {
                "id": source_id,
                "input": input_path,
                "enrollment": enrollment_path,
                "source_offset": source_offset,
                "provenance": str(item.get("provenance") or "unspecified"),
            }
        )
        ids.add(source_id)
    return sources


def _source_audio_path(source: dict[str, Any], work_dir: Path) -> Path:
    digest = hashlib.sha256(str(source["input"]).encode("utf-8")).hexdigest()[:16]
    return work_dir / "audio" / f"{source['id']}-{digest}.wav"


def _source_profile_record(
    *,
    source: dict[str, Any],
    classifier: Any,
    audio_path: Path,
) -> dict[str, Any]:
    sample_rate, data = _read_wav(audio_path)
    ranges, enrollment_meta = load_voice_enrollment(
        Path(source["enrollment"]),
        source_offset=float(source["source_offset"]),
        audio_duration=len(data) / sample_rate,
    )
    profiles: dict[str, Any] = {}
    for name, speaker_ranges in ranges.items():
        windows = _windows_from_ranges(data, sample_rate, speaker_ranges)
        active_seconds = sum(float(window["end"]) - float(window["start"]) for window in windows)
        if active_seconds < 3.0:
            raise ValueError(
                f"voice registry source {source['id']} has only {active_seconds:.2f}s of active speech for {name}"
            )
        embeddings, valid_windows = _encode_windows_with_classifier(classifier, data, sample_rate, windows)
        if len(embeddings) == 0:
            raise ValueError(f"voice registry source {source['id']} produced no embeddings for {name}")
        profiles[name] = {
            "embeddings": [[round(float(value), 8) for value in vector] for vector in embeddings],
            "embedding_count": int(len(embeddings)),
            "active_speech_seconds": round(active_seconds, 3),
            "speech_windows": len(valid_windows),
        }
    return {
        "id": str(source["id"]),
        "input": str(source["input"]),
        "enrollment": str(source["enrollment"]),
        "source_offset": float(source["source_offset"]),
        "provenance": str(source["provenance"]),
        "enrollment_meta": enrollment_meta,
        "profiles": profiles,
    }


def _profile_summaries(source_records: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    by_name: dict[str, list[Any]] = {}
    active_seconds: dict[str, float] = {}
    source_ids: dict[str, set[str]] = {}
    for source in source_records:
        for name, profile in source["profiles"].items():
            vectors = np.asarray(profile["embeddings"], dtype="float32")
            by_name.setdefault(name, []).append(vectors)
            active_seconds[name] = active_seconds.get(name, 0.0) + float(profile["active_speech_seconds"])
            source_ids.setdefault(name, set()).add(str(source["id"]))

    summaries: dict[str, Any] = {}
    for name, matrices in by_name.items():
        vectors = np.concatenate(matrices, axis=0)
        centroid = _normalize_rows(np.mean(vectors, axis=0, keepdims=True))[0]
        summaries[name] = {
            "centroid": [round(float(value), 8) for value in centroid],
            "embedding_count": int(len(vectors)),
            "active_speech_seconds": round(active_seconds[name], 3),
            "source_ids": sorted(source_ids[name]),
        }
    if len(summaries) < 2:
        raise ValueError("voice registry needs at least two named profiles")
    return summaries


def _select_threshold(positive_scores: list[float], negative_scores: list[float], target_far: float) -> tuple[float | None, float | None, float | None]:
    if not positive_scores or not negative_scores:
        return None, None, None
    candidates = sorted({*positive_scores, *negative_scores, 1.000001})
    eligible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        false_accept_rate = sum(score >= threshold for score in negative_scores) / len(negative_scores)
        true_accept_rate = sum(score >= threshold for score in positive_scores) / len(positive_scores)
        if false_accept_rate <= target_far:
            eligible.append((threshold, true_accept_rate, false_accept_rate))
    if not eligible:
        return None, None, None
    threshold, true_accept_rate, false_accept_rate = max(eligible, key=lambda row: (row[1], -row[0]))
    return round(float(threshold), 4), round(float(true_accept_rate), 4), round(float(false_accept_rate), 4)


def calibrate_voice_registry(source_records: list[dict[str, Any]], *, target_far: float = 0.01) -> dict[str, Any]:
    import numpy as np

    if target_far <= 0 or target_far >= 1:
        raise ValueError("voice registry target FAR must be between 0 and 1")
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    positive_margins: list[float] = []
    source_ids = [str(record["id"]) for record in source_records]
    all_names = sorted({name for record in source_records for name in record["profiles"]})

    for held_out in source_records:
        held_out_id = str(held_out["id"])
        centroids: dict[str, Any] = {}
        for name in all_names:
            matrices = []
            for source in source_records:
                if str(source["id"]) == held_out_id or name not in source["profiles"]:
                    continue
                matrices.append(np.asarray(source["profiles"][name]["embeddings"], dtype="float32"))
            if matrices:
                centroids[name] = _normalize_rows(np.mean(np.concatenate(matrices, axis=0), axis=0, keepdims=True))[0]
        if len(centroids) < 2:
            continue
        candidate_names = sorted(centroids)
        profile_matrix = _normalize_rows(np.stack([centroids[name] for name in candidate_names]))
        for actual_name, profile in held_out["profiles"].items():
            if actual_name not in centroids:
                continue
            embeddings = _normalize_rows(np.asarray(profile["embeddings"], dtype="float32"))
            scores = embeddings @ profile_matrix.T
            actual_index = candidate_names.index(actual_name)
            for row in scores:
                positive = float(row[actual_index])
                others = [float(value) for index, value in enumerate(row) if index != actual_index]
                positive_scores.append(positive)
                negative_scores.extend(others)
                positive_margins.append(positive - max(others))

    threshold, true_accept_rate, false_accept_rate = _select_threshold(positive_scores, negative_scores, target_far)
    status = "calibrated" if len(positive_scores) >= 8 and len(negative_scores) >= 16 and threshold is not None else "limited"
    margin = None
    if positive_margins:
        margin = round(float(max(0.06, np.percentile(positive_margins, 10))), 4)
    return {
        "status": status,
        "target_false_accept_rate": target_far,
        "positive_trials": len(positive_scores),
        "negative_trials": len(negative_scores),
        "source_count": len(source_ids),
        "suggested_threshold": threshold,
        "suggested_margin": margin,
        "true_accept_rate": true_accept_rate,
        "false_accept_rate": false_accept_rate,
        "note": (
            "Threshold is based on leave-one-source-out cross-recording trials."
            if status == "calibrated"
            else "Not enough cross-recording trials for automatic acceptance; use only as an audit signal."
        ),
    }


def build_voice_registry(
    manifest_path: Path,
    output_path: Path,
    *,
    work_dir: Path | None = None,
    speechbrain_cache: Path | None = None,
    target_far: float = 0.01,
    speechbrain_model: str = SPEECHBRAIN_ECAPA_MODEL,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    sources = load_voice_registry_sources(manifest_path)
    work_dir = (work_dir or output_path.parent / "voice_registry_work").expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    model_cache = _speechbrain_cache(output_path, speechbrain_cache or work_dir / "speechbrain_models")
    classifier = _load_speechbrain_classifier(speechbrain_model, model_cache)

    source_records: list[dict[str, Any]] = []
    for source in sources:
        audio_path = _source_audio_path(source, work_dir)
        if not audio_path.exists():
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            extract_audio(Path(source["input"]), audio_path)
        source_records.append(_source_profile_record(source=source, classifier=classifier, audio_path=audio_path))

    profiles = _profile_summaries(source_records)
    payload = {
        "format": VOICE_REGISTRY_FORMAT,
        "model": speechbrain_model,
        "model_cache": str(model_cache),
        "profiles": profiles,
        "sources": source_records,
        "calibration": calibrate_voice_registry(source_records, target_far=target_far),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_voice_registry(registry_path: Path) -> dict[str, Any]:
    registry_path = registry_path.expanduser().resolve()
    payload = _read_json_object(registry_path)
    if payload.get("format") != VOICE_REGISTRY_FORMAT:
        raise ValueError(f"Unsupported voice registry format in {registry_path}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or len(profiles) < 2:
        raise ValueError("voice registry needs at least two profiles")
    dimensions: set[int] = set()
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(profile, dict):
            raise ValueError("voice registry profile names and records must be valid")
        centroid = profile.get("centroid")
        if not isinstance(centroid, list) or not centroid:
            raise ValueError(f"voice registry profile {name} is missing a centroid")
        dimensions.add(len(centroid))
    if len(dimensions) != 1:
        raise ValueError("voice registry profile centroids must have equal dimensions")
    return payload


def _registry_thresholds(
    registry: dict[str, Any],
    *,
    threshold: float | None,
    margin: float | None,
) -> tuple[float, float]:
    calibration = registry.get("calibration") if isinstance(registry.get("calibration"), dict) else {}
    if threshold is None:
        threshold = calibration.get("suggested_threshold") if calibration.get("status") == "calibrated" else 0.72
    if margin is None:
        margin = calibration.get("suggested_margin") if calibration.get("status") == "calibrated" else 0.1
    threshold = float(threshold)
    margin = float(margin)
    if not -1 <= threshold <= 1:
        raise ValueError("voice registry threshold must be between -1 and 1")
    if margin < 0:
        raise ValueError("voice registry margin must be non-negative")
    return threshold, margin


def attach_registry_scores(
    segments: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    names: list[str],
    scores: Any,
    *,
    threshold: float,
    margin: float,
    minimum_vote_share: float = 0.67,
) -> dict[str, Any]:
    import numpy as np

    if len(windows) != len(scores):
        raise ValueError("voice registry windows and scores must have the same length")
    if len(names) < 2:
        raise ValueError("voice registry needs at least two names")
    accepted_windows = 0
    evidence_by_segment: list[dict[str, list[dict[str, float]]]] = [{} for _ in segments]
    for window, row in zip(windows, scores):
        row_values = np.asarray(row, dtype="float32")
        ranked = np.argsort(row_values)[::-1]
        best_index = int(ranked[0])
        runner_up = float(row_values[int(ranked[1])])
        best_score = float(row_values[best_index])
        best_margin = best_score - runner_up
        if best_score < threshold or best_margin < margin:
            continue
        accepted_windows += 1
        for segment_index, segment in enumerate(segments):
            overlap = max(
                0.0,
                min(float(segment["end"]), float(window["end"])) - max(float(segment["start"]), float(window["start"])),
            )
            if overlap <= 0:
                continue
            evidence_by_segment[segment_index].setdefault(names[best_index], []).append(
                {"overlap": overlap, "score": best_score, "margin": best_margin}
            )

    assigned = 0
    skipped_trusted = 0
    for segment, candidates in zip(segments, evidence_by_segment):
        if not candidates:
            continue
        if str(segment.get("name_source") or "") in _TRUSTED_NAME_SOURCES:
            skipped_trusted += 1
            continue
        ranked_candidates = sorted(
            candidates.items(),
            key=lambda item: sum(record["overlap"] for record in item[1]),
            reverse=True,
        )
        name, records = ranked_candidates[0]
        all_records = [record for values in candidates.values() for record in values]
        vote_share = sum(record["overlap"] for record in records) / sum(record["overlap"] for record in all_records)
        segment_duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        minimum_windows = 1 if segment_duration <= 1.8 else 2
        if len(records) < minimum_windows or vote_share < minimum_vote_share:
            continue
        mean_score = sum(record["score"] for record in records) / len(records)
        mean_margin = sum(record["margin"] for record in records) / len(records)
        confidence = min(0.98, max(0.0, 0.45 + mean_score * 0.35 + mean_margin * 1.2))
        if segment.get("speaker_confidence") is not None:
            confidence = min(confidence, float(segment["speaker_confidence"]))
        segment["name"] = name
        segment["name_source"] = "voice_registry"
        segment["name_confidence"] = round(confidence, 3)
        segment["voice_registry_evidence"] = {
            "accepted_windows": len(records),
            "vote_share": round(vote_share, 3),
            "mean_score": round(mean_score, 4),
            "mean_margin": round(mean_margin, 4),
            "threshold": round(threshold, 4),
            "margin_threshold": round(margin, 4),
        }
        assigned += 1
    return {
        "accepted_windows": accepted_windows,
        "assigned_segments": assigned,
        "skipped_trusted_segments": skipped_trusted,
        "unresolved_segments": len(segments) - assigned,
    }


def apply_voice_registry(
    audio_path: Path,
    segments: list[dict[str, Any]],
    registry_path: Path,
    *,
    speechbrain_cache: Path | None = None,
    threshold: float | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    registry_path = registry_path.expanduser().resolve()
    registry = load_voice_registry(registry_path)
    names = sorted(registry["profiles"])
    profiles = _normalize_rows(
        np.asarray([registry["profiles"][name]["centroid"] for name in names], dtype="float32")
    )
    resolved_threshold, resolved_margin = _registry_thresholds(registry, threshold=threshold, margin=margin)
    sample_rate, data = _read_wav(audio_path)
    windows, vad_status = _speech_windows(data, sample_rate)
    if not windows:
        return {
            "engine": "speechbrain-ecapa-voice-registry",
            "status": "empty_audio",
            "registry": str(registry_path),
            "profiles": names,
        }
    model = str(registry.get("model") or SPEECHBRAIN_ECAPA_MODEL)
    cache_dir = _speechbrain_cache(audio_path, speechbrain_cache)
    classifier = _load_speechbrain_classifier(model, cache_dir)
    embeddings, windows = _encode_windows_with_classifier(classifier, data, sample_rate, windows)
    if len(embeddings) == 0:
        return {
            "engine": "speechbrain-ecapa-voice-registry",
            "status": "insufficient_embeddings",
            "registry": str(registry_path),
            "profiles": names,
        }
    if embeddings.shape[1] != profiles.shape[1]:
        raise ValueError("voice registry embedding dimension does not match the active SpeechBrain model")
    summary = attach_registry_scores(
        segments,
        windows,
        names,
        embeddings @ profiles.T,
        threshold=resolved_threshold,
        margin=resolved_margin,
    )
    return {
        "engine": "speechbrain-ecapa-voice-registry",
        "status": "ok",
        "registry": str(registry_path),
        "profiles": names,
        "speech_windows": len(windows),
        "candidate_windows": vad_status["candidate_windows"],
        "threshold": resolved_threshold,
        "margin": resolved_margin,
        "calibration": registry.get("calibration", {}),
        "model": model,
        "model_cache": str(cache_dir),
        **summary,
        "note": "Names were attached from a cross-recording voice registry only when score and runner-up margin gates passed.",
    }
