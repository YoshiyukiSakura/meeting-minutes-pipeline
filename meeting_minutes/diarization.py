from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SPEECHBRAIN_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def _read_wav(audio_path: Path) -> tuple[int, Any]:
    from scipy.io import wavfile  # type: ignore
    import numpy as np

    sample_rate, data = wavfile.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in {"i", "u"}:
        max_value = float(np.iinfo(data.dtype).max)
        data = data.astype("float32") / max_value
    else:
        data = data.astype("float32")
    return int(sample_rate), data


def _band_log_energy(signal: Any, sample_rate: int) -> Any:
    import numpy as np

    if len(signal) < int(sample_rate * 0.2):
        return None
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    bands = np.geomspace(80, min(7600, sample_rate / 2 - 1), 25)
    values = []
    for low, high in zip(bands[:-1], bands[1:]):
        mask = (freqs >= low) & (freqs < high)
        values.append(float(np.log(np.mean(spectrum[mask]) + 1e-10)) if mask.any() else -23.0)
    rms = float(np.sqrt(np.mean(signal**2) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(signal)))))
    return np.array(values + [np.log(rms + 1e-10), zcr], dtype="float32")


def _kmeans(features: Any, k: int, *, iterations: int = 60) -> tuple[Any, Any]:
    import numpy as np

    n = len(features)
    if n == 0:
        return np.array([], dtype=int), np.empty((0, 0))
    k = max(1, min(k, n))
    centers = [features[0]]
    while len(centers) < k:
        distances = np.min(
            np.stack([np.sum((features - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        centers.append(features[int(np.argmax(distances))])
    centers = np.stack(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        distances = np.stack([np.sum((features - center) ** 2, axis=1) for center in centers], axis=1)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for index in range(k):
            assigned = features[labels == index]
            if len(assigned):
                centers[index] = assigned.mean(axis=0)
    return labels, centers


def _normalize_rows(matrix: Any) -> Any:
    import numpy as np

    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)


def _cosine_kmeans(features: Any, k: int, *, iterations: int = 60) -> tuple[Any, Any]:
    import numpy as np

    n = len(features)
    if n == 0:
        return np.array([], dtype=int), np.empty((0, 0))
    k = max(1, min(k, n))
    features = _normalize_rows(features)
    centers = [features[0]]
    while len(centers) < k:
        similarities = np.stack([features @ center for center in centers], axis=1)
        best = np.max(similarities, axis=1)
        centers.append(features[int(np.argmin(best))])
    centers = _normalize_rows(np.stack(centers))
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        next_labels = np.argmax(features @ centers.T, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for index in range(k):
            assigned = features[labels == index]
            if len(assigned):
                centers[index] = assigned.mean(axis=0)
        centers = _normalize_rows(centers)
    return labels, centers


def _smooth_labels(windows: list[dict[str, Any]], labels: Any, *, passes: int = 2) -> Any:
    import numpy as np

    smoothed = np.array(labels, dtype=int).copy()
    if len(smoothed) < 3:
        return smoothed
    for _ in range(passes):
        next_labels = smoothed.copy()
        for index in range(1, len(smoothed) - 1):
            if smoothed[index - 1] == smoothed[index + 1] != smoothed[index]:
                left_gap = float(windows[index]["start"]) - float(windows[index - 1]["end"])
                right_gap = float(windows[index + 1]["start"]) - float(windows[index]["end"])
                if left_gap <= 1.0 and right_gap <= 1.0:
                    next_labels[index] = smoothed[index - 1]
        smoothed = next_labels
    return smoothed


def _reorder_labels_by_first_time(windows: list[dict[str, Any]], labels: Any) -> dict[int, str]:
    first_seen: dict[int, float] = {}
    for window, label in zip(windows, labels):
        first_seen.setdefault(int(label), float(window["start"]))
    ordered = sorted(first_seen, key=lambda label: first_seen[label])
    return {label: f"Speaker {index + 1}" for index, label in enumerate(ordered)}


def _merge_windows(windows: list[dict[str, Any]], labels: Any, label_names: dict[int, str]) -> list[dict[str, Any]]:
    adjusted_pairs: list[tuple[dict[str, Any], int]] = []
    for index, window in enumerate(windows):
        adjusted = dict(window)
        start = float(adjusted["start"])
        end = float(adjusted["end"])
        center = (start + end) / 2
        if index > 0:
            prev = windows[index - 1]
            prev_center = (float(prev["start"]) + float(prev["end"])) / 2
            start = max(start, (prev_center + center) / 2)
        if index < len(windows) - 1:
            nxt = windows[index + 1]
            next_center = (float(nxt["start"]) + float(nxt["end"])) / 2
            end = min(end, (center + next_center) / 2)
        if end <= start:
            continue
        adjusted["start"] = start
        adjusted["end"] = end
        adjusted_pairs.append((adjusted, int(labels[index])))

    merged: list[dict[str, Any]] = []
    for window, label in adjusted_pairs:
        speaker = label_names[int(label)]
        if merged and merged[-1]["speaker"] == speaker and float(window["start"]) - float(merged[-1]["end"]) <= 1.0:
            merged[-1]["end"] = float(window["end"])
            merged[-1]["confidence_values"].append(float(window["confidence"]))
        else:
            merged.append(
                {
                    "start": float(window["start"]),
                    "end": float(window["end"]),
                    "speaker": speaker,
                    "confidence_values": [float(window["confidence"])],
                }
            )
    turns: list[dict[str, Any]] = []
    for turn in merged:
        if float(turn["end"]) - float(turn["start"]) < 0.35:
            continue
        values = turn.pop("confidence_values")
        turn["confidence"] = round(sum(values) / len(values), 3)
        turns.append(turn)
    return turns


def _speech_windows(
    data: Any,
    sample_rate: int,
    *,
    window_seconds: float = 1.6,
    hop_seconds: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    window_size = max(1, int(sample_rate * window_seconds))
    hop_size = max(1, int(sample_rate * hop_seconds))
    starts = range(0, max(1, len(data) - window_size + 1), hop_size)
    rms_values = []
    for start in starts:
        chunk = data[start : min(len(data), start + window_size)]
        if len(chunk):
            rms_values.append(float(np.sqrt(np.mean(chunk**2) + 1e-12)))
    if not rms_values:
        return [], {"candidate_windows": 0, "rms_threshold": 0.0}
    threshold = max(float(np.percentile(rms_values, 55)) * 0.55, float(np.max(rms_values)) * 0.015, 1e-5)
    windows: list[dict[str, Any]] = []
    for start in range(0, max(1, len(data) - window_size + 1), hop_size):
        end = min(len(data), start + window_size)
        chunk = data[start:end]
        if not len(chunk):
            continue
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        if rms < threshold:
            continue
        windows.append(
            {
                "start": start / sample_rate,
                "end": end / sample_rate,
                "sample_start": start,
                "sample_end": end,
                "rms": rms,
                "confidence": 0.5,
            }
        )
    return windows, {"candidate_windows": len(rms_values), "rms_threshold": threshold}


def _speechbrain_cache(audio_path: Path, speechbrain_cache: Path | None) -> Path:
    if speechbrain_cache:
        return speechbrain_cache.expanduser().resolve()
    return audio_path.parent / "speechbrain_models" / "spkrec-ecapa-voxceleb"


def _load_speechbrain_classifier(model_source: str, savedir: Path) -> Any:
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    savedir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(savedir / "hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(savedir / "hf_home" / "hub"))
    return EncoderClassifier.from_hparams(source=model_source, savedir=str(savedir))


def _encode_windows_with_classifier(
    classifier: Any,
    data: Any,
    sample_rate: int,
    windows: list[dict[str, Any]],
    *,
    batch_size: int = 32,
    minimum_seconds: float = 0.45,
) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np
    import torch

    valid_windows = [
        window
        for window in windows
        if int(window["sample_end"]) - int(window["sample_start"]) >= int(sample_rate * minimum_seconds)
    ]
    if not valid_windows:
        return np.empty((0, 0), dtype="float32"), []

    batches = []
    for offset in range(0, len(valid_windows), batch_size):
        batch_windows = valid_windows[offset : offset + batch_size]
        chunks = [
            data[int(window["sample_start"]) : int(window["sample_end"])].astype("float32")
            for window in batch_windows
        ]
        target_length = max(len(chunk) for chunk in chunks)
        payload = np.zeros((len(chunks), target_length), dtype="float32")
        wav_lens = np.zeros(len(chunks), dtype="float32")
        for index, chunk in enumerate(chunks):
            payload[index, : len(chunk)] = chunk
            wav_lens[index] = len(chunk) / target_length
        batch = torch.from_numpy(payload)
        lens = torch.from_numpy(wav_lens)
        with torch.inference_mode():
            try:
                encoded = classifier.encode_batch(batch, wav_lens=lens, normalize=False)
            except TypeError:  # pragma: no cover - older SpeechBrain API
                encoded = classifier.encode_batch(batch)
        vectors = encoded.squeeze().detach().cpu().numpy()
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        batches.append(vectors.astype("float32"))
    embeddings = _normalize_rows(np.concatenate(batches, axis=0))
    return embeddings, valid_windows


def _classification_confidences(scores: Any, labels: Any, *, unknown_label: int | None = None) -> list[float]:
    import numpy as np

    confidences: list[float] = []
    for row, label in zip(scores, labels):
        if unknown_label is not None and int(label) == unknown_label:
            confidences.append(0.0)
            continue
        ordered = np.sort(row)[::-1]
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
        score = float(ordered[0])
        confidences.append(round(max(0.35, min(0.95, 0.45 + score * 0.25 + margin * 1.2)), 3))
    return confidences


def diarize_audio_local_cluster(
    audio_path: Path,
    *,
    expected_speakers: int,
    window_seconds: float = 1.6,
    hop_seconds: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    sample_rate, data = _read_wav(audio_path)
    windows, vad_status = _speech_windows(
        data,
        sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    features = []
    if not windows:
        return [], {"engine": "local-mfcc-kmeans", "status": "empty_audio"}
    for window in windows:
        chunk = data[int(window["sample_start"]) : int(window["sample_end"])]
        feature = _band_log_energy(chunk, sample_rate)
        if feature is None:
            continue
        features.append(feature)
    if len(features) < expected_speakers:
        return [], {
            "engine": "local-mfcc-kmeans",
            "status": "insufficient_speech_windows",
            "windows": len(features),
        }
    windows = windows[: len(features)]
    matrix = np.stack(features)
    matrix = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-6)
    labels, centers = _kmeans(matrix, expected_speakers)
    distances = np.stack([np.sum((matrix - center) ** 2, axis=1) for center in centers], axis=1)
    sorted_distances = np.sort(distances, axis=1)
    margins = (sorted_distances[:, 1] - sorted_distances[:, 0]) / (sorted_distances[:, 1] + 1e-6) if expected_speakers > 1 else np.ones(len(matrix))
    for window, margin in zip(windows, margins):
        window["confidence"] = float(max(0.35, min(0.85, 0.45 + margin * 0.4)))
    label_names = _reorder_labels_by_first_time(windows, labels)
    turns = _merge_windows(windows, labels, label_names)
    return turns, {
        "engine": "local-mfcc-kmeans",
        "status": "ok",
        "speakers": expected_speakers,
        "speech_windows": len(windows),
        "candidate_windows": vad_status["candidate_windows"],
        "turns": len(turns),
        "note": "Experimental local voice clustering. It separates recurring voice characteristics but does not know real names.",
    }


def diarize_audio_speechbrain_cluster(
    audio_path: Path,
    *,
    expected_speakers: int,
    window_seconds: float = 1.6,
    hop_seconds: float = 0.8,
    speechbrain_cache: Path | None = None,
    speechbrain_model: str = SPEECHBRAIN_ECAPA_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    sample_rate, data = _read_wav(audio_path)
    windows, vad_status = _speech_windows(
        data,
        sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    if len(windows) < expected_speakers:
        return [], {
            "engine": "speechbrain-ecapa-kmeans",
            "status": "insufficient_speech_windows",
            "speech_windows": len(windows),
        }
    cache_dir = _speechbrain_cache(audio_path, speechbrain_cache)
    try:
        classifier = _load_speechbrain_classifier(speechbrain_model, cache_dir)
        embeddings, windows = _encode_windows_with_classifier(classifier, data, sample_rate, windows)
    except Exception as exc:  # pragma: no cover - optional model/runtime dependent
        return [], {
            "engine": "speechbrain-ecapa-kmeans",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "model": speechbrain_model,
            "model_cache": str(cache_dir),
        }
    if len(embeddings) < expected_speakers:
        return [], {
            "engine": "speechbrain-ecapa-kmeans",
            "status": "insufficient_embeddings",
            "speech_windows": len(windows),
            "model": speechbrain_model,
            "model_cache": str(cache_dir),
        }
    labels, centers = _cosine_kmeans(embeddings, expected_speakers)
    labels = _smooth_labels(windows, labels)
    scores = embeddings @ _normalize_rows(centers).T
    for window, confidence in zip(windows, _classification_confidences(scores, labels)):
        window["confidence"] = confidence
    label_names = _reorder_labels_by_first_time(windows, labels)
    turns = _merge_windows(windows, labels, label_names)
    return turns, {
        "engine": "speechbrain-ecapa-kmeans",
        "status": "ok",
        "speakers": expected_speakers,
        "speech_windows": len(windows),
        "candidate_windows": vad_status["candidate_windows"],
        "turns": len(turns),
        "model": speechbrain_model,
        "model_cache": str(cache_dir),
        "note": "SpeechBrain ECAPA clustering separates voices but does not know real names without enrollment or a participant map.",
    }


def _parse_enrollment_item(item: Any) -> tuple[float, float]:
    if isinstance(item, dict):
        return float(item["start"]), float(item["end"])
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return float(item[0]), float(item[1])
    raise ValueError("voice enrollment ranges must be {start,end} objects or [start,end] pairs")


def load_voice_enrollment(
    enrollment_path: Path,
    *,
    source_offset: float = 0.0,
    audio_duration: float | None = None,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]:
    import json

    payload = json.loads(enrollment_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("voice enrollment must be a JSON object")
    reference = str(payload.get("enrollment_audio_reference", "effective_clip"))
    if reference not in {"effective_clip", "original_file"}:
        raise ValueError("enrollment_audio_reference must be effective_clip or original_file")
    speakers_payload = payload.get("speakers", payload)
    if not isinstance(speakers_payload, dict):
        raise ValueError("voice enrollment speakers must be a JSON object")

    ranges: dict[str, list[tuple[float, float]]] = {}
    all_ranges: list[tuple[float, float, str]] = []
    for raw_name, items in speakers_payload.items():
        if raw_name in {"enrollment_audio_reference", "speakers"}:
            continue
        if not isinstance(items, list):
            raise ValueError(f"voice enrollment for {raw_name} must be a list of ranges")
        name = str(raw_name).strip()
        if not name:
            raise ValueError("voice enrollment speaker names cannot be empty")
        speaker_ranges: list[tuple[float, float]] = []
        for item in items:
            start, end = _parse_enrollment_item(item)
            if reference == "original_file":
                start -= source_offset
                end -= source_offset
            if end <= start:
                raise ValueError(f"voice enrollment range for {name} has end <= start")
            if audio_duration is not None and (start < 0 or end > audio_duration):
                raise ValueError(
                    f"voice enrollment range for {name} ({start:.2f}-{end:.2f}) falls outside audio duration {audio_duration:.2f}s"
                )
            speaker_ranges.append((start, end))
            all_ranges.append((start, end, name))
        ranges[name] = speaker_ranges

    if len(ranges) < 2:
        raise ValueError("voice enrollment needs at least two speakers for closed-set identification")

    for name, speaker_ranges in ranges.items():
        ordered = sorted(speaker_ranges)
        for current, nxt in zip(ordered, ordered[1:]):
            if current[1] > nxt[0]:
                raise ValueError(f"voice enrollment ranges overlap within {name}")
    ordered_all = sorted(all_ranges)
    for current, nxt in zip(ordered_all, ordered_all[1:]):
        if current[1] > nxt[0]:
            raise ValueError(f"voice enrollment ranges overlap across {current[2]} and {nxt[2]}")

    return ranges, {
        "path": str(enrollment_path),
        "enrollment_audio_reference": reference,
        "source_offset": source_offset,
        "speakers": sorted(ranges),
    }


def _windows_from_ranges(
    data: Any,
    sample_rate: int,
    ranges: list[tuple[float, float]],
    *,
    window_seconds: float = 1.6,
    hop_seconds: float = 0.8,
) -> list[dict[str, Any]]:
    import numpy as np

    window_size = max(1, int(sample_rate * window_seconds))
    hop_size = max(1, int(sample_rate * hop_seconds))
    candidates: list[dict[str, Any]] = []
    for start_s, end_s in ranges:
        start_sample = max(0, int(start_s * sample_rate))
        end_sample = min(len(data), int(end_s * sample_rate))
        if end_sample <= start_sample:
            continue
        if end_sample - start_sample <= window_size:
            starts = [start_sample]
        else:
            starts = list(range(start_sample, end_sample - window_size + 1, hop_size))
            last_start = max(start_sample, end_sample - window_size)
            if starts[-1] != last_start:
                starts.append(last_start)
        for start in starts:
            end = min(end_sample, start + window_size)
            chunk = data[start:end]
            if len(chunk) < int(sample_rate * 0.45):
                continue
            rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
            candidates.append(
                {
                    "start": start / sample_rate,
                    "end": end / sample_rate,
                    "sample_start": start,
                    "sample_end": end,
                    "rms": rms,
                    "confidence": 0.5,
                }
            )
    if not candidates:
        return []
    rms_values = np.array([float(item["rms"]) for item in candidates], dtype="float32")
    threshold = max(float(np.percentile(rms_values, 40)) * 0.55, float(np.max(rms_values)) * 0.015, 1e-5)
    return [item for item in candidates if float(item["rms"]) >= threshold]


def _build_voice_profiles(
    audio_path: Path,
    enrollment_path: Path,
    *,
    source_offset: float = 0.0,
    speechbrain_cache: Path | None = None,
    speechbrain_model: str = SPEECHBRAIN_ECAPA_MODEL,
    minimum_enrolled_seconds: float = 3.0,
) -> tuple[list[str], Any, dict[str, Any], Any, int, Any]:
    import numpy as np

    sample_rate, data = _read_wav(audio_path)
    audio_duration = len(data) / sample_rate
    ranges, enrollment_meta = load_voice_enrollment(
        enrollment_path,
        source_offset=source_offset,
        audio_duration=audio_duration,
    )
    cache_dir = _speechbrain_cache(audio_path, speechbrain_cache)
    classifier = _load_speechbrain_classifier(speechbrain_model, cache_dir)
    names: list[str] = []
    profiles = []
    profile_status: dict[str, Any] = {}
    for name, speaker_ranges in ranges.items():
        windows = _windows_from_ranges(data, sample_rate, speaker_ranges)
        enrolled_seconds = sum(float(window["end"]) - float(window["start"]) for window in windows)
        if enrolled_seconds < minimum_enrolled_seconds:
            raise ValueError(
                f"voice enrollment for {name} has only {enrolled_seconds:.2f}s active speech after VAD; need at least {minimum_enrolled_seconds:.2f}s"
            )
        embeddings, valid_windows = _encode_windows_with_classifier(classifier, data, sample_rate, windows)
        if len(embeddings) == 0:
            raise ValueError(f"voice enrollment for {name} produced no embeddings")
        profile = _normalize_rows(np.mean(embeddings, axis=0, keepdims=True))[0]
        names.append(name)
        profiles.append(profile)
        profile_status[name] = {
            "ranges": len(speaker_ranges),
            "speech_windows": len(valid_windows),
            "active_speech_seconds": round(enrolled_seconds, 3),
        }
    return names, _normalize_rows(np.stack(profiles)), {
        **enrollment_meta,
        "profiles": profile_status,
        "model": speechbrain_model,
        "model_cache": str(cache_dir),
    }, classifier, sample_rate, data


def diarize_audio_voiceprint(
    audio_path: Path,
    *,
    enrollment_path: Path,
    source_offset: float = 0.0,
    window_seconds: float = 1.6,
    hop_seconds: float = 0.8,
    similarity_threshold: float = 0.4,
    similarity_margin: float = 0.06,
    speechbrain_cache: Path | None = None,
    speechbrain_model: str = SPEECHBRAIN_ECAPA_MODEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    names, profiles, enrollment_meta, classifier, sample_rate, data = _build_voice_profiles(
        audio_path,
        enrollment_path,
        source_offset=source_offset,
        speechbrain_cache=speechbrain_cache,
        speechbrain_model=speechbrain_model,
    )
    windows, vad_status = _speech_windows(
        data,
        sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    embeddings, windows = _encode_windows_with_classifier(classifier, data, sample_rate, windows)
    if len(embeddings) == 0:
        return [], {
            "engine": "speechbrain-ecapa-voiceprint",
            "status": "insufficient_embeddings",
            "enrollment": enrollment_meta,
        }
    scores = embeddings @ profiles.T
    best = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)[:, ::-1]
    margins = sorted_scores[:, 0] - sorted_scores[:, 1] if len(names) > 1 else np.ones(len(scores))
    labels = []
    unknown_label = -1
    for index, (label, score, margin) in enumerate(zip(best, sorted_scores[:, 0], margins)):
        if float(score) >= similarity_threshold and float(margin) >= similarity_margin:
            labels.append(int(label))
        else:
            labels.append(unknown_label)
        windows[index]["confidence"] = round(max(0.0, min(0.98, 0.5 + float(score) * 0.25 + float(margin) * 1.2)), 3)
        windows[index]["voice_scores"] = {
            name: round(float(value), 4)
            for name, value in sorted(zip(names, scores[index]), key=lambda item: float(item[1]), reverse=True)
        }
    labels = np.array(labels, dtype=int)
    accepted = int(np.sum(labels != unknown_label))
    label_names = {index: name for index, name in enumerate(names)}
    label_names[unknown_label] = "Speaker Unknown"
    turns = _merge_windows(windows, labels, label_names)
    for turn in turns:
        if turn["speaker"] != "Speaker Unknown":
            turn["name"] = turn["speaker"]
            turn["name_source"] = "voice_enrollment"
            turn["name_confidence"] = float(turn["confidence"])
    return turns, {
        "engine": "speechbrain-ecapa-voiceprint",
        "status": "ok",
        "speakers": names,
        "speech_windows": len(windows),
        "candidate_windows": vad_status["candidate_windows"],
        "accepted_windows": accepted,
        "unknown_windows": len(windows) - accepted,
        "turns": len(turns),
        "similarity_threshold": similarity_threshold,
        "similarity_margin": similarity_margin,
        "enrollment": enrollment_meta,
        "note": "Real names were assigned only from explicit voice enrollment ranges; low-similarity windows remain Speaker Unknown.",
    }


def diarize_audio_pyannote(audio_path: Path, *, expected_speakers: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return [], {
            "engine": "pyannote.audio",
            "status": "skipped",
            "reason": "HF_TOKEN/HUGGINGFACE_TOKEN is not set; keeping Speaker Unknown and OCR-based names only.",
        }

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return [], {
            "engine": "pyannote.audio",
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        kwargs: dict[str, Any] = {}
        if expected_speakers:
            kwargs["num_speakers"] = expected_speakers
        diarization = pipeline(str(audio_path), **kwargs)
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        return [], {
            "engine": "pyannote.audio",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    turns: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker).replace("_", " ").title(),
                "confidence": 0.75,
            }
        )

    return turns, {
        "engine": "pyannote.audio",
        "status": "ok",
        "turns": len(turns),
    }


def diarize_audio(
    audio_path: Path,
    *,
    expected_speakers: int | None = None,
    backend: str = "auto",
    enrollment_path: Path | None = None,
    source_offset: float = 0.0,
    speechbrain_cache: Path | None = None,
    similarity_threshold: float = 0.4,
    similarity_margin: float = 0.06,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if enrollment_path:
        return diarize_audio_voiceprint(
            audio_path,
            enrollment_path=enrollment_path,
            source_offset=source_offset,
            speechbrain_cache=speechbrain_cache,
            similarity_threshold=similarity_threshold,
            similarity_margin=similarity_margin,
        )
    if backend == "none":
        return [], {"engine": "none", "status": "skipped"}
    if backend == "local-cluster":
        if not expected_speakers or expected_speakers < 2:
            return [], {
                "engine": "local-mfcc-kmeans",
                "status": "failed",
                "reason": "expected_speakers >= 2 is required for anonymous clustering.",
            }
        return diarize_audio_local_cluster(audio_path, expected_speakers=expected_speakers)
    if backend == "speechbrain-cluster":
        if not expected_speakers or expected_speakers < 2:
            return [], {
                "engine": "speechbrain-ecapa-kmeans",
                "status": "failed",
                "reason": "expected_speakers >= 2 is required for anonymous clustering.",
            }
        return diarize_audio_speechbrain_cluster(
            audio_path,
            expected_speakers=expected_speakers,
            speechbrain_cache=speechbrain_cache,
        )
    if backend == "pyannote":
        return diarize_audio_pyannote(audio_path, expected_speakers=expected_speakers)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        turns, status = diarize_audio_pyannote(audio_path, expected_speakers=expected_speakers)
        if status.get("status") == "ok":
            return turns, status
    if expected_speakers and expected_speakers >= 2:
        turns, status = diarize_audio_speechbrain_cluster(
            audio_path,
            expected_speakers=expected_speakers,
            speechbrain_cache=speechbrain_cache,
        )
        if status.get("status") == "ok":
            return turns, status
        return diarize_audio_local_cluster(audio_path, expected_speakers=expected_speakers)
    return [], {
        "engine": "auto",
        "status": "skipped",
        "reason": "Set --expected-speakers 2 for local/SpeechBrain clustering, set --enrollment for voiceprint names, or set HF_TOKEN for pyannote.",
    }


def attach_speakers(segments: list[dict[str, Any]], turns: list[dict[str, Any]]) -> None:
    if not turns:
        return
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        segment_duration = max(0.1, end - start)
        speaker_scores: dict[str, dict[str, Any]] = {}
        total_speech_overlap = 0.0
        for turn in turns:
            overlap = max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))
            if overlap <= 0:
                continue
            total_speech_overlap += overlap
            speaker = str(turn["speaker"])
            bucket = speaker_scores.setdefault(
                speaker,
                {
                    "overlap": 0.0,
                    "confidence_sum": 0.0,
                    "name_confidence_sum": 0.0,
                    "turn": turn,
                },
            )
            bucket["overlap"] += overlap
            bucket["confidence_sum"] += overlap * float(turn.get("confidence", 0.0))
            bucket["name_confidence_sum"] += overlap * float(turn.get("name_confidence", turn.get("confidence", 0.0)))
            if overlap > max(0.0, min(end, float(bucket["turn"]["end"])) - max(start, float(bucket["turn"]["start"]))):
                bucket["turn"] = turn
        if speaker_scores and total_speech_overlap > 0:
            best_speaker, best = max(speaker_scores.items(), key=lambda item: float(item[1]["overlap"]))
            best_overlap = float(best["overlap"])
            best_turn = best["turn"]
            speech_share = best_overlap / max(0.1, total_speech_overlap)
            weighted_confidence = float(best["confidence_sum"]) / max(0.1, best_overlap)
            speaker_confidence = min(0.95, max(0.0, weighted_confidence * speech_share))
            segment["speaker"] = best_turn["speaker"]
            segment["speaker_confidence"] = round(speaker_confidence, 3)
            segment["speaker_speech_share"] = round(speech_share, 3)
            segment["speaker_speech_overlap_seconds"] = round(best_overlap, 3)
            segment["speaker_segment_coverage"] = round(best_overlap / segment_duration, 3)
            if best_turn.get("name"):
                segment["name"] = best_turn["name"]
                segment["name_source"] = best_turn.get("name_source", "voice_enrollment")
                weighted_name_confidence = float(best["name_confidence_sum"]) / max(0.1, best_overlap)
                segment["name_confidence"] = min(
                    weighted_name_confidence,
                    float(segment["speaker_confidence"]),
                )
