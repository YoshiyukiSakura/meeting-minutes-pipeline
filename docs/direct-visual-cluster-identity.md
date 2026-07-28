# Direct Visual Cluster Identity

This optional same-recording stage extends direct active-speaker visual evidence through a diarization cluster only when the cluster is independently validated. It is intended for recordings where the active-speaker border is visible but the visual layout changes during screen sharing.

The stage never derives a name from a roster, an avatar position, or a diarization label alone.

## Evidence Gate

1. A frame must already contain a direct active-speaker cue and a reviewed visual-profile slot or an in-tile nameplate OCR result.
2. Frames are assigned to a diarization turn only after a fixed boundary erosion interval. Frames near a turn change are discarded.
3. Each turn produces at most one name vote after its internal frame votes agree. Repeated frames in a long turn do not increase support.
4. The turn votes are split chronologically. The early block must identify and validate the late block, and the late block must independently identify and validate the early block.
5. The candidate must remain stable in all four chronological quartiles and across the fixed boundary-erosion stability sweep.
6. Each accepted name may map to exactly one diarization cluster. A collision rejects every cluster for that name.
7. Propagation applies only to unnamed, high-confidence diarized segments inside contiguous direct-visual support intervals. A gap above 120 seconds is not bridged, even when the cluster remains accepted on both sides.

Mixed clusters, `Speaker Unknown`, low-confidence diarized segments, and segments outside direct visual support intervals remain unnamed.

## Reruns and Cross-Checks

- A rerun first removes only names previously propagated by this stage, then recalculates them from the current evidence and settings. This retraction also happens when the rerun must skip because direct visual evidence is missing, ambiguous, or belongs to another recording.
- Refreshing either static or dynamic visual identity explicitly invalidates the prior cluster artifact and stage status before a new cluster pass. A standalone cluster revalidation also persists its retraction before surfacing a configuration or artifact failure.
- The main pipeline passes the exact visual artifact it created in the current run and binds it to the resolved recording path, duration, size, and content SHA-256. The standalone command refuses to guess when both static and dynamic visual artifacts are present; pass `--visual-identity-path` explicitly.
- Propagation requires explicit support intervals in the current artifact. Legacy full-recording bounds are not used as a fallback.
- A later same-session voiceprint stage preserves direct visual names but may independently confirm or correct a cluster-propagated name after its held-out precision gate passes. The correction is recorded in internal transcript evidence.

## Command

```bash
meeting-minutes visual-cluster-identify \
  --output-dir /absolute/path/to/output \
  --visual-identity-path /absolute/path/to/output/visual_identity.json
```

## Artifacts

- `direct_visual_cluster_identity.json`: turn-level evidence, bidirectional validation results, quartile and erosion stability checks, accepted/rejected clusters, and contiguous propagation intervals.
- `direct_visual_cluster_identity_report.md`: concise internal audit report.
- `transcript.json`: propagated names have source `direct_visual_voice_cluster_consensus` and include cluster-validation evidence.
