# Same-Session Visual Voiceprint

This optional stage extends direct same-frame visual evidence to other speech segments in the same recording. It is designed for precision, not blanket coverage.

Enrollment eligibility:

1. The source frame must already have a direct same-frame active border plus either in-tile nameplate OCR or a reviewed visual-profile slot.
2. A person needs multiple samples separated in time and spanning a configured duration.
3. Time-ordered direct samples are split into separate profile-training, threshold-calibration, and final held-out sets. The held-out set is never used to choose the score threshold.
4. Eligibility is evaluated per person. A name needs the configured minimum number of accepted held-out trials, zero held-out false accepts for that name, and sufficient impostor trials.
5. A person who does not qualify remains direct-visual-only for that recording. Their profile still competes during scoring, so a segment whose highest score is ineligible is left unnamed rather than assigned to a lower-scoring eligible person.

The registry is not reusable across recordings. It does not name a participant from one short utterance, a diarization cluster, or a sidebar roster alone. A reviewed visual profile is allowed only when its active tile and participant mapping were calibrated against the recording.

Apply it after dynamic visual identity or reviewed visual-profile identity:

- Direct frame-level visual identities remain authoritative and are preserved.
- A name propagated only by the direct visual-cluster stage is not treated as final: a precision-gated voiceprint result can independently confirm it or replace it, and the prior cluster name remains in internal evidence.
- A rerun first retracts only prior `same_session_visual_voiceprint` labels. If the current visual artifact is unavailable or unreadable, ambiguous, mismatched to the recording, a configuration is invalid, or recalibration fails, the retraction and failure status are persisted so those labels cannot carry forward as stale voice evidence.
- Refreshing static or dynamic visual identity invalidates the existing same-session voice registry and report, so a later voiceprint pass must rebuild enrollment from the refreshed visual evidence.
- The command uses one verified visual artifact. When static and dynamic artifacts coexist, pass `--visual-identity-path`; its resolved recording path, duration, size, and content SHA-256 must match the current output metadata.

```bash
meeting-minutes visual-voice-identify \
  --output-dir /absolute/path/to/output \
  --visual-identity-path /absolute/path/to/output/visual_identity.json
```

Artifacts:

- `same_session_visual_voice_registry.json`: direct visual enrollment frames, training-only voice centroids, the calibration split, per-person held-out results, and eligible/ineligible profiles.
- `same_session_visual_voice_report.md`: enrolled participants, threshold provenance, eligible profiles, abstentions, and segment results.
- `transcript.json`: additional names have source `same_session_visual_voiceprint` and per-segment score evidence.
