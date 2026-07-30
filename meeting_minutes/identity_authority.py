"""Shared identity provenance and overwrite authority rules.

Identity stages run independently and may be rerun in a different order.  This
module is the single authority for which named evidence may be replaced by a
weaker downstream inference.
"""

from __future__ import annotations

from typing import Any

VOICE_ENROLLMENT_SOURCE = "voice_enrollment"
VOICE_REGISTRY_SOURCE = "voice_registry"
PARTICIPANT_MAP_SOURCE = "participant_map"
USER_CONFIRMED_SPEAKER_VOLUME_SOURCE = "user_confirmed_speaker_volume_mapping"
MANUAL_REVIEW_SOURCE = "manual_review"
HUMAN_REVIEW_SOURCE = "human_review"

REVIEWED_SLOT_SOURCE = "visual_profile_reviewed_slot"
DYNAMIC_NAMEPLATE_SOURCE = "dynamic_visual_in_tile_nameplate_ocr"
ACTIVE_SPEAKER_HIGHLIGHT_SOURCE = "visual_active_speaker_highlight"
AVATAR_TEMPLATE_SOURCE = "visual_avatar_template_match"
DIRECT_VISUAL_CLUSTER_SOURCE = "direct_visual_voice_cluster_consensus"
SAME_SESSION_VISUAL_VOICE_SOURCE = "same_session_visual_voiceprint"
ROSTER_AVATAR_SOURCE = "visual_roster_avatar_match"

HUMAN_REVIEWED_SOURCES = frozenset(
    {
        VOICE_ENROLLMENT_SOURCE,
        PARTICIPANT_MAP_SOURCE,
        USER_CONFIRMED_SPEAKER_VOLUME_SOURCE,
        MANUAL_REVIEW_SOURCE,
        HUMAN_REVIEW_SOURCE,
    }
)
HUMAN_REVIEWED_SOURCE_PREFIXES = ("reviewed_", "manual_", "user_confirmed_")
DIRECT_VISUAL_SOURCES = frozenset(
    {
        REVIEWED_SLOT_SOURCE,
        DYNAMIC_NAMEPLATE_SOURCE,
        ACTIVE_SPEAKER_HIGHLIGHT_SOURCE,
        AVATAR_TEMPLATE_SOURCE,
        DIRECT_VISUAL_CLUSTER_SOURCE,
        SAME_SESSION_VISUAL_VOICE_SOURCE,
    }
)


def name_source(segment: dict[str, Any]) -> str:
    """Return a normalized provenance source from a transcript segment."""

    return str(segment.get("name_source") or "")


def has_named_identity(segment: dict[str, Any]) -> bool:
    return bool(segment.get("name"))


def has_human_reviewed_identity(segment: dict[str, Any]) -> bool:
    """Return whether a name is supplied or explicitly reviewed by a person."""

    source = name_source(segment)
    return has_named_identity(segment) and (
        source in HUMAN_REVIEWED_SOURCES or source.startswith(HUMAN_REVIEWED_SOURCE_PREFIXES)
    )


def has_operator_locked_identity(segment: dict[str, Any]) -> bool:
    """Return whether a later participant-map command must not replace the name."""

    return has_human_reviewed_identity(segment) and name_source(segment) != PARTICIPANT_MAP_SOURCE


def has_roster_avatar_identity(segment: dict[str, Any]) -> bool:
    """Return whether calibrated same-frame roster-avatar evidence is present."""

    return has_named_identity(segment) and name_source(segment) == ROSTER_AVATAR_SOURCE


def is_protected_from_ocr_overwrite(segment: dict[str, Any]) -> bool:
    """Return whether generic nearby OCR must not change the existing name."""

    source = name_source(segment)
    return has_named_identity(segment) and (
        has_human_reviewed_identity(segment)
        or source == VOICE_REGISTRY_SOURCE
        or source == ROSTER_AVATAR_SOURCE
        or source in DIRECT_VISUAL_SOURCES
    )


def is_protected_from_voice_registry_overwrite(segment: dict[str, Any]) -> bool:
    """Return whether cross-recording voice inference must preserve a name."""

    return is_protected_from_ocr_overwrite(segment)
