from meeting_minutes.identity import attach_names, extract_candidate_names


def test_extract_candidate_names_chinese_and_english():
    text = "录制\n张三\nAlice Chen\n共享屏幕\n10:23"
    assert extract_candidate_names(text) == ["张三", "Alice Chen"]


def test_extract_candidate_names_filters_long_ui_text():
    text = "Participants\nhttps://example.com\n这是一个非常长的会议界面提示文本"
    assert extract_candidate_names(text) == []


def test_voice_enrollment_name_has_priority_over_participant_map():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Alice",
            "name": "Alice",
            "name_source": "voice_enrollment",
            "name_confidence": 0.8,
            "text": "hello",
        }
    ]
    attach_names(segments, [], {"Alice": "Bob"})
    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_source"] == "voice_enrollment"


def test_user_confirmed_name_has_priority_over_participant_map():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Alice",
            "name": "Alice",
            "name_source": "user_confirmed_speaker_volume_mapping",
            "name_confidence": 0.8,
            "text": "hello",
        }
    ]
    attach_names(segments, [], {"Alice": "Bob"})
    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_source"] == "user_confirmed_speaker_volume_mapping"


def test_voice_registry_name_has_priority_over_ocr_candidates():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Alice",
            "name_source": "voice_registry",
            "name_confidence": 0.8,
            "text": "hello",
        }
    ]

    attach_names(segments, [{"time": 1.0, "text": "Bob Smith", "path": "/tmp/frame.jpg"}])

    assert segments[0]["name"] == "Alice"
    assert segments[0]["name_source"] == "voice_registry"


def test_calibrated_roster_avatar_identity_has_priority_over_ocr_candidates():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": "visual_roster_avatar_match",
            "name_confidence": 0.81,
            "text": "hello",
        }
    ]

    attach_names(
        segments,
        [{"time": 1.0, "text": "Xin Chen", "path": "/tmp/frame.jpg"}],
        allow_ocr_names=True,
    )

    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "visual_roster_avatar_match"


def test_participant_map_identity_has_priority_over_ocr_candidates():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Billy",
            "name_source": "participant_map",
            "name_confidence": 0.95,
            "text": "hello",
        }
    ]

    attach_names(
        segments,
        [{"time": 1.0, "text": "Xin Chen", "path": "/tmp/frame.jpg"}],
        allow_ocr_names=True,
    )

    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "participant_map"


def test_participant_map_overrides_voice_registry_name():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Speaker 1",
            "name": "Alice",
            "name_source": "voice_registry",
            "name_confidence": 0.8,
            "text": "hello",
        }
    ]

    attach_names(segments, [], {"Speaker 1": "Bob"})

    assert segments[0]["name"] == "Bob"
    assert segments[0]["name_source"] == "participant_map"
