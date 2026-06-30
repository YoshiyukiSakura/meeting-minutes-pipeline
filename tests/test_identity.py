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
            "speaker": "Billy",
            "name": "Billy",
            "name_source": "voice_enrollment",
            "name_confidence": 0.8,
            "text": "hello",
        }
    ]
    attach_names(segments, [], {"Billy": "Xin"})
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "voice_enrollment"


def test_user_confirmed_name_has_priority_over_participant_map():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "Billy",
            "name": "Billy",
            "name_source": "user_confirmed_speaker_volume_mapping",
            "name_confidence": 0.8,
            "text": "hello",
        }
    ]
    attach_names(segments, [], {"Billy": "Xin"})
    assert segments[0]["name"] == "Billy"
    assert segments[0]["name_source"] == "user_confirmed_speaker_volume_mapping"
