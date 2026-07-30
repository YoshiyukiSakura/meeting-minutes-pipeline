from meeting_minutes.sample_evidence import unique_frames_for_requests


def test_unique_frames_for_requests_deduplicates_collapsed_video_times():
    frame = {"time": 4.2, "path": "/tmp/frame.jpg"}

    frames = unique_frames_for_requests(
        [
            {"video_time": 4.2},
            {"video_time": 4.2001},
            {"video_time": 4.2},
        ],
        {4.2: frame},
    )

    assert frames == [frame]
