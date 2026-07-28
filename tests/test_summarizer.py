from meeting_minutes.summarizer import _segment_chunks, build_minutes_prompt


def _segment(index: int) -> dict[str, object]:
    return {
        "start": float(index * 10),
        "end": float(index * 10 + 9),
        "speaker": f"Speaker {index}",
        "text": f"segment-{index} " + "details " * 10,
    }


def test_segment_chunks_preserve_every_segment_in_order():
    segments = [_segment(index) for index in range(6)]

    chunks = _segment_chunks(segments, max_chars=180)

    assert len(chunks) > 1
    assert [item["text"] for chunk in chunks for item in chunk] == [item["text"] for item in segments]


def test_minutes_prompt_uses_fixed_topic_structure_for_each_chunk():
    prompt = build_minutes_prompt(
        segments=[_segment(1)],
        keyframes=[],
        metadata={"input": "/recording.mov", "duration": 120.0},
        chunk_index=2,
        chunk_count=4,
    )

    assert "第 2/4 个连续时间片段" in prompt
    assert "### 简洁议题名称（开始时间-结束时间）" in prompt
    assert "不得输出一级或二级标题、行动项、负责人" in prompt
