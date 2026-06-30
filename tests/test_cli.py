from pathlib import Path
from types import SimpleNamespace

import pytest

from meeting_minutes.cli import write_voice_template
from meeting_minutes.jsonio import read_json


def test_voice_template_uses_generic_speaker_count(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, names=None, speaker_count=3)
    assert write_voice_template(args) == 0

    template = read_json(tmp_path / "voice_enrollment.template.json")
    assert list(template["speakers"]) == ["Speaker 1", "Speaker 2", "Speaker 3"]

    guide = (tmp_path / "voice_enrollment_guide.md").read_text(encoding="utf-8")
    assert '"Speaker 3"' in guide
    assert "Billy" not in guide
    assert "Xin" not in guide


def test_voice_template_accepts_multiple_known_names(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, names=["Billy", "Xin", "Alice"], speaker_count=2)
    assert write_voice_template(args) == 0

    template = read_json(tmp_path / "voice_enrollment.template.json")
    assert list(template["speakers"]) == ["Billy", "Xin", "Alice"]


def test_voice_template_requires_names_or_speaker_count(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, names=None, speaker_count=0)
    with pytest.raises(ValueError, match="speaker-count"):
        write_voice_template(args)
