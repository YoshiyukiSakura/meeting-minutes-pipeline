from meeting_minutes.time_utils import format_ts, parse_ts


def test_format_ts():
    assert format_ts(83) == "01:23"
    assert format_ts(3661) == "01:01:01"


def test_parse_ts():
    assert parse_ts("01:23") == 83
    assert parse_ts("01:01:01") == 3661

