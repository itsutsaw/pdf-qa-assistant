from datetime import date

from usage import DailyLimit


def test_daily_limit_stops_at_limit_per_kind():
    limit = DailyLimit({"pdfs": 2, "questions": 1}, today=lambda: date(2026, 11, 3))
    assert [limit.take("pdfs") for _ in range(3)] == [True, True, False]
    assert limit.take("questions") is True
    assert limit.take("questions") is False


def test_daily_limit_resets_on_a_new_day():
    day = [date(2026, 11, 3)]
    limit = DailyLimit({"questions": 1}, today=lambda: day[0])
    assert limit.take("questions") is True
    assert limit.take("questions") is False
    day[0] = date(2026, 11, 4)
    assert limit.take("questions") is True
