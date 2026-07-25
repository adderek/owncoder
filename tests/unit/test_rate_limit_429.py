"""Daily-quota 429 detection (OpenRouter free-tier daily limit vs burst)."""
from agent.core.turn_errors import is_daily_quota_429, retry_after_seconds


class _E(Exception):
    def __init__(self, msg="", body=None, retry_after=None):
        super().__init__(msg)
        self.message = msg
        self.body = body
        if retry_after is not None:
            self.response = type("R", (), {"headers": {"retry-after": str(retry_after)}})()


def test_retry_after_parsing():
    assert retry_after_seconds(_E(retry_after=42)) == 42.0
    assert retry_after_seconds(_E()) == 0.0


def test_daily_by_message():
    assert is_daily_quota_429(_E("Rate limit exceeded: free-models-per-day"), 0)
    assert is_daily_quota_429(_E("You exceeded your daily quota"), 0)
    assert is_daily_quota_429(_E("limit is 50 requests per day"), 0)


def test_daily_by_body():
    e = _E("429", body={"error": {"message": "free tokens per day exhausted"}})
    assert is_daily_quota_429(e, 0)


def test_daily_by_long_retry_after():
    assert is_daily_quota_429(_E(retry_after=3600), retry_after_seconds(_E(retry_after=3600)))
    # 5 min or under is a burst limit, not daily
    assert not is_daily_quota_429(_E(retry_after=120), 120)


def test_burst_not_daily():
    assert not is_daily_quota_429(_E("Too many requests, slow down"), 0)
    assert not is_daily_quota_429(_E("rate limited"), 5)
