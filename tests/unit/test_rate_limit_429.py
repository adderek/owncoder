"""Daily-quota 429 detection (OpenRouter free-tier daily limit vs burst)."""
from agent.core.turn import _is_daily_quota_429, _retry_after_seconds


class _E(Exception):
    def __init__(self, msg="", body=None, retry_after=None):
        super().__init__(msg)
        self.message = msg
        self.body = body
        if retry_after is not None:
            self.response = type("R", (), {"headers": {"retry-after": str(retry_after)}})()


def test_retry_after_parsing():
    assert _retry_after_seconds(_E(retry_after=42)) == 42.0
    assert _retry_after_seconds(_E()) == 0.0


def test_daily_by_message():
    assert _is_daily_quota_429(_E("Rate limit exceeded: free-models-per-day"), 0)
    assert _is_daily_quota_429(_E("You exceeded your daily quota"), 0)
    assert _is_daily_quota_429(_E("limit is 50 requests per day"), 0)


def test_daily_by_body():
    e = _E("429", body={"error": {"message": "free tokens per day exhausted"}})
    assert _is_daily_quota_429(e, 0)


def test_daily_by_long_retry_after():
    assert _is_daily_quota_429(_E(retry_after=3600), _retry_after_seconds(_E(retry_after=3600)))
    # 5 min or under is a burst limit, not daily
    assert not _is_daily_quota_429(_E(retry_after=120), 120)


def test_burst_not_daily():
    assert not _is_daily_quota_429(_E("Too many requests, slow down"), 0)
    assert not _is_daily_quota_429(_E("rate limited"), 5)
