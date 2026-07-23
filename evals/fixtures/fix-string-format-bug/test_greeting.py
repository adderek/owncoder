from greeting import format_greeting


def test_format_greeting():
    assert format_greeting("Ann", 3) == "Hello Ann, you have 3 new messages"
