from parser import parse_int


def test_parses_valid_int():
    assert parse_int("42") == 42


def test_returns_none_on_garbage():
    assert parse_int("not a number") is None
