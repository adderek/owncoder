from basket import add_item


def test_independent_calls_do_not_share_state():
    a = add_item("apple")
    b = add_item("banana")
    assert a == ["apple"]
    assert b == ["banana"]
