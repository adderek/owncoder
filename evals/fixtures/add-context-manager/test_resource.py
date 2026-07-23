from resource import Resource


def test_with_block_acquires_and_releases():
    r = Resource()
    assert r.open is False
    with r as ctx:
        assert r.open is True
    assert r.open is False


def test_releases_even_on_exception():
    r = Resource()
    try:
        with r:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert r.open is False
