from settings import CONFIG, get_timeout


def test_get_timeout():
    assert get_timeout(CONFIG) == 30
