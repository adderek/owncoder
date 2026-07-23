from stats import running_totals


def test_running_totals():
    assert running_totals([1, 2, 3, 4]) == [1, 3, 6, 10]


def test_running_totals_single():
    assert running_totals([5]) == [5]
