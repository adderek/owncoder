import pytest
from account import withdraw


def test_normal_withdraw():
    assert withdraw(100, 30) == 70


def test_negative_amount_raises():
    with pytest.raises(ValueError):
        withdraw(100, -10)


def test_amount_exceeding_balance_raises():
    with pytest.raises(ValueError):
        withdraw(100, 150)
