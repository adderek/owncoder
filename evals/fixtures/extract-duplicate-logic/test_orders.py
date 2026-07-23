from orders import total_with_tax_us, total_with_tax_ca, total_with_tax


def test_us_rate():
    assert total_with_tax_us(10, 2) == 21.6


def test_ca_rate():
    assert total_with_tax_ca(10, 2) == 22.6


def test_shared_helper_exists_and_matches():
    assert total_with_tax(10, 2, 0.08) == 21.6
    assert total_with_tax(10, 2, 0.13) == 22.6
