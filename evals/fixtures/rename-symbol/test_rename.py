from module_a import process
from module_b import run


def test_process():
    assert process(3) == 6


def test_run_uses_process():
    assert run(3) == 7
