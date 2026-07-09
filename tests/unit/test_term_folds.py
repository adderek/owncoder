"""OSC-777 tilix fold escapes (ui/term_folds.py)."""
import io
import sys

from agent.ui import term_folds


def _capture(fn, *a, **kw):
    old = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        fn(*a, **kw)
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_fold_start_format():
    out = _capture(term_folds.fold_start, "turn-1", title="ls /tmp", group="agent-rounds")
    assert out == "\033]777;tilix-fold-start;id=turn-1;title=ls /tmp;group=agent-rounds\007"


def test_fold_end_format_and_status_whitelist():
    out = _capture(term_folds.fold_end, "turn-1", summary="exit=0 0.3s", status="success")
    assert out == "\033]777;tilix-fold-end;id=turn-1;summary=exit=0 0.3s;status=success\007"
    out = _capture(term_folds.fold_end, "turn-1", summary="x", status="bogus")
    assert "status=" not in out


def test_sanitizes_separators_and_control_chars():
    out = _capture(term_folds.fold_start, "id;1", title="a;b\x1b[31m\ncd")
    # ';' and control chars must not survive inside values
    assert out.startswith("\033]777;tilix-fold-start;id=id 1;")
    body = out[len("\033]777;"):-1]
    assert "\x1b" not in body and "\n" not in body
    assert "title=a b [31m cd" in out


def test_title_truncated():
    out = _capture(term_folds.fold_start, "t", title="x" * 500)
    assert "x" * 120 in out and "x" * 121 not in out
