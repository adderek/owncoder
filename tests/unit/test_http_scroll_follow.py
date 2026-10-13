"""Auto-follow keeps the final answer in view.

stickScroll() used to ask "near the bottom?" after the append. A whole answer
bubble grows the log far past the 80px slack, so the check read "user
scrolled away" and left the answer below the fold until "scroll to end".
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
STICK = APP_JS[APP_JS.index("function stickScroll()"):APP_JS.index("function toBottom()")]


def test_follow_is_decided_before_the_append():
    assert "if (following)" in STICK
    assert "atEnd()" not in STICK


def test_only_scrolling_changes_follow():
    i = APP_JS.index("log.addEventListener('scroll'")
    assert "following = end;" in APP_JS[i:i + 200]


def test_follow_scroll_is_instant():
    """Smooth scroll's mid-animation events would read as scrolled away."""
    assert "behavior: 'instant'" in STICK


def test_scroll_to_end_resumes_follow():
    i = APP_JS.index("function toBottom()")
    assert "following = true;" in APP_JS[i:i + 120]
