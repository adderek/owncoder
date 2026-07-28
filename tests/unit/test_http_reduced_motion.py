"""A request for no motion is honoured by everything that moves.

The reduce block exempted .row and the drawers; the ask box, retry bar, slash
palette and @-file list all slid in on a translate regardless.
"""
import re
from pathlib import Path

APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
FIRST_REDUCE = APP_CSS.index("@media (prefers-reduced-motion: reduce)")


class TestFade:
    def test_the_default_fade_moves(self):
        """The override only matters because the default translates."""
        i = APP_CSS.index("@keyframes fadein")
        assert "translateY" in APP_CSS[i:i + 160]

    def test_the_override_lands_after_it(self):
        """Same specificity — the later definition has to be the reduce one."""
        override = APP_CSS.index("@keyframes fadein", FIRST_REDUCE)
        assert override > FIRST_REDUCE

    def test_the_override_removes_the_movement_but_keeps_the_fade(self):
        override = APP_CSS.index("@keyframes fadein", FIRST_REDUCE)
        block = APP_CSS[override:APP_CSS.index("}", override) + 1]
        assert "translateY" not in block
        assert "opacity: 0" in block

    def test_it_covers_every_fading_element_at_once(self):
        """Naming them one by one would miss the next one added."""
        faders = re.findall(r"animation: fadein", APP_CSS)
        assert len(faders) >= 4


class TestIndicators:
    def test_the_looping_indicators_are_silenced(self):
        block = APP_CSS[APP_CSS.rindex("@media (prefers-reduced-motion: reduce)"):]
        for sel in (".wspin", ".streaming::after", "details.tool .mark.pend",
                    "#dot.busy"):
            assert sel in block, sel

    def test_the_activity_bar_still_reads_as_a_state(self):
        """Silencing it must not make a stalled backend look like an idle one."""
        block = APP_CSS[APP_CSS.rindex("@media (prefers-reduced-motion: reduce)"):]
        i = block.index("body[data-activity]")
        assert "background-color: var(--act)" in block[i:i + 300]


class TestStreamingBubble:
    def test_nothing_animates_inside_it(self):
        """It re-renders every 120ms — an animation there restarts 8x a second."""
        rules = re.findall(r"\.streaming[^{]*\{[^}]*\}", APP_CSS)
        inner = [r for r in rules if re.search(r"\.streaming\s+\.?\w", r)
                 and "animation:" in r]
        assert inner == []
