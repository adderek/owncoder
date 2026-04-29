"""Terminal color utilities."""
from __future__ import annotations


def _hex_to_ansi(hex_color: str) -> str:
    """Convert #RRGGBB to an ANSI 24-bit foreground escape sequence."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"
