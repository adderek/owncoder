"""Terminal color utilities."""
from __future__ import annotations


def _hex_to_ansi(hex_color: str) -> str:
    """Convert #RRGGBB to an ANSI 24-bit foreground escape sequence.

    Theme color fields accept any Rich color string (named colors like "green",
    short "#fff", etc.), so a user-configured non-6-hex value must not crash the
    readline prompt — fall back to no color escape (default terminal color).
    """
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return ""
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return ""
    return f"\033[38;2;{r};{g};{b}m"
