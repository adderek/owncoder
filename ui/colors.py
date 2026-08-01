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


# GNU readline measures the prompt to place the cursor and to redraw the line.
# It cannot tell a colour escape from a printable character unless the escape is
# bracketed by these markers, so an unbracketed prompt miscounts its own width —
# and on redraw the terminal shows the raw "[38;2;56;142;60m>" instead of a
# green ">".
_RL_IGNORE_START = "\001"
_RL_IGNORE_END = "\002"


def readline_prompt(escape: str, text: str, reset: str = "\033[0m") -> str:
    """*text* coloured by *escape*, safe to hand to ``input()`` under readline."""
    if not escape:
        return text
    return (f"{_RL_IGNORE_START}{escape}{_RL_IGNORE_END}{text}"
            f"{_RL_IGNORE_START}{reset}{_RL_IGNORE_END}")
