"""The UI can be installed on a phone.

A LAN agent is mostly reached from a phone, and every visit opened inside
browser chrome: no manifest, no touch icon, no theme colour.
"""
import json
import struct
import zlib
from pathlib import Path

import pytest

from agent.ui.http_loop import _MANIFEST, _PAGE, _icon_png

HEAD = _PAGE[:_PAGE.index("</head>")]
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class TestManifest:
    def test_it_is_valid_json_with_what_a_launcher_needs(self):
        m = json.loads(_MANIFEST)
        assert m["name"] and m["start_url"] == "/"
        assert m["display"] == "standalone"
        assert m["icons"]

    def test_the_icons_it_names_are_served(self):
        for icon in json.loads(_MANIFEST)["icons"]:
            assert icon["src"] in ("/icon-192.png", "/icon-512.png")
        assert 'self.path in ("/icon-192.png", "/icon-512.png")' in HTTP_LOOP

    def test_one_icon_is_maskable(self):
        """Android crops a non-maskable icon into whatever shape it likes."""
        purposes = [i.get("purpose", "") for i in json.loads(_MANIFEST)["icons"]]
        assert any("maskable" in p for p in purposes)

    def test_the_colours_match_the_ui(self):
        m = json.loads(_MANIFEST)
        assert m["background_color"] == "#16181d"      # --bg
        assert m["theme_color"] == "#1e2128"           # --panel


class TestPage:
    def test_the_page_links_it(self):
        assert '<link rel="manifest" href="/manifest.webmanifest">' in HEAD

    def test_ios_gets_a_png_icon(self):
        """iOS ignores the manifest and the SVG favicon both."""
        assert '<link rel="apple-touch-icon" href="/icon-192.png">' in HEAD

    def test_the_browser_chrome_is_themed(self):
        assert '<meta name="theme-color" content="#1e2128">' in HEAD


class TestIconRaster:
    @pytest.mark.parametrize("size", [192, 512])
    def test_it_is_a_real_png_of_that_size(self, size):
        data = _icon_png(size)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", data[16:24])
        assert (w, h) == (size, size)

    def test_the_corners_are_transparent_and_the_middle_is_not(self):
        """A square icon in a round launcher slot looks broken."""
        size = 64
        data = _png_pixels(_icon_png(size), size)
        assert data[_px(0, 0, size)][3] == 0            # corner cut away
        assert data[_px(size // 2, size // 2, size)][3] == 255

    def test_the_dot_is_the_accent_colour(self):
        size = 64
        data = _png_pixels(_icon_png(size), size)
        assert data[_px(size // 2, size // 2, size)][:3] == (0x6a, 0xa6, 0xff)

    def test_rasterising_is_done_once(self):
        assert _icon_png(192) is _icon_png(192)


def _px(x, y, size):
    return y * size + x


def _png_pixels(data, size):
    """Decode our own PNG back to RGBA tuples (filter 0 on every row)."""
    idat = b""
    i = 8
    while i < len(data):
        ln = struct.unpack(">I", data[i:i + 4])[0]
        tag = data[i + 4:i + 8]
        if tag == b"IDAT":
            idat += data[i + 8:i + 8 + ln]
        i += 12 + ln
    raw = zlib.decompress(idat)
    out = []
    stride = size * 4 + 1
    for y in range(size):
        row = raw[y * stride + 1:(y + 1) * stride]
        for x in range(size):
            out.append(tuple(row[x * 4:x * 4 + 4]))
    return out
