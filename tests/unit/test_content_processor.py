"""Unit tests for Layer 3: Content Processor."""
from __future__ import annotations

import pytest
from agent.tools.web_search import content_processor


class TestBinaryDetection:
    def test_null_bytes_rejected(self):
        data = b"\x00" * 100 + b"some text"
        result = content_processor.process(data)
        assert result["binary_rejected"]

    def test_high_null_ratio_rejected(self):
        data = b"ab" + b"\x00" * 10 + b"cd"  # >5% nulls
        result = content_processor.process(data)
        assert result["binary_rejected"]

    def test_few_nulls_passed(self):
        """A single null byte (<5%) should not trigger binary rejection."""
        data = b"hello world" + b"\x00" + b"more text here that makes this long enough to dilute the null ratio"
        result = content_processor.process(data)
        assert not result["binary_rejected"]

    def test_plain_text_passes(self):
        result = content_processor.process(b"Hello World")
        assert not result["binary_rejected"]
        assert "Hello World" in result["text"]

    def test_empty_body(self):
        result = content_processor.process(b"")
        assert not result["binary_rejected"]
        assert result["text"] == ""


class TestHTMLStripping:
    def test_script_tags_stripped(self):
        html = b"<html><body><p>Hello</p><script>alert('xss')</script></body></html>"
        result = content_processor.process(html, content_type="text/html")
        assert "Hello" in result["text"]
        assert "alert" not in result["text"]

    def test_style_tags_stripped(self):
        html = b"<html><head><style>.evil{color:red}</style></head><body>Text</body></html>"
        result = content_processor.process(html, content_type="text/html")
        assert "Text" in result["text"]
        assert ".evil" not in result["text"]

    def test_iframe_stripped(self):
        html = b"<html><body><p>Good</p><iframe src='evil'></iframe></body></html>"
        result = content_processor.process(html, content_type="text/html")
        assert "Good" in result["text"]
        assert "evil" not in result["text"]

    def test_html_entities_decoded(self):
        html = b"<html><body>&amp; &lt; &gt; &quot; &#39;</body></html>"
        result = content_processor.process(html, content_type="text/html")
        assert "&" in result["text"]
        assert "<" in result["text"]
        assert ">" in result["text"]

    def test_title_extracted(self):
        html = b"<html><head><title>My Page</title></head><body>Content</body></html>"
        result = content_processor.process(html, content_type="text/html")
        assert "My Page" in result["text"]
        assert "Content" in result["text"]


class TestCharsetHandling:
    def test_respects_content_type_header(self):
        data = "café".encode("latin-1")
        result = content_processor.process(data, content_type="text/html; charset=iso-8859-1")
        assert "café" in result["text"]

    def test_detects_meta_charset(self):
        html = b'<html><head><meta charset="utf-8"></head><body>\xc3\xa9</body></html>'
        result = content_processor.process(html)
        assert "é" in result["text"]

    def test_fallback_to_utf8(self):
        data = "hello".encode("utf-8")
        result = content_processor.process(data)
        assert "hello" in result["text"]
        assert result["charset_used"] == "utf-8"

    def test_invalid_utf8_replaced(self):
        data = b"valid text \xff\xfe invalid"
        result = content_processor.process(data)
        assert "valid text" in result["text"]
        # U+FFFD replacement character should be present
        assert "�" in result["text"]


class TestSizeCapping:
    def test_content_truncated_at_max(self):
        big = b"a" * 40000
        result = content_processor.process(big, max_chars=100)
        assert result["truncated"]
        assert len(result["text"]) == 100

    def test_content_not_truncated_when_small(self):
        small = b"short text"
        result = content_processor.process(small, max_chars=32000)
        assert not result["truncated"]


class TestHashing:
    def test_hash_consistent(self):
        r1 = content_processor.process(b"hello")
        r2 = content_processor.process(b"hello")
        assert r1["hash"] == r2["hash"]

    def test_hash_different_for_different_content(self):
        r1 = content_processor.process(b"hello")
        r2 = content_processor.process(b"world")
        assert r1["hash"] != r2["hash"]

    def test_hash_is_hex_string(self):
        result = content_processor.process(b"test")
        assert len(result["hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["hash"])
