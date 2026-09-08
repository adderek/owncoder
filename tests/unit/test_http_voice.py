"""Browser dictation — POST /api/voice into the shared SpeechIntake.

A browser cannot speak the notify relay protocol (that needs the e2e key), so
it posts recorded WAV over the already-authenticated HTTP channel. These tests
pin the Python wiring, and drive the browser's WAV encoder under node so the
container it produces is checked against the same decoder the server uses
(``soundfile``) rather than by grepping the source.
"""
import base64
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from agent.config.models import UIConfig
from agent.ui.http_loop import _HttpUI, _bind_server

ROOT = Path(__file__).resolve().parents[2]
HTTP_LOOP = (ROOT / "ui" / "http_loop.py").read_text(encoding="utf-8")
APP_JS = (ROOT / "ui" / "static" / "app.js").read_text(encoding="utf-8")

# The encoder is pure: slice it out of app.js and run it under node.
_WAV_JS = APP_JS[APP_JS.index("function wavHeader("):APP_JS.index("function b64(")]


def _node_works() -> bool:
    """node is present in most dev images but may be unusable (sandboxed V8
    cannot reserve its CodeRange) — skip rather than report a false failure."""
    exe = shutil.which("node")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-e", "0"], capture_output=True,
                              timeout=30).returncode == 0
    except Exception:
        return False


requires_node = pytest.mark.skipif(not _node_works(), reason="node unavailable")


class _Loop:
    def __init__(self):
        self.calls = []

    def call_soon_threadsafe(self, fn, *args):
        self.calls.append((fn, args))


class _Intake:
    def __init__(self):
        self.fed = []

    def feed(self, wire):
        self.fed.append(wire)


class _Server:
    def __init__(self, intake=None):
        self._intake = intake


def _ui(intake=None):
    loop = _Loop()
    return _HttpUI(_Server(intake), session=None, loop=loop), loop


class TestVoiceFeed:
    def test_disabled_speech_reports_why(self):
        ui, loop = _ui(intake=None)
        out = ui.voice_feed({"id": "u1", "data": "AAAA"})
        assert out["ok"] is False and "enabled" in out["msg"]
        assert loop.calls == []

    def test_missing_id_is_rejected(self):
        intake = _Intake()
        ui, _ = _ui(intake)
        out = ui.voice_feed({"data": "AAAA"})
        assert out["ok"] is False and "id" in out["msg"]
        assert intake.fed == []

    def test_frame_is_marshalled_to_the_loop_thread(self):
        """SpeechIntake.feed calls create_task, so it must not run on the
        handler thread — the loop owns transcription."""
        intake = _Intake()
        ui, loop = _ui(intake)
        out = ui.voice_feed({"id": "u1", "seq": 3, "last": True,
                             "fmt": "wav", "lang": "pl", "data": "AAAA"})
        assert out == {"ok": True}
        assert intake.fed == []            # nothing ran inline
        assert len(loop.calls) == 1
        fn, args = loop.calls[0]
        assert fn == intake.feed
        wire = args[0]
        assert wire["type"] == "voice" and wire["id"] == "u1"
        assert wire["seq"] == 3 and wire["last"] is True
        assert wire["lang"] == "pl" and wire["fmt"] == "wav"

    def test_answer_to_survives_the_round_trip(self):
        intake = _Intake()
        ui, loop = _ui(intake)
        ui.voice_feed({"id": "u1", "answer_to": "q-7", "data": "AAAA"})
        assert loop.calls[0][1][0]["answer_to"] == "q-7"

    def test_route_exists(self):
        assert 'elif self.path == "/api/voice":' in HTTP_LOOP
        assert "ui.voice_feed(payload)" in HTTP_LOOP


class TestPage:
    def test_the_composer_has_a_mic_and_a_language_picker(self):
        assert 'id="mic"' in HTTP_LOOP
        assert 'id="miclang"' in HTTP_LOOP
        assert '<option value="pl">pl</option>' in HTTP_LOOP
        assert '<option value="">auto</option>' in HTTP_LOOP

    def test_the_client_posts_to_the_endpoint(self):
        assert "'/api/voice'" in APP_JS


@requires_node
class TestWavEncoder:
    """The bytes the browser sends must decode with the server's decoder."""

    def _wav(self, samples: list[float], rate: int) -> bytes:
        js = (_WAV_JS
              + f"\nconst wav = toWav([new Float32Array({samples})], {rate});\n"
              + "process.stdout.write(Buffer.from(wav).toString('base64'));\n")
        out = subprocess.run(["node", "-e", js], capture_output=True,
                             text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        return base64.b64decode(out.stdout.strip())

    def test_it_is_a_riff_wave(self):
        data = self._wav([0, 0.5, -0.5], 16000)
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        assert data[12:16] == b"fmt " and data[36:40] == b"data"

    def test_the_declared_length_matches_the_payload(self):
        data = self._wav([0, 0.5, -0.5], 16000)
        assert struct.unpack("<I", data[40:44])[0] == 6      # 3 samples × 2B
        assert struct.unpack("<I", data[4:8])[0] == 36 + 6
        assert len(data) == 44 + 6

    def test_the_rate_is_carried_through(self):
        """Phones hand us 44.1/48 kHz, not 16 kHz — the header must say so."""
        data = self._wav([0.1, 0.2], 48000)
        assert struct.unpack("<I", data[24:28])[0] == 48000

    def test_soundfile_decodes_it_as_mono(self):
        """soundfile is what FasterWhisperSTT._decode uses."""
        sf = pytest.importorskip("soundfile")
        import io
        data, rate = sf.read(io.BytesIO(self._wav([0, 0.5, -0.5, 0.25], 16000)),
                             dtype="float32", always_2d=True)
        assert rate == 16000
        assert data.shape[1] == 1
        assert abs(float(data[1][0]) - 0.5) < 0.01


class TestTls:
    def test_config_has_the_knobs(self):
        cfg = UIConfig()
        assert cfg.http_tls_cert == "" and cfg.http_tls_key == ""

    def test_bind_server_accepts_a_cert_and_key(self):
        import inspect
        params = inspect.signature(_bind_server).parameters
        assert "tls_cert" in params and "tls_key" in params

    def test_server_startup_passes_them(self):
        assert "http_tls_cert" in HTTP_LOOP and "http_tls_key" in HTTP_LOOP
        assert "_bind_server(_make_handler(ui), host, port, tls_cert, tls_key)" in HTTP_LOOP

    def test_banner_shows_the_scheme(self):
        assert 'scheme = "https" if (tls_cert and tls_key) else "http"' in HTTP_LOOP
