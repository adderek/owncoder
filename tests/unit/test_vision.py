"""Images reach the model as pixels — but only when the model can see them.

The marker (`[image: path]`) lives in the message text for the whole internal
life of a turn; only the wire copy carries content blocks. These tests pin the
two halves of that: what gets expanded, and what must never be.
"""
import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.config.models import Config
from agent.core import vision
from agent.core.turn_setup import normalize_api_messages

# 1x1 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.llm.model = "qwen2-vl-7b-instruct"    # hinted vision
    return c


@pytest.fixture()
def shot(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(PNG)
    return p


def user(text):
    return [{"role": "user", "content": text}]


APP_JS = Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"


def _attach_ref(resp: dict, name: str) -> dict:
    """Run app.js's attachRef under node on *resp* and return its result.

    Only that function is extracted from app.js — the rest of the file needs a
    DOM. node is optional here (it is not an agent runtime dependency).
    """
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    src = APP_JS.read_text(encoding="utf-8")
    fn = src[src.index("// ATTACHREF_START"):src.index("// ATTACHREF_END")]
    script = (fn + "\nprocess.stdout.write(JSON.stringify("
                   "attachRef(JSON.parse(process.argv[1]), process.argv[2])));")
    proc = subprocess.run(["node", "-e", script, json.dumps(resp), name],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class TestCapability:
    def test_a_vision_model_id_is_recognised(self):
        for mid in ("qwen2-vl-7b", "llava-1.6", "gpt-4o-mini", "gemma-3-12b-it"):
            assert vision.hints_vision(mid), mid

    def test_a_text_model_id_is_not(self):
        for mid in ("qwen3-coder-30b", "deepseek-r1", "nomic-embed-text"):
            assert not vision.hints_vision(mid), mid

    def test_explicit_on_beats_the_hint(self, cfg):
        cfg.llm.model = "qwen3-coder-30b"
        cfg.llm.vision = "on"
        assert vision.supports_images(cfg)

    def test_explicit_off_beats_the_hint(self, cfg):
        cfg.llm.vision = "off"
        assert not vision.supports_images(cfg)

    def test_the_section_switch_disables_everything(self, cfg):
        cfg.vision.enabled = False
        assert not vision.supports_images(cfg)


class TestExpansion:
    def test_a_marker_becomes_an_image_block(self, cfg, shot):
        out = vision.expand_image_markers(user("look: [image: shot.png]"), cfg)
        parts = out[0]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_the_text_survives_next_to_the_image(self, cfg, shot):
        out = vision.expand_image_markers(user("what is this? [image: shot.png]"), cfg)
        assert "what is this?" in out[0]["content"][0]["text"]

    def test_a_text_only_model_keeps_the_path(self, cfg, shot):
        cfg.llm.vision = "off"
        msgs = user("look: [image: shot.png]")
        assert vision.expand_image_markers(msgs, cfg) == msgs

    def test_nothing_is_touched_without_a_marker(self, cfg):
        msgs = user("plain question")
        assert vision.expand_image_markers(msgs, cfg) is msgs

    def test_only_user_messages_are_expanded(self, cfg, shot):
        msgs = [{"role": "assistant", "content": "see [image: shot.png]"}]
        assert vision.expand_image_markers(msgs, cfg) == msgs

    def test_history_older_than_keep_last_falls_back_to_text(self, cfg, shot):
        cfg.vision.keep_last_turns = 1
        msgs = user("first [image: shot.png]") + [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second [image: shot.png]"},
        ]
        out = vision.expand_image_markers(msgs, cfg)
        assert isinstance(out[0]["content"], str)      # old turn: marker only
        assert isinstance(out[2]["content"], list)     # newest turn: pixels

    def test_the_per_request_image_cap_holds(self, cfg, shot):
        cfg.vision.max_images = 1
        out = vision.expand_image_markers(
            user("[image: shot.png] [image: shot.png]"), cfg)
        blocks = [p for p in out[0]["content"] if p["type"] == "image_url"]
        assert len(blocks) == 1
        assert "image limit reached" in out[0]["content"][0]["text"]


class TestRefusals:
    """A marker names a file that gets shipped to whatever endpoint is active,
    possibly remote — so the path rules are a security boundary, not polish."""

    def test_a_path_outside_the_working_dir_is_refused(self, cfg, tmp_path):
        outside = tmp_path.parent / "secret.png"
        outside.write_bytes(PNG)
        path, why = vision.resolve_image(str(outside), cfg)
        assert path is None and "working directory" in why

    def test_a_traversal_escape_is_refused(self, cfg):
        path, why = vision.resolve_image("../../etc/passwd.png", cfg)
        assert path is None

    def test_a_missing_file_is_reported_not_raised(self, cfg):
        out = vision.expand_image_markers(user("[image: nope.png]"), cfg)
        assert "no such file" in out[0]["content"][0]["text"]
        assert not [p for p in out[0]["content"] if p["type"] == "image_url"]

    def test_a_non_image_extension_is_refused(self, cfg, tmp_path):
        (tmp_path / "notes.txt").write_text("hi")
        path, why = vision.resolve_image("notes.txt", cfg)
        assert path is None and "unsupported" in why

    def test_an_oversized_image_is_skipped_not_sent(self, cfg, tmp_path):
        cfg.vision.max_side = 0            # no downscale path
        cfg.vision.max_bytes = 10
        (tmp_path / "big.png").write_bytes(PNG)
        out = vision.expand_image_markers(user("[image: big.png]"), cfg)
        assert "cap" in out[0]["content"][0]["text"]


class TestBudget:
    def test_images_are_charged_against_the_context(self, cfg, shot):
        msgs = user("[image: shot.png]")
        assert vision.estimated_image_tokens(msgs, cfg) == vision.TOKENS_PER_IMAGE

    def test_a_text_only_model_is_charged_nothing(self, cfg, shot):
        cfg.llm.vision = "off"
        assert vision.estimated_image_tokens(user("[image: shot.png]"), cfg) == 0


class TestConfigWiring:
    @pytest.fixture(autouse=True)
    def _no_user_layer(self, tmp_path, monkeypatch):
        """The developer's own ~/.config/agent must not decide these."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no_home")

    def _load(self, tmp_path, text):
        from agent.config.loader import load_config
        p = tmp_path / "agent.toml"
        p.write_text(text, encoding="utf-8")
        return load_config(p)

    def test_the_section_is_known_and_merged(self, tmp_path):
        c = self._load(tmp_path, "[vision]\nmax_images = 2\nkeep_last_turns = 1\n")
        assert (c.vision.max_images, c.vision.keep_last_turns) == (2, 1)

    def test_a_vision_tag_on_the_entry_declares_the_capability(self, tmp_path):
        c = self._load(tmp_path, '[models.vlm]\nmodel = "some-local-build"\n'
                                 'tags = ["vision"]\n\n[models]\ndefault = "vlm"\n')
        assert c.llm.vision == "on"

    def test_a_per_entry_setting_wins_over_the_global_one(self, tmp_path):
        c = self._load(tmp_path, '[agent]\nvision = "on"\n\n[models.txt]\n'
                                 'model = "qwen3-coder-30b"\nvision = "off"\n\n'
                                 '[models]\ndefault = "txt"\n')
        assert c.llm.vision == "off"


class TestUploadUI:
    """The browser has to say which of the two things happened to an image.

    attachRef is run under node (see _attach_ref): the browser code decides
    this, so asserting on its source text would pass just as happily on JS
    that never executes.
    """

    def test_an_image_upload_inserts_the_marker(self):
        got = _attach_ref({"kind": "image", "path": ".agent/uploads/a.png",
                           "vision": True}, "a.png")
        assert got["marker"] == "[image: .agent/uploads/a.png]"

    def test_a_non_image_stays_a_plain_path(self):
        got = _attach_ref({"kind": "file", "path": ".agent/uploads/log.txt"}, "log.txt")
        assert got["marker"] == "[attached: .agent/uploads/log.txt]"
        assert "vision" not in got["note"]

    def test_the_user_is_told_the_image_was_sent(self):
        got = _attach_ref({"kind": "image", "path": "u/a.png", "vision": True}, "a.png")
        assert "sent to the model as an image" in got["note"]

    def test_the_user_is_told_when_the_model_cannot_see_it(self):
        got = _attach_ref({"kind": "image", "path": "u/a.png", "vision": False}, "a.png")
        assert "no vision" in got["note"] and "path only" in got["note"]


class TestApiBoundary:
    def test_normalize_expands_when_given_a_config(self, cfg, shot):
        out = normalize_api_messages(user("[image: shot.png]"), cfg)
        assert isinstance(out[0]["content"], list)

    def test_normalize_without_a_config_stays_text(self, shot):
        out = normalize_api_messages(user("[image: shot.png]"))
        assert isinstance(out[0]["content"], str)
