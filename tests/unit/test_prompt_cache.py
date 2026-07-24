"""Unit tests for core.prompt_cache — breakpoints, telemetry, prefix stability."""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.config.models import ModelEntry
import agent.core.prompt_cache as pc


@pytest.fixture()
def cfg():
    c = Config()
    pc.reset()
    yield c
    pc.reset()


def _msgs():
    return [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


class TestBreakpoints:
    def test_off_by_default_leaves_request_untouched(self, cfg):
        msgs = _msgs()
        assert pc.apply_breakpoints(msgs, cfg) == msgs

    def test_anthropic_marks_system_and_last_message(self, cfg):
        cfg.llm.cache_breakpoints = "anthropic"
        out = pc.apply_breakpoints(_msgs(), cfg)
        assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert out[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert out[1]["content"] == "hello", "middle messages stay plain"

    def test_marks_last_of_several_system_messages(self, cfg):
        cfg.llm.cache_breakpoints = "anthropic"
        msgs = [{"role": "system", "content": "a"}, {"role": "system", "content": "b"},
                {"role": "user", "content": "q"}]
        out = pc.apply_breakpoints(msgs, cfg)
        assert isinstance(out[0]["content"], str), "only the last system block is marked"
        assert out[1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_structured_content_keeps_its_blocks(self, cfg):
        cfg.llm.cache_breakpoints = "anthropic"
        msgs = [{"role": "system", "content": [{"type": "text", "text": "a"},
                                               {"type": "text", "text": "b"}]}]
        out = pc.apply_breakpoints(msgs, cfg)
        assert len(out[0]["content"]) == 2
        assert "cache_control" not in out[0]["content"][0]
        assert out[0]["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_message_without_markable_content_is_left_alone(self, cfg):
        cfg.llm.cache_breakpoints = "anthropic"
        msgs = [{"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}]
        assert pc.apply_breakpoints(msgs, cfg)[0]["content"] is None

    def test_unknown_style_is_treated_as_off(self, cfg):
        cfg.llm.cache_breakpoints = "gemini-flavored"
        msgs = _msgs()
        assert pc.apply_breakpoints(msgs, cfg) == msgs

    def test_empty_request_is_safe(self, cfg):
        cfg.llm.cache_breakpoints = "anthropic"
        assert pc.apply_breakpoints([], cfg) == []

    def test_original_messages_are_not_mutated(self, cfg):
        cfg.llm.cache_breakpoints = "anthropic"
        msgs = _msgs()
        pc.apply_breakpoints(msgs, cfg)
        assert msgs[0]["content"] == "you are an agent"


class TestEntryResolution:
    def test_entry_setting_wins_over_global(self, cfg):
        cfg.llm.base_url = "https://api.example/v1"
        cfg.llm.model = "big"
        cfg.llm.cache_breakpoints = "off"
        cfg.model_entries = {"big": ModelEntry(base_url="https://api.example/v1",
                                               model="big", cache_breakpoints="anthropic")}
        assert pc._style(cfg) == "anthropic"

    def test_entry_is_matched_on_endpoint_not_name(self, cfg):
        # Mid-turn routing rewrites config.llm; the entry that no longer matches
        # must not keep dictating the dialect.
        cfg.llm.base_url = "http://localhost:8080/v1"
        cfg.llm.model = "local"
        cfg.model_entries = {"remote": ModelEntry(base_url="https://api.example/v1",
                                                  model="big", cache_breakpoints="anthropic")}
        assert pc._style(cfg) == "off"

    def test_blank_entry_value_inherits_global(self, cfg):
        cfg.llm.base_url = "https://api.example/v1"
        cfg.llm.model = "big"
        cfg.llm.cache_breakpoints = "anthropic"
        cfg.model_entries = {"big": ModelEntry(base_url="https://api.example/v1", model="big")}
        assert pc._style(cfg) == "anthropic"


class TestCachedTokens:
    def test_openai_style_details(self):
        class Details:
            cached_tokens = 1234

        class Usage:
            prompt_tokens = 5000
            prompt_tokens_details = Details()

        assert pc.extract_cached_tokens(Usage()) == 1234

    def test_anthropic_style_flat_field(self):
        class Usage:
            cache_read_input_tokens = 99

        assert pc.extract_cached_tokens(Usage()) == 99

    def test_dict_usage(self):
        assert pc.extract_cached_tokens({"prompt_tokens_details": {"cached_tokens": 7}}) == 7

    def test_nothing_reported_is_zero(self):
        class Usage:
            prompt_tokens = 10

        assert pc.extract_cached_tokens(Usage()) == 0
        assert pc.extract_cached_tokens(None) == 0


class TestPrefixStability:
    def test_first_request_is_stable(self, cfg):
        assert pc.check_prefix_stable(_msgs(), cfg) is True

    def test_identical_prefix_stays_stable(self, cfg):
        pc.check_prefix_stable(_msgs(), cfg)
        msgs = _msgs() + [{"role": "user", "content": "more"}]
        assert pc.check_prefix_stable(msgs, cfg) is True, "appending must not disturb the prefix"

    def test_changed_prefix_is_reported(self, cfg):
        pc.check_prefix_stable(_msgs(), cfg)
        changed = _msgs()
        changed[0] = {"role": "system", "content": "you are a different agent"}
        assert pc.check_prefix_stable(changed, cfg) is False

    def test_tracked_per_endpoint(self, cfg):
        pc.check_prefix_stable(_msgs(), cfg)
        cfg.llm.model = "other-model"
        assert pc.check_prefix_stable(_msgs(), cfg) is True

    def test_unserialisable_content_does_not_raise(self, cfg):
        assert pc.prefix_signature([{"role": "system", "content": object()}])


class TestPrepare:
    def test_prepare_applies_breakpoints(self, cfg):
        cfg.llm.cache_breakpoints = "anthropic"
        assert isinstance(pc.prepare(_msgs(), cfg)[0]["content"], list)

    def test_prepare_never_breaks_the_request(self, cfg, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("nope")

        monkeypatch.setattr(pc, "check_prefix_stable", boom)
        msgs = _msgs()
        assert pc.prepare(msgs, cfg) == msgs
