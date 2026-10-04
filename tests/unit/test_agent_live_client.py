"""Agent._live_client: a failover that repoints config.llm must not leave the
next turn on the old endpoint's client."""
from types import SimpleNamespace

from agent.config import Config
from agent.core.agent import Agent
from agent.core.llm_client import make_llm_client


def _fake_agent(base_url: str):
    config = Config()
    config.llm.base_url = base_url
    config.llm.api_key = "k"
    return SimpleNamespace(config=config, _client=make_llm_client(config))


def test_client_kept_while_endpoint_unchanged():
    a = _fake_agent("https://openrouter.ai/api/v1")
    before = a._client
    assert Agent._live_client(a) is before


def test_client_rebuilt_after_failover_repoints_config():
    a = _fake_agent("https://openrouter.ai/api/v1")
    a.config.llm.base_url = "http://192.168.31.42:8081/v1"   # switch_to_entry
    a.config.llm.model = "ornith10-35B-iq4nl"
    client = Agent._live_client(a)
    assert str(client.base_url).rstrip("/") == "http://192.168.31.42:8081/v1"
    assert a._client is client
