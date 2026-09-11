"""Per-endpoint concurrency tracking (local vs cloud split in the status bar)."""
from agent.core import model_status as ms


def test_provider_label_local_and_cloud():
    assert ms.provider_label("http://localhost:8080/v1") == "local"
    assert ms.provider_label("http://127.0.0.1:1234") == "local"
    assert ms.provider_label(None) == "local"
    assert ms.provider_label("stdio://x") == "local"
    assert ms.provider_label("https://api.groq.com/openai/v1") == "groq"
    assert ms.provider_label("https://api.cerebras.ai/v1") == "cerebras"
    assert ms.provider_label("https://openrouter.ai/api/v1") == "openrouter"
    assert ms.provider_label("https://api.mistral.ai/v1") == "mistral"


def test_endpoint_counts_track_concurrency():
    # Simulate 2 local + 3 cloud calls overlapping, like a free-hybrid fan-out.
    with ms.track_sync("main", "local"), ms.track_sync("main", "local"), \
         ms.track_sync("bg", "groq"), ms.track_sync("bg", "groq"), \
         ms.track_sync("sec", "cerebras"):
        eps = ms.get_endpoint_counts()
        assert eps == {"local": 2, "groq": 2, "cerebras": 1}
    # All released → no stale endpoint entries.
    assert ms.get_endpoint_counts() == {}


def test_role_counts_fold_labels_onto_config_roles():
    # "main"/"sum"/"name"/"emb" are internal labels; the models panel renders
    # config roles, so the read side maps them (chat's "main" -> "default").
    with ms.track_sync("main"), ms.track_sync("main"), ms.track_sync("name"):
        assert ms.get_role_counts() == {"default": 2, "namer": 1}
    assert ms.get_role_counts() == {}
    # A label with no role name passes through rather than being dropped.
    with ms.track_sync("sec"):
        assert ms.get_role_counts() == {"sec": 1}


def test_endpoint_optional_keeps_role_counts():
    # Endpoint arg is optional; role counters still work without it.
    with ms.track_sync("main"):
        assert ms.get_counts().get("main", 0) >= 1
        assert ms.get_endpoint_counts() == {}
