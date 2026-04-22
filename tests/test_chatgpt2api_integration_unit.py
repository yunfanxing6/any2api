from app.control.model import registry as model_registry
from app.platform.auth.key_registry import ALL_PROVIDERS, _normalize_multi
from app.platform.auth.middleware import _infer_provider_from_model
from app.providers.chatgpt2api import is_chatgpt_model_name


def test_chatgpt_models_are_registered_and_detected():
    spec = model_registry.get("gpt-image-1")
    assert spec is not None
    assert spec.is_image()
    assert is_chatgpt_model_name("gpt-image-1") is True
    assert _infer_provider_from_model("gpt-image-1") == "chatgpt2api"


def test_chatgpt_provider_is_exposed_in_global_key_registry():
    assert "chatgpt2api" in ALL_PROVIDERS


def test_legacy_full_access_provider_lists_expand_to_chatgpt2api():
    assert _normalize_multi(["grok", "qwen"], ALL_PROVIDERS) == list(ALL_PROVIDERS)
