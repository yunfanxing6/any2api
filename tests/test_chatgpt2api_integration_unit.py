from app.control.model import registry as model_registry
from app.platform.auth.key_registry import ALL_PROVIDERS, _normalize_multi
from app.platform.auth.middleware import _infer_provider_from_model
from app.providers.chatgpt2api import _normalize_chatgpt_model_ids, is_chatgpt_model_name


def test_chatgpt_models_are_registered_and_detected():
    for model_name in ("gpt-image-1", "gpt-image-2", "codex-gpt-image-2", "auto", "gpt-5-3", "gpt-5-mini", "gpt-5.4"):
        assert model_registry.get(model_name) is not None
        assert is_chatgpt_model_name(model_name) is True
        assert _infer_provider_from_model(model_name) == "chatgpt2api"


def test_chatgpt_model_list_keeps_legacy_image_aliases():
    assert _normalize_chatgpt_model_ids(["gpt-image-2", "codex-gpt-image-2"]) == (
        "gpt-image-2",
        "codex-gpt-image-2",
        "gpt-image-1",
    )


def test_chatgpt_provider_is_exposed_in_global_key_registry():
    assert "chatgpt2api" in ALL_PROVIDERS


def test_legacy_full_access_provider_lists_expand_to_chatgpt2api():
    assert _normalize_multi(["grok", "qwen"], ALL_PROVIDERS) == list(ALL_PROVIDERS)
