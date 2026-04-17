import time
import uuid


VARIANT_SUFFIXES = {"auto", "fast", "thinking"}


def split_requested_model(model: str) -> tuple[str, str]:
    model = (model or "").strip()
    if not model:
        return "qwen3.6-plus", "auto"
    if ":" not in model:
        return model, "auto"
    base, suffix = model.rsplit(":", 1)
    suffix = suffix.strip().lower()
    if suffix in VARIANT_SUFFIXES:
        return base, suffix
    return model, "auto"


def normalize_upstream_model(model: str) -> str:
    base, _ = split_requested_model(model)
    return base


def build_feature_config_for_model(model: str, has_custom_tools: bool = False) -> dict:
    _, variant = split_requested_model(model)
    feature_config = {
        **CUSTOM_TOOL_COMPAT_FEATURE_CONFIG,
        "function_calling": False,
        "enable_tools": False,
        "enable_function_call": False,
        "tool_choice": "none",
    }
    if variant == "fast":
        feature_config.update(CUSTOM_TOOL_LOW_LATENCY_OVERRIDES)
    elif variant == "thinking":
        # Keep the same known-good upstream settings as AUTO. The key fix is
        # mapping the public variant back to the base upstream model name.
        feature_config.update({
            "thinking_enabled": True,
            "auto_thinking": True,
            "thinking_mode": "Auto",
            "thinking_format": "summary",
        })
    if has_custom_tools:
        feature_config.update(CUSTOM_TOOL_LOW_LATENCY_OVERRIDES)
    return feature_config

CUSTOM_TOOL_COMPAT_FEATURE_CONFIG = {
    "thinking_enabled": True,
    "output_schema": "phase",
    "research_mode": "normal",
    "auto_thinking": True,
    "thinking_mode": "Auto",
    "thinking_format": "summary",
    "auto_search": False,
    "code_interpreter": False,
    "plugins_enabled": False,
}

CUSTOM_TOOL_LOW_LATENCY_OVERRIDES = {
    "thinking_enabled": False,
    "auto_thinking": False,
}


def build_chat_payload(chat_id: str, model: str, content: str, has_custom_tools: bool = False, files: list[dict] | None = None) -> dict:
    ts = int(time.time())
    upstream_model = normalize_upstream_model(model)
    feature_config = build_feature_config_for_model(model, has_custom_tools)
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": upstream_model,
        "parent_id": None,
        "messages": [
            {
                "fid": str(uuid.uuid4()),
                "parentId": None,
                "childrenIds": [str(uuid.uuid4())],
                "role": "user",
                "content": content,
                "user_action": "chat",
                "files": files or [],
                "timestamp": ts,
                "models": [upstream_model],
                "chat_type": "t2t",
                "feature_config": feature_config,
                "extra": {"meta": {"subChatType": "t2t"}},
                "sub_chat_type": "t2t",
                "parent_id": None,
            }
        ],
        "timestamp": ts,
    }
