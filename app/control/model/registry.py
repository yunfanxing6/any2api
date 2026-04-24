"""Model registry — all supported model variants defined in one place."""

from .enums import Capability, ModeId, Tier
from .spec import ModelSpec

# ---------------------------------------------------------------------------
# Master model list.
# Add new models here; no other files need to change.
# ---------------------------------------------------------------------------

MODELS: tuple[ModelSpec, ...] = (
    # === Chat ==============================================================

    # Qwen
    ModelSpec("qwen3.6-plus",                           ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.6 Plus (Auto)"),
    ModelSpec("qwen3.6-plus:auto",                      ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.6 Plus (Auto)"),
    ModelSpec("qwen3.6-plus:fast",                      ModeId.FAST,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.6 Plus (Fast)"),
    ModelSpec("qwen3.6-plus:thinking",                  ModeId.EXPERT, Tier.BASIC, Capability.CHAT,       True, "Qwen 3.6 Plus (Thinking)"),
    ModelSpec("qwen3.5-plus",                           ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.5 Plus (Auto)"),
    ModelSpec("qwen3.5-plus:auto",                      ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.5 Plus (Auto)"),
    ModelSpec("qwen3.5-plus:fast",                      ModeId.FAST,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.5 Plus (Fast)"),
    ModelSpec("qwen3.5-plus:thinking",                  ModeId.EXPERT, Tier.BASIC, Capability.CHAT,       True, "Qwen 3.5 Plus (Thinking)"),
    ModelSpec("qwen3.5-flash",                          ModeId.FAST,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.5 Flash"),
    ModelSpec("qwen3.5-omni-plus",                      ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "Qwen 3.5 Omni Plus"),

    # ChatGPT2API text/chat
    ModelSpec("auto",                                   ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "ChatGPT Auto"),
    ModelSpec("gpt-5",                                  ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "GPT-5"),
    ModelSpec("gpt-5-1",                                ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "GPT-5.1"),
    ModelSpec("gpt-5-2",                                ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "GPT-5.2"),
    ModelSpec("gpt-5-3",                                ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "GPT-5.3"),
    ModelSpec("gpt-5-3-mini",                           ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "GPT-5.3 Mini"),
    ModelSpec("gpt-5-mini",                             ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "GPT-5 Mini"),
    ModelSpec("gpt-5.4",                                ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "GPT-5.4"),

    # Basic+
    ModelSpec("grok-4.20-0309-non-reasoning",           ModeId.FAST,   Tier.BASIC, Capability.CHAT,       True, "Grok 4.20 0309 Non-Reasoning"),
    ModelSpec("grok-4.20-0309",                         ModeId.AUTO,   Tier.BASIC, Capability.CHAT,       True, "Grok 4.20 0309"),
    ModelSpec("grok-4.20-0309-reasoning",               ModeId.EXPERT, Tier.BASIC, Capability.CHAT,       True, "Grok 4.20 0309 Reasoning"),
    # Super+
    ModelSpec("grok-4.20-0309-non-reasoning-super",     ModeId.FAST,   Tier.SUPER, Capability.CHAT,       True, "Grok 4.20 0309 Non-Reasoning Super"),
    ModelSpec("grok-4.20-0309-super",                   ModeId.AUTO,   Tier.SUPER, Capability.CHAT,       True, "Grok 4.20 0309 Super"),
    ModelSpec("grok-4.20-0309-reasoning-super",         ModeId.EXPERT, Tier.SUPER, Capability.CHAT,       True, "Grok 4.20 0309 Reasoning Super"),
    # Heavy+
    ModelSpec("grok-4.20-0309-non-reasoning-heavy",     ModeId.FAST,   Tier.HEAVY, Capability.CHAT,       True, "Grok 4.20 0309 Non-Reasoning Heavy"),
    ModelSpec("grok-4.20-0309-heavy",                   ModeId.AUTO,   Tier.HEAVY, Capability.CHAT,       True, "Grok 4.20 0309 Heavy"),
    ModelSpec("grok-4.20-0309-reasoning-heavy",         ModeId.EXPERT, Tier.HEAVY, Capability.CHAT,       True, "Grok 4.20 0309 Reasoning Heavy"),
    ModelSpec("grok-4.20-multi-agent-0309",             ModeId.HEAVY,  Tier.HEAVY, Capability.CHAT,       True, "Grok 4.20 Multi-Agent 0309"),
    
    # === Image ==============================================================

    # Basic+
    ModelSpec("grok-imagine-image-lite",                ModeId.FAST,   Tier.BASIC, Capability.IMAGE,      True, "Grok Imagine Image Lite"),
    ModelSpec("gpt-image-1",                            ModeId.AUTO,   Tier.BASIC, Capability.IMAGE,      True, "ChatGPT Image 1"),
    ModelSpec("gpt-image-2",                            ModeId.AUTO,   Tier.BASIC, Capability.IMAGE,      True, "ChatGPT Image 2"),
    ModelSpec("codex-gpt-image-2",                      ModeId.AUTO,   Tier.BASIC, Capability.IMAGE,      True, "Codex GPT Image 2"),
    # Super+
    ModelSpec("grok-imagine-image",                     ModeId.AUTO,   Tier.SUPER, Capability.IMAGE,      True, "Grok Imagine Image"),
    ModelSpec("grok-imagine-image-pro",                 ModeId.AUTO,   Tier.SUPER, Capability.IMAGE,      True, "Grok Imagine Image Pro"),
    
    # === Image Edit =========================================================

    # Super+
    ModelSpec("grok-imagine-image-edit",                ModeId.AUTO,   Tier.SUPER, Capability.IMAGE_EDIT, True, "Grok Imagine Image Edit"),
    
    # === Video ==============================================================

    # Super+
    ModelSpec("grok-imagine-video",                     ModeId.AUTO,   Tier.SUPER, Capability.VIDEO,      True, "Grok Imagine Video"),
)

# ---------------------------------------------------------------------------
# Internal lookup structures — built once at import time.
# ---------------------------------------------------------------------------

_BY_NAME: dict[str, ModelSpec] = {m.model_name: m for m in MODELS}

_BY_CAP: dict[int, list[ModelSpec]] = {}
for _m in MODELS:
    _BY_CAP.setdefault(int(_m.capability), []).append(_m)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get(model_name: str) -> ModelSpec | None:
    """Return the spec for *model_name*, or ``None`` if not registered."""
    return _BY_NAME.get(model_name)


def resolve(model_name: str) -> ModelSpec:
    """Return the spec for *model_name*; raise ``ValueError`` if unknown."""
    spec = _BY_NAME.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model: {model_name!r}")
    return spec


def list_enabled() -> list[ModelSpec]:
    """Return all enabled models in registration order."""
    return [m for m in MODELS if m.enabled]


def list_by_capability(cap: Capability) -> list[ModelSpec]:
    """Return enabled models that include *cap* in their capability mask."""
    return [m for m in MODELS if m.enabled and bool(m.capability & cap)]


__all__ = ["MODELS", "get", "resolve", "list_enabled", "list_by_capability"]
