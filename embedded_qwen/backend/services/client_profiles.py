from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

CLAUDE_CODE_OPENAI_PROFILE = "claude_code_openai"
OPENCLAW_OPENAI_PROFILE = "openclaw_openai"
QWEN_CODE_OPENAI_PROFILE = "qwen_code_openai"

OPENCLAW_STARTUP_PATTERNS = (
    "A new session was started via /new or /reset.",
    "If runtime-provided startup context is included for this first turn",
)
OPENCLAW_UNTRUSTED_METADATA_PREFIX = "Sender (untrusted metadata):"
OPENCODE_SYSTEM_PREFIX = "you are opencode"


def sanitize_openclaw_user_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    if any(marker in cleaned for marker in OPENCLAW_STARTUP_PATTERNS):
        return ""
    if cleaned.startswith(OPENCLAW_UNTRUSTED_METADATA_PREFIX):
        match = re.search(r"\n\n(\[[^\n]+\]\s*[\s\S]*)$", cleaned)
        if match:
            cleaned = match.group(1).strip()
        else:
            return ""
    return cleaned


def looks_like_opencode_system_prompt(system_prompt: str) -> bool:
    if not isinstance(system_prompt, str):
        return False
    return system_prompt.strip().lower().startswith(OPENCODE_SYSTEM_PREFIX)


def header_value(headers: Mapping[str, Any] | Any, header_name: str) -> str:
    value = ""
    if hasattr(headers, "get"):
        value = headers.get(header_name, "")
    elif isinstance(headers, Mapping):
        value = headers.get(header_name, "")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)
