from __future__ import annotations

import re

_REFUSAL_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"I[''\u2019]\s*m\s+sorry", re.IGNORECASE),
    re.compile(r"I\s+am\s+sorry", re.IGNORECASE),
    re.compile(r"I\s+cannot\s+help\s+with", re.IGNORECASE),
    re.compile(r"I\s+can\s+only\s+answer", re.IGNORECASE),
    re.compile(r"I\s+only\s+answer", re.IGNORECASE),
    re.compile(r"Tool\s+[\w.:-]+\s+does\s+not\s+exists?", re.IGNORECASE),
    re.compile(r"I\s+cannot\s+execute\s+this\s+tool", re.IGNORECASE),
    re.compile(r"我只能回答"),
    re.compile(r"工具.*?不存在"),
    re.compile(r"我无法执行.*?工具"),
)


def is_refusal_text(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


def clean_refusal_messages(messages: list, *, replacement_text: str = "") -> tuple[list, int]:
    if not messages:
        return messages, 0

    placeholder = replacement_text or "[earlier assistant turn omitted by proxy]"
    out: list = []
    replaced = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            out.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if is_refusal_text(content):
                new_msg = dict(msg)
                new_msg["content"] = placeholder
                out.append(new_msg)
                replaced += 1
            else:
                out.append(msg)
            continue
        if not isinstance(content, list):
            out.append(msg)
            continue

        has_tool_use = any(isinstance(part, dict) and part.get("type") == "tool_use" for part in content)
        new_content = []
        mutated = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and is_refusal_text(part.get("text", "")):
                if has_tool_use:
                    mutated = True
                    continue
                new_content.append({"type": "text", "text": placeholder})
                mutated = True
                continue
            new_content.append(part)

        if mutated:
            new_msg = dict(msg)
            new_msg["content"] = new_content or [{"type": "text", "text": placeholder}]
            out.append(new_msg)
            replaced += 1
        else:
            out.append(msg)
    return out, replaced
