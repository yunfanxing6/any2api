from __future__ import annotations

import os
import re
from typing import Any

_SMART_DOUBLE_QUOTES = {"\u00ab", "\u201c", "\u201d", "\u275e", "\u201f", "\u201e", "\u275d", "\u00bb"}
_SMART_SINGLE_QUOTES = {"\u2018", "\u2019", "\u201a", "\u201b"}
_DOUBLE_QUOTE_CLASS = '["\u00ab\u201c\u201d\u275e\u201f\u201e\u275d\u00bb]'
_SINGLE_QUOTE_CLASS = "['\u2018\u2019\u201a\u201b]"


def replace_smart_quotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = []
    for ch in text:
        if ch in _SMART_DOUBLE_QUOTES:
            out.append('"')
        elif ch in _SMART_SINGLE_QUOTES:
            out.append("'")
        else:
            out.append(ch)
    return "".join(out)


def _build_fuzzy_pattern(text: str) -> str:
    parts = []
    for ch in text:
        if ch in _SMART_DOUBLE_QUOTES or ch == '"':
            parts.append(_DOUBLE_QUOTE_CLASS)
        elif ch in _SMART_SINGLE_QUOTES or ch == "'":
            parts.append(_SINGLE_QUOTE_CLASS)
        elif ch in (" ", "\t"):
            parts.append(r"\s+")
        elif ch == "\\":
            parts.append(r"\\{1,2}")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def repair_exact_match(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        return args
    lower = (tool_name or "").lower()
    if not any(key in lower for key in ("edit", "str_replace", "strreplace", "search_replace")):
        return args

    old_string = args.get("old_string") or args.get("old_str")
    file_path = args.get("file_path") or args.get("path")
    if not isinstance(old_string, str) or not old_string or not isinstance(file_path, str) or not file_path:
        return args

    try:
        if not os.path.exists(file_path):
            return args
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except Exception:
        return args

    if old_string in content:
        _normalize_new_string(args)
        return args

    try:
        matches = list(re.finditer(_build_fuzzy_pattern(old_string), content))
    except re.error:
        return args

    if len(matches) != 1:
        return args

    matched_text = matches[0].group(0)
    if "old_string" in args:
        args["old_string"] = matched_text
    elif "old_str" in args:
        args["old_str"] = matched_text
    _normalize_new_string(args)
    return args


def _normalize_new_string(args: dict[str, Any]) -> None:
    for key in ("new_string", "new_str"):
        if key in args and isinstance(args[key], str):
            args[key] = replace_smart_quotes(args[key])


def fix_tool_call_arguments(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        return args
    return repair_exact_match(tool_name, args)
