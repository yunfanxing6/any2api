from __future__ import annotations

import re

TOOL_NAME_ALIASES: dict[str, str] = {
    "Read": "fs_open_file",
    "Write": "fs_put_file",
    "Edit": "fs_patch_file",
    "Bash": "shell_run",
    "Grep": "text_search",
    "Glob": "path_find",
    "NotebookEdit": "notebook_patch",
    "WebFetch": "http_get_url",
    "WebSearch": "web_query",
}

REVERSE_ALIASES: dict[str, str] = {value: key for key, value in TOOL_NAME_ALIASES.items()}
_AUTO_PREFIX = "u_"


def to_qwen_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return name
    if name in TOOL_NAME_ALIASES:
        return TOOL_NAME_ALIASES[name]
    if name in REVERSE_ALIASES or name.startswith(_AUTO_PREFIX):
        return name
    return _AUTO_PREFIX + name


def from_qwen_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        return name
    if name in REVERSE_ALIASES:
        return REVERSE_ALIASES[name]
    if name.startswith(_AUTO_PREFIX):
        return name[len(_AUTO_PREFIX):]
    return name


_BARE_NAME_PATTERN = re.compile(r"\b(" + "|".join(sorted(TOOL_NAME_ALIASES.keys(), key=len, reverse=True)) + r")\b")


def obfuscate_bare_names(text: str) -> str:
    if not text:
        return text
    return _BARE_NAME_PATTERN.sub(lambda match: TOOL_NAME_ALIASES[match.group(1)], text)
