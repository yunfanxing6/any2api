from __future__ import annotations

import re
from typing import Any

_CORE_TOOL_PATTERNS = [
    re.compile(r"^(Read|read_file|ReadFile)$", re.IGNORECASE),
    re.compile(r"^(Write|write_to_file|WriteFile|write_file)$", re.IGNORECASE),
    re.compile(r"^(Bash|execute_command|RunCommand|run_command)$", re.IGNORECASE),
    re.compile(r"^(ListDir|list_dir|list_directory|ListDirectory|list_files)$", re.IGNORECASE),
    re.compile(r"^(Search|search_files|SearchFiles|grep_search|codebase_search|Grep|Glob)$", re.IGNORECASE),
    re.compile(r"^(Edit|edit_file|EditFile|replace_in_file)$", re.IGNORECASE),
]


def _is_core_tool(name: str) -> bool:
    return any(pattern.match(name) for pattern in _CORE_TOOL_PATTERNS)


def _tool_namespace(name: str) -> str:
    if not name:
        return ""
    match = re.match(r"^(mcp__[^_]+)", name)
    if match:
        return match.group(1)
    match = re.match(r"^([^_]+)__", name)
    if match:
        return match.group(1)
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[0]
    return name


def _example_params_for_core(name: str) -> dict[str, Any]:
    low = name.lower()
    if "read" in low:
        return {"file_path": "src/index.ts"}
    if "write" in low:
        return {"file_path": "output.txt", "content": "..."}
    if "bash" in low or "command" in low:
        return {"command": "ls -la"}
    if "list" in low:
        return {"path": "."}
    if "glob" in low:
        return {"pattern": "**/*.py"}
    if "search" in low or "grep" in low:
        return {"pattern": "TODO"}
    if "edit" in low:
        return {"file_path": "src/main.ts", "old_string": "old", "new_string": "new"}
    return {"input": "value"}


def _example_params_from_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("parameters") or tool.get("input_schema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict):
        return {"input": "value"}
    out: dict[str, Any] = {}
    for key, spec in list(props.items())[:2]:
        if not isinstance(spec, dict):
            out[key] = "value"
            continue
        t = spec.get("type", "string")
        if t == "boolean":
            out[key] = True
        elif t in ("number", "integer"):
            out[key] = 1
        elif t == "array":
            out[key] = []
        elif t == "object":
            out[key] = {}
        else:
            out[key] = "value"
    return out or {"input": "value"}


def pick_few_shot_tools(tools: list[dict[str, Any]], max_third_party: int = 4) -> list[dict[str, Any]]:
    if not tools:
        return []

    core_tools = [tool for tool in tools if _is_core_tool(tool.get("name", ""))]
    third_party = [tool for tool in tools if not _is_core_tool(tool.get("name", ""))]
    chosen: list[dict[str, Any]] = []

    core_pick = next((tool for tool in core_tools if tool.get("name") == "Read"), None) or next((tool for tool in core_tools if tool.get("name") == "Bash"), None)
    if core_pick is None and core_tools:
        core_pick = core_tools[0]
    if core_pick is not None:
        chosen.append(core_pick)

    groups: dict[str, list[dict[str, Any]]] = {}
    for tool in third_party:
        groups.setdefault(_tool_namespace(tool.get("name", "")), []).append(tool)

    for _, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(chosen) >= 1 + max_third_party:
            break
        chosen.append(max(items, key=lambda item: len(item.get("description", "") or "")))

    if not chosen and tools:
        chosen.append(tools[0])
    return chosen


def build_example_params(tool: dict[str, Any]) -> dict[str, Any]:
    name = tool.get("name", "")
    if _is_core_tool(name):
        return _example_params_for_core(name)
    return _example_params_from_schema(tool)


def render_few_shot_turn(few_shot_tools: list[dict[str, Any]], render_tool_call, thinking_enabled: bool = False) -> tuple[str, str]:
    actions = [render_tool_call(tool.get("name", ""), build_example_params(tool)) for tool in few_shot_tools]
    user_text = (
        "[FEW-SHOT WARM-UP] Now show me how you would emit multiple action markers in a single "
        "response. Use representatives from different action categories."
    )
    body = "\n\n".join(actions)
    if thinking_enabled:
        assistant_text = f"<thinking>Emit multiple action markers.</thinking>\n\nHere are examples across action categories:\n\n{body}"
    else:
        assistant_text = f"Understood. Here are example markers across action categories:\n\n{body}"
    return user_text, assistant_text


def tool_summary_for_log(few_shot_tools: list[dict[str, Any]]) -> str:
    return ", ".join(tool.get("name", "?") for tool in few_shot_tools)
