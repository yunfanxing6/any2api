from __future__ import annotations

from typing import Any


def _type_of(prop: dict[str, Any]) -> str:
    if not isinstance(prop, dict):
        return "any"
    if "enum" in prop and isinstance(prop["enum"], list) and prop["enum"]:
        return "|".join(str(value) for value in prop["enum"])
    base_type = prop.get("type", "any")
    if base_type == "array":
        items = prop.get("items")
        return f"{_type_of(items)}[]" if isinstance(items, dict) else "any[]"
    if base_type == "object" and isinstance(prop.get("properties"), dict):
        return compact_schema(prop)
    if isinstance(base_type, list):
        return "|".join(str(value) for value in base_type)
    return str(base_type) if base_type else "any"


def compact_schema(schema: dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return "{}"
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return "{}"
    required = set(schema.get("required", []) if isinstance(schema.get("required"), list) else [])
    parts = []
    for name, spec in props.items():
        parts.append(f"{name}{'!' if name in required else '?'}: {_type_of(spec if isinstance(spec, dict) else {})}")
    return "{" + ", ".join(parts) + "}"
