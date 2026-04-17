"""Persistent global API key registry with metadata."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from app.platform.config.snapshot import config
from app.platform.paths import data_path
from app.platform.runtime.clock import now_ms


ALL_PROVIDERS = ("grok", "qwen")
ALL_SCOPES = ("models", "chat", "responses", "images", "videos", "files", "anthropic")


@dataclass(slots=True)
class APIKeyRecord:
    key: str
    label: str = ""
    note: str = ""
    enabled: bool = True
    providers: list[str] | None = None
    scopes: list[str] | None = None
    created_at: int = 0
    updated_at: int = 0
    last_used_at: int = 0

    def normalize(self) -> "APIKeyRecord":
        self.key = str(self.key or "").strip()
        self.label = str(self.label or "").strip()
        self.note = str(self.note or "").strip()
        self.enabled = bool(self.enabled)
        self.providers = _normalize_multi(self.providers, ALL_PROVIDERS)
        self.scopes = _normalize_multi(self.scopes, ALL_SCOPES)
        self.created_at = int(self.created_at or now_ms())
        self.updated_at = int(self.updated_at or now_ms())
        self.last_used_at = int(self.last_used_at or 0)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "note": self.note,
            "enabled": self.enabled,
            "providers": list(self.providers or []),
            "scopes": list(self.scopes or []),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
        }


def _normalize_multi(values: list[str] | tuple[str, ...] | None, allowed: tuple[str, ...]) -> list[str]:
    if not values:
        return list(allowed)
    seen: list[str] = []
    allowed_set = set(allowed)
    for raw in values:
        item = str(raw or "").strip().lower()
        if not item or item in seen:
            continue
        if item == "all":
            return list(allowed)
        if item in allowed_set:
            seen.append(item)
    # Backward compatibility: records saved before the `files` scope existed
    # should keep behaving like full-access keys after the scope list expands.
    if "files" in allowed_set and set(seen) == (allowed_set - {"files"}):
        return list(allowed)
    if allowed == ALL_SCOPES and "files" not in seen and any(item in seen for item in ("chat", "responses", "images", "anthropic")):
        seen.append("files")
    return seen or list(allowed)


def _parse_config_keys() -> list[str]:
    raw = config.get("app.api_key", "")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


class APIKeyRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or data_path("api_keys.meta.json")
        self._lock = asyncio.Lock()
        self._loaded = False
        self._records: dict[str, APIKeyRecord] = {}
        self._last_config_keys: tuple[str, ...] = ()

    async def ensure_loaded(self) -> None:
        if self._loaded:
            async with self._lock:
                await self._sync_with_config_keys_locked()
            return
        async with self._lock:
            if not self._loaded:
                self._records = await asyncio.to_thread(self._read_records)
                self._loaded = True
            await self._sync_with_config_keys_locked()

    async def sync_with_config_keys(self) -> None:
        async with self._lock:
            if not self._loaded:
                self._records = await asyncio.to_thread(self._read_records)
                self._loaded = True
            await self._sync_with_config_keys_locked()

    def _read_records(self) -> dict[str, APIKeyRecord]:
        if not self._path.exists():
            return {}
        try:
            data = orjson.loads(self._path.read_bytes())
        except Exception:
            return {}
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return {}
        records: dict[str, APIKeyRecord] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized = {
                "key": item.get("key", ""),
                "label": item.get("label", ""),
                "note": item.get("note", ""),
                "enabled": item.get("enabled", True),
                "providers": item.get("providers"),
                "scopes": item.get("scopes"),
                "created_at": item.get("created_at", 0),
                "updated_at": item.get("updated_at", 0),
                "last_used_at": item.get("last_used_at", 0),
            }
            record = APIKeyRecord(**normalized).normalize()
            if record.key:
                records[record.key] = record
        return records

    def _write_records(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": [record.to_dict() for record in sorted(self._records.values(), key=lambda item: item.created_at)],
        }
        self._path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    async def _sync_with_config_keys_locked(self) -> None:
        keys = tuple(_parse_config_keys())
        if self._last_config_keys == keys and self._records:
            return

        desired = set(keys)
        changed = False
        for key in keys:
            if key not in self._records:
                self._records[key] = APIKeyRecord(key=key).normalize()
                changed = True
        for existing in list(self._records.keys()):
            if existing not in desired:
                del self._records[existing]
                changed = True

        self._last_config_keys = keys
        if changed:
            await asyncio.to_thread(self._write_records)

    async def list_records(self) -> list[dict[str, Any]]:
        await self.ensure_loaded()
        return [record.to_dict() for record in sorted(self._records.values(), key=lambda item: item.created_at)]

    async def get_record(self, key: str) -> APIKeyRecord | None:
        await self.ensure_loaded()
        return self._records.get(key)

    async def upsert_record(self, payload: dict[str, Any]) -> APIKeyRecord:
        async with self._lock:
            if not self._loaded:
                self._records = await asyncio.to_thread(self._read_records)
                self._loaded = True
            await self._sync_with_config_keys_locked()

            key = str(payload.get("key") or "").strip()
            if not key:
                raise ValueError("key is required")

            existing = self._records.get(key)
            if existing is None:
                record = APIKeyRecord(key=key)
            else:
                record = APIKeyRecord(**existing.to_dict())

            for field in ("label", "note", "enabled", "providers", "scopes"):
                if field in payload:
                    setattr(record, field, payload[field])
            record.updated_at = now_ms()
            record.normalize()
            self._records[key] = record
            self._last_config_keys = tuple(self._records.keys())
            await self._sync_config_value_locked()
            await asyncio.to_thread(self._write_records)
            return record

    async def delete_record(self, key: str) -> None:
        async with self._lock:
            if not self._loaded:
                self._records = await asyncio.to_thread(self._read_records)
                self._loaded = True
            await self._sync_with_config_keys_locked()
            if key not in self._records:
                raise KeyError(key)
            del self._records[key]
            self._last_config_keys = tuple(self._records.keys())
            await self._sync_config_value_locked()
            await asyncio.to_thread(self._write_records)

    async def record_success(self, key: str) -> None:
        async with self._lock:
            if not self._loaded:
                self._records = await asyncio.to_thread(self._read_records)
                self._loaded = True
            await self._sync_with_config_keys_locked()
            record = self._records.get(key)
            if record is None:
                return
            record.last_used_at = now_ms()
            record.updated_at = now_ms()
            await asyncio.to_thread(self._write_records)

    async def authorize(self, key: str, *, provider: str | None, scope: str | None) -> tuple[bool, str | None, APIKeyRecord | None]:
        await self.ensure_loaded()
        record = self._records.get(key)
        if record is None:
            return False, "Invalid API key.", None
        if not record.enabled:
            return False, "API key is disabled.", record
        if scope and scope not in (record.scopes or list(ALL_SCOPES)):
            return False, f"API key is not allowed to access scope '{scope}'.", record
        if provider and provider not in (record.providers or list(ALL_PROVIDERS)):
            return False, f"API key is not allowed to access provider '{provider}'.", record
        return True, None, record

    async def _sync_config_value_locked(self) -> None:
        joined = ",".join(self._records.keys())
        await config.update({"app": {"api_key": joined}})
        await config.load()


registry = APIKeyRegistry()


__all__ = ["APIKeyRegistry", "APIKeyRecord", "registry", "ALL_PROVIDERS", "ALL_SCOPES"]
