"""Unified Any2API API-key management endpoints."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException

from app.platform.auth.key_registry import ALL_PROVIDERS, ALL_SCOPES, registry


router = APIRouter(tags=["Admin - Keys"])


def _normalize_payload(payload: dict) -> dict:
    data = dict(payload or {})
    data["key"] = str(data.get("key") or "").strip()
    data["label"] = str(data.get("label") or "").strip()
    data["note"] = str(data.get("note") or "").strip()
    data["enabled"] = bool(data.get("enabled", True))
    data["providers"] = [str(item).strip().lower() for item in (data.get("providers") or list(ALL_PROVIDERS)) if str(item).strip()]
    data["scopes"] = [str(item).strip().lower() for item in (data.get("scopes") or list(ALL_SCOPES)) if str(item).strip()]
    return data


@router.get("/keys")
async def list_keys():
    items = await registry.list_records()
    return {"keys": [item["key"] for item in items], "items": items}


@router.post("/keys")
async def add_key(payload: dict | None = None):
    payload = payload or {}
    key = str(payload.get("key") or "").strip()
    if not key:
        key = f"sk-any-{secrets.token_urlsafe(18).replace('-', '').replace('_', '')[:28]}"
    payload = _normalize_payload({**payload, "key": key})
    try:
        record = await registry.upsert_record(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "item": record.to_dict(), "key": record.key}


@router.put("/keys/{key}")
async def update_key(key: str, payload: dict | None = None):
    payload = _normalize_payload({**(payload or {}), "key": key})
    try:
        record = await registry.upsert_record(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "item": record.to_dict()}


@router.delete("/keys/{key}")
async def delete_key(key: str):
    try:
        await registry.delete_record(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"ok": True}


__all__ = ["router"]
