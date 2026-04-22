"""Internal APIs used by every2api for sync and runtime inspection."""

from collections import Counter
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from app.control.account.state_machine import derive_status, is_manageable, is_selectable
from app.control.model import registry as model_registry
from app.control.model.enums import Capability
from app.platform.auth.middleware import verify_internal_key
from app.platform.meta import get_project_version

router = APIRouter(prefix="/internal", dependencies=[Depends(verify_internal_key)])
_TAG_INTERNAL = "Internal - Sync"


def _detect_provider(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("qwen"):
        return "qwen"
    if lowered.startswith("gpt-image-"):
        return "chatgpt2api"
    if lowered.startswith("grok"):
        return "grok"
    return "unknown"


def _capabilities(spec) -> list[str]:
    caps: list[str] = []
    if spec.capability & Capability.CHAT:
        caps.append("chat")
    if spec.capability & Capability.IMAGE:
        caps.append("image")
    if spec.capability & Capability.IMAGE_EDIT:
        caps.append("image_edit")
    if spec.capability & Capability.VIDEO:
        caps.append("video")
    if spec.capability & Capability.VOICE:
        caps.append("voice")
    if spec.capability & Capability.ASSET:
        caps.append("asset")
    return caps


def _mask_token(token: str) -> str:
    if len(token) <= 8:
        return token[:2] + "***"
    return f"{token[:4]}***{token[-4:]}"


async def _runtime_snapshot(request: Request):
    repo = getattr(request.app.state, "repository", None)
    if repo is None:
        return None
    return await repo.runtime_snapshot()


async def _qwen_summary(request: Request) -> dict:
    provider = getattr(request.app.state, "qwen_provider", None)
    if provider is None:
        return {"enabled": False, "connected": False}
    try:
        return await provider.summary()
    except Exception as exc:
        return {"enabled": True, "connected": False, "error": str(exc)}


async def _qwen_accounts(request: Request) -> list[dict[str, Any]]:
    provider = getattr(request.app.state, "qwen_provider", None)
    if provider is None:
        return []
    try:
        return await provider.accounts()
    except Exception:
        return []


async def _chatgpt_summary(request: Request) -> dict:
    provider = getattr(request.app.state, "chatgpt_provider", None)
    if provider is None:
        return {"enabled": False, "connected": False}
    try:
        return await provider.summary()
    except Exception as exc:
        return {"enabled": True, "connected": False, "error": str(exc)}


@router.get("/health", tags=[_TAG_INTERNAL])
async def internal_health(request: Request):
    snapshot = await _runtime_snapshot(request)
    qwen = await _qwen_summary(request)
    chatgpt = await _chatgpt_summary(request)
    return {
        "status": "ok",
        "service": {
            "name": "any2api",
            "version": get_project_version(),
        },
        "runtime": {
            "accounts_ready": snapshot is not None,
            "revision": snapshot.revision if snapshot is not None else 0,
        },
        "providers": {
            "qwen": qwen,
            "chatgpt2api": chatgpt,
        },
    }


@router.get("/models", tags=[_TAG_INTERNAL])
async def list_internal_models():
    models = []
    for spec in model_registry.list_enabled():
        models.append(
            {
                "id": spec.model_name,
                "name": spec.public_name,
                "provider": _detect_provider(spec.model_name),
                "mode": spec.mode_id.to_api_str(),
                "tier": spec.pool_name(),
                "capabilities": _capabilities(spec),
            }
        )
    return Response(
        content=orjson.dumps({"object": "list", "data": models}),
        media_type="application/json",
    )


@router.get("/accounts", tags=[_TAG_INTERNAL])
async def list_internal_accounts(request: Request):
    snapshot = await _runtime_snapshot(request)
    qwen_items = await _qwen_accounts(request)
    if snapshot is None:
        return Response(
            content=orjson.dumps({"object": "list", "data": {"grok": [], "qwen": qwen_items}, "summary": {"total": len(qwen_items)}}),
            media_type="application/json",
        )

    pool_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    manageable = 0
    selectable = 0
    items = []
    for record in snapshot.items:
        effective_status = derive_status(record).value
        pool_counts[record.pool] += 1
        status_counts[effective_status] += 1
        record_manageable = is_manageable(record)
        record_selectable = any(is_selectable(record, mode_id) for mode_id in (0, 1, 2, 3))
        if record_manageable:
            manageable += 1
        if record_selectable:
            selectable += 1
        items.append(
            {
                "token": _mask_token(record.token),
                "pool": record.pool,
                "status": effective_status,
                "manageable": record_manageable,
                "selectable": record_selectable,
                "tags": record.tags,
                "usage": {
                    "use_count": record.usage_use_count,
                    "fail_count": record.usage_fail_count,
                    "sync_count": record.usage_sync_count,
                },
                "last_use_at": record.last_use_at,
                "last_fail_at": record.last_fail_at,
                "last_sync_at": record.last_sync_at,
                "state_reason": record.state_reason,
            }
        )

    return Response(
        content=orjson.dumps(
            {
                "object": "list",
                "data": {
                    "grok": items,
                    "qwen": qwen_items,
                },
                "summary": {
                    "revision": snapshot.revision,
                    "total": len(snapshot.items) + len(qwen_items),
                    "manageable": manageable,
                    "selectable": selectable,
                    "pools": dict(pool_counts),
                    "statuses": dict(status_counts),
                },
            }
        ),
        media_type="application/json",
    )


@router.get("/providers/summary", tags=[_TAG_INTERNAL])
async def providers_summary(request: Request):
    snapshot = await _runtime_snapshot(request)
    qwen = await _qwen_summary(request)
    chatgpt = await _chatgpt_summary(request)
    model_items = model_registry.list_enabled()

    model_counts: Counter[str] = Counter()
    for spec in model_items:
        model_counts[_detect_provider(spec.model_name)] += 1

    pool_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    manageable = 0
    selectable = 0
    total_accounts = 0
    if snapshot is not None:
        for record in snapshot.items:
            pool_counts[record.pool] += 1
            status_counts[derive_status(record).value] += 1
            if is_manageable(record):
                manageable += 1
            if any(is_selectable(record, mode_id) for mode_id in (0, 1, 2, 3)):
                selectable += 1
        total_accounts += len(snapshot.items)

    qwen_accounts = qwen.get("accounts", {}) if isinstance(qwen, dict) else {}
    if isinstance(qwen_accounts, dict):
        total_accounts += int(qwen_accounts.get("total", 0) or 0)

    chatgpt_accounts = chatgpt.get("accounts", {}) if isinstance(chatgpt, dict) else {}
    if isinstance(chatgpt_accounts, dict):
        total_accounts += int(chatgpt_accounts.get("total", 0) or 0)

    return Response(
        content=orjson.dumps(
            {
                "status": "ok",
                "service": {
                    "name": "any2api",
                    "version": get_project_version(),
                },
                "models": {
                    "total": len(model_items),
                    "providers": dict(model_counts),
                },
                "accounts": {
                    "revision": snapshot.revision if snapshot is not None else 0,
                    "total": total_accounts,
                    "manageable": manageable,
                    "selectable": selectable,
                    "pools": dict(pool_counts),
                    "statuses": dict(status_counts),
                },
                "provider_status": {
                    "qwen": qwen,
                    "chatgpt2api": chatgpt,
                },
            }
        ),
        media_type="application/json",
    )


__all__ = ["router"]
