"""Admin endpoints for chatgpt2api sidecar management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.platform.config.snapshot import config
from app.providers.chatgpt2api import _chatgpt2api_settings_from_config


router = APIRouter(prefix="/chatgpt2api", tags=["Admin - ChatGPT2API"])


def _provider(request: Request):
    provider = getattr(request.app.state, "chatgpt_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="chatgpt2api provider is unavailable")
    return provider


def _to_response(resp) -> Response:
    media_type = resp.headers.get("content-type", "application/json").split(";", 1)[0]
    return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)


def _settings_patch(data: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {"providers": {"chatgpt2api": {}}}
    target = patch["providers"]["chatgpt2api"]
    mapping = {
        "enabled": "enabled",
        "base_url": "base_url",
        "auth_key": "auth_key",
        "timeout_sec": "timeout_sec",
    }
    for src, dst in mapping.items():
        if src in data:
            target[dst] = data[src]
    return patch


def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "total": len(items),
        "normal": 0,
        "limited": 0,
        "disabled": 0,
        "invalid": 0,
    }
    for item in items:
        status = str(item.get("status") or "").strip()
        if status == "正常":
            counts["normal"] += 1
        elif status == "限流":
            counts["limited"] += 1
        elif status == "禁用":
            counts["disabled"] += 1
        else:
            counts["invalid"] += 1
    return counts


@router.get("/status")
async def chatgpt2api_status(request: Request):
    provider = _provider(request)
    summary = await provider.summary()
    accounts_resp = await provider.request("GET", "/api/accounts") if summary.get("configured") else None
    items: list[dict[str, Any]] = []
    if accounts_resp is not None and 200 <= accounts_resp.status_code < 300 and accounts_resp.content:
        payload = accounts_resp.json()
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            items = payload["items"]
    summary["accounts"] = {
        **summary.get("accounts", {}),
        **_status_counts(items),
    }
    summary["settings"] = provider.settings_payload()
    return summary


@router.get("/settings")
async def chatgpt2api_settings(request: Request):
    return _provider(request).settings_payload()


@router.put("/settings")
async def chatgpt2api_update_settings(request: Request):
    body = await request.json()
    patch = _settings_patch(body if isinstance(body, dict) else {})
    await config.update(patch)
    await config.load()
    return _provider(request).settings_payload()


@router.get("/accounts")
async def chatgpt2api_accounts(request: Request):
    provider = _provider(request)
    if not provider.is_configured():
        return {"items": []}
    resp = await provider.request("GET", "/api/accounts")
    return _to_response(resp)


@router.post("/accounts")
async def chatgpt2api_add_accounts(request: Request):
    body = await request.json()
    resp = await _provider(request).request("POST", "/api/accounts", json_body=body)
    return _to_response(resp)


@router.delete("/accounts")
async def chatgpt2api_delete_accounts(request: Request):
    body = await request.json()
    resp = await _provider(request).request("DELETE", "/api/accounts", json_body=body)
    return _to_response(resp)


@router.post("/accounts/refresh")
async def chatgpt2api_refresh_accounts(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {"access_tokens": []}
    resp = await _provider(request).request("POST", "/api/accounts/refresh", json_body=body)
    return _to_response(resp)


@router.post("/accounts/update")
async def chatgpt2api_update_account(request: Request):
    body = await request.json()
    resp = await _provider(request).request("POST", "/api/accounts/update", json_body=body)
    return _to_response(resp)


@router.get("/models")
async def chatgpt2api_models(request: Request):
    resp = await _provider(request).request("GET", "/v1/models")
    return _to_response(resp)


@router.get("/version")
async def chatgpt2api_version(request: Request):
    resp = await _provider(request).request("GET", "/version")
    return _to_response(resp)


@router.get("/health")
async def chatgpt2api_health(request: Request):
    provider = _provider(request)
    settings = _chatgpt2api_settings_from_config()
    return {
        "enabled": bool(settings.get("enabled")),
        "configured": provider.is_configured(),
        "base_url": settings.get("base_url"),
    }


__all__ = ["router"]
