"""Admin endpoints for embedded qwen provider management and unified settings."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.platform.auth.middleware import get_internal_key
from app.platform.auth.key_registry import registry as key_registry
from app.platform.config.snapshot import config
from app.providers.qwen_embed import create_embedded_qwen_provider, _qwen_runtime_settings_from_config


router = APIRouter(prefix="/qwen", tags=["Admin - Qwen"])


def _to_response(resp) -> Response:
    media_type = resp.headers.get("content-type", "application/json").split(";", 1)[0]
    return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)


def _compat_status_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    runtime = _qwen_runtime_settings_from_config()
    data = dict(payload or {})
    request_runtime = data.get("request_runtime") if isinstance(data.get("request_runtime"), dict) else {}
    browser_automation = data.get("browser_automation") if isinstance(data.get("browser_automation"), dict) else {}
    data.setdefault("enabled", True)
    data.setdefault("connected", True)
    data.setdefault("engine_mode", runtime.get("engine_mode", "httpx"))
    data.setdefault("browser_engine", {
        "started": browser_automation.get("mode") not in (None, "disabled", "none"),
        "mode": browser_automation.get("mode", "on_demand_registration_only"),
        "description": browser_automation.get("description", "仅注册/激活时按需启动浏览器"),
        "pool_size": runtime.get("browser_pool_size", 0),
    })
    data.setdefault("httpx_engine", {
        "started": request_runtime.get("mode") == "direct_http",
        "mode": request_runtime.get("mode", "direct_http"),
        "description": request_runtime.get("description", "普通请求直连 HTTP"),
    })
    data.setdefault("hybrid_engine", None)
    return data


async def _boot_or_sync_provider(request: Request):
    runtime_settings = _qwen_runtime_settings_from_config()
    provider = getattr(request.app.state, "qwen_provider", None)

    if not runtime_settings.get("enabled", True):
        if provider is not None and getattr(provider, "started", False):
            await provider.stop()
        request.app.state.qwen_provider = None
        return None

    if provider is None:
        provider = create_embedded_qwen_provider(internal_key=get_internal_key())
        request.app.state.qwen_provider = provider

    if provider is None or getattr(provider, "app", None) is None:
        raise HTTPException(status_code=503, detail=getattr(provider, "error", "Qwen provider is unavailable"))

    await provider.apply_runtime_settings(runtime_settings)
    if not provider.started:
        await provider.start()
    return provider


def _require_provider(request: Request):
    provider = getattr(request.app.state, "qwen_provider", None)
    if provider is None or not provider.is_ready():
        raise HTTPException(status_code=503, detail="Qwen provider is unavailable")
    return provider


def _settings_patch(data: dict[str, Any]) -> dict[str, Any]:
    aliases = data.get("model_aliases")
    if aliases is not None and not isinstance(aliases, dict):
        raise HTTPException(status_code=400, detail="model_aliases must be a JSON object")
    patch: dict[str, Any] = {"providers": {"qwen": {}}}
    target = patch["providers"]["qwen"]
    mapping = {
        "enabled": "enabled",
        "engine_mode": "engine_mode",
        "browser_pool_size": "browser_pool_size",
        "max_inflight_per_account": "max_inflight_per_account",
        "stream_keepalive_interval": "stream_keepalive_interval",
        "register_secret": "register_secret",
        "native_tool_passthrough": "native_tool_passthrough",
        "account_min_interval_ms": "account_min_interval_ms",
        "request_jitter_min_ms": "request_jitter_min_ms",
        "request_jitter_max_ms": "request_jitter_max_ms",
        "rate_limit_base_cooldown": "rate_limit_base_cooldown",
        "rate_limit_max_cooldown": "rate_limit_max_cooldown",
        "model_aliases": "model_aliases",
    }
    for src, dst in mapping.items():
        if src in data:
            target[dst] = data[src]
    return patch


@router.get("/status")
async def qwen_status(request: Request):
    provider = await _boot_or_sync_provider(request)
    if provider is None:
        return _compat_status_payload({"enabled": False, "connected": False, "settings": _qwen_runtime_settings_from_config()})
    resp = await provider.request("GET", "/api/admin/status")
    if resp.status_code < 200 or resp.status_code >= 300:
        return _to_response(resp)
    payload = resp.json() if resp.content else {}
    return _compat_status_payload(payload)


@router.get("/settings")
async def qwen_settings(request: Request):
    provider = getattr(request.app.state, "qwen_provider", None)
    payload = provider.settings_payload() if provider is not None else _qwen_runtime_settings_from_config()
    payload["enabled"] = bool(payload.get("enabled", True)) if isinstance(payload, dict) else True
    return payload


@router.put("/settings")
async def qwen_update_settings(request: Request):
    body = await request.json()
    patch = _settings_patch(body)
    await config.update(patch)
    await config.load()
    provider = await _boot_or_sync_provider(request)
    if provider is None:
        return {"ok": True, "restarted": False, "settings": _qwen_runtime_settings_from_config()}
    return await provider.apply_runtime_settings(_qwen_runtime_settings_from_config())


@router.get("/accounts")
async def qwen_accounts(request: Request):
    resp = await _require_provider(request).request("GET", "/api/admin/accounts")
    return _to_response(resp)


@router.post("/accounts")
async def qwen_add_account(request: Request):
    body = await request.json()
    resp = await _require_provider(request).request("POST", "/api/admin/accounts", json_body=body)
    return _to_response(resp)


@router.post("/accounts/register-verify")
async def qwen_register_verify(request: Request):
    body = await request.json()
    secret = str((body or {}).get("secret") or "").strip()
    configured = str(_qwen_runtime_settings_from_config().get("register_secret") or "").strip()
    if not configured:
        return {"ok": True, "message": "未配置 register secret，默认放行"}
    return {"ok": secret == configured, "message": "ok" if secret == configured else "register secret mismatch"}


@router.post("/accounts/register")
async def qwen_register_account(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else None
    configured = str(_qwen_runtime_settings_from_config().get("register_secret") or "").strip()
    provided = str((body or {}).get("secret") or "").strip()
    if configured and configured != provided:
        raise HTTPException(status_code=403, detail="register secret mismatch")
    resp = await _require_provider(request).request("POST", "/api/admin/accounts/register")
    return _to_response(resp)


@router.post("/verify")
async def qwen_verify_all(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else None
    resp = await _require_provider(request).request("POST", "/api/admin/verify", json_body=body)
    return _to_response(resp)


@router.post("/accounts/{email}/activate")
async def qwen_activate_account(email: str, request: Request):
    resp = await _require_provider(request).request("POST", f"/api/admin/accounts/{quote(email, safe='')}/activate")
    return _to_response(resp)


@router.post("/accounts/{email}/verify")
async def qwen_verify_account(email: str, request: Request):
    resp = await _require_provider(request).request("POST", f"/api/admin/accounts/{quote(email, safe='')}/verify")
    return _to_response(resp)


@router.delete("/accounts/{email}")
async def qwen_delete_account(email: str, request: Request):
    resp = await _require_provider(request).request("DELETE", f"/api/admin/accounts/{quote(email, safe='')}")
    return _to_response(resp)


@router.get("/healthz")
async def qwen_healthz(request: Request):
    resp = await _require_provider(request).request("GET", "/healthz")
    return _to_response(resp)


@router.get("/readyz")
async def qwen_readyz(request: Request):
    resp = await _require_provider(request).request("GET", "/readyz")
    return _to_response(resp)


@router.get("/captures")
async def qwen_captures(request: Request):
    resp = await _require_provider(request).request("GET", "/admin/dev/captures")
    return _to_response(resp)


@router.delete("/captures")
async def qwen_clear_captures(request: Request):
    resp = await _require_provider(request).request("DELETE", "/admin/dev/captures")
    return _to_response(resp)


@router.get("/keys")
async def qwen_keys(request: Request):
    items = await key_registry.list_records()
    return {"keys": [item["key"] for item in items], "items": items, "scope": "global"}


__all__ = ["router"]
