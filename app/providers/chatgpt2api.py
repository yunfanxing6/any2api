"""chatgpt2api reverse-proxy helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import orjson
from fastapi.responses import Response

from app.platform.config.snapshot import get_config
from app.platform.logging.logger import logger

_MODELS_CACHE_TTL = 30.0
_CHATGPT_IMAGE_MODELS = {"gpt-image", "gpt-image-1", "gpt-image-2", "codex-gpt-image-2"}
_CHATGPT_TEXT_MODELS = {"auto"}


def is_chatgpt_model_name(model_name: str) -> bool:
    lowered = (model_name or "").strip().lower()
    return bool(lowered and (lowered in _CHATGPT_IMAGE_MODELS or lowered in _CHATGPT_TEXT_MODELS or lowered.startswith("gpt-5")))


def _normalize_chatgpt_model_ids(model_ids: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for model_id in model_ids:
        candidate = str(model_id or "").strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)

    if {"gpt-image", "gpt-image-1", "gpt-image-2"} & set(normalized):
        for alias in ("gpt-image-1", "gpt-image-2"):
            if alias not in normalized:
                normalized.append(alias)

    return tuple(normalized)


def _chatgpt2api_settings_from_config() -> dict[str, Any]:
    return {
        "enabled": bool(get_config("providers.chatgpt2api.enabled", False)),
        "base_url": str(get_config("providers.chatgpt2api.base_url", "http://chatgpt2api") or "").strip().rstrip("/"),
        "auth_key": str(get_config("providers.chatgpt2api.auth_key", "") or "").strip(),
        "timeout_sec": int(get_config("providers.chatgpt2api.timeout_sec", 180) or 180),
    }


@dataclass(slots=True)
class ChatGPT2APIProvider:
    _client: httpx.AsyncClient | None = None
    _models_cache: tuple[str, ...] = ()
    _models_fetched_at: float = 0.0

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=True)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _settings(self) -> dict[str, Any]:
        return _chatgpt2api_settings_from_config()

    def is_enabled(self) -> bool:
        return bool(self._settings().get("enabled", False))

    def is_configured(self) -> bool:
        settings = self._settings()
        return bool(settings.get("enabled") and settings.get("base_url") and settings.get("auth_key"))

    def settings_payload(self) -> dict[str, Any]:
        settings = self._settings()
        return {
            "enabled": bool(settings["enabled"]),
            "base_url": str(settings["base_url"]),
            "timeout_sec": int(settings["timeout_sec"]),
            "configured": bool(settings["base_url"] and settings["auth_key"]),
            "auth_key_configured": bool(settings["auth_key"]),
        }

    def _headers(self) -> dict[str, str]:
        settings = self._settings()
        return {
            "Authorization": f"Bearer {settings['auth_key']}",
            "Accept": "application/json",
            "User-Agent": "any2api-chatgpt2api/0.1",
        }

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.start()
        if self._client is None:
            raise RuntimeError("chatgpt2api client is unavailable")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        data: list[tuple[str, str]] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    ) -> httpx.Response:
        settings = self._settings()
        if not settings["enabled"]:
            raise RuntimeError("chatgpt2api provider is disabled")
        if not settings["base_url"] or not settings["auth_key"]:
            raise RuntimeError("chatgpt2api provider is not configured")
        client = await self._ensure_client()
        return await client.request(
            method,
            f"{settings['base_url']}{path}",
            headers=self._headers(),
            json=json_body,
            data=data,
            files=files,
            timeout=httpx.Timeout(float(settings["timeout_sec"])),
        )

    async def request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> httpx.Response:
        return await self._request(method, path, json_body=json_body)

    def _error_response(self, message: str, *, status_code: int = 503) -> Response:
        return Response(
            content=orjson.dumps({"error": {"message": message, "type": "server_error"}}),
            status_code=status_code,
            media_type="application/json",
        )

    async def supported_model_ids(self, *, force_refresh: bool = False) -> tuple[str, ...]:
        if not self.is_configured():
            return ()

        now = time.time()
        if not force_refresh and self._models_cache and (now - self._models_fetched_at) < _MODELS_CACHE_TTL:
            return self._models_cache

        try:
            resp = await self._request("GET", "/v1/models")
            if resp.status_code < 200 or resp.status_code >= 300:
                logger.warning("chatgpt2api models request failed: status={}", resp.status_code)
                self._models_cache = ()
                self._models_fetched_at = now
                return ()
            payload = resp.json() if resp.content else {}
        except Exception as exc:
            logger.warning("chatgpt2api models request failed: error={}", exc)
            self._models_cache = ()
            self._models_fetched_at = now
            return ()

        items = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids: list[str] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                model_id = str(item.get("id") or "").strip()
                if model_id and model_id not in model_ids:
                    model_ids.append(model_id)

        self._models_cache = _normalize_chatgpt_model_ids(model_ids)
        self._models_fetched_at = now
        return self._models_cache

    async def summary(self) -> dict[str, Any]:
        payload = self.settings_payload()
        summary = {
            "enabled": payload["enabled"],
            "configured": payload["configured"],
            "base_url": payload["base_url"],
            "connected": False,
            "models": list(await self.supported_model_ids()) if payload["configured"] else [],
            "accounts": {"total": 0},
        }
        if not payload["enabled"]:
            return summary
        if not payload["configured"]:
            summary["error"] = "chatgpt2api provider is not configured"
            return summary

        try:
            login_resp = await self._request("POST", "/auth/login")
            summary["connected"] = 200 <= login_resp.status_code < 300
            if not summary["connected"]:
                summary["error"] = f"chatgpt2api auth failed: HTTP {login_resp.status_code}"
                return summary

            version_resp = await self._request("GET", "/version")
            if 200 <= version_resp.status_code < 300 and version_resp.content:
                version_payload = version_resp.json()
                if isinstance(version_payload, dict):
                    summary["version"] = version_payload.get("version")

            accounts_resp = await self._request("GET", "/api/accounts")
            if 200 <= accounts_resp.status_code < 300 and accounts_resp.content:
                accounts_payload = accounts_resp.json()
                items = accounts_payload.get("items", []) if isinstance(accounts_payload, dict) else []
                if isinstance(items, list):
                    summary["accounts"] = {"total": len(items)}
        except Exception as exc:
            summary["error"] = str(exc)
        return summary

    async def forward_json(self, path: str, body: dict[str, Any], *, method: str = "POST") -> Response:
        if not self.is_configured():
            return self._error_response("chatgpt2api provider is unavailable")
        try:
            resp = await self._request(method, path, json_body=body)
        except Exception as exc:
            logger.warning("chatgpt2api forward failed: path={} error={}", path, exc)
            return self._error_response(str(exc))
        media_type = resp.headers.get("content-type", "application/json").split(";", 1)[0]
        return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)

    async def forward_form(
        self,
        path: str,
        *,
        data: list[tuple[str, str]] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        method: str = "POST",
    ) -> Response:
        if not self.is_configured():
            return self._error_response("chatgpt2api provider is unavailable")
        try:
            resp = await self._request(method, path, data=data, files=files)
        except Exception as exc:
            logger.warning("chatgpt2api multipart forward failed: path={} error={}", path, exc)
            return self._error_response(str(exc))
        media_type = resp.headers.get("content-type", "application/json").split(";", 1)[0]
        return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)


def create_chatgpt2api_provider() -> ChatGPT2APIProvider:
    return ChatGPT2APIProvider()


__all__ = [
    "ChatGPT2APIProvider",
    "create_chatgpt2api_provider",
    "is_chatgpt_model_name",
    "_normalize_chatgpt_model_ids",
    "_chatgpt2api_settings_from_config",
]
