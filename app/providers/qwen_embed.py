"""Embedded qwen2API runtime and internal proxy helpers."""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import orjson
from fastapi.responses import Response, StreamingResponse

from app.platform.config.snapshot import get_config
from app.platform.logging.logger import logger
from app.platform.paths import data_path


def is_qwen_model_name(model_name: str) -> bool:
    return (model_name or "").strip().lower().startswith("qwen")


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_qwen_root() -> Path:
    embedded = Path(__file__).resolve().parents[2] / "embedded_qwen"
    if embedded.is_dir():
        return embedded
    return Path(__file__).resolve().parents[3] / "qwen2API"


def _resolve_qwen_root() -> Path | None:
    raw = os.getenv("ANY2API_QWEN2API_PATH", "").strip()
    candidate = Path(raw).expanduser().resolve() if raw else _default_qwen_root().resolve()
    backend_dir = candidate / "backend"
    if backend_dir.is_dir() and (backend_dir / "main.py").is_file():
        return candidate
    return None


def _resolve_embedded_admin_key(explicit_key: str | None = None) -> str:
    if explicit_key:
        return explicit_key
    for env_name in ("ANY2API_INTERNAL_API_KEY", "ANY2API_API_KEY", "ANY2API_APP_KEY"):
        raw = os.getenv(env_name, "").strip()
        if raw:
            return raw
    return "any2api-internal"


def _qwen_runtime_settings_from_config() -> dict[str, Any]:
    model_aliases = get_config("providers.qwen.model_aliases", {})
    if not isinstance(model_aliases, dict):
        model_aliases = {}
    return {
        "enabled": bool(get_config("providers.qwen.enabled", True)),
        "engine_mode": str(get_config("providers.qwen.engine_mode", os.getenv("ANY2API_QWEN_ENGINE_MODE", "httpx")) or "httpx"),
        "browser_pool_size": int(get_config("providers.qwen.browser_pool_size", 2) or 2),
        "max_inflight_per_account": int(get_config("providers.qwen.max_inflight_per_account", 1) or 1),
        "stream_keepalive_interval": int(get_config("providers.qwen.stream_keepalive_interval", 5) or 5),
        "register_secret": str(get_config("providers.qwen.register_secret", "") or ""),
        "native_tool_passthrough": bool(get_config("providers.qwen.native_tool_passthrough", True)),
        "account_min_interval_ms": int(get_config("providers.qwen.account_min_interval_ms", 1200) or 1200),
        "request_jitter_min_ms": int(get_config("providers.qwen.request_jitter_min_ms", 120) or 120),
        "request_jitter_max_ms": int(get_config("providers.qwen.request_jitter_max_ms", 360) or 360),
        "rate_limit_base_cooldown": int(get_config("providers.qwen.rate_limit_base_cooldown", 600) or 600),
        "rate_limit_max_cooldown": int(get_config("providers.qwen.rate_limit_max_cooldown", 3600) or 3600),
        "model_aliases": dict(model_aliases),
    }


def _qwen_data_env(root: Path) -> dict[str, str]:
    data_dir = data_path("qwen")
    generated_dir = data_dir / "context_files"
    data_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ACCOUNTS_FILE": str(data_dir / "accounts.json"),
        "USERS_FILE": str(data_dir / "users.json"),
        "CAPTURES_FILE": str(data_dir / "captures.json"),
        "CONFIG_FILE": str(data_dir / "config.json"),
        "CONTEXT_CACHE_FILE": str(data_dir / "context_cache.json"),
        "UPLOADED_FILES_FILE": str(data_dir / "uploaded_files.json"),
        "CONTEXT_AFFINITY_FILE": str(data_dir / "session_affinity.json"),
        "CONTEXT_GENERATED_DIR": str(generated_dir),
    }


@dataclass(slots=True)
class EmbeddedQwenProvider:
    root: Path
    app: Any
    internal_key: str
    config_module: Any | None = None
    enabled: bool = True
    started: bool = False
    error: str = ""
    _lifespan: AbstractAsyncContextManager[Any] | None = None
    _client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self.started:
            return
        self._lifespan = self.app.router.lifespan_context(self.app)
        await self._lifespan.__aenter__()
        transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
        self._client = httpx.AsyncClient(transport=transport, base_url="http://qwen.internal")
        self.started = True
        self.error = ""

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._lifespan is not None:
            await self._lifespan.__aexit__(None, None, None)
            self._lifespan = None
        self.started = False

    def is_ready(self) -> bool:
        return self.enabled and self.started and self._client is not None and not self.error

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.internal_key}",
            "X-Internal-Key": self.internal_key,
            "Accept": "application/json",
            "User-Agent": "any2api-qwen-embed/0.1",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        content: bytes | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        if self._client is None or not self.started:
            raise RuntimeError("Embedded qwen provider is not started")
        headers = self._headers()
        if extra_headers:
            headers.update({key: value for key, value in extra_headers.items() if value is not None})
        kwargs: dict[str, Any] = {"headers": headers}
        if json_body is not None:
            kwargs["json"] = json_body
        if content is not None:
            kwargs["content"] = content
        return await self._client.request(method, path, **kwargs)

    async def request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> httpx.Response:
        if not self.is_ready():
            raise RuntimeError(self.error or "qwen provider unavailable")
        return await self._request(method, path, json_body=json_body)

    def settings_payload(self) -> dict[str, Any]:
        payload = _qwen_runtime_settings_from_config()
        payload["enabled"] = self.enabled
        if self.config_module is not None:
            settings = self.config_module.settings
            payload["version"] = getattr(self.config_module, "VERSION", "2.0.0")
            payload["max_inflight_per_account"] = int(getattr(settings, "MAX_INFLIGHT_PER_ACCOUNT", payload["max_inflight_per_account"]))
            payload["account_min_interval_ms"] = int(getattr(settings, "ACCOUNT_MIN_INTERVAL_MS", payload["account_min_interval_ms"]))
            payload["request_jitter_min_ms"] = int(getattr(settings, "REQUEST_JITTER_MIN_MS", payload["request_jitter_min_ms"]))
            payload["request_jitter_max_ms"] = int(getattr(settings, "REQUEST_JITTER_MAX_MS", payload["request_jitter_max_ms"]))
            payload["rate_limit_base_cooldown"] = int(getattr(settings, "RATE_LIMIT_BASE_COOLDOWN", payload["rate_limit_base_cooldown"]))
            payload["rate_limit_max_cooldown"] = int(getattr(settings, "RATE_LIMIT_MAX_COOLDOWN", payload["rate_limit_max_cooldown"]))
            payload["model_aliases"] = dict(getattr(self.config_module, "MODEL_MAP", payload["model_aliases"]))
        else:
            payload["version"] = "2.0.0"
        return payload

    async def apply_runtime_settings(self, runtime_settings: dict[str, Any]) -> dict[str, Any]:
        if self.config_module is None:
            raise RuntimeError("embedded qwen config is unavailable")

        runtime_settings = dict(runtime_settings or {})
        model_aliases = runtime_settings.get("model_aliases")
        if not isinstance(model_aliases, dict):
            model_aliases = dict(getattr(self.config_module, "MODEL_MAP", {}))

        settings = self.config_module.settings
        settings.MAX_INFLIGHT_PER_ACCOUNT = int(runtime_settings.get("max_inflight_per_account", getattr(settings, "MAX_INFLIGHT_PER_ACCOUNT", 1)) or 1)
        settings.ACCOUNT_MIN_INTERVAL_MS = int(runtime_settings.get("account_min_interval_ms", getattr(settings, "ACCOUNT_MIN_INTERVAL_MS", 0)) or 0)
        settings.REQUEST_JITTER_MIN_MS = int(runtime_settings.get("request_jitter_min_ms", getattr(settings, "REQUEST_JITTER_MIN_MS", 0)) or 0)
        settings.REQUEST_JITTER_MAX_MS = int(runtime_settings.get("request_jitter_max_ms", getattr(settings, "REQUEST_JITTER_MAX_MS", 0)) or 0)
        settings.RATE_LIMIT_BASE_COOLDOWN = int(runtime_settings.get("rate_limit_base_cooldown", getattr(settings, "RATE_LIMIT_BASE_COOLDOWN", 600)) or 600)
        settings.RATE_LIMIT_MAX_COOLDOWN = int(runtime_settings.get("rate_limit_max_cooldown", getattr(settings, "RATE_LIMIT_MAX_COOLDOWN", 3600)) or 3600)
        settings.RATE_LIMIT_COOLDOWN = settings.RATE_LIMIT_BASE_COOLDOWN

        self.config_module.MODEL_MAP.clear()
        self.config_module.MODEL_MAP.update(model_aliases)

        if getattr(getattr(self.app, "state", None), "account_pool", None) is not None:
            if hasattr(self.app.state.account_pool, "set_max_inflight"):
                self.app.state.account_pool.set_max_inflight(settings.MAX_INFLIGHT_PER_ACCOUNT)
            else:
                self.app.state.account_pool.max_inflight = settings.MAX_INFLIGHT_PER_ACCOUNT

        return {
            "ok": True,
            "restarted": False,
            "settings": self.settings_payload(),
        }

    async def summary(self) -> dict[str, Any]:
        if not self.is_ready():
            return {
                "enabled": self.enabled,
                "connected": False,
                "error": self.error or "qwen provider unavailable",
            }
        resp = await self._request("GET", "/api/admin/status")
        data = resp.json() if resp.content else {}
        accounts = data.get("accounts", {}) if isinstance(data, dict) else {}
        return {
            "enabled": True,
            "connected": 200 <= resp.status_code < 300,
            "engine_mode": self.settings_payload().get("engine_mode"),
            "accounts": accounts,
            "browser_engine": data.get("browser_engine") or data.get("browser_automation"),
            "httpx_engine": data.get("httpx_engine") or data.get("request_runtime"),
            "hybrid_engine": data.get("hybrid_engine"),
        }

    async def accounts(self) -> list[dict[str, Any]]:
        if not self.is_ready():
            return []
        resp = await self._request("GET", "/api/admin/accounts")
        if resp.status_code < 200 or resp.status_code >= 300:
            return []
        data = resp.json() if resp.content else {}
        items = data.get("accounts", []) if isinstance(data, dict) else []
        return items if isinstance(items, list) else []

    async def forward_json(self, path: str, body: dict[str, Any]) -> Response:
        if not self.is_ready():
            return Response(
                content=orjson.dumps({"error": {"message": self.error or "qwen provider unavailable", "type": "server_error"}}),
                status_code=503,
                media_type="application/json",
            )
        resp = await self._request("POST", path, json_body=body)
        media_type = resp.headers.get("content-type", "application/json").split(";", 1)[0]
        return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)

    async def forward_content(
        self,
        method: str,
        path: str,
        *,
        content: bytes | None = None,
        content_type: str | None = None,
        accept: str | None = None,
    ) -> Response:
        if not self.is_ready():
            return Response(
                content=orjson.dumps({"error": {"message": self.error or "qwen provider unavailable", "type": "server_error"}}),
                status_code=503,
                media_type="application/json",
            )
        extra_headers: dict[str, str] = {}
        if content_type:
            extra_headers["Content-Type"] = content_type
        if accept:
            extra_headers["Accept"] = accept
        resp = await self._request(method, path, content=content, extra_headers=extra_headers)
        media_type = resp.headers.get("content-type", "application/json").split(";", 1)[0]
        return Response(content=resp.content, status_code=resp.status_code, media_type=media_type)

    def forward_stream(self, path: str, body: dict[str, Any]) -> StreamingResponse:
        if not self.is_ready() or self._client is None:
            async def _error_stream():
                payload = orjson.dumps({"error": {"message": self.error or "qwen provider unavailable", "type": "server_error"}}).decode()
                yield f"event: error\ndata: {payload}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_error_stream(), media_type="text/event-stream")

        async def _gen():
            async with self._client.stream("POST", path, headers=self._headers(), json=body) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    content = await resp.aread()
                    payload = content.decode("utf-8", errors="replace")
                    if payload.strip().startswith("{"):
                        yield f"event: error\ndata: {payload}\n\n"
                    else:
                        wrapped = orjson.dumps({"error": {"message": payload or f"HTTP {resp.status_code}", "type": "server_error"}}).decode()
                        yield f"event: error\ndata: {wrapped}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                async for chunk in resp.aiter_text():
                    if chunk:
                        yield chunk

        return StreamingResponse(_gen(), media_type="text/event-stream")


def create_embedded_qwen_provider(*, internal_key: str | None = None) -> EmbeddedQwenProvider | None:
    runtime_settings = _qwen_runtime_settings_from_config()
    if not runtime_settings.get("enabled", True) and not _bool_env("ANY2API_QWEN_ENABLED", False):
        return None

    root = _resolve_qwen_root()
    if root is None:
        logger.warning("embedded qwen provider disabled: qwen2API checkout not found")
        return None

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    internal_key = _resolve_embedded_admin_key(internal_key)
    os.environ["ENGINE_MODE"] = str(runtime_settings.get("engine_mode", os.getenv("ANY2API_QWEN_ENGINE_MODE", "httpx")))
    os.environ["WORKERS"] = "1"
    os.environ["BROWSER_POOL_SIZE"] = str(runtime_settings.get("browser_pool_size", 2))
    os.environ["MAX_INFLIGHT"] = str(runtime_settings.get("max_inflight_per_account", 1))
    os.environ["ACCOUNT_MIN_INTERVAL_MS"] = str(runtime_settings.get("account_min_interval_ms", 1200))
    os.environ["REQUEST_JITTER_MIN_MS"] = str(runtime_settings.get("request_jitter_min_ms", 120))
    os.environ["REQUEST_JITTER_MAX_MS"] = str(runtime_settings.get("request_jitter_max_ms", 360))
    os.environ["RATE_LIMIT_BASE_COOLDOWN"] = str(runtime_settings.get("rate_limit_base_cooldown", 600))
    os.environ["RATE_LIMIT_MAX_COOLDOWN"] = str(runtime_settings.get("rate_limit_max_cooldown", 3600))
    os.environ["ADMIN_KEY"] = internal_key
    os.environ["REGISTER_SECRET"] = str(runtime_settings.get("register_secret", "") or "")
    for env_name, env_value in _qwen_data_env(root).items():
        os.environ[env_name] = env_value

    try:
        config_module = importlib.import_module("backend.core.config")
        main_module = importlib.import_module("backend.main")
    except Exception as exc:
        logger.exception("failed to import embedded qwen provider: error={}", exc)
        provider = EmbeddedQwenProvider(root=root, app=None, internal_key=internal_key, config_module=None, enabled=True, error=str(exc))  # type: ignore[arg-type]
        return provider

    try:
        config_module.settings.ADMIN_KEY = internal_key
        config_module.API_KEYS.add(internal_key)
    except Exception as exc:
        logger.warning("failed to align embedded qwen auth: error={}", exc)

    provider = EmbeddedQwenProvider(root=root, app=main_module.app, internal_key=internal_key, config_module=config_module)
    return provider


__all__ = ["EmbeddedQwenProvider", "create_embedded_qwen_provider", "is_qwen_model_name", "_qwen_runtime_settings_from_config"]
