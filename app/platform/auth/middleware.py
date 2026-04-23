"""API-key authentication dependencies for FastAPI routes."""

import hmac

import orjson
from fastapi import Header, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.platform.config.snapshot import get_config
from .key_registry import registry

_security = HTTPBearer(auto_error=False, scheme_name="API Key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_keys() -> list[str]:
    raw = get_config("app.api_key", "")
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(k).strip() for k in raw if str(k).strip()]
    return [k.strip() for k in str(raw).split(",") if k.strip()]


def get_admin_key() -> str:
    """Return configured ``app.app_key`` (admin password)."""
    return str(get_config("app.app_key", "grok2api") or "")


def get_webui_key() -> str:
    """Return configured ``app.webui_key`` (webui access key)."""
    return str(get_config("app.webui_key", "") or "")


def get_webui_email() -> str:
    """Return configured ``app.webui_email``."""
    return str(get_config("app.webui_email", "") or "")


def get_webui_password() -> str:
    """Return configured ``app.webui_password``."""
    return str(get_config("app.webui_password", "") or "")


def get_effective_webui_token() -> str:
    """Return the bearer token accepted for WebUI APIs."""
    return get_webui_key() or get_admin_key()


def has_webui_credentials() -> bool:
    return bool(get_webui_email().strip() and get_webui_password())


def get_internal_key() -> str:
    """Return configured ``app.internal_api_key`` (service-to-service access key)."""
    return str(get_config("app.internal_api_key", "") or "")


def is_webui_enabled() -> bool:
    """Whether the webui entry is enabled."""
    val = get_config("app.webui_enabled", False)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

async def verify_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Validate Bearer token against configured ``api_key``."""
    allowed_keys = _get_keys()
    if not allowed_keys:
        return

    token = _extract_bearer(authorization)
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid Authorization header.")

    if not any(hmac.compare_digest(token, k) for k in allowed_keys):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid API key.")

    scope, provider = await _resolve_request_scope_provider(request)
    ok, message, record = await registry.authorize(token, provider=provider, scope=scope)
    if not ok:
        raise HTTPException(status.HTTP_403_FORBIDDEN, message or "API key access denied.")

    request.state.api_key_scope = scope
    request.state.api_key_provider = provider
    request.state.api_key_record = record.to_dict() if record is not None else None


async def verify_admin_key(
    authorization: str | None = Header(default=None),
    app_key: str | None = Query(default=None),
) -> None:
    """Validate Bearer token against ``app.app_key`` (admin access).

    Accepts either ``Authorization: Bearer <key>`` header or ``?app_key=<key>``
    query parameter (the latter is needed for EventSource which cannot send headers).
    """
    key = get_admin_key()
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Admin key is not configured.")

    token = _extract_bearer(authorization) or app_key
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing authentication token.")

    if not hmac.compare_digest(token, key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication token.")


async def verify_webui_key(
    authorization: str | None = Header(default=None),
) -> None:
    """Validate Bearer token for webui endpoints."""
    webui_key = get_effective_webui_token()

    if not webui_key:
        if is_webui_enabled():
            return
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "WebUI access is disabled.")

    token = _extract_bearer(authorization)
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing authentication token.")

    if not hmac.compare_digest(token, webui_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication token.")


async def verify_internal_key(
    authorization: str | None = Header(default=None),
    x_internal_key: str | None = Header(default=None, alias="X-Internal-Key"),
) -> None:
    """Validate the service-to-service key for internal sync endpoints."""
    internal_key = get_internal_key()
    allowed_keys = []
    if internal_key:
        allowed_keys.append(internal_key)
    allowed_keys.extend(_get_keys())
    if not allowed_keys:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Internal API key is not configured.")

    token = _extract_bearer(authorization) or x_internal_key
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing authentication token.")

    if not any(hmac.compare_digest(token, key) for key in allowed_keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid authentication token.")


def _infer_provider_from_model(model: str | None) -> str | None:
    lowered = str(model or "").strip().lower()
    if not lowered:
        return None
    if lowered.startswith("grok"):
        return "grok"
    if lowered.startswith("qwen"):
        return "qwen"
    if lowered.startswith("gpt-image-"):
        return "chatgpt2api"

    aliases = get_config("providers.qwen.model_aliases", {})
    if isinstance(aliases, dict):
        for key, target in aliases.items():
            if lowered == str(key).strip().lower() and str(target).strip().lower().startswith("qwen"):
                return "qwen"
    return None


async def _extract_model_from_request(request: Request) -> str | None:
    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    try:
        if content_type == "application/json":
            body = await request.body()
            if not body:
                return None
            data = orjson.loads(body)
            if isinstance(data, dict):
                return str(data.get("model") or "").strip() or None
        if content_type in {"multipart/form-data", "application/x-www-form-urlencoded"} or content_type.startswith("multipart/"):
            form = await request.form()
            model = form.get("model")
            return str(model).strip() if model else None
    except Exception:
        return None
    return None


async def _resolve_request_scope_provider(request: Request) -> tuple[str | None, str | None]:
    path = request.url.path
    scope: str | None = None
    if path.endswith("/v1/models") or "/v1/models/" in path:
        scope = "models"
    elif path.endswith("/v1/files") or "/v1/files/" in path:
        scope = "files"
        provider = "qwen"
        return scope, provider
    elif path.endswith("/chat/completions"):
        scope = "chat"
    elif path.endswith("/responses"):
        scope = "responses"
    elif path.endswith("/images/generations") or path.endswith("/images/edits"):
        scope = "images"
    elif path.endswith("/videos") or "/videos/" in path:
        scope = "videos"
    elif path.endswith("/messages"):
        scope = "anthropic"

    model = await _extract_model_from_request(request)
    provider = _infer_provider_from_model(model)
    return scope, provider

__all__ = [
    "verify_api_key",
    "verify_admin_key",
    "verify_webui_key",
    "verify_internal_key",
    "get_admin_key",
    "get_webui_key",
    "get_webui_email",
    "get_webui_password",
    "get_effective_webui_token",
    "has_webui_credentials",
    "get_internal_key",
    "is_webui_enabled",
]
