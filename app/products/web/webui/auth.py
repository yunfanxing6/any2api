"""WebUI login endpoint for email/password auth."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.platform.auth.middleware import (
    get_effective_webui_token,
    get_webui_email,
    get_webui_password,
    has_webui_credentials,
)


router = APIRouter(prefix="/webui/api", tags=["WebUI - System"])


class WebUILoginRequest(BaseModel):
    email: str = ""
    password: str = ""


@router.post("/login")
async def webui_login(req: WebUILoginRequest):
    configured_email = get_webui_email().strip()
    configured_password = get_webui_password()
    token = get_effective_webui_token()

    if not has_webui_credentials():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "WebUI email/password login is not configured.")

    email = str(req.email or "").strip()
    password = str(req.password or "")
    if not email or not password:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "email and password are required")

    if not hmac.compare_digest(email, configured_email) or not hmac.compare_digest(password, configured_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    return {"ok": True, "token": token, "email": configured_email}


__all__ = ["router"]
