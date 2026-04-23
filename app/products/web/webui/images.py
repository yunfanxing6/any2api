"""WebUI image generation and edit routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import JSONResponse

from app.platform.auth.middleware import verify_webui_key
from app.platform.errors import AppError
from app.platform.errors import ValidationError
from app.platform.logging.logger import logger
from app.products.openai.router import image_edits as openai_image_edits
from app.products.openai.router import image_generations as openai_image_generations
from app.products.openai.schemas import ImageGenerationRequest


router = APIRouter(prefix="/webui/api", dependencies=[Depends(verify_webui_key)], tags=["WebUI - Images"])


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_app_error(exc: BaseException) -> AppError | None:
    if isinstance(exc, AppError):
        return exc
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, tuple):
        for item in nested:
            found = _extract_app_error(item)
            if found is not None:
                return found
    return None


@router.post("/images/generations")
async def webui_image_generations(req: ImageGenerationRequest, request: Request):
    try:
        return await openai_image_generations(req, request)
    except AppError as exc:
        return JSONResponse(exc.to_dict(), status_code=exc.status)
    except Exception as exc:
        extracted = _extract_app_error(exc)
        if extracted is not None:
            return JSONResponse(extracted.to_dict(), status_code=extracted.status)
        logger.exception("webui image generation failed: error={}", exc)
        return JSONResponse(
            {"error": {"message": "Internal server error", "type": "server_error"}},
            status_code=500,
        )


@router.post("/images/edits")
async def webui_image_edits(request: Request):
    form = await request.form()

    model = str(form.get("model") or "").strip()
    prompt = str(form.get("prompt") or "").strip()
    size = str(form.get("size") or "1024x1024").strip() or "1024x1024"
    response_format = str(form.get("response_format") or "b64_json").strip() or "b64_json"
    n = _coerce_int(form.get("n"), default=1)

    image_items = [
        item
        for item in [*form.getlist("image"), *form.getlist("image[]")]
        if hasattr(item, "read")
    ]
    if not image_items:
        raise ValidationError("image is required", param="image")

    mask_item = form.get("mask")
    mask = mask_item if hasattr(mask_item, "read") else None

    try:
        return await openai_image_edits(
            model=model,
            prompt=prompt,
            image=image_items,
            mask=mask,
            n=n,
            size=size,
            response_format=response_format,
            request=request,
        )
    except AppError as exc:
        return JSONResponse(exc.to_dict(), status_code=exc.status)
    except Exception as exc:
        extracted = _extract_app_error(exc)
        if extracted is not None:
            return JSONResponse(extracted.to_dict(), status_code=extracted.status)
        logger.exception("webui image edit failed: error={}", exc)
        return JSONResponse(
            {"error": {"message": "Internal server error", "type": "server_error"}},
            status_code=500,
        )


__all__ = ["router"]
