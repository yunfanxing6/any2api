"""WebUI chat API routes."""

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.control.model import registry as model_registry
from app.platform.auth.middleware import verify_webui_key
from app.products.openai.router import chat_completions_endpoint
from app.products.openai.schemas import ChatCompletionRequest

router = APIRouter(prefix="/webui/api", dependencies=[Depends(verify_webui_key)], tags=["WebUI - Chat"])


def _capability_name(spec) -> str:
    if spec.is_image_edit():
        return "image_edit"
    if spec.is_image():
        return "image"
    if spec.is_video():
        return "video"
    return "chat"


def _model_owner(model_name: str) -> str:
    lowered = str(model_name or "").strip().lower()
    if lowered.startswith("qwen"):
        return "qwen"
    if lowered.startswith("gpt-image-"):
        return "chatgpt2api"
    return "xai"


@router.get("/models")
async def list_webui_models(request: Request):
    models = [
        {
            "id": spec.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": _model_owner(spec.model_name),
            "name": spec.public_name,
            "capability": _capability_name(spec),
        }
        for spec in model_registry.list_enabled()
    ]

    qwen_provider = getattr(request.app.state, "qwen_provider", None)
    if qwen_provider is not None and qwen_provider.is_ready():
        for model_id, model_name in [
            ("qwen-image", "Qwen Image"),
            ("qwen-image-plus", "Qwen Image Plus"),
            ("qwen-image-turbo", "Qwen Image Turbo"),
        ]:
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen",
                    "name": model_name,
                    "capability": "image",
                }
            )
    return JSONResponse({"object": "list", "data": models})


@router.post("/chat/completions")
async def webui_chat_completions(req: ChatCompletionRequest):
    return await chat_completions_endpoint(req)


__all__ = ["router"]
