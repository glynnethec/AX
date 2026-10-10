from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json
from .nvidia_provider import NVIDIAProviderEngine, get_available_models

router = APIRouter(prefix="/api/v1", tags=["GLYNNE Models Provider (build.nvidia.com Gateway)"])

engine = NVIDIAProviderEngine()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    top_p: Optional[float] = None

def verify_sub_api_key(authorization: Optional[str] = Header(None)):
    """
    Verifica que la petición contenga una Sub-API Key válida.
    Acepta cualquier Bearer token configurado por el usuario o token live `gly_sub_live_*`.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Header Authorization faltante o inválido. Formato esperado: Bearer gly_sub_live_..."
        )
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Sub-API Key vacía.")
    return token

@router.options("/chat/completions")
async def chat_completions_options():
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )

@router.get("/models")
async def list_models():
    """
    Retorna el catálogo de modelos disponibles en formato OpenAI /v1/models.
    """
    raw_models = get_available_models()
    formatted_data = []
    for item in raw_models:
        formatted_data.append({
            "id": item["id"],
            "object": "model",
            "created": 1735689600,
            "owned_by": "axglynne-ai",
            "permission": [],
            "root": item["id"],
            "parent": None,
            "description": item["description"],
            "context_length": item["context_length"],
        })
    return {
        "object": "list",
        "data": formatted_data
    }

@router.post("/chat/completions")
async def create_chat_completion(
    req: ChatCompletionRequest,
    sub_api_key: str = Depends(verify_sub_api_key)
):
    """
    Gateway Proxy de inferencia de modelos NVIDIA.
    Recibe la petición del cliente y la redirige de forma transparente a build.nvidia.com.
    """
    messages_payload = [{"role": msg.role, "content": msg.content} for msg in req.messages]

    if req.stream:
        # Respuesta en Streaming SSE
        def event_generator():
            try:
                for chunk in engine.stream_chat_completion(
                    model=req.model,
                    messages=messages_payload,
                    temperature=req.temperature or 0.7,
                    max_tokens=req.max_tokens or 2048,
                    top_p=req.top_p,
                ):
                    yield chunk
            except Exception as e:
                err_payload = json.dumps({"error": str(e)})
                yield f"data: {err_payload}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Powered-By": "AXGLYNNE Model Provider Engine",
                "Server": "GLYNNE-AI-Gateway/1.0",
                "Access-Control-Allow-Origin": "*",
            }
        )

    # Respuesta Síncrona JSON
    try:
        response_data = engine.create_chat_completion(
            model=req.model,
            messages=messages_payload,
            temperature=req.temperature or 0.7,
            max_tokens=req.max_tokens or 2048,
            stream=False,
            top_p=req.top_p,
        )
        return JSONResponse(
            content=response_data,
            headers={
                "X-Powered-By": "AXGLYNNE Model Provider Engine",
                "Server": "GLYNNE-AI-Gateway/1.0",
                "Access-Control-Allow-Origin": "*",
            }
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en GLYNNE AI Gateway: {str(e)}")

@router.get("/provider/health")
async def provider_health():
    """
    Verifica el estado del módulo Models Provider y las credenciales del .env.
    """
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    has_key = bool(nvidia_key and len(nvidia_key) > 5)
    return {
        "status": "active" if has_key else "configured_without_api_key",
        "nvidia_key_present": has_key,
        "supported_models_count": len(get_available_models()),
        "base_url": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    }
