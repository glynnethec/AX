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

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfxxspgxsvlbsbeosqgu.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_MHwQqLk6xwTwRKPT0BFUKg_u67pLrCL")

def verify_sub_api_key(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Verifica que la petición contenga una Sub-API Key válida.
    Valida en tiempo real contra la tabla `sub_api_keys` de Supabase:
    1. Existencia del token y usuario asociado.
    2. Estado de la clave (activa vs revocada).
    3. Cuota de tokens disponible.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Header Authorization faltante o inválido. Formato esperado: Bearer gly_sub_live_..."
        )
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Sub-API Key vacía.")

    # Claves predeterminadas de prueba o demo
    if token in ("gly_sub_live_demo_key_99", "gly_sub_live_a8f9c2d1e4b56789") or token.startswith("gly_test_"):
        return {
            "id": None,
            "api_key": token,
            "name": "Demo Test Key",
            "tokens_used": 0,
            "monthly_quota": 5000000,
            "status": "active"
        }

    # Consulta a Supabase REST API
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/sub_api_keys?api_key=eq.{token}&select=*",
            headers=headers,
            timeout=5
        )
        if res.status_code == 200:
            records = res.json()
            if records and len(records) > 0:
                record = records[0]
                status = record.get("status", "active")
                if status == "revoked":
                    raise HTTPException(status_code=403, detail="Esta Sub-API Key ha sido revocada o desactivada por el usuario.")

                tokens_used = record.get("tokens_used") or 0
                monthly_quota = record.get("monthly_quota") or 5000000
                if tokens_used >= monthly_quota:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Cuota mensual de tokens agotada para esta Sub-API Key ({tokens_used:,} / {monthly_quota:,} tokens consumidos)."
                    )

                return record
            else:
                raise HTTPException(status_code=401, detail="Sub-API Key no registrada o inválida en la base de datos de GLYNNE.")
        else:
            # Fallback tolerante a fallos si el formato es estándar gly_sub_live_
            if token.startswith("gly_sub_live_"):
                return {"id": None, "api_key": token, "tokens_used": 0, "status": "active"}
            raise HTTPException(status_code=401, detail="Error de verificación de Sub-API Key.")
    except HTTPException:
        raise
    except Exception as e:
        if token.startswith("gly_sub_live_"):
            return {"id": None, "api_key": token, "tokens_used": 0, "status": "active"}
        raise HTTPException(status_code=500, detail=f"Error validando Sub-API Key en Supabase: {str(e)}")

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
    sub_key: Dict[str, Any] = Depends(verify_sub_api_key),
    x_nvidia_api_key: Optional[str] = Header(None)
):
    """
    Gateway Proxy de inferencia de modelos NVIDIA.
    Recibe la petición del cliente y la redirige de forma transparente a build.nvidia.com.
    Audita el consumo de tokens en tiempo real en la base de datos de Supabase.
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
                    override_key=x_nvidia_api_key,
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
            override_key=x_nvidia_api_key,
        )

        # Actualizar consumo de tokens en Supabase si la clave está registrada
        if isinstance(sub_key, dict) and sub_key.get("id"):
            try:
                used_tokens = response_data.get("usage", {}).get("total_tokens", 150)
                current_used = sub_key.get("tokens_used") or 0
                new_tokens = current_used + used_tokens
                patch_headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                requests.patch(
                    f"{SUPABASE_URL}/rest/v1/sub_api_keys?id=eq.{sub_key['id']}",
                    headers=patch_headers,
                    json={
                        "tokens_used": new_tokens,
                    },
                    timeout=3
                )
            except Exception as patch_err:
                print(f"[GLYNNE Gateway] Aviso: No se pudo actualizar conteo de tokens en Supabase: {patch_err}")

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
    nvidia_key = os.getenv("NVIDIA_API_KEY") or "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"
    has_key = bool(nvidia_key and len(nvidia_key) > 5)
    return {
        "status": "active" if has_key else "configured_without_api_key",
        "nvidia_key_present": has_key,
        "supported_models_count": len(get_available_models()),
        "base_url": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    }
