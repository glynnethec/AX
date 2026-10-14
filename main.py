import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import base64
import os
import json
import time
import asyncio
import re
import subprocess
import tempfile
import threading
from elevenlabs.client import ElevenLabs


from AX_Chat.AX_Agent import run_ax_agent
from AX_Voice.AX_Voice_Agent import run_ax_voice_agent, llm, SYSTEM_PROMPT, SYSTEM_PROMPT_EN
from AX_Voice.AX_Action_Agent import run_ax_action_agent_async
from AX_Trainer.AX_Dataset_Generator import generate_dataset_with_groq, extract_text_from_file_bytes
from AX_LibraryModel.AX_HuggingFace_Bridge import get_curated_open_weights_catalog, search_huggingface_models, get_huggingface_model_files, HF_RESOLVE_BASE
from models_provider.router import router as models_provider_router
from core.auth import AuthUser, get_current_user, resolve_user_id
from core.rate_limit import check_rate_limit
from core.jobs import training_jobs, get_modal_cmd, safe_user_id, user_model_dir, user_downloads_dir, MODAL_VOLUME, MODAL_APP_NAME, BASE_DIR
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Al apagar el servidor: cancelar entrenamientos en curso para no dejar GPUs huérfanas en Modal
    training_jobs.shutdown()


IS_PRODUCTION = os.getenv("ENVIRONMENT", "").lower() == "production" or bool(os.getenv("RENDER"))

app = FastAPI(
    title="AX Glynne Core",
    version="1.0.0",
    lifespan=lifespan,
    # La documentación interactiva no se expone públicamente en producción
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://axglynne.com",
        "https://www.axglynne.com",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    # Regex anclada: solo subdominios reales de estos dominios (antes aceptaba p.ej. axglynne.com.evil.com)
    allow_origin_regex=os.getenv(
        "CORS_ORIGIN_REGEX",
        r"^https://([a-z0-9-]+\.)*(axglynne\.com|vercel\.app|onrender\.com)$",
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Compresión de respuestas grandes (logs de entrenamiento, catálogos, audio base64)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


@app.get("/")
@app.get("/health")
async def root_health():
    return {"status": "ok", "service": "AX Glynne Core", "version": "1.0.0"}

# Integración del nuevo módulo Models Provider (build.nvidia.com Gateway)
app.include_router(models_provider_router)


class MessageModel(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[MessageModel]
    use_mock_tts: bool = False
    user_id: Optional[str] = "default_user"
    language: Optional[str] = "es"

MessageModel.model_rebuild()
ChatRequest.model_rebuild()

USAGE_FILE = "tts_usage.json"
MAX_CHARS = 2000
RESET_SECONDS = 48 * 3600
_usage_lock = threading.Lock()

def load_all_tts_usage() -> dict:
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def save_all_tts_usage(all_data: dict):
    # Escritura atómica: nunca deja el archivo a medias si dos peticiones coinciden
    try:
        directory = os.path.dirname(os.path.abspath(USAGE_FILE))
        fd, tmp_path = tempfile.mkstemp(prefix=".tts_usage_", dir=directory)
        with os.fdopen(fd, "w") as f:
            json.dump(all_data, f)
        os.replace(tmp_path, USAGE_FILE)
    except Exception as e:
        print(f"Error saving TTS usage: {e}")

def get_user_tts_usage(user_id: str) -> dict:
    with _usage_lock:
        all_data = load_all_tts_usage()
        u_data = all_data.get(user_id, {})
        current_time = time.time()

        start_time = u_data.get("start_time", current_time)
        chars_used = u_data.get("chars_used", 0)

        if current_time - start_time > RESET_SECONDS:
            start_time = current_time
            chars_used = 0
            all_data[user_id] = {"start_time": start_time, "chars_used": chars_used}
            save_all_tts_usage(all_data)

        return {"start_time": start_time, "chars_used": chars_used}

def update_user_tts_usage(user_id: str, new_chars_used: int, start_time: float):
    with _usage_lock:
        all_data = load_all_tts_usage()
        all_data[user_id] = {
            "start_time": start_time,
            "chars_used": new_chars_used
        }
        save_all_tts_usage(all_data)

def split_into_sentences(text: str) -> List[str]:
    """Divide el texto en oraciones naturales para TTS."""
    parts = re.split(r'(?<=[.!?…])\s+|(?<=\.\.\.)\s*', text)
    return [p.strip() for p in parts if p.strip()]

async def tts_edge_sentence(sentence: str, language: str = "es") -> bytes:
    """Genera audio para una sola oración con edge_tts."""
    import edge_tts
    voice_name = "en-US-ChristopherNeural" if language == "en" else "es-CO-SalomeNeural"
    communicate = edge_tts.Communicate(sentence, voice_name, rate="+20%", volume="+5%")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

async def tts_elevenlabs_sentence(sentence: str, api_key: str, language: str = "es") -> bytes:
    """Genera audio para una sola oración con ElevenLabs (en executor para no bloquear)."""
    def _sync():
        client = ElevenLabs(api_key=api_key)
        voice_id = "ut2XM2wJyIZLTtW6lFzZ" if language == "en" else "VmejBeYhbrcTPwDniox7"
        gen = client.text_to_speech.convert(
            text=sentence,
            voice_id=voice_id,
            model_id="eleven_turbo_v2_5",  # modelo más rápido de ElevenLabs
        )
        return b"".join(gen)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync)


class ClientDisconnected(Exception):
    """El usuario abandonó la página: se detiene el procesamiento para no gastar recursos."""


async def ensure_connected(http_request: Request):
    if await http_request.is_disconnected():
        raise ClientDisconnected()


@app.post("/api/chat")
async def process_chat(request: ChatRequest, http_request: Request, user: AuthUser = Depends(get_current_user)):
    user_id = resolve_user_id(user, request.user_id)
    check_rate_limit(f"chat:{user_id}", limit=30, window_seconds=60)
    try:
        langchain_msgs = []
        for msg in request.messages:
            role = msg.role.lower()
            if role == "user":
                langchain_msgs.append(HumanMessage(content=msg.content))
            elif role in ["ai", "assistant"]:
                langchain_msgs.append(AIMessage(content=msg.content))

        if not langchain_msgs:
            raise HTTPException(status_code=400, detail="El historial está vacío.")

        # El agente es síncrono: se ejecuta fuera del event loop para no bloquear a otros usuarios
        ai_response = await run_in_threadpool(run_ax_agent, langchain_msgs)
        return {"status": "success", "reply": ai_response}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")

@app.post("/api/voice_chat")
async def process_voice_chat(request: ChatRequest, http_request: Request, user: AuthUser = Depends(get_current_user)):
    user_id = resolve_user_id(user, request.user_id)
    check_rate_limit(f"voice:{user_id}", limit=30, window_seconds=60)
    action_task = None
    try:
        langchain_msgs = []
        user_message_text = ""
        for msg in request.messages:
            role = msg.role.lower()
            if role == "user":
                langchain_msgs.append(HumanMessage(content=msg.content))
                user_message_text = msg.content
            elif role in ["ai", "assistant"]:
                langchain_msgs.append(AIMessage(content=msg.content))

        if not langchain_msgs:
            raise HTTPException(status_code=400, detail="El historial está vacío.")

        # ── 1. STREAMING DEL LLM ────────────────────────────────────────────
        # Limitar historial
        MAX_HISTORY = 6
        recent_history = langchain_msgs[-MAX_HISTORY:] if len(langchain_msgs) > MAX_HISTORY else langchain_msgs
        prompt_to_use = SYSTEM_PROMPT_EN if request.language == "en" else SYSTEM_PROMPT
        messages = [SystemMessage(content=prompt_to_use)] + recent_history

        # Acumular texto en oraciones completas mientras hace stream
        full_text = ""
        sentence_buffer = ""
        sentences: List[str] = []

        async for chunk in llm.astream(messages):
            token = chunk.content
            full_text += token
            sentence_buffer += token
            # Detectar fin de oración natural
            if re.search(r'[.!?…]\s*$', sentence_buffer) or '...' in sentence_buffer:
                candidate = sentence_buffer.strip()
                if len(candidate) > 3:
                    sentences.append(candidate)
                sentence_buffer = ""

        # Añadir el buffer restante si no terminó en puntuación
        if sentence_buffer.strip() and len(sentence_buffer.strip()) > 3:
            sentences.append(sentence_buffer.strip())

        ai_response = full_text.strip()
        if not sentences:
            sentences = [ai_response] if ai_response else ["Entendido."]

        # Si el usuario ya salió de AX_voice, no se genera audio (ahorra cuota de ElevenLabs)
        await ensure_connected(http_request)

        # Lanzar el Agente de Acción en paralelo para evaluar intenciones de UI
        action_task = asyncio.create_task(run_ax_action_agent_async(user_message_text, ai_response))

        # ── 2. TTS CONCURRENTE POR ORACIÓN (PER USER) ──────────────────────
        use_mock_tts = request.use_mock_tts
        user_usage = get_user_tts_usage(user_id)
        audio_data = b""

        if not use_mock_tts:
            response_len = len(ai_response)
            eleven_api_key = os.getenv("ELEVENLABS_API_KEY")

            if eleven_api_key and user_usage["chars_used"] + response_len <= MAX_CHARS:
                try:
                    # Generar audio de todas las oraciones EN PARALELO
                    tasks = [tts_elevenlabs_sentence(s, eleven_api_key, request.language) for s in sentences]
                    audio_chunks = await asyncio.gather(*tasks)
                    audio_data = b"".join(audio_chunks)

                    user_usage["chars_used"] += response_len
                    update_user_tts_usage(user_id, user_usage["chars_used"], user_usage["start_time"])
                except Exception as e:
                    print(f"Error con ElevenLabs paralelo, usando edge_tts: {e}")
                    use_mock_tts = True
            else:
                if not eleven_api_key:
                    print("Falta ELEVENLABS_API_KEY. Usando edge_tts.")
                else:
                    print(f"Límite de ElevenLabs superado para usuario {safe_user_id(user_id)[:8]}…. Usando edge_tts.")
                use_mock_tts = True

        # Fallback edge_tts: también concurrente por oraciones
        if use_mock_tts:
            await ensure_connected(http_request)
            try:
                tasks = [tts_edge_sentence(s, request.language) for s in sentences]
                audio_chunks = await asyncio.gather(*tasks)
                audio_data = b"".join(audio_chunks)
            except Exception as e:
                print(f"Error con edge_tts paralelo: {e}")
                # Último fallback: edge_tts sobre texto completo
                import edge_tts
                voice_name = "en-US-ChristopherNeural" if request.language == "en" else "es-CO-SalomeNeural"
                communicate = edge_tts.Communicate(ai_response, voice_name, rate="+20%", volume="+5%")
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]

        # Esperar resultado del Action Agent
        url_to_open = await action_task

        # ── 3. RESPUESTA ─────────────────────────────────────────────────────
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        used_engine = "edge" if use_mock_tts else "elevenlabs"

        user_usage = get_user_tts_usage(user_id)
        current_time = time.time()
        time_until_reset = max(0.0, RESET_SECONDS - (current_time - user_usage["start_time"]))
        hours_until_reset = round(time_until_reset / 3600.0, 1)
        available_chars = max(0, MAX_CHARS - user_usage["chars_used"])

        return {
            "status": "success",
            "reply": ai_response,
            "audio_base64": audio_base64,
            "used_engine": used_engine,
            "chars_used": user_usage["chars_used"],
            "max_chars": MAX_CHARS,
            "available_chars": available_chars,
            "hours_until_reset": hours_until_reset,
            "url_to_open": url_to_open
        }
    except ClientDisconnected:
        if action_task and not action_task.done():
            action_task.cancel()
        # 499: el cliente cerró la conexión (nadie recibirá esta respuesta)
        raise HTTPException(status_code=499, detail="Cliente desconectado.")
    except HTTPException:
        raise
    except Exception as e:
        if action_task and not action_task.done():
            action_task.cancel()
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")

@app.get("/api/tts_status")
async def get_tts_status(user_id: Optional[str] = "default_user", user: AuthUser = Depends(get_current_user)):
    user_id = resolve_user_id(user, user_id)
    user_usage = get_user_tts_usage(user_id)
    current_time = time.time()

    eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
    chars_used = user_usage.get("chars_used", 0)
    available_chars = max(0, MAX_CHARS - chars_used)
    active_engine = "elevenlabs" if (eleven_api_key and chars_used < MAX_CHARS) else "edge"

    time_until_reset = max(0.0, RESET_SECONDS - (current_time - user_usage.get("start_time", current_time)))
    hours_until_reset = round(time_until_reset / 3600.0, 1)

    return {
        "status": "success",
        "used_engine": active_engine,
        "chars_used": chars_used,
        "max_chars": MAX_CHARS,
        "available_chars": available_chars,
        "hours_until_reset": hours_until_reset
    }

# ── ENDPOINTS DE GENERACIÓN Y ENTRENAMIENTO QLORA ──────────────────────────────

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB por archivo de conocimiento
MAX_DATASET_EXAMPLES = 200

class GenerateDatasetRequest(BaseModel):
    personality_info: Optional[str] = ""
    business_info: Optional[str] = ""
    num_examples: Optional[int] = 25

def _clamp_examples(n: Optional[int]) -> int:
    return max(1, min(int(n or 25), MAX_DATASET_EXAMPLES))

async def _read_upload_limited(upload: UploadFile) -> bytes:
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"El archivo {upload.filename} supera el límite de 10 MB.")
    return content

@app.post("/api/train/generate_dataset")
async def generate_dataset_endpoint(
    personality_text: Optional[str] = Form(""),
    business_text: Optional[str] = Form(""),
    num_examples: Optional[int] = Form(25),
    personality_file: Optional[UploadFile] = File(None),
    business_file: Optional[UploadFile] = File(None),
    user: AuthUser = Depends(get_current_user),
):
    check_rate_limit(f"dataset:{resolve_user_id(user)}", limit=6, window_seconds=60)
    try:
        final_personality = personality_text or ""
        final_business = business_text or ""

        if personality_file:
            bytes_content = await _read_upload_limited(personality_file)
            extracted = await run_in_threadpool(extract_text_from_file_bytes, personality_file.filename, bytes_content)
            if extracted:
                final_personality += f"\n\n[Archivo Personalidad: {personality_file.filename}]\n" + extracted

        if business_file:
            bytes_content = await _read_upload_limited(business_file)
            extracted = await run_in_threadpool(extract_text_from_file_bytes, business_file.filename, bytes_content)
            if extracted:
                final_business += f"\n\n[Archivo Negocio: {business_file.filename}]\n" + extracted

        if not final_personality.strip() and not final_business.strip():
            raise HTTPException(status_code=400, detail="Debes proporcionar al menos información del negocio o de personalidad.")

        dataset = await run_in_threadpool(
            generate_dataset_with_groq,
            personality_info=final_personality,
            business_info=final_business,
            num_examples=_clamp_examples(num_examples),
        )

        return {
            "status": "success",
            "count": len(dataset),
            "dataset": dataset
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando dataset sintético: {str(e)}")


@app.post("/api/train/generate_dataset_json")
async def generate_dataset_json_endpoint(req: GenerateDatasetRequest, user: AuthUser = Depends(get_current_user)):
    check_rate_limit(f"dataset:{resolve_user_id(user)}", limit=6, window_seconds=60)
    try:
        if not (req.personality_info or "").strip() and not (req.business_info or "").strip():
            raise HTTPException(status_code=400, detail="Debes proporcionar información de personalidad o del negocio.")

        dataset = await run_in_threadpool(
            generate_dataset_with_groq,
            personality_info=req.personality_info,
            business_info=req.business_info,
            num_examples=_clamp_examples(req.num_examples),
        )

        return {
            "status": "success",
            "count": len(dataset),
            "dataset": dataset
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando dataset sintético: {str(e)}")


class DatasetItem(BaseModel):
    instruction: Optional[str] = "Eres un asistente virtual de IA."
    input: Optional[str] = ""
    output: Optional[str] = ""

class TrainRequest(BaseModel):
    model_name: Optional[str] = "unsloth/Qwen2.5-0.5B-Instruct"
    dataset: List[DatasetItem] = []

# Solo se permiten modelos base del catálogo (evita descargar repos arbitrarios en la GPU)
ALLOWED_BASE_MODEL_PATTERN = re.compile(r"^unsloth/[A-Za-z0-9._-]+$")
MAX_TRAIN_ROWS = 2000


@app.post("/api/train")
async def start_training(req: TrainRequest, user: AuthUser = Depends(get_current_user)):
    user_id = resolve_user_id(user)
    target_model = req.model_name or "unsloth/Qwen2.5-0.5B-Instruct"
    if not ALLOWED_BASE_MODEL_PATTERN.match(target_model):
        raise HTTPException(status_code=400, detail="Modelo base no permitido.")
    if len(req.dataset) > MAX_TRAIN_ROWS:
        raise HTTPException(status_code=413, detail=f"El dataset supera el máximo de {MAX_TRAIN_ROWS} ejemplos.")
    if not training_jobs.is_running(user_id):
        check_rate_limit(f"train:{user_id}", limit=5, window_seconds=600, message="Has iniciado demasiados entrenamientos. Espera unos minutos.")

    dataset_list = [item.model_dump() for item in req.dataset] if req.dataset else []
    return training_jobs.start(user_id, target_model, dataset_list)

@app.get("/api/train/status")
async def get_train_status(user: AuthUser = Depends(get_current_user)):
    # Cada consulta actúa como heartbeat: si la página deja de consultar, el job se cancela solo
    return training_jobs.get_status(resolve_user_id(user))


@app.post("/api/train/reset")
@app.get("/api/train/reset")
async def reset_train_status(user: AuthUser = Depends(get_current_user)):
    # Cancela ÚNICAMENTE el entrenamiento de este usuario (su subproceso `modal run` y su app efímera).
    # Ya no se ejecuta `modal app stop` global, que apagaba la inferencia de todos los usuarios.
    return training_jobs.reset(resolve_user_id(user))


def _find_gguf_files(directory: str) -> List[str]:
    found = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".gguf"):
                found.append(os.path.join(root, f))
    return found


def _fetch_user_gguf(user_id: str) -> List[str]:
    """Busca el .gguf del usuario en disco y, si no existe, lo descarga de SU ruta en el volumen."""
    downloads_dir = user_downloads_dir(user_id)
    os.makedirs(downloads_dir, exist_ok=True)

    gguf_files = _find_gguf_files(downloads_dir)
    if gguf_files:
        return gguf_files

    modal_path = get_modal_cmd()
    model_dir = user_model_dir(user_id)
    for remote_path in (f"{model_dir}_gguf", model_dir):
        try:
            cmd = [modal_path, "volume", "get", MODAL_VOLUME, remote_path, downloads_dir + os.sep]
            subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=600)
            gguf_files = _find_gguf_files(downloads_dir)
            if gguf_files:
                return gguf_files
        except Exception as e:
            print(f"Error descargando desde Modal volume {remote_path}: {e}")
    return []


@app.get("/api/train/download")
async def download_trained_model(user: AuthUser = Depends(get_current_user)):
    """
    Descarga o sirve el archivo .gguf del modelo entrenado POR ESTE USUARIO.
    Si el archivo no existe localmente en ./downloads/<uid>, intenta bajarlo desde su ruta en el volumen de Modal.
    """
    user_id = resolve_user_id(user)
    check_rate_limit(f"download:{user_id}", limit=10, window_seconds=600)
    gguf_files = await run_in_threadpool(_fetch_user_gguf, user_id)

    if not gguf_files:
        raise HTTPException(status_code=404, detail="No se encontró ningún modelo .gguf entrenado disponible para descargar.")

    target_file = gguf_files[0]
    filename = os.path.basename(target_file)
    return FileResponse(path=target_file, filename=filename, media_type="application/octet-stream")


class TestChatRequest(BaseModel):
    prompt: str
    instruction: Optional[str] = "Responder la duda del cliente."

@app.post("/api/train/test_chat")
async def test_chat_trained_model(req: TestChatRequest, user: AuthUser = Depends(get_current_user)):
    """
    Realiza inferencia en tiempo real sobre el modelo entrenado por ESTE usuario en Modal.
    """
    user_id = resolve_user_id(user)
    check_rate_limit(f"test_chat:{user_id}", limit=30, window_seconds=60)
    try:
        import modal
        f = modal.Function.from_name(MODAL_APP_NAME, "inferir_modelo_remote")
        try:
            response_text = await f.remote.aio(
                req.prompt,
                req.instruction or "Responder la duda del cliente de forma amable y concisa.",
                safe_user_id(user_id),
            )
        except Exception as call_err:
            err_msg = str(call_err)
            if "takes from 1 to 2 positional arguments" in err_msg or "positional arguments" in err_msg:
                # Fallback retrocompatible con la app actualmente desplegada en la nube de Modal
                response_text = await f.remote.aio(
                    req.prompt,
                    req.instruction or "Responder la duda del cliente de forma amable y concisa.",
                )
            else:
                raise call_err
        return {"status": "success", "response": response_text}
    except Exception as e:
        print(f"Error en inferencia de prueba con Modal: {e}")
        return {"status": "error", "response": f"Error al consultar el modelo en Modal: {str(e)}"}
# ═══════════════════════════════════════════════════════════════════════════
# AX_LIBRARYMODEL — OPEN-WEIGHTS & HUGGING FACE DOWNLOAD BRIDGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

# IDs de Hugging Face: "owner/repo" con caracteres seguros (evita path traversal / SSRF)
HF_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
HF_FILENAME_PATTERN = re.compile(r"^(?!.*\.\.)[A-Za-z0-9._/-]+$")

def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

@app.get("/api/library/catalog")
async def get_library_catalog():
    """Retorna el catálogo curado de modelos open-weights."""
    return {"status": "success", "models": get_curated_open_weights_catalog()}

@app.get("/api/library/search")
async def search_library_models(request: Request, q: Optional[str] = "", filter_tag: Optional[str] = "gguf", limit: Optional[int] = 24):
    """Busca modelos en tiempo real en la API de Hugging Face."""
    check_rate_limit(f"lib_search:{_client_key(request)}", limit=60, window_seconds=60)
    safe_limit = max(1, min(int(limit or 24), 100))
    results = await run_in_threadpool(search_huggingface_models, query=q or "", filter_tag=filter_tag or "gguf", limit=safe_limit)
    return {"status": "success", "models": results}

@app.get("/api/library/files")
async def get_library_model_files(model_id: str, request: Request):
    """Obtiene los archivos descargables de pesos (.gguf, .safetensors) para un modelo específico."""
    if not HF_MODEL_ID_PATTERN.match(model_id or ""):
        raise HTTPException(status_code=400, detail="model_id inválido.")
    check_rate_limit(f"lib_files:{_client_key(request)}", limit=60, window_seconds=60)
    files = await run_in_threadpool(get_huggingface_model_files, model_id=model_id)
    return {"status": "success", "model_id": model_id, "files": files}

@app.get("/api/library/download_proxy")
async def download_model_weight_proxy(model_id: str, filename: str, request: Request):
    """
    Actúa como puente de descarga (proxy streaming) directamente desde Hugging Face hacia el usuario final.
    """
    if not HF_MODEL_ID_PATTERN.match(model_id or "") or not HF_FILENAME_PATTERN.match(filename or ""):
        raise HTTPException(status_code=400, detail="Parámetros de descarga inválidos.")
    check_rate_limit(f"lib_dl:{_client_key(request)}", limit=20, window_seconds=600)
    try:
        import requests
        target_url = f"{HF_RESOLVE_BASE}/{model_id}/resolve/main/{filename}"
        headers = {"User-Agent": "AXGLYNNE-Core/1.0"}

        req = await run_in_threadpool(requests.get, target_url, stream=True, headers=headers, timeout=15)
        if req.status_code != 200:
            req.close()
            raise HTTPException(status_code=req.status_code, detail=f"No se pudo acceder al archivo en Hugging Face ({req.status_code})")

        content_type = req.headers.get("content-type", "application/octet-stream")
        content_length = req.headers.get("content-length")

        def iter_file():
            # Generador síncrono: Starlette lo itera en un threadpool. Si el usuario cancela la
            # descarga, el generador se cierra y se libera la conexión con Hugging Face.
            try:
                for chunk in req.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        yield chunk
            finally:
                req.close()

        safe_name = os.path.basename(filename).replace('"', "")
        response_headers = {
            "Content-Disposition": f'attachment; filename="{safe_name}"'
        }
        if content_length:
            response_headers["Content-Length"] = content_length

        return StreamingResponse(
            iter_file(),
            media_type=content_type,
            headers=response_headers
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en proxy de descarga de modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error en puente de descarga GLYNNE: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Render asigna dinámicamente un puerto a través de la variable de entorno PORT
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, proxy_headers=True, forwarded_allow_ips="*")
