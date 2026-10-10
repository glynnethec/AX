import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
import json
import time
import asyncio
import re
import subprocess
import sys
import shutil
from elevenlabs.client import ElevenLabs


from AX_Chat.AX_Agent import run_ax_agent
from AX_Voice.AX_Voice_Agent import run_ax_voice_agent, llm, SYSTEM_PROMPT, SYSTEM_PROMPT_EN
from AX_Voice.AX_Action_Agent import run_ax_action_agent_async
from AX_Trainer.AX_QLoRA_Trainer import run_ax_trainer
from AX_Trainer.AX_Dataset_Generator import generate_dataset_with_groq, extract_text_from_file_bytes
from AX_LibraryModel.AX_HuggingFace_Bridge import get_curated_open_weights_catalog, search_huggingface_models, get_huggingface_model_files, HF_RESOLVE_BASE
from models_provider.router import router as models_provider_router
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


app = FastAPI(title="AX Glynne Core", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(all_data, f)
    except Exception as e:
        print(f"Error saving TTS usage: {e}")

def get_user_tts_usage(user_id: str) -> dict:
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync)

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
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
            
        ai_response = run_ax_agent(langchain_msgs)
        return {"status": "success", "reply": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")

@app.post("/api/voice_chat")
async def process_voice_chat(request: ChatRequest):
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

        # Lanzar el Agente de Acción en paralelo para evaluar intenciones de UI
        action_task = asyncio.create_task(run_ax_action_agent_async(user_message_text, ai_response))

        # ── 2. TTS CONCURRENTE POR ORACIÓN (PER USER) ──────────────────────
        use_mock_tts = request.use_mock_tts
        user_id = request.user_id or "default_user"
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
                    print(f"Límite de ElevenLabs superado para usuario {user_id}. Usando edge_tts.")
                use_mock_tts = True

        # Fallback edge_tts: también concurrente por oraciones
        if use_mock_tts:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")

@app.get("/api/tts_status")
async def get_tts_status(user_id: Optional[str] = "default_user"):
    user_id = user_id or "default_user"
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

class GenerateDatasetRequest(BaseModel):
    personality_info: Optional[str] = ""
    business_info: Optional[str] = ""
    num_examples: Optional[int] = 25

@app.post("/api/train/generate_dataset")
async def generate_dataset_endpoint(
    personality_text: Optional[str] = Form(""),
    business_text: Optional[str] = Form(""),
    num_examples: Optional[int] = Form(25),
    personality_file: Optional[UploadFile] = File(None),
    business_file: Optional[UploadFile] = File(None),
):
    try:
        final_personality = personality_text or ""
        final_business = business_text or ""

        if personality_file:
            bytes_content = await personality_file.read()
            extracted = extract_text_from_file_bytes(personality_file.filename, bytes_content)
            if extracted:
                final_personality += f"\n\n[Archivo Personalidad: {personality_file.filename}]\n" + extracted

        if business_file:
            bytes_content = await business_file.read()
            extracted = extract_text_from_file_bytes(business_file.filename, bytes_content)
            if extracted:
                final_business += f"\n\n[Archivo Negocio: {business_file.filename}]\n" + extracted

        if not final_personality.strip() and not final_business.strip():
            raise HTTPException(status_code=400, detail="Debes proporcionar al menos información del negocio o de personalidad.")

        dataset = generate_dataset_with_groq(
            personality_info=final_personality,
            business_info=final_business,
            num_examples=num_examples or 25
        )

        return {
            "status": "success",
            "count": len(dataset),
            "dataset": dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando dataset sintético: {str(e)}")


@app.post("/api/train/generate_dataset_json")
async def generate_dataset_json_endpoint(req: GenerateDatasetRequest):
    try:
        if not req.personality_info.strip() and not req.business_info.strip():
            raise HTTPException(status_code=400, detail="Debes proporcionar información de personalidad o del negocio.")

        dataset = generate_dataset_with_groq(
            personality_info=req.personality_info,
            business_info=req.business_info,
            num_examples=req.num_examples or 25
        )

        return {
            "status": "success",
            "count": len(dataset),
            "dataset": dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando dataset sintético: {str(e)}")


current_training_process = None

training_job_state = {
    "status": "idle",  # idle, running, completed, error
    "model_name": "",
    "progress": 0,
    "logs": [],
    "error_message": None,
    "download_command": None
}

class DatasetItem(BaseModel):
    instruction: str
    input: Optional[str] = ""
    output: str

class TrainRequest(BaseModel):
    model_name: str = "unsloth/Qwen2.5-0.5B-Instruct"
    dataset: List[DatasetItem]

def get_modal_cmd():
    modal_bin = shutil.which("modal")
    if modal_bin:
        return modal_bin
    venv_dir = os.path.dirname(sys.executable)
    potential_modal = os.path.join(venv_dir, "modal")
    if os.path.exists(potential_modal):
        return potential_modal
    base_dir = os.path.dirname(os.path.abspath(__file__))
    relative_modal = os.path.join(base_dir, "venv", "bin", "modal")
    if os.path.exists(relative_modal):
        return relative_modal
    return "modal"


def execute_training_task(model_name: str, dataset_raw: List[dict]):
    global training_job_state, current_training_process
    training_job_state["status"] = "running"
    training_job_state["model_name"] = model_name
    training_job_state["progress"] = 10
    training_job_state["logs"] = [f"[GLYNNE CORE] Conectando con la nube de Modal para {model_name}..."]
    training_job_state["error_message"] = None
    
    try:
        dataset_str = json.dumps(dataset_raw)
        modal_path = get_modal_cmd()
        cmd = [
            modal_path, "run", "AX_Trainer/AX_QLoRA_Trainer.py",
            "--model-name", model_name,
            "--dataset-json", dataset_str
        ]
        training_job_state["logs"].append(f"[MODAL] Solicitando GPU remota en la nube...")
        training_job_state["progress"] = 20

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        current_training_process = process

        for line in iter(process.stdout.readline, ''):
            if line:
                cleaned = line.strip()
                training_job_state["logs"].append(cleaned)
                if "[1/4]" in cleaned or "Cargando modelo" in cleaned:
                    training_job_state["progress"] = 35
                elif "[2/4]" in cleaned or "LoRA" in cleaned:
                    training_job_state["progress"] = 55
                elif "[3/4]" in cleaned or "datos" in cleaned:
                    training_job_state["progress"] = 70
                elif "[4/4]" in cleaned or "SFTTrainer" in cleaned:
                    training_job_state["progress"] = 85
                elif "Guardando modelo" in cleaned or "GGUF" in cleaned:
                    training_job_state["progress"] = 95

        process.stdout.close()
        return_code = process.wait()

        if training_job_state["status"] == "idle":
            # Si ya fue reseteado voluntariamente por el usuario
            return

        if return_code == 0:
            training_job_state["progress"] = 100
            training_job_state["status"] = "completed"
            training_job_state["download_command"] = "modal volume get ax-qlora-models ax_model/unsloth.Q4_K_M.gguf ./"
            training_job_state["logs"].append("[SISTEMA] ✅ Proceso de entrenamiento finalizado con éxito. Modelo .GGUF listo para descarga.")
        else:
            training_job_state["status"] = "error"
            training_job_state["error_message"] = "El comando de Modal finalizó con error o fue cancelado."
            training_job_state["logs"].append("[ERROR] El proceso fue abortado o devolvió un código de error.")
    except Exception as e:
        if training_job_state["status"] != "idle":
            training_job_state["status"] = "error"
            training_job_state["error_message"] = str(e)
            training_job_state["logs"].append(f"[EXCEPTION] {str(e)}")
    finally:
        current_training_process = None


@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    global training_job_state
    if training_job_state["status"] == "running":
        return {"status": "error", "message": "Ya hay un entrenamiento en curso."}
    
    dataset_list = [item.model_dump() for item in req.dataset]
    background_tasks.add_task(execute_training_task, req.model_name, dataset_list)
    return {"status": "started", "message": f"Entrenamiento de {req.model_name} iniciado en la nube de Modal."}

@app.get("/api/train/status")
async def get_train_status():
    return training_job_state


@app.post("/api/train/reset")
@app.get("/api/train/reset")
async def reset_train_status():
    global training_job_state, current_training_process
    
    # 1. Matar el proceso Python en ejecución si existe
    if current_training_process is not None:
        try:
            current_training_process.terminate()
            current_training_process.kill()
        except Exception as e:
            print(f"Error al terminar el proceso de entrenamiento: {e}")
        finally:
            current_training_process = None

    # 2. Detener cualquier app activa de entrenamiento en la nube de Modal
    try:
        modal_path = get_modal_cmd()
        subprocess.run([modal_path, "app", "stop", "ax-glynne-trainer"], capture_output=True, text=True, timeout=5)
    except Exception as e:
        print(f"Error al detener app en Modal: {e}")

    training_job_state = {
        "status": "idle",
        "model_name": "",
        "progress": 0,
        "logs": [],
        "error_message": None,
        "download_command": None
    }
    return {"status": "reset", "message": "Proceso cancelado y estado reseteado correctamente."}


@app.get("/api/train/download")
async def download_trained_model():
    """
    Descarga o sirve el archivo .gguf del modelo entrenado.
    Si el archivo no existe localmente en ./downloads, intenta bajarlo desde el volumen de Modal.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(base_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    
    gguf_files = []
    for root, dirs, files in os.walk(downloads_dir):
        for f in files:
            if f.endswith(".gguf"):
                gguf_files.append(os.path.join(root, f))
    
    modal_path = get_modal_cmd()
    if not gguf_files:
        try:
            cmd = [modal_path, "volume", "get", "ax-qlora-models", "ax_model_gguf", "downloads/"]
            subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
            for root, dirs, files in os.walk(downloads_dir):
                for f in files:
                    if f.endswith(".gguf"):
                        gguf_files.append(os.path.join(root, f))
        except Exception as e:
            print(f"Error descargando desde Modal volume ax_model_gguf: {e}")

    if not gguf_files:
        try:
            cmd = [modal_path, "volume", "get", "ax-qlora-models", "ax_model", "downloads/"]
            subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True)
            for root, dirs, files in os.walk(downloads_dir):
                for f in files:
                    if f.endswith(".gguf"):
                        gguf_files.append(os.path.join(root, f))
        except Exception:
            pass

    if not gguf_files:
        raise HTTPException(status_code=404, detail="No se encontró ningún modelo .gguf entrenado disponible para descargar.")
    
    target_file = gguf_files[0]
    filename = os.path.basename(target_file)
    return FileResponse(path=target_file, filename=filename, media_type="application/octet-stream")


class TestChatRequest(BaseModel):
    prompt: str
    instruction: Optional[str] = "Responder la duda del cliente."

@app.post("/api/train/test_chat")
async def test_chat_trained_model(req: TestChatRequest):
    """
    Realiza inferencia en tiempo real sobre el modelo entrenado en Modal.
    """
    try:
        import modal
        f = modal.Function.from_name("ax-glynne-trainer", "inferir_modelo_remote")
        response_text = await f.remote.aio(req.prompt, req.instruction or "Responder la duda del cliente de forma amable y concisa.")
        return {"status": "success", "response": response_text}
    except Exception as e:
        print(f"Error en inferencia de prueba con Modal: {e}")
# ═══════════════════════════════════════════════════════════════════════════
# AX_LIBRARYMODEL — OPEN-WEIGHTS & HUGGING FACE DOWNLOAD BRIDGE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/library/catalog")
async def get_library_catalog():
    """Retorna el catálogo curado de modelos open-weights."""
    return {"status": "success", "models": get_curated_open_weights_catalog()}

@app.get("/api/library/search")
async def search_library_models(q: Optional[str] = "", filter_tag: Optional[str] = "gguf", limit: Optional[int] = 24):
    """Busca modelos en tiempo real en la API de Hugging Face."""
    results = search_huggingface_models(query=q or "", filter_tag=filter_tag or "gguf", limit=limit or 24)
    return {"status": "success", "models": results}

@app.get("/api/library/files")
async def get_library_model_files(model_id: str):
    """Obtiene los archivos descargables de pesos (.gguf, .safetensors) para un modelo específico."""
    files = get_huggingface_model_files(model_id=model_id)
    return {"status": "success", "model_id": model_id, "files": files}

@app.get("/api/library/download_proxy")
async def download_model_weight_proxy(model_id: str, filename: str):
    """
    Actúa como puente de descarga (proxy streaming) directamente desde Hugging Face hacia el usuario final.
    """
    try:
        import requests
        target_url = f"{HF_RESOLVE_BASE}/{model_id}/resolve/main/{filename}"
        headers = {"User-Agent": "AXGLYNNE-Core/1.0"}
        
        req = requests.get(target_url, stream=True, headers=headers, timeout=15)
        if req.status_code != 200:
            raise HTTPException(status_code=req.status_code, detail=f"No se pudo acceder al archivo en Hugging Face ({req.status_code})")

        content_type = req.headers.get("content-type", "application/octet-stream")
        content_length = req.headers.get("content-length")

        def iter_file():
            for chunk in req.iter_content(chunk_size=64 * 1024):
                if chunk:
                    yield chunk

        response_headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        if content_length:
            response_headers["Content-Length"] = content_length

        return StreamingResponse(
            iter_file(),
            media_type=content_type,
            headers=response_headers
        )
    except Exception as e:
        print(f"Error en proxy de descarga de modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error en puente de descarga GLYNNE: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Render asigna dinámicamente un puerto a través de la variable de entorno PORT
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)



