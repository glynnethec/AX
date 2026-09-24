import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
import json
import time
import asyncio
import re
from elevenlabs.client import ElevenLabs

from AX_Chat.AX_Agent import run_ax_agent
from AX_Voice.AX_Voice_Agent import run_ax_voice_agent, llm, SYSTEM_PROMPT, SYSTEM_PROMPT_EN
from AX_Voice.AX_Action_Agent import run_ax_action_agent_async
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

app = FastAPI(title="AX Glynne Core", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

if __name__ == "__main__":
    import uvicorn
    # Render asigna dinámicamente un puerto a través de la variable de entorno PORT
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


