import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64
import os
import json
import time
import asyncio
import re
from elevenlabs.client import ElevenLabs

from AX_Chat.AX_Agent import run_ax_agent
from AX_Voice.AX_Voice_Agent import run_ax_voice_agent, llm, SYSTEM_PROMPT
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

USAGE_FILE = "tts_usage.json"
MAX_CHARS = 2000
RESET_SECONDS = 48 * 3600

def get_tts_usage():
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"start_time": time.time(), "chars_used": 0}

def save_tts_usage(data):
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving TTS usage: {e}")

def split_into_sentences(text: str) -> List[str]:
    """Divide el texto en oraciones naturales para TTS."""
    parts = re.split(r'(?<=[.!?…])\s+|(?<=\.\.\.)\s*', text)
    return [p.strip() for p in parts if p.strip()]

async def tts_edge_sentence(sentence: str) -> bytes:
    """Genera audio para una sola oración con edge_tts."""
    import edge_tts
    communicate = edge_tts.Communicate(sentence, "es-CO-SalomeNeural")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

async def tts_elevenlabs_sentence(sentence: str, api_key: str) -> bytes:
    """Genera audio para una sola oración con ElevenLabs (en executor para no bloquear)."""
    def _sync():
        client = ElevenLabs(api_key=api_key)
        gen = client.text_to_speech.convert(
            text=sentence,
            voice_id="VmejBeYhbrcTPwDniox7",
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
        for msg in request.messages:
            role = msg.role.lower()
            if role == "user":
                langchain_msgs.append(HumanMessage(content=msg.content))
            elif role in ["ai", "assistant"]:
                langchain_msgs.append(AIMessage(content=msg.content))
                
        if not langchain_msgs:
            raise HTTPException(status_code=400, detail="El historial está vacío.")

        # ── 1. STREAMING DEL LLM ────────────────────────────────────────────
        # Limitar historial
        MAX_HISTORY = 6
        recent_history = langchain_msgs[-MAX_HISTORY:] if len(langchain_msgs) > MAX_HISTORY else langchain_msgs
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent_history

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

        # ── 2. TTS CONCURRENTE POR ORACIÓN ──────────────────────────────────
        use_mock_tts = request.use_mock_tts
        audio_data = b""

        if not use_mock_tts:
            # Verificar cuota de ElevenLabs
            usage_data = get_tts_usage()
            current_time = time.time()
            if current_time - usage_data["start_time"] > RESET_SECONDS:
                usage_data["start_time"] = current_time
                usage_data["chars_used"] = 0

            response_len = len(ai_response)
            eleven_api_key = os.getenv("ELEVENLABS_API_KEY")

            if eleven_api_key and usage_data["chars_used"] + response_len <= MAX_CHARS:
                try:
                    # Generar audio de todas las oraciones EN PARALELO
                    tasks = [tts_elevenlabs_sentence(s, eleven_api_key) for s in sentences]
                    audio_chunks = await asyncio.gather(*tasks)
                    audio_data = b"".join(audio_chunks)

                    usage_data["chars_used"] += response_len
                    save_tts_usage(usage_data)
                except Exception as e:
                    print(f"Error con ElevenLabs paralelo, usando edge_tts: {e}")
                    use_mock_tts = True
            else:
                if not eleven_api_key:
                    print("Falta ELEVENLABS_API_KEY. Usando edge_tts.")
                else:
                    print(f"Límite de ElevenLabs superado. Usando edge_tts.")
                use_mock_tts = True

        # Fallback edge_tts: también concurrente por oraciones
        if use_mock_tts:
            try:
                tasks = [tts_edge_sentence(s) for s in sentences]
                audio_chunks = await asyncio.gather(*tasks)
                audio_data = b"".join(audio_chunks)
            except Exception as e:
                print(f"Error con edge_tts paralelo: {e}")
                # Último fallback: edge_tts sobre texto completo
                import edge_tts
                communicate = edge_tts.Communicate(ai_response, "es-CO-SalomeNeural")
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]

        # ── 3. RESPUESTA ─────────────────────────────────────────────────────
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        
        return {
            "status": "success", 
            "reply": ai_response,
            "audio_base64": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Render asigna dinámicamente un puerto a través de la variable de entorno PORT
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


