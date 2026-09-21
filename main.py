import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64
import os
import json
import time
from elevenlabs.client import ElevenLabs

from AX_Chat.AX_Agent import run_ax_agent
from AX_Voice.AX_Voice_Agent import run_ax_voice_agent
from langchain_core.messages import HumanMessage, AIMessage

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
            
        # 1. Obtener la respuesta de texto del agente
        ai_response = run_ax_voice_agent(langchain_msgs)
        
        # 2. Generar el audio
        use_mock_tts = request.use_mock_tts
        audio_data = b""
        
        # Lógica dinámica para ElevenLabs (2000 chars / 48 horas)
        if not use_mock_tts:
            usage_data = get_tts_usage()
            current_time = time.time()
            
            # Reiniciar si han pasado 48 horas
            if current_time - usage_data["start_time"] > RESET_SECONDS:
                usage_data["start_time"] = current_time
                usage_data["chars_used"] = 0
                
            response_len = len(ai_response)
            
            if usage_data["chars_used"] + response_len <= MAX_CHARS:
                # Intentar usar ElevenLabs
                eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
                if eleven_api_key:
                    try:
                        client = ElevenLabs(api_key=eleven_api_key)
                        audio_generator = client.text_to_speech.convert(
                            text=ai_response,
                            voice_id="VmejBeYhbrcTPwDniox7", 
                            model_id="eleven_multilingual_v2"
                        )
                        audio_data = b"".join(audio_generator)
                        
                        # Actualizar consumo y guardar
                        usage_data["chars_used"] += response_len
                        save_tts_usage(usage_data)
                    except Exception as e:
                        print(f"Error con ElevenLabs, usando fallback a edge_tts: {str(e)}")
                        use_mock_tts = True
                else:
                    print("Falta ELEVENLABS_API_KEY en .env, usando fallback a edge_tts.")
                    use_mock_tts = True
            else:
                print(f"Límite de caracteres de ElevenLabs superado ({usage_data['chars_used']}/{MAX_CHARS}). Usando Edge TTS.")
                use_mock_tts = True
                
        # Fallback a Edge TTS si ElevenLabs falló, superó el límite, o si use_mock_tts estaba activo
        if use_mock_tts:
            import edge_tts
            communicate = edge_tts.Communicate(ai_response, "es-CO-SalomeNeural")
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
                
        # 3. Convertir el audio a base64
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
