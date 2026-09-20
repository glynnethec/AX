import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64
import os
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
        
        # 2. Generar el audio con ElevenLabs (Hiperrealista)
        eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not eleven_api_key:
            raise HTTPException(status_code=500, detail="Falta ELEVENLABS_API_KEY en .env")
            
        client = ElevenLabs(api_key=eleven_api_key)
        
        # Usamos el ID de Sarah (EXAVITQu4vr4xnSDxMaL) que es gratuito
        audio_generator = client.text_to_speech.convert(
            text=ai_response,
            voice_id="EXAVITQu4vr4xnSDxMaL", 
            model_id="eleven_multilingual_v2"
        )
        
        # Convertir el generador en bytes
        audio_data = b"".join(audio_generator)
                
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
