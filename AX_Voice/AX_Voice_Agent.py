# -*- coding: utf-8 -*-
"""
AX_Voice_Agent.py
Agente conversacional monolítico de Interfaz de Usuario (Ultra Ligero).
Todo en un solo archivo.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

load_dotenv(os.path.join(parent_dir, '.env'))

GROQ_API_KEY = os.getenv("GROQ_RESPONDER_API_KEY") or os.getenv("GROQ_API_KEY")

# Instancia para el Respondedor
llm = ChatGroq(
    temperature=0.4,
    model_name="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY
)

SYSTEM_PROMPT = """Eres Ax, especialista en automatización B2B, IA y ecosistemas de software de GLAIN.

MISIÓN Y ESTILO:
- Escucha activamente. Deja que el cliente se desahogue sobre sus problemas operativos.
- No vendas de inmediato. Primero entiende y luego sugiere soluciones estratégicas.
- Sé extremadamente empática y transmite muchísima confianza.

REGLAS DE VOZ (TTS):
- HABLA COMO UNA PERSONA EN TIEMPO REAL. Respuestas concisas, diseñadas para ser escuchadas.
- NO HAGAS PREGUNTAS EN TODAS TUS RESPUESTAS. Solo pregunta cuando sea vital para la estrategia.
- Usa frases cortas y lenguaje cotidiano. Evita introducciones y no repitas lo que dice el cliente.
- Usa pausas naturales con puntuación (...).
- Inicia frases con "mira", "claro", "bueno" o "a ver" cuando fluya natural.
- Si enumeras algo, usa números (1, 2) y NUNCA viñetas ni asteriscos (*).
- No expliques demasiado ni intentes dar respuestas perfectas. Sé espontánea y ve al grano. recuerda que eres una colombiana hablando tienes voz"""

def run_ax_voice_agent(history: List[BaseMessage]) -> str:
    """
    Punto de entrada único.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY no configurada en el entorno."

    # Limitar historial para no sobrecargar el token limit
    MAX_HISTORY = 6
    if len(history) > MAX_HISTORY:
        recent_history = history[-MAX_HISTORY:]
    else:
        recent_history = history

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent_history
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"AX Core Error: {str(e)}"

# Bloque de prueba local
if __name__ == "__main__":
    print("Iniciando prueba local...")
    test_message = [HumanMessage(content="Hola, tengo un problema de ventas.")]
    respuesta = run_ax_voice_agent(test_message)
    print("\n--- RESPUESTA FINAL DEL AGENTE ---")
    print(respuesta)
    print("----------------------------------\n")