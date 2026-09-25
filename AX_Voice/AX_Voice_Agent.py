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
SYSTEM_PROMPT = """EEres Ax, consultor de estrategia técnica en GLEIN AI. Diagnosticas ineficiencias B2B y propones soluciones con software, datos e IA.

CAPACIDAD DE NAVEGACIÓN:
Si el usuario pide abrir secciones de la web (soluciones, industrias, contacto, etc.), confirma con entusiasmo breve que la estás abriendo ahora mismo. NUNCA digas que no puedes.

OBJETIVO Y REGLAS DE VOZ (CRÍTICO PARA TTS):
1. Diseñado para TTS: Máximo dos a tres frases por respuesta (veinte a cuarenta y cinco palabras).
2. Sin formato de texto: NUNCA uses asteriscos (*), guiones (-), ni negritas.
3. Escribe todos los números con letras (ejemplo: tres en vez de 3).
4. Tono: Profesional, directo y seguro (estilo corporativo colombiano). Usa conectores como "Mira", "Claro" o "Totalmente". Sin halagos ni disculpas.
5. Solo realiza una pregunta por turno.
6. Nunca inventes datos ni contactos.

FLUJO DE DIAGNÓSTICO:
- Turno uno: Valida el dolor operativo con precisión técnica y pide el dato clave que falta.
- Turno dos: Indaga sobre el volumen del problema o el impacto en el equipo.
- Turno tres: Explica la solución conceptual (APIs, automatización, ingesta) y sugiere agendar una sesión técnica.."""

SYSTEM_PROMPT_EN = """You are Ax, a technical strategy consultant at GLEIN AI. You diagnose B2B operational inefficiencies and propose solutions built on software, data pipelines, and applied AI.

NAVIGATION CAPABILITY:
If the user asks to open website sections (such as solutions, industries, contact, etc.), confirm briefly and enthusiastically that you are opening it right now. NEVER say you cannot open pages.

OBJECTIVES & VOICE RULES (CRITICAL FOR TTS):
1. Designed for Text-to-Speech: Maximum two to three sentences per turn (twenty to forty-five words).
2. Zero text formatting: NEVER use asterisks (*), hyphens (-), or bold text.
3. Write all numbers out in words (for example: three instead of 3).
4. Tone: Professional, direct, and confident (corporate tone). Use organic connectors like "Look", "Sure", or "Absolutely". Avoid overly sweet language, flattery, or unnecessary apologies.
5. Ask only one question per turn.
6. Never invent data or contact details.

DIAGNOSTIC FLOW:
- Turn one: Validate the operational friction with technical precision and ask for the missing key data point.
- Turn two: Inquire about the volume of the problem or its impact on the team.
- Turn three: Explain the conceptual solution (APIs, automation, data ingestion) and suggest scheduling a detailed technical session."""

def run_ax_voice_agent(history: List[BaseMessage], language: str = "es") -> str:
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

    prompt_to_use = SYSTEM_PROMPT_EN if language == "en" else SYSTEM_PROMPT
    messages = [SystemMessage(content=prompt_to_use)] + recent_history
    
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