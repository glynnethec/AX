# -*- coding: utf-8 -*-
"""
AX_Agent.py
Agente conversacional de Interfaz de Usuario.
Arquitectura Multi-Agente Zero-Waste: Orquestador Router -> Loader -> Memory -> LLM Respondedor.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from AX_Chat.Router_Agent import route_intent
from AX_Chat.Context_Loader import load_context
from AX_Chat.Memory_Agent import summarize_old_history

load_dotenv(os.path.join(parent_dir, '.env'))

GROQ_API_KEY = os.getenv("GROQ_RESPONDER_API_KEY") or os.getenv("GROQ_API_KEY")
PRIMARY_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = "openai/gpt-oss-20b"

llm = ChatGroq(
    temperature=0.45,
    model_name=PRIMARY_MODEL,
    groq_api_key=GROQ_API_KEY
)

fallback_llm = ChatGroq(
    temperature=0.45,
    model_name=FALLBACK_MODEL,
    groq_api_key=GROQ_API_KEY
)

DEFAULT_SYSTEM_PROMPT = (
    "Eres Ax, el Asesor Estratégico y Guía de Plataforma de GLYNNE S.A.S. y AX GLYNNE. "
    "Responde de forma clara, profesional y directa sin repetir información ya dada en el historial."
)

FAST_GREETINGS = {
    "hola", "hola!", "buenas", "buenos dias", "buenos días", "buenas tardes", "buenas noches",
    "hi", "hello", "hey", "gracias", "muchas gracias", "thanks", "thank you", "ok", "vale", "listo"
}


def _format_recent_context(history: List[BaseMessage], limit: int = 4) -> str:
    """Formatea los últimos mensajes para dar contexto al Router."""
    if len(history) <= 1:
        return ""
    snippet = history[-(limit + 1):-1]
    lines = []
    for msg in snippet:
        msg_type = getattr(msg, "type", "")
        role = "Usuario" if msg_type in ["human", "user"] else "Ax"
        lines.append(f"{role}: {str(msg.content)[:240]}")
    return "\n".join(lines)


def run_ax_agent(history: List[BaseMessage], user_name: Optional[str] = None) -> str:
    """
    Punto de entrada: Orquesta Router (con contexto) -> Loader -> Memory -> LLM Respondedor.
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY no configurada en el entorno."

    last_user_msg = ""
    for msg in reversed(history):
        if (hasattr(msg, 'type') and msg.type in ["human", "user"]) or isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    selected_modules = []
    normalized_msg = last_user_msg.strip().lower().rstrip(".!?¡¿")
    if last_user_msg and normalized_msg not in FAST_GREETINGS:
        recent_context = _format_recent_context(history)
        selected_modules = route_intent(last_user_msg, recent_context)
        print(f"\n[AX Router] Módulos adicionales inyectados: {selected_modules}\n")

    SYSTEM_PROMPT = load_context(selected_modules) or DEFAULT_SYSTEM_PROMPT

    if user_name and user_name.strip() and user_name.strip().lower() not in {"default_user", "ax_user", "user"}:
        clean_name = user_name.strip().split()[0]
        SYSTEM_PROMPT += (
            f"\n\n[CONTEXTO DEL USUARIO]: Estás conversando con {clean_name}. "
            "Puedes usar su nombre de forma natural en el primer saludo o en un momento clave, "
            "pero NUNCA lo repitas en todos los mensajes."
        )

    # Mantener hasta 14 mensajes (7 turnos) para preservar coherencia y evitar repeticiones
    MAX_HISTORY = 14
    if len(history) > MAX_HISTORY:
        recent_history = history[-MAX_HISTORY:]
        old_history = history[:-MAX_HISTORY]

        memory_summary = summarize_old_history(old_history)
        if memory_summary:
            print(f"\n[AX Memory Agent] Resumen generado:\n{memory_summary}\n")
            SYSTEM_PROMPT += f"\n\n[MEMORIA DE TURNOS ANTERIORES - NO REPETIR LO YA EXPLICADO]:\n{memory_summary}"
    else:
        recent_history = history

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent_history

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as primary_err:
        print(f"[AX Agent] Fallback a {FALLBACK_MODEL} por error en {PRIMARY_MODEL}: {primary_err}")
        try:
            response = fallback_llm.invoke(messages)
            return response.content
        except Exception as fallback_err:
            return f"AX Core Error: {str(fallback_err)}"


if __name__ == "__main__":
    print("Iniciando prueba local Multi-Agente...")
    test_message = [HumanMessage(content="¿Qué herramientas tiene el Panel de GLYNNE y cómo uso la API gratis?")]
    respuesta = run_ax_agent(test_message, user_name="Alexander")
    print("\n--- RESPUESTA FINAL DEL AGENTE ---")
    print(respuesta)
    print("----------------------------------\n")
