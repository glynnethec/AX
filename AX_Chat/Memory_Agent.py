# -*- coding: utf-8 -*-
"""
Memory_Agent.py
Agente resumidor que comprime el historial antiguo preservando qué temas ya se explicaron para evitar repeticiones.
"""

import os
from typing import List
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
load_dotenv(os.path.join(parent_dir, '.env'))

GROQ_API_KEY = os.getenv("GROQ_ROUTER_API_KEY") or os.getenv("GROQ_API_KEY")

memory_llm = ChatGroq(
    temperature=0.0,
    model_name=os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=GROQ_API_KEY
)

def summarize_old_history(old_history: List[BaseMessage]) -> str:
    """
    Toma los mensajes antiguos y devuelve un resumen estructurado compacto para evitar que el asesor repita información.
    """
    if not old_history or len(old_history) < 2:
        return ""

    conversation_text = ""
    for msg in old_history:
        msg_type = getattr(msg, "type", "")
        role = "User" if msg_type in ["human", "user"] else "AI"
        conversation_text += f"{role}: {msg.content}\n"

    prompt = f"""Eres el módulo de memoria conversacional del Asesor de GLYNNE.
Resume los siguientes turnos anteriores en máximo 4 líneas usando exactamente este formato (en el idioma del usuario):
- Tema/objetivo del usuario: ...
- Puntos y enlaces ya explicados (NO repetir): ...
- Dudas o siguientes pasos pendientes: ...

Conversación anterior:
{conversation_text}

Resumen estructurado:"""

    try:
        response = memory_llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[Memory Agent Error] {e}")
        return ""
