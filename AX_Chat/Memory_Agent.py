# -*- coding: utf-8 -*-
"""
Memory_Agent.py
Agente resumidor ultrarrápido que comprime el historial antiguo de la conversación.
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

# Usamos el modelo más rápido posible (Llama 3 8B) para no frenar la respuesta principal
memory_llm = ChatGroq(
    temperature=0.0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

def summarize_old_history(old_history: List[BaseMessage]) -> str:
    """
    Toma los mensajes antiguos y devuelve un resumen compacto de 1-3 oraciones.
    """
    if not old_history or len(old_history) < 2:
        return ""
        
    conversation_text = ""
    for msg in old_history:
        role = "User" if msg.type in ["human", "user"] else "AI"
        conversation_text += f"{role}: {msg.content}\n"
        
    prompt = f"""Eres una IA de resumen de memoria para GLYNNE. 
Tu tarea es leer la siguiente conversación antigua y resumir los temas centrales discutidos, la intención principal del usuario y cualquier detalle importante (como nombres, empresas o solicitudes específicas). 
Mantenlo EXTREMADAMENTE corto (1 a 3 oraciones máximo). No uses relleno. Responde en el mismo idioma en el que habla el usuario.

Conversación:
{conversation_text}

Resumen:"""

    try:
        response = memory_llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"[Memory Agent Error] {e}")
        return ""
