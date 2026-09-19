# -*- coding: utf-8 -*-
"""
AX_Agent.py
Agente conversacional de Interfaz de Usuario (Ultra Ligero).
Arquitectura Multi-Agente Zero-Waste: Punto de Entrada y Respondedor Principal.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # Para que AX_Chat. sea resoluble si se ejecuta localmente

from AX_Chat.Router_Agent import route_intent
from AX_Chat.Context_Loader import load_context
from AX_Chat.Memory_Agent import summarize_old_history
load_dotenv(os.path.join(parent_dir, '.env'))

GROQ_API_KEY = os.getenv("GROQ_RESPONDER_API_KEY") or os.getenv("GROQ_API_KEY")

# Instancia para el Respondedor (Equilibrada, ahora más cálida)
llm = ChatGroq(
    temperature=0.4,
    model_name="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY
)

def run_ax_agent(history: List[BaseMessage]) -> str:
    """
    Punto de entrada: Orquesta Router -> Loader -> LLM Responder
    """
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY no configurada en el entorno."

    # 1. Extraer el último mensaje para el enrutador
    last_user_msg = ""
    for msg in reversed(history):
        if hasattr(msg, 'type') and msg.type in ["human", "user"]:
            last_user_msg = msg.content
            break
        elif isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
            
    # 2. Agente 1 (Router) decide módulos
    selected_modules = []
    if last_user_msg:
        selected_modules = route_intent(last_user_msg)
        print(f"\n[AX Router] Módulos inyectados para esta consulta: {selected_modules}\n")

    # 3. Cargar el contexto dinámico (Zero-Waste)
    SYSTEM_PROMPT = load_context(selected_modules)
    if not SYSTEM_PROMPT:
        SYSTEM_PROMPT = "Eres un Ingeniero de Soluciones Operativas de GLYNNE. Tu objetivo exclusivo es diagnosticar, solucionar problemas y ejecutar tareas para el usuario. No ofrezcas descripciones corporativas a menos que se te pregunte explícitamente"

    # 4. Preparar historial limitando a los últimos 6 mensajes (3 turnos) para evitar lentitud
    MAX_HISTORY = 6
    if len(history) > MAX_HISTORY:
        recent_history = history[-MAX_HISTORY:]
        old_history = history[:-MAX_HISTORY]
        
        # Ejecutar el Memory Agent sobre los mensajes viejos
        memory_summary = summarize_old_history(old_history)
        if memory_summary:
            print(f"\n[AX Memory Agent] Resumen generado: {memory_summary}\n")
            SYSTEM_PROMPT += f"\n\n[PAST CONVERSATION MEMORY]:\n{memory_summary}"
    else:
        recent_history = history

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + recent_history
    
    # 5. Agente 2 (Responder) genera respuesta final
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"AX Core Error: {str(e)}"

# Bloque de prueba local
if __name__ == "__main__":
    print("Iniciando prueba local Multi-Agente Separado...")
    test_message = [HumanMessage(content="¿Cómo automatizaron la auditoría de Servex?")]
    respuesta = run_ax_agent(test_message)
    print("\n--- RESPUESTA FINAL DEL AGENTE ---")
    print(respuesta)
    print("----------------------------------\n")
