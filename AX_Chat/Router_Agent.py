# -*- coding: utf-8 -*-
"""
Router_Agent.py
Agente Supervisor. Lee la intención del usuario y decide qué contexto inyectar.
"""

import os
import json
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
load_dotenv(os.path.join(parent_dir, '.env'))

GROQ_API_KEY = os.getenv("GROQ_ROUTER_API_KEY") or os.getenv("GROQ_API_KEY")

router_llm = ChatGroq(
    temperature=0.0,
    model_name="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

def route_intent(user_input: str) -> List[str]:
    """
    Analiza el mensaje y devuelve los módulos requeridos en formato JSON.
    """
    router_prompt = f"""Eres el motor central de enrutamiento de intenciones para el Agente de IA Operativo de GLYNNE.
Tu tarea es analizar el mensaje del usuario y determinar qué módulos de contexto o flujos de acción se requieren para resolver su problema o responder su consulta.

Módulos:

[OPERATIVO Y RESOLUCIÓN DE PROBLEMAS - Prioriza estos para problemas/tareas]
- support_sops (Diagnósticos, resolución de errores, solución de problemas paso a paso, manuales técnicos)
- operations (Ejecución de tareas, estado del sistema, flujos de trabajo internos, solicitudes procesables, disparadores de llamadas a funciones)

[CORPORATIVO E INFORMACIÓN - Usa solo para preguntas directas sobre la empresa]
- knowledge (Antecedentes de la empresa, lo que hacemos, marco general)
- legal (NIT, dirección, teléfono, registro)
- servex (Caso de Estudio Servex, reducción de tiempo, BPO, automatización de catálogos de CET)
- team (CEO Alexander Quiroga, liderazgo, perfil)
- methodology (Framework, "Auditar antes de IA", determinismo, multi-agente)
- industries (Sectores como finanzas, legal, logística, retail, etc.)
- solutions (Soluciones técnicas, plataformas RAG y MCP, extracción de datos)
- onboarding (Cómo contratar, pasos, sesión de arquitectura)
- policies (Políticas de privacidad, cookies, términos de servicio)
- about (Misión, posicionamiento de marca, "La tecnología no es una capa")

Instrucciones Críticas:
1. PRIORIZA LA RESOLUCIÓN DE PROBLEMAS: Si el usuario informa un error, pide ayuda técnica o solicita que se realice una tarea, DEBES enrutar a "support_sops" y/o "operations". Evita estrictamente enrutar a "knowledge" o "about" en estos casos para prevenir relleno corporativo.
2. Responde ÚNICAMENTE con un array JSON puro de strings. Ej: ["support_sops", "operations"]
3. Si el mensaje es un simple saludo o no requiere contexto extra ("Hola", "Gracias"), retorna: []
4. NO incluyas formato Markdown (```json) ni explicaciones, solo emite el array JSON crudo.

Mensaje del usuario: "{user_input}"
"""
    try:
        response = router_llm.invoke([SystemMessage(content=router_prompt)])
        content = response.content.strip()
        
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        selected = json.loads(content)
        if isinstance(selected, list):
            return [f"{s}.md" for s in selected]
        return []
    except Exception as e:
        print(f"[AX Router] Fallo en parseo JSON: {e}")
        return ["about.md", "knowledge.md"]
