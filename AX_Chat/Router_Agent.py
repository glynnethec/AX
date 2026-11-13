# -*- coding: utf-8 -*-
"""
Router_Agent.py
Agente Supervisor. Lee la intención del usuario (con contexto reciente) y decide qué módulos .md inyectar.
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
    model_name=os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=GROQ_API_KEY
)

VALID_MODULES = {
    "platform_tools",
    "knowledge",
    "about",
    "methodology",
    "solutions",
    "servex",
    "industries",
    "onboarding",
    "team",
    "legal",
    "policies",
}

def route_intent(user_input: str, recent_context: str = "") -> List[str]:
    """
    Analiza el mensaje actual junto con el contexto reciente y devuelve los módulos requeridos.
    """
    context_block = f"\nContexto reciente de la conversación:\n{recent_context}\n" if recent_context else ""

    router_prompt = f"""Eres el motor de enrutamiento de contexto para el Asesor de GLYNNE.
Analiza el último mensaje del usuario (y el contexto reciente si existe) y selecciona ÚNICAMENTE los módulos estrictamente necesarios para responder con precisión.

Módulos disponibles:
- platform_tools (Herramientas del Panel: Free AI Models API / Sub-API keys, Create Your Own Model / QLoRA, Model Library / GGUF-FP16-EXL2, AX Chat, AX Voice)
- knowledge (Qué hace GLYNNE, servicios B2B principales, diferenciador frente a chatbots genéricos)
- about (Filosofía, tesis, modelo dual: GLYNNE S.A.S. B2B vs. AX GLYNNE I+D)
- methodology (Cómo trabajamos: Auditar antes de IA, gobernanza determinista, multi-agente, HITL)
- solutions (Soluciones técnicas: QA Autónomo, Extracción/Orquestación de Datos, Infraestructura RAG/LLMs)
- servex (Caso de estudio Servex US, catálogos CET, reducción del 99% en tiempo)
- industries (Sectores cubiertos: finanzas, legal, logística, retail, salud, etc.)
- onboarding (Cómo contratar, proceso comercial, Sesión de Arquitectura, contacto)
- team (CEO Alexander Quiroga, liderazgo, perfil)
- legal (NIT, dirección, teléfono, email, registro legal de GLYNNE S.A.S.)
- policies (Privacidad, términos de servicio, política de cookies)

Reglas:
1. Si el usuario hace una pregunta de seguimiento ("¿y cómo funciona eso?", "cuéntame más", "¿cuál me sirve?"), usa el contexto reciente para mantener o ampliar los módulos relevantes del tema actual.
2. Selecciona máximo 3 módulos por turno para mantener el contexto limpio y sin redundancia.
3. Si es un saludo simple o despedida ("Hola", "Gracias", "Ok") que no pide información, retorna [].
4. Responde ÚNICAMENTE con un array JSON válido de strings (sin Markdown ni texto adicional). Ej: ["platform_tools", "onboarding"]
{context_block}
Último mensaje del usuario: "{user_input}"
"""
    try:
        response = router_llm.invoke([SystemMessage(content=router_prompt)])
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        selected = json.loads(content)
        if isinstance(selected, list):
            return [f"{s}.md" for s in selected if isinstance(s, str) and s in VALID_MODULES]
        return []
    except Exception as e:
        print(f"[AX Router] Fallo en parseo JSON: {e}")
        return ["knowledge.md", "platform_tools.md"]
