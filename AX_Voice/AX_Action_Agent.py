import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

load_dotenv(os.path.join(parent_dir, '.env'))

GROQ_API_KEY = os.getenv("GROQ_RESPONDER_API_KEY") or os.getenv("GROQ_API_KEY")

llm_action = ChatGroq(
    temperature=0.0,  # Temperature 0 for deterministic JSON extraction
    model_name="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}}
)

ACTION_SYSTEM_PROMPT = """Eres un agente de extracción de intenciones estricto.
Tu única tarea es leer el último mensaje del usuario y la respuesta de la IA, y determinar si la IA ha decidido abrirle una de las siguientes páginas de GLYNNE al usuario.

Páginas disponibles:
- "about": https://axglynne.com/About
- "solutions": https://axglynne.com/Solutions
- "ia_vailable": https://axglynne.com/ia_vailable
- "contact": https://axglynne.com/contact
- "industries": https://axglynne.com/Industries
- "terms": https://axglynne.com/terms-of-service
- "linkedin": https://www.linkedin.com/company/glynne/posts/?viewAsMember=true
- "legal": https://www.informacolombia.com/directorio-empresas/informacion-empresa/glynne-sas

DEBES retornar ÚNICAMENTE un JSON válido con este formato:
{
  "action": "open_url" | "none",
  "url": "https://..." | null
}

Reglas:
- Si la IA dice algo como "te voy a abrir la página", "te muestro las industrias", o el usuario pide ver el linkedin y la IA acepta, debes extraer la URL correcta de la lista.
- Si no hay intención de abrir nada, devuelve {"action": "none", "url": null}.
- No inventes URLs, usa solo las de la lista.
"""

async def run_ax_action_agent_async(user_message: str, ai_response: str) -> str:
    """Evalúa asíncronamente si se debe abrir una URL."""
    if not GROQ_API_KEY:
        return None
        
    messages = [
        SystemMessage(content=ACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Mensaje del Usuario: {user_message}\n\nRespuesta de la IA: {ai_response}")
    ]
    
    try:
        response = await llm_action.ainvoke(messages)
        data = json.loads(response.content)
        if data.get("action") == "open_url" and data.get("url"):
            return data.get("url")
        return None
    except Exception as e:
        print(f"Action Agent Error: {e}")
        return None
