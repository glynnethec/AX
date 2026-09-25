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

URL_MAPPING = {
    "about": "https://axglynne.com/About",
    "solutions": "https://axglynne.com/Solutions",
    "ia_vailable": "https://axglynne.com/ia_vailable",
    "contact": "https://axglynne.com/contact",
    "industries": "https://axglynne.com/Industries",
    "terms": "https://axglynne.com/terms-of-service",
    "linkedin": "https://www.linkedin.com/company/glynne/posts/?viewAsMember=true",
    "legal": "https://www.informacolombia.com/directorio-empresas/informacion-empresa/glynne-sas"
}

ACTION_SYSTEM_PROMPT = """You are an intent extractor. Determine if the user or AI requests opening a GLYNNE URL.

URL Mapping:
- about: https://axglynne.com/About
- solutions: https://axglynne.com/Solutions
- ia_vailable: https://axglynne.com/ia_vailable
- contact: https://axglynne.com/contact
- industries: https://axglynne.com/Industries
- terms: https://axglynne.com/terms-of-service
- linkedin: https://www.linkedin.com/company/glynne/posts/?viewAsMember=true
- legal: https://www.informacolombia.com/directorio-empresas/informacion-empresa/glynne-sas

Output Rules:
1. Respond ONLY in valid JSON: {"action": "open_url" | "none", "url": "https://..." | null}
2. Return "open_url" and the exact matching URL if the user asks to see/open/visit a section or if the AI confirms opening it.
3. If there is no clear intent, return {"action": "none", "url": null}.
4. NEVER invent URLs. Use exclusively those in the mapping.
"""

def fallback_extract_url(user_message: str, ai_response: str) -> str:
    """Extracción por palabras clave cuando el LLM no está disponible o falla."""
    text = f"{user_message} {ai_response}".lower()
    
    # Intención explícita de navegación/apertura o mención de sección
    nav_verbs = ["abrir", "abre", "abreme", "ábreme", "ver", "muestra", "muestrame", "muéstrame", "ir", "lleva", "llevame", "llévame", "mostrar", "open", "show"]
    has_nav_intent = any(v in text for v in nav_verbs)
    
    if "linkedin" in text:
        return URL_MAPPING["linkedin"]
    if "solucion" in text or "soluciones" in text or "solutions" in text:
        return URL_MAPPING["solutions"]
    if "industria" in text or "industrias" in text or "industries" in text:
        return URL_MAPPING["industries"]
    if "contacto" in text or "contactar" in text or "contact" in text:
        return URL_MAPPING["contact"]
    if "sobre nosotros" in text or "about" in text or "quienes somos" in text or "quiénes somos" in text:
        return URL_MAPPING["about"]
    if "ia_vailable" in text or "ia vailable" in text:
        return URL_MAPPING["ia_vailable"]
    if "terminos" in text or "términos" in text or "condiciones" in text or "terms" in text:
        return URL_MAPPING["terms"]
    if "legal" in text or "directorio" in text or "informacolombia" in text:
        return URL_MAPPING["legal"]
        
    return None

async def run_ax_action_agent_async(user_message: str, ai_response: str) -> str:
    """Evalúa asíncronamente si se debe abrir una URL."""
    url_found = None
    
    if GROQ_API_KEY:
        messages = [
            SystemMessage(content=ACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"Mensaje del Usuario: {user_message}\n\nRespuesta de la IA: {ai_response}")
        ]
        try:
            response = await llm_action.ainvoke(messages)
            data = json.loads(response.content)
            if data.get("action") == "open_url" and data.get("url"):
                url_found = data.get("url")
        except Exception as e:
            print(f"Action Agent LLM Error: {e}")

    # Fallback determinista si el LLM falló o no retornó URL
    if not url_found:
        url_found = fallback_extract_url(user_message, ai_response)
        
    return url_found

