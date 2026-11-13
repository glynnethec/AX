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
    temperature=0.0,
    model_name=os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b"),
    groq_api_key=GROQ_API_KEY,
    model_kwargs={"response_format": {"type": "json_object"}}
)

URL_MAPPING = {
    "panel": "/Panel",
    "models_provider": "/models_provider",
    "create_model": "/Create_you_GLYNNE_model",
    "library_model": "/Librarymodel",
    "ax_chat": "/AX_chat",
    "about": "https://axglynne.com/About",
    "methodology": "https://axglynne.com/Methodology",
    "solutions": "https://axglynne.com/Solutions",
    "industries": "https://axglynne.com/Industries",
    "security": "https://axglynne.com/Segurity",
    "ceo": "https://axglynne.com/CEO_GLYNNE",
    "contact": "https://axglynne.com/contact",
    "terms": "https://axglynne.com/terms-of-service",
    "linkedin": "https://www.linkedin.com/company/glynne/posts/?viewAsMember=true",
    "legal": "https://www.informacolombia.com/directorio-empresas/informacion-empresa/glynne-sas"
}

ACTION_SYSTEM_PROMPT = """You are a strict navigation intent extractor for GLYNNE's Voice Advisor.
Determine if the user EXPLICITLY asked to open, visit, or navigate to a specific GLYNNE page or Panel tool.

URL Mapping:
- panel (Main Panel / Hub): /Panel
- models_provider (Free AI Models API / API Key Gateway): /models_provider
- create_model (Create Your Own Model / QLoRA Training): /Create_you_GLYNNE_model
- library_model (Model Library / Open-Source Weights GGUF): /Librarymodel
- ax_chat (GLYNNE Chat Advisor): /AX_chat
- about (About GLYNNE / Who we are): https://axglynne.com/About
- methodology (Methodology / Audit before AI): https://axglynne.com/Methodology
- solutions (Solutions / Servex Case Study): https://axglynne.com/Solutions
- industries (B2B Industries): https://axglynne.com/Industries
- security (Cybersecurity / Data Protection): https://axglynne.com/Segurity
- ceo (CEO / Alexander Quiroga): https://axglynne.com/CEO_GLYNNE
- contact (Contact / Schedule Architecture Session): https://axglynne.com/contact
- terms (Terms of Service): https://axglynne.com/terms-of-service
- linkedin (GLYNNE LinkedIn): https://www.linkedin.com/company/glynne/posts/?viewAsMember=true
- legal (Legal Registry InformaColombia): https://www.informacolombia.com/directorio-empresas/informacion-empresa/glynne-sas

Output Rules:
1. Respond ONLY in valid JSON: {"action": "open_url" | "none", "url": "..." | null}
2. Return "open_url" ONLY when the user explicitly requests opening/showing/going to a page or when the AI explicitly confirms opening it right now.
3. Do NOT trigger "open_url" if they are merely talking about a topic without navigation intent.
4. NEVER invent URLs. Use exclusively the values in the mapping.
"""


def fallback_extract_url(user_message: str, ai_response: str) -> str:
    """Extracción determinista por palabras clave solo cuando existe intención explícita de abrir/navegar."""
    user_lower = (user_message or "").lower()
    ai_lower = (ai_response or "").lower()

    nav_verbs = [
        "abrir", "abre", "abreme", "ábreme", "llevame", "llévame",
        "ir a", "ve a", "muestrame", "muéstrame", "open", "take me to", "go to", "show me"
    ]
    ai_confirmations = ["abriendo", "te llevo a", "opening", "taking you to"]

    has_nav_intent = any(v in user_lower for v in nav_verbs) or any(c in ai_lower for c in ai_confirmations)
    if not has_nav_intent:
        return None

    combined = f"{user_lower} {ai_lower}"

    if any(k in combined for k in ["api", "models provider", "proveedor de modelos"]):
        return URL_MAPPING["models_provider"]
    if any(k in combined for k in ["crear modelo", "create model", "entrenar", "qlora", "fine-tuning"]):
        return URL_MAPPING["create_model"]
    if any(k in combined for k in ["biblioteca", "library", "gguf", "descargar modelo"]):
        return URL_MAPPING["library_model"]
    if any(k in combined for k in ["chat", "asesor por texto"]):
        return URL_MAPPING["ax_chat"]
    if "panel" in combined:
        return URL_MAPPING["panel"]
    if "linkedin" in combined:
        return URL_MAPPING["linkedin"]
    if any(k in combined for k in ["metodologia", "metodología", "methodology"]):
        return URL_MAPPING["methodology"]
    if any(k in combined for k in ["solucion", "soluciones", "solutions", "servex"]):
        return URL_MAPPING["solutions"]
    if any(k in combined for k in ["industria", "industrias", "industries"]):
        return URL_MAPPING["industries"]
    if any(k in combined for k in ["seguridad", "ciberseguridad", "security"]):
        return URL_MAPPING["security"]
    if any(k in combined for k in ["ceo", "alexander", "quiroga"]):
        return URL_MAPPING["ceo"]
    if any(k in combined for k in ["contacto", "contactar", "contact", "agendar"]):
        return URL_MAPPING["contact"]
    if any(k in combined for k in ["sobre nosotros", "about", "quienes somos", "quiénes somos"]):
        return URL_MAPPING["about"]
    if any(k in combined for k in ["terminos", "términos", "condiciones", "terms"]):
        return URL_MAPPING["terms"]
    if any(k in combined for k in ["legal", "directorio", "informacolombia", "nit"]):
        return URL_MAPPING["legal"]

    return None


async def run_ax_action_agent_async(user_message: str, ai_response: str) -> str:
    """Evalúa asíncronamente si se debe abrir una URL o ruta del Panel."""
    url_found = None

    if GROQ_API_KEY:
        messages = [
            SystemMessage(content=ACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"Mensaje del Usuario: {user_message}\n\nRespuesta de la IA: {ai_response}")
        ]
        try:
            response = await llm_action.ainvoke(messages)
            data = json.loads(response.content)
            candidate = data.get("url")
            if data.get("action") == "open_url" and candidate in URL_MAPPING.values():
                url_found = candidate
        except Exception as e:
            print(f"Action Agent LLM Error: {e}")

    if not url_found:
        url_found = fallback_extract_url(user_message, ai_response)

    return url_found
