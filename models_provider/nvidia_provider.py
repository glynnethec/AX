import os
import json
import requests
from typing import Dict, Any, List, Generator, Optional
from dotenv import load_dotenv

load_dotenv()

# Base URL para build.nvidia.com NIM APIs
NVIDIA_NIM_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Mapeo de IDs públicos de GLYNNE a los nombres exactos de modelos activos en build.nvidia.com
DEFAULT_MODEL_MAP: Dict[str, str] = {
    # Alias para retrocompatibilidad con scripts existentes
    "llama-3.2-11b-vision": "meta/llama-3.1-8b-instruct",
    "llama-3.2-90b-vision": "meta/llama-3.3-70b-instruct",

    # Modelos Meta Llama activos
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
    "llama-3.1-8b": "meta/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",

    # Modelos NVIDIA y Mistral probados
    "mistral-nemo-12b": "nv-mistralai/mistral-nemo-12b-instruct",
    "nvidia-nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",

    # DeepSeek y Qwen
    "deepseek-r1": "deepseek-ai/deepseek-r1",
    "deepseek-v3": "deepseek-ai/deepseek-v3",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "qwen-2.5-coder-32b": "qwen/qwen2.5-coder-32b-instruct",
    "qwen-2.5-72b": "qwen/qwen2.5-72b-instruct",

    # Microsoft
    "phi-3.5-moe": "microsoft/phi-3.5-moe-instruct",
}


# Metadatos del catálogo de modelos
CATALOG_METADATA = [
    {
        "id": "llama-3.3-70b",
        "name": "Meta Llama 3.3 70B Instruct",
        "nvidia_target": "meta/llama-3.3-70b-instruct",
        "provider": "Meta AI / NVIDIA NIM",
        "description": "El modelo insignia de Meta Llama más avanzado para razonamiento complejo y agentes.",
        "context_length": 131072,
    },
    {
        "id": "llama-3.1-8b",
        "name": "Meta Llama 3.1 8B Instruct",
        "nvidia_target": "meta/llama-3.1-8b-instruct",
        "provider": "Meta AI / NVIDIA NIM",
        "description": "Modelo de ultra baja latencia, eficiente y versátil para tareas conversacionales.",
        "context_length": 131072,
    },
    {
        "id": "llama-3.2-11b-vision",
        "name": "Meta Llama 3.1 (Compatibilidad)",
        "nvidia_target": "meta/llama-3.1-8b-instruct",
        "provider": "Meta AI / NVIDIA NIM",
        "description": "Ruta de compatibilidad retroactiva optimizada con Llama 3.1.",
        "context_length": 131072,
    },
    {
        "id": "mistral-nemo-12b",
        "name": "Mistral NeMo 12B Instruct",
        "nvidia_target": "nv-mistralai/mistral-nemo-12b-instruct",
        "provider": "Mistral AI / NVIDIA NIM",
        "description": "Modelo insignia optimizado por Mistral y NVIDIA para respuestas rápidas y precisas.",
        "context_length": 131072,
    },
    {
        "id": "nvidia-nemotron-70b",
        "name": "NVIDIA Nemotron 70B Instruct",
        "nvidia_target": "nvidia/llama-3.1-nemotron-70b-instruct",
        "provider": "NVIDIA NIM",
        "description": "Modelo optimizado por NVIDIA con RLAIF para máxima fidelidad y apego a instrucciones.",
        "context_length": 131072,
    },
    {
        "id": "qwen-2.5-coder-32b",
        "name": "Qwen 2.5 Coder 32B Instruct",
        "nvidia_target": "qwen/qwen2.5-coder-32b-instruct",
        "provider": "Qwen / NVIDIA NIM",
        "description": "Líder en generación, autocompletado, refactorización y depuración de código.",
        "context_length": 131072,
    },
    {
        "id": "deepseek-r1",
        "name": "DeepSeek R1 Reasoning",
        "nvidia_target": "deepseek-ai/deepseek-r1",
        "provider": "DeepSeek AI / NVIDIA NIM",
        "description": "Modelo de razonamiento paso a paso de alto nivel para problemas complejos y matemáticas.",
        "context_length": 65536,
    },
    {
        "id": "deepseek-v3",
        "name": "DeepSeek V3",
        "nvidia_target": "deepseek-ai/deepseek-v3",
        "provider": "DeepSeek AI / NVIDIA NIM",
        "description": "Modelo fundacional de arquitectura MoE de gran capacidad y velocidad.",
        "context_length": 65536,
    },
    {
        "id": "phi-3.5-moe",
        "name": "Microsoft Phi-3.5 MoE",
        "nvidia_target": "microsoft/phi-3.5-moe-instruct",
        "provider": "Microsoft / NVIDIA NIM",
        "description": "Modelo Mixture-of-Experts para lógica, ciencias y matemáticas.",
        "context_length": 131072,
    },
]


# La clave de NVIDIA se lee exclusivamente del entorno (.env / variables de Render). Nunca hardcodeada.
DEFAULT_NVIDIA_API_KEY = None

class NVIDIAProviderEngine:
    def __init__(self):
        self.base_url = NVIDIA_NIM_BASE_URL
        self.model_map = DEFAULT_MODEL_MAP

    def _get_api_key(self, model_alias: str, override_key: Optional[str] = None) -> Optional[str]:
        """
        Obtiene la API Key de NVIDIA desde override_key o .env.
        Soporta llaves específicas por modelo o una llave general NVIDIA_API_KEY.
        """
        if override_key and len(override_key) > 5:
            return override_key
        model_env_var = f"NVIDIA_API_KEY_{model_alias.upper().replace('-', '_')}"
        specific_key = os.getenv(model_env_var)
        if specific_key:
            return specific_key
        return os.getenv("NVIDIA_API_KEY") or DEFAULT_NVIDIA_API_KEY

    def resolve_nvidia_model(self, model_alias: str) -> str:
        """Traduce el alias del modelo al identificador exacto de NVIDIA NIM."""
        return self.model_map.get(model_alias, model_alias)

    def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        top_p: Optional[float] = None,
        override_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta una llamada síncrona a build.nvidia.com y retorna la respuesta en formato OpenAI.
        """
        api_key = self._get_api_key(model, override_key=override_key)
        if not api_key:
            raise ValueError(
                f"No se ha configurado la API Key de NVIDIA en el .env (NVIDIA_API_KEY o NVIDIA_API_KEY_{model.upper().replace('-', '_')})"
            )

        target_model = self.resolve_nvidia_model(model)
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if top_p is not None:
            payload["top_p"] = top_p

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )

        STABLE_FALLBACK_MODEL = "nv-mistralai/mistral-nemo-12b-instruct"

        if response.status_code in (404, 410) and target_model != STABLE_FALLBACK_MODEL:
            # Si el modelo fue retirado (410) o no existe (404), fallback transparente a modelo activo y probado
            fallback_payload = dict(payload)
            fallback_payload["model"] = STABLE_FALLBACK_MODEL
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=fallback_payload,
                headers=headers,
                timeout=120,
            )

        if response.status_code != 200:
            try:
                err_detail = response.json()
            except Exception:
                err_detail = response.text
            raise RuntimeError(f"NVIDIA API Error [{response.status_code}]: {err_detail}")

        data = response.json()
        if isinstance(data, dict) and "model" in data:
            data["model"] = model  # Firma con la marca del cliente/GLYNNE
            data["owned_by"] = "axglynne-ai-provider"

        return data

    def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: Optional[float] = None,
        override_key: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Ejecuta una llamada en streaming (SSE) a build.nvidia.com y genera líneas data: ...
        """
        api_key = self._get_api_key(model, override_key=override_key)
        if not api_key:
            yield f"data: {json.dumps({'error': 'Missing NVIDIA_API_KEY in .env'})}\n\n"
            return

        target_model = self.resolve_nvidia_model(model)
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if top_p is not None:
            payload["top_p"] = top_p

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            stream=True,
            timeout=120,
        )

        STABLE_FALLBACK_MODEL = "nv-mistralai/mistral-nemo-12b-instruct"
        if response.status_code in (404, 410) and target_model != STABLE_FALLBACK_MODEL:
            fallback_payload = dict(payload)
            fallback_payload["model"] = STABLE_FALLBACK_MODEL
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=fallback_payload,
                headers=headers,
                stream=True,
                timeout=120,
            )

        for line in response.iter_lines(decode_unicode=True):
            if line:
                yield f"{line}\n"

def get_available_models() -> List[Dict[str, Any]]:
    """Devuelve el catálogo de modelos disponibles."""
    return CATALOG_METADATA
