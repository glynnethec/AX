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
    "llama-3.2-11b-vision": "meta/llama-3.2-11b-vision-instruct",
    "llama-3.2-90b-vision": "meta/llama-3.2-90b-vision-instruct",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-v4-flash": "deepseek-ai/deepseek-v4.1-flash",
    "nvidia-nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistral-nemo-12b": "nv-mistralai/mistral-nemo-12b-instruct",
    "phi-3.5-moe": "microsoft/phi-3.5-moe-instruct",
}


# Metadatos del catálogo de modelos
CATALOG_METADATA = [
    {
        "id": "llama-3.2-11b-vision",
        "name": "Meta Llama 3.2 11B Vision",
        "nvidia_target": "meta/llama-3.2-11b-vision-instruct",
        "provider": "Meta AI / NVIDIA NIM",
        "description": "Modelo multimodal avanzado de visión e imágenes en tiempo real.",
        "context_length": 131072,
    },
    {
        "id": "llama-3.2-90b-vision",
        "name": "Meta Llama 3.2 90B Vision",
        "nvidia_target": "meta/llama-3.2-90b-vision-instruct",
        "provider": "Meta AI / NVIDIA NIM",
        "description": "Modelo visión insignia para análisis complejo de imágenes, gráficos y PDFs.",
        "context_length": 131072,
    },
    {
        "id": "deepseek-coder-6.7b",
        "name": "DeepSeek Coder 6.7B",
        "nvidia_target": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "provider": "DeepSeek AI / NVIDIA NIM",
        "description": "Especializado en autocompletado de código y refactorización ultrarrápida.",
        "context_length": 65536,
    },
    {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek V4.1 Flash",
        "nvidia_target": "deepseek-ai/deepseek-v4.1-flash",
        "provider": "DeepSeek AI / NVIDIA NIM",
        "description": "Modelo general de ultra baja latencia y alta precisión.",
        "context_length": 65536,
    },
    {
        "id": "nvidia-nemotron-70b",
        "name": "NVIDIA Nemotron 70B",
        "nvidia_target": "nvidia/llama-3.1-nemotron-70b-instruct",
        "provider": "NVIDIA NIM",
        "description": "Modelo optimizado por NVIDIA con RLAIF para alta fidelidad de respuesta.",
        "context_length": 131072,
    },
    {
        "id": "mistral-nemo-12b",
        "name": "Mistral NeMo 12B",
        "nvidia_target": "nv-mistralai/mistral-nemo-12b-instruct",
        "provider": "Mistral AI / NVIDIA NIM",
        "description": "Modelo ligero y rápido optimizado para agentes conversacionales en tiempo real.",
        "context_length": 131072,
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


class NVIDIAProviderEngine:
    def __init__(self):
        self.base_url = NVIDIA_NIM_BASE_URL
        self.model_map = DEFAULT_MODEL_MAP

    def _get_api_key(self, model_alias: str) -> Optional[str]:
        """
        Obtiene la API Key de NVIDIA desde .env.
        Soporta llaves específicas por modelo o una llave general NVIDIA_API_KEY.
        Ej: NVIDIA_API_KEY_QWEN, NVIDIA_API_KEY_DEEPSEEK, etc.
        """
        model_env_var = f"NVIDIA_API_KEY_{model_alias.upper().replace('-', '_')}"
        specific_key = os.getenv(model_env_var)
        if specific_key:
            return specific_key
        return os.getenv("NVIDIA_API_KEY")

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
    ) -> Dict[str, Any]:
        """
        Ejecuta una llamada síncrona a build.nvidia.com y retorna la respuesta en formato OpenAI.
        """
        api_key = self._get_api_key(model)
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
    ) -> Generator[str, None, None]:
        """
        Ejecuta una llamada en streaming (SSE) a build.nvidia.com y genera líneas data: ...
        """
        api_key = self._get_api_key(model)
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

        for line in response.iter_lines(decode_unicode=True):
            if line:
                yield f"{line}\n"

def get_available_models() -> List[Dict[str, Any]]:
    """Devuelve el catálogo de modelos disponibles."""
    return CATALOG_METADATA
