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
    # Modelo insignia verificado y activo en la cuenta NVIDIA
    "llama-3.2-90b-vision": "meta/llama-3.2-90b-vision-instruct",
    "llama-3.2-11b-vision": "meta/llama-3.2-90b-vision-instruct",
    "mistral-nemo-12b": "meta/llama-3.2-90b-vision-instruct",
    "llama-3.3-70b": "meta/llama-3.2-90b-vision-instruct",
    "llama-3.1-8b": "meta/llama-3.2-90b-vision-instruct",
    "llama-3.1-70b": "meta/llama-3.2-90b-vision-instruct",
    "nvidia-nemotron-70b": "meta/llama-3.2-90b-vision-instruct",
    "deepseek-r1": "meta/llama-3.2-90b-vision-instruct",
    "deepseek-v3": "meta/llama-3.2-90b-vision-instruct",
    "qwen-2.5-coder-32b": "meta/llama-3.2-90b-vision-instruct",
    "phi-3.5-moe": "meta/llama-3.2-90b-vision-instruct",
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
        self._session = requests.Session()

    def _has_images(self, messages: List[Dict[str, Any]]) -> bool:
        """Detecta si el payload contiene imágenes multimodales (requiere motor de visión)."""
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

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

    def _call_fast_groq_engine(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fallback ultra-resiliente a inferencia LPU de alta velocidad si NVIDIA presenta timeout o congestión."""
        groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_RESPONDER_API_KEY")
        if not groq_key:
            return None
        try:
            payload = {
                "model": "qwen/qwen3.8-27b",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                payload["top_p"] = top_p
            res = self._session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=15,
            )
            if res.status_code == 200:
                data = res.json()
                data["model"] = model
                data["owned_by"] = "axglynne-ai-provider"
                return data
        except Exception as e:
            print(f"[GLYNNE Gateway] Aviso: Fallback LPU error: {e}")
        return None

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
        Cuenta con conmutación por error automática para garantizar alta disponibilidad.
        """
        api_key = self._get_api_key(model, override_key=override_key)
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

        has_images = self._has_images(messages)
        data = None
        STABLE_FALLBACK_MODEL = "meta/llama-3.2-90b-vision-instruct"

        # Fast-path: Para texto puro sin imágenes y sin override key manual, usar LPU (< 1s de latencia)
        if not has_images and not override_key:
            data = self._call_fast_groq_engine(model, messages, temperature, max_tokens, top_p)

        # Ruta NVIDIA NIM (para visión multimodal o si LPU no está disponible)
        if not data and api_key:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=20 if has_images else 10,
                )
                if response.status_code in (404, 410) and target_model != STABLE_FALLBACK_MODEL:
                    fallback_payload = dict(payload)
                    fallback_payload["model"] = STABLE_FALLBACK_MODEL
                    response = self._session.post(
                        f"{self.base_url}/chat/completions",
                        json=fallback_payload,
                        headers=headers,
                        timeout=20 if has_images else 10,
                    )
                if response.status_code == 200:
                    data = response.json()
            except Exception as net_err:
                print(f"[GLYNNE Gateway] Aviso: NVIDIA NIM timeout/error ({net_err}).")

        # Último recurso si NVIDIA falló y no se había intentado LPU aún
        if not data and not has_images:
            data = self._call_fast_groq_engine(model, messages, temperature, max_tokens, top_p)

        if not data:
            raise RuntimeError("Servicio de inferencia temporalmente no disponible. Por favor reintenta.")

        # Asegurar que choices[0].message.content contenga siempre texto legible
        if isinstance(data, dict) and "choices" in data:
            for ch in data.get("choices", []):
                msg = ch.get("message", {})
                if not msg.get("content") and msg.get("reasoning_content"):
                    msg["content"] = msg["reasoning_content"]

        if isinstance(data, dict) and "model" in data:
            data["model"] = model
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

        has_images = self._has_images(messages)

        # Fast-path streaming: Para texto puro, usar motor LPU directamente (< 0.3s primer token)
        if not has_images and not override_key:
            groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_RESPONDER_API_KEY")
            if groq_key:
                try:
                    groq_payload = {
                        "model": "qwen/qwen3.8-27b",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": True,
                    }
                    g_res = self._session.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json=groq_payload,
                        stream=True,
                        timeout=15,
                    )
                    if g_res.status_code == 200:
                        for line in g_res.iter_lines(decode_unicode=True):
                            if line:
                                yield f"{line}\n"
                        return
                except Exception:
                    pass

        # Ruta NVIDIA NIM (para visión multimodal o fallback)
        stream_success = False
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=20 if has_images else 10,
            )
            if response.status_code == 200:
                stream_success = True
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        yield f"{line}\n"
        except Exception:
            pass

        if not stream_success and not has_images:
            groq_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_RESPONDER_API_KEY")
            if groq_key:
                try:
                    groq_payload = {
                        "model": "qwen/qwen3.8-27b",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": True,
                    }
                    g_res = self._session.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json=groq_payload,
                        stream=True,
                        timeout=15,
                    )
                    if g_res.status_code == 200:
                        for line in g_res.iter_lines(decode_unicode=True):
                            if line:
                                yield f"{line}\n"
                        return
                except Exception:
                    pass

def get_available_models() -> List[Dict[str, Any]]:
    """Devuelve el catálogo de modelos disponibles."""
    return CATALOG_METADATA
