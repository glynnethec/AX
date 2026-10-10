"""
models_provider Package
Módulo de Integración con build.nvidia.com (NVIDIA NIM APIs)
Provee inferencia de modelos de lenguaje en formato OpenAI-compatible para clientes de GLYNNE.
"""

from .nvidia_provider import NVIDIAProviderEngine, get_available_models
from .router import router as models_provider_router

__all__ = ["NVIDIAProviderEngine", "get_available_models", "models_provider_router"]
