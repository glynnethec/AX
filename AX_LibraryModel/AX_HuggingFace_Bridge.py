"""
AX_HuggingFace_Bridge.py — Hugging Face & Open-Weights Download Bridge for GLYNNE Core
Provides live Hugging Face model search, repository file inspection, curated open-weights catalog,
and high-speed streaming proxy downloads for air-gapped/offline model deployment.
"""

import os
import requests
import json
from typing import List, Dict, Any, Optional

HF_API_BASE = "https://huggingface.co/api/models"
HF_RESOLVE_BASE = "https://huggingface.co"

# Curated high-performance open-weights catalog
CURATED_MODELS = [
    {
        "id": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "name": "Llama 3.3 70B Instruct (GGUF)",
        "author": "Meta AI / bartowski",
        "category": "Reasoning & General",
        "params": "70 Billion",
        "context": "128K Tokens",
        "minVram": "40 GB VRAM (Q4_K_M) / 80 GB (FP16)",
        "license": "Llama 3 Community",
        "useCase": "Complex reasoning, enterprise document analysis, and autonomous workflow orchestration in private data centers.",
        "recommendedFile": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "formats": ["GGUF Q4_K_M", "GGUF Q8_0", "EXL2 4.5bpw"],
        "downloads": 485000,
        "likes": 3420,
        "featured": True
    },
    {
        "id": "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "name": "DeepSeek R1 Distill 70B (GGUF)",
        "author": "DeepSeek AI / Unsloth",
        "category": "Reasoning & Math",
        "params": "70 Billion",
        "context": "128K Tokens",
        "minVram": "42 GB VRAM (Q4_K_M) / Dual RTX 4090",
        "license": "MIT License",
        "useCase": "Chain-of-thought mathematical reasoning, logic synthesis, and automated code generation in closed environments.",
        "recommendedFile": "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
        "formats": ["GGUF Q4_K_M", "GGUF Q5_K_M", "GGUF Q8_0"],
        "downloads": 620000,
        "likes": 5100,
        "featured": True
    },
    {
        "id": "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "name": "Qwen 2.5 Coder 32B Instruct (GGUF)",
        "author": "Alibaba Cloud",
        "category": "Code & Software",
        "params": "32 Billion",
        "context": "128K Tokens",
        "minVram": "20 GB VRAM (Q4_K_M) / Single RTX 3090/4090",
        "license": "Apache 2.0",
        "useCase": "On-premise code completion, refactoring, vulnerability scanning, and internal dev stack automation without cloud dependencies.",
        "recommendedFile": "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        "formats": ["GGUF Q4_K_M", "GGUF Q8_0", "FP16"],
        "downloads": 890000,
        "likes": 6800,
        "featured": True
    },
    {
        "id": "bartowski/Mistral-Large-Instruct-2407-GGUF",
        "name": "Mistral Large 2 (123B GGUF)",
        "author": "Mistral AI / bartowski",
        "category": "Reasoning & General",
        "params": "123 Billion",
        "context": "128K Tokens",
        "minVram": "64 GB VRAM (Q4_K_M) / Dual A100/H100",
        "license": "Mistral Commercial / Open Weights",
        "useCase": "Multilingual legal auditing, financial contract parsing, and high-scale corporate data processing.",
        "recommendedFile": "Mistral-Large-Instruct-2407-Q4_K_M.gguf",
        "formats": ["GGUF Q4_K_M", "EXL2 4.0bpw"],
        "downloads": 310000,
        "likes": 2150,
        "featured": False
    },
    {
        "id": "unsloth/phi-4-GGUF",
        "name": "Phi-4 14B Reasoning (GGUF)",
        "author": "Microsoft Research / Unsloth",
        "category": "Lightweight & Local",
        "params": "14 Billion",
        "context": "16K Tokens",
        "minVram": "10 GB VRAM (Q4_K_M) / M1/M2/M3 Mac 16GB",
        "license": "MIT License",
        "useCase": "Edge computing, workstation-local execution, sensitive local file auditing, and offline mobile workstation integration.",
        "recommendedFile": "phi-4-Q4_K_M.gguf",
        "formats": ["GGUF Q4_K_M", "GGUF Q8_0", "FP16"],
        "downloads": 540000,
        "likes": 3890,
        "featured": False
    },
    {
        "id": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "name": "Qwen 2.5 72B Instruct (GGUF)",
        "author": "Alibaba Cloud",
        "category": "Reasoning & General",
        "params": "72 Billion",
        "context": "128K Tokens",
        "minVram": "44 GB VRAM (Q4_K_M) / RTX 6000 Ada",
        "license": "Apache 2.0",
        "useCase": "High-throughput RAG search over proprietary PDF/vector databases in completely isolated local networks.",
        "recommendedFile": "qwen2.5-72b-instruct-q4_k_m.gguf",
        "formats": ["GGUF Q4_K_M", "GGUF Q8_0"],
        "downloads": 730000,
        "likes": 4900,
        "featured": False
    }
]


def format_bytes_human(size_in_bytes: int) -> str:
    """Convierte bytes a formato legible (MB, GB)."""
    if not size_in_bytes:
        return "Unknown size"
    if size_in_bytes >= 1024 ** 3:
        return f"{size_in_bytes / (1024 ** 3):.2f} GB"
    elif size_in_bytes >= 1024 ** 2:
        return f"{size_in_bytes / (1024 ** 2):.1f} MB"
    elif size_in_bytes >= 1024:
        return f"{size_in_bytes / 1024:.0f} KB"
    return f"{size_in_bytes} Bytes"


def get_curated_open_weights_catalog() -> List[Dict[str, Any]]:
    """Devuelve el catálogo curado de modelos open-weights."""
    return CURATED_MODELS


def search_huggingface_models(query: str = "", filter_tag: str = "gguf", limit: int = 24) -> List[Dict[str, Any]]:
    """
    Busca modelos en la API pública de Hugging Face.
    """
    try:
        params = {
            "search": query if query else "gguf",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(limit)
        }
        if filter_tag and filter_tag.lower() != "all":
            params["filter"] = filter_tag.lower()

        headers = {"User-Agent": "AXGLYNNE-Core/1.0"}
        response = requests.get(HF_API_BASE, params=params, headers=headers, timeout=8)
        
        if response.status_code != 200:
            return CURATED_MODELS

        data = response.json()
        results = []

        for item in data:
            model_id = item.get("id", "")
            if not model_id:
                continue

            author = model_id.split("/")[0] if "/" in model_id else "Community"
            name = model_id.split("/")[-1]

            tags = item.get("tags", [])
            pipeline_tag = item.get("pipeline_tag", "text-generation")
            downloads = item.get("downloads", 0)
            likes = item.get("likes", 0)

            results.append({
                "id": model_id,
                "name": name,
                "author": author,
                "pipeline_tag": pipeline_tag,
                "tags": tags[:6],
                "downloads": downloads,
                "likes": likes,
                "downloadUrl": f"{HF_RESOLVE_BASE}/{model_id}"
            })

        return results if results else CURATED_MODELS

    except Exception as e:
        print(f"Error searching Hugging Face models: {e}")
        return CURATED_MODELS


def get_huggingface_model_files(model_id: str) -> List[Dict[str, Any]]:
    """
    Inspecciona los archivos disponibles dentro de un repositorio de Hugging Face.
    """
    try:
        url = f"{HF_API_BASE}/{model_id}/tree/main"
        headers = {"User-Agent": "AXGLYNNE-Core/1.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return []

        items = response.json()
        files = []

        for item in items:
            if item.get("type") == "file":
                filename = item.get("path", "")
                size = item.get("size", 0)
                
                # Filtrar archivos relevantes de pesos de modelos
                ext = os.path.splitext(filename)[-1].lower()
                if ext in [".gguf", ".safetensors", ".bin", ".pt", ".onnx", ".json"]:
                    files.append({
                        "filename": filename,
                        "size_bytes": size,
                        "size_human": format_bytes_human(size),
                        "download_url": f"{HF_RESOLVE_BASE}/{model_id}/resolve/main/{filename}",
                        "proxy_url": f"/api/library/download_proxy?model_id={model_id}&filename={filename}"
                    })

        # Ordenar poniendo los archivos .gguf y .safetensors al principio
        files.sort(key=lambda x: 0 if x["filename"].endswith(".gguf") else (1 if x["filename"].endswith(".safetensors") else 2))
        return files

    except Exception as e:
        print(f"Error inspecting model tree for {model_id}: {e}")
        return []
