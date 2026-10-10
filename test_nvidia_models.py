import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
url = "https://integrate.api.nvidia.com/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    models = response.json()
    model_ids = [m["id"] for m in models.get("data", [])]
    print("MATCHING ACTIVE MODELS IN NVIDIA NIM:")
    for m in model_ids:
        if any(k in m for k in ["meta", "deepseek", "qwen", "nvidia", "mistral", "microsoft"]):
            print(f" - {m}")
