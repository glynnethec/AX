import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
url = "https://integrate.api.nvidia.com/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

r = requests.get(url, headers=headers)
models = [m["id"] for m in r.json().get("data", [])]

print(f"Testing key across {len(models)} models...")

for m in models:
    test_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    p = {
        "model": m,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
    resp = requests.post(test_url, headers=headers, json=p)
    if resp.status_code == 200:
        print(f"✅ SUCCESS on model: {m}")
        print(f"   Content: {resp.json()}")
        break
    elif resp.status_code != 404 and resp.status_code != 410:
        print(f"⚠️ Status {resp.status_code} on {m}: {resp.text[:100]}")
