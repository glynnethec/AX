import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "nv-mistralai/mistral-nemo-12b-instruct",
    "messages": [
        {"role": "user", "content": "Escribe una funcion rapida en Python para sumar dos numeros."}
    ],
    "max_tokens": 100,
    "stream": False
}


response = requests.post(invoke_url, headers=headers, json=payload)
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("SUCCESSFUL RESPONSE FROM QWEN CODER:")
    print(data["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.text}")
