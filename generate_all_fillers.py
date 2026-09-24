import os
import asyncio
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import edge_tts

load_dotenv()

fillers = {
    "filler_1.mp3": "A ver...",
    "filler_2.mp3": "Déjame pensar...",
    "filler_3.mp3": "Claro...",
    "filler_6.mp3": "Comprendo...",
    "filler_7.mp3": "Veamos...",
    "filler_8.mp3": "Perfecto...",
    "filler_9.mp3": "Déjame analizarlo...",
    "filler_10.mp3": "Buena pregunta...",
    "filler_11.mp3": "Vale...",
    "filler_12.mp3": "Déjame ver...",
    "filler_14.mp3": "Entiendo la idea...",
    "filler_15.mp3": "Bien..."
}

base_dir = "/Users/glynne/Desktop/GLYNNE_SITE_2026/public/fillers"
eleven_dir = os.path.join(base_dir, "elevenlabs")
edge_dir = os.path.join(base_dir, "edge")

os.makedirs(eleven_dir, exist_ok=True)
os.makedirs(edge_dir, exist_ok=True)

# 1. Generate ElevenLabs Fillers
api_key = os.getenv("ELEVENLABS_API_KEY")
if api_key:
    client = ElevenLabs(api_key=api_key)
    for filename, text in fillers.items():
        print(f"Generating ElevenLabs {filename} ({text})...")
        try:
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id="VmejBeYhbrcTPwDniox7", 
                model_id="eleven_multilingual_v2"
            )
            filepath = os.path.join(eleven_dir, filename)
            with open(filepath, "wb") as f:
                for chunk in audio_generator:
                    if chunk:
                        f.write(chunk)
            print(f"Saved ElevenLabs {filepath}")
        except Exception as e:
            print(f"Error ElevenLabs {filename}: {e}")
else:
    print("No ELEVENLABS_API_KEY found, skipping ElevenLabs filler generation.")

# 2. Generate Edge TTS Fillers
async def generate_edge_fillers():
    for filename, text in fillers.items():
        print(f"Generating Edge TTS {filename} ({text})...")
        communicate = edge_tts.Communicate(text, "es-CO-SalomeNeural", rate="+20%", volume="+5%")
        filepath = os.path.join(edge_dir, filename)
        await communicate.save(filepath)
        print(f"Saved Edge TTS {filepath}")

asyncio.run(generate_edge_fillers())
print("All 15 fillers updated successfully!")
