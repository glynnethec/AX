import os
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
)

fillers = {
    "filler_1.mp3": "Mmm...",
    "filler_2.mp3": "Déjame pensar...",
    "filler_3.mp3": "Claro...",
    "filler_4.mp3": "Okay..."
}

output_dir = "/Users/glynne/Desktop/GLYNNE_SITE_2026/public/fillers"
os.makedirs(output_dir, exist_ok=True)

for filename, text in fillers.items():
    print(f"Generating {filename}...")
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id="VmejBeYhbrcTPwDniox7", 
        model_id="eleven_multilingual_v2"
    )
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        for chunk in audio_generator:
            if chunk:
                f.write(chunk)
    print(f"Saved {filepath}")

print("All fillers generated!")
