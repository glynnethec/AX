import torch
from transformers import pipeline

def test():
    print("Loading VibeVoice pipeline...")
    # Use MPS (Metal Performance Shaders) if available on Mac, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    synthesizer = pipeline(
        "text-to-speech",
        model="microsoft/VibeVoice-Realtime-0.5B",
        device=device,
        trust_remote_code=True
    )
    
    print("Synthesizing audio...")
    text = "Hello, this is a test of VibeVoice."
    # The pipeline should return a dictionary with 'audio' and 'sampling_rate'
    audio_output = synthesizer(text)
    
    print("Output keys:", audio_output.keys())
    print("Sampling rate:", audio_output.get("sampling_rate"))
    print("Audio shape:", audio_output.get("audio").shape)

if __name__ == "__main__":
    test()
