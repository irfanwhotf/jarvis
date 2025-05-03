import os
import io
import time
import requests
import numpy as np
import torch
import sounddevice as sd
import warnings
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from colorama import Fore, Style

PROVIDER_HEADERS = {
    "accept": "/",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9",
    "dnt": "1",
    "origin": "https://www.openai.fm",
    "referer": "https://www.openai.fm/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}
PROVIDER_URL = "https://www.openai.fm/api/generate"
AUDIO_CACHE = {}
MAX_CACHE_SIZE = 50
CUDA_AVAILABLE = torch.cuda.is_available()

def generate_speech(text):
    cache_key = text
    if cache_key in AUDIO_CACHE:
        return AUDIO_CACHE[cache_key]
    payload = {
        "input": text,
        "prompt": "Speak clearly and naturally.",
        "voice": "nova",
        "vibe": "null"
    }
    try:
        response = requests.post(PROVIDER_URL, headers=PROVIDER_HEADERS, data=payload, timeout=30)
        response.raise_for_status()
        if len(AUDIO_CACHE) >= MAX_CACHE_SIZE:
            AUDIO_CACHE.pop(next(iter(AUDIO_CACHE)))
        AUDIO_CACHE[cache_key] = response.content
        return response.content
    except Exception as e:
        print(f"Error generating speech: {e}")
        raise

def play_audio(audio_bytes):
    try:
        audio_io = io.BytesIO(audio_bytes)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample_rate, audio_data = wavfile.read(audio_io)
        except Exception:
            audio_io.seek(0)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_bytes)
            import subprocess
            try:
                subprocess.run(["start", temp_path], shell=True, check=True)
                time.sleep(0.5)
                return
            except Exception as e:
                print(f"Could not play audio via system player: {e}")
                raise
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        if CUDA_AVAILABLE:
            audio_tensor = torch.tensor(audio_data, device='cuda')
            audio_data = audio_tensor.cpu().numpy()
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Could not play audio: {e}")
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_bytes)
            os.system(f'start {temp_path}')
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

def generate_notification_sound(duration=0.5):
    """Generate a more distinct two-tone notification sound"""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate two tones for a more distinct sound
    freq1 = 880  # Higher frequency
    freq2 = 660  # Lower frequency

    # First half is higher tone, second half is lower tone
    half_point = int(len(t) / 2)
    note = np.zeros_like(t)
    note[:half_point] = np.sin(freq1 * t[:half_point] * 2 * np.pi)
    note[half_point:] = np.sin(freq2 * t[half_point:] * 2 * np.pi)

    # Apply a smooth envelope to avoid clicks
    envelope = np.ones_like(note)
    attack = int(0.05 * sample_rate)
    release = int(0.05 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)

    # Apply a slight fade between the two tones
    mid_fade = int(0.05 * sample_rate)
    if half_point > mid_fade:
        envelope[half_point-mid_fade:half_point+mid_fade] = np.concatenate([
            np.linspace(1, 0.7, mid_fade),
            np.linspace(0.7, 1, mid_fade)
        ])

    note = note * envelope

    # Make it louder
    note = note * 0.9

    # Convert to float32 for sounddevice
    audio = note.astype(np.float32)
    return audio, sample_rate

def play_notification():
    """Play a notification sound when wake word is detected"""
    print(f"{Fore.CYAN}Wake word detected!{Style.RESET_ALL}")
    audio, sample_rate = generate_notification_sound()
    sd.play(audio, sample_rate)
    sd.wait()

def speak(text):
    print(f"{Fore.CYAN}Speaking...{Style.RESET_ALL}")
    audio_bytes = generate_speech(text)
    play_audio(audio_bytes)