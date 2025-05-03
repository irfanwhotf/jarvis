import pvporcupine
import sounddevice as sd
import struct
import numpy as np
import torch
import time
from faster_whisper import WhisperModel
from colorama import Fore, Style

DEFAULT_WHISPER_MODEL = "small.en"  # Upgraded from small.en for better accuracy
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
COMPUTE_TYPE = "float16" if CUDA_AVAILABLE else "int8"
whisper_model = None

RATE = 16000
VAD_THRESHOLD = 0.01  # Voice activity detection threshold
SILENCE_DURATION = 1.5  # Seconds of silence to stop recording

def initialize_whisper(model_size=DEFAULT_WHISPER_MODEL):
    global whisper_model
    print(f"{Fore.CYAN}Initializing Whisper model ({model_size})...{Style.RESET_ALL}")
    try:
        whisper_model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
        print(f"{Fore.GREEN}Whisper model initialized successfully{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error initializing Whisper model: {e}{Style.RESET_ALL}")
        raise

def transcribe_audio(audio_data, sample_rate=RATE):
    if whisper_model is None:
        initialize_whisper()
    print(f"{Fore.CYAN}Transcribing...{Style.RESET_ALL}")
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    # Improved transcription with larger beam size and language hint
    segments, info = whisper_model.transcribe(
        audio_data,
        beam_size=8,  # Increased from 5 for better accuracy
        language="en",  # Explicitly set language to English
        initial_prompt="This is a voice command for an AI assistant."  # Context hint
    )
    text = " ".join(segment.text for segment in segments)
    return text

def is_silent(audio_chunk, threshold=VAD_THRESHOLD):
    """Detect if an audio chunk is silent based on RMS amplitude"""
    # Convert to float for calculation if needed
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32) / np.iinfo(np.int16).max
    # Calculate RMS amplitude
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms < threshold

def listen_for_speech():
    from tts import play_notification  # Import here to avoid circular imports

    porcupine = None
    stream = None
    try:
        # Create porcupine with higher sensitivity (0.7) for better wake word detection
        # Sensitivity range is 0-1, where 1 is most sensitive (may have more false positives)
        # Default is 0.5, we're increasing to 0.7 for better detection
        porcupine = pvporcupine.create(keywords=["jarvis"], sensitivities=[0.7])
        stream = sd.InputStream(
            samplerate=porcupine.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=porcupine.frame_length
        )
        stream.start()
        print(f"{Fore.GREEN}Say 'jarvis' to activate...{Style.RESET_ALL}")

        # Wait for wake word
        while True:
            pcm = stream.read(porcupine.frame_length)[0]
            pcm = struct.unpack_from("%dh" % porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                # Play notification sound when wake word is detected
                play_notification()
                print(f"{Fore.CYAN}Listening...{Style.RESET_ALL}")
                break

        # Dynamic recording with voice activity detection
        print(f"{Fore.CYAN}Recording...{Style.RESET_ALL}")

        # Setup recording parameters
        audio_chunks = []
        silence_counter = 0
        max_duration = 15  # Maximum recording duration in seconds
        chunk_duration = 0.1  # Duration of each audio chunk in seconds
        chunk_samples = int(RATE * chunk_duration)

        # Create a new stream for recording with smaller chunks for better VAD
        rec_stream = sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype='int16',
            blocksize=chunk_samples
        )
        rec_stream.start()

        # Wait for voice to start
        voice_started = False
        start_time = time.time()

        while time.time() - start_time < max_duration:
            chunk, _ = rec_stream.read(chunk_samples)
            chunk = chunk.flatten()

            # If voice hasn't started yet, wait for it
            if not voice_started:
                if not is_silent(chunk):
                    voice_started = True
                    audio_chunks.append(chunk)
                continue

            # Add the chunk to our recording
            audio_chunks.append(chunk)

            # Check if this chunk is silent
            if is_silent(chunk):
                silence_counter += 1
                # If we've had enough silence, stop recording
                if silence_counter >= int(SILENCE_DURATION / chunk_duration):
                    print(f"{Fore.CYAN}Voice input complete.{Style.RESET_ALL}")
                    break
            else:
                # Reset silence counter if we hear something
                silence_counter = 0

        # Close the recording stream
        rec_stream.stop()
        rec_stream.close()

        # Combine all audio chunks
        if not audio_chunks:
            print(f"{Fore.RED}No speech detected.{Style.RESET_ALL}")
            return ""

        audio_data = np.concatenate(audio_chunks)
        return transcribe_audio(audio_data, RATE)
    except Exception as e:
        print(f"{Fore.RED}Error in STT: {e}{Style.RESET_ALL}")
        return ""
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        if porcupine is not None:
            porcupine.delete()