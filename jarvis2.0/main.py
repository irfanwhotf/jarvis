import os
from dotenv import load_dotenv
import asyncio
from tts import speak
from stt import listen_for_speech, initialize_whisper
from gemini_api import GeminiAssistant
from colorama import init, Fore, Style

# Always load .env from the directory where main.py is located
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path)

init()

def print_system_info():
    print(f"{Fore.GREEN}=== Jarvis 2.0 Voice Assistant ===")
    print(f"Say 'jarvis' to activate. A sound will play when activated.")
    print(f"After the sound, start speaking and pause when done.")
    print(f"Say 'exit' or 'quit' to end the chat{Style.RESET_ALL}")
    print()

async def run_async(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

async def voice_assistant():
    print(f"{Fore.CYAN}Initializing voice assistant...{Style.RESET_ALL}")
    await run_async(initialize_whisper)
    gemini = GeminiAssistant()
    print_system_info()
    greeting = "Hello! I'm your voice assistant powered by Gemini. Say 'jarvis' to activate."
    print(f"{Fore.YELLOW}Assistant: {greeting}{Style.RESET_ALL}")
    await run_async(speak, greeting)

    while True:
        user_input = await run_async(listen_for_speech)

        # Handle empty input
        if not user_input.strip():
            print(f"{Fore.RED}No speech detected. Please try again.{Style.RESET_ALL}")
            continue

        # Clean up the transcription a bit
        user_input = user_input.strip()
        print(f"{Fore.GREEN}You: {user_input}{Style.RESET_ALL}")

        # Check for exit commands
        if any(exit_word in user_input.lower() for exit_word in ['exit', 'quit', 'bye']):
            farewell = "Goodbye! Have a great day!"
            print(f"{Fore.YELLOW}Assistant: {farewell}{Style.RESET_ALL}")
            await run_async(speak, farewell)
            break

        # Get response from Gemini
        response_text = await run_async(gemini.get_response, user_input)
        print(f"{Fore.YELLOW}Assistant: {response_text}{Style.RESET_ALL}")
        await run_async(speak, response_text)

def main():
    try:
        asyncio.run(voice_assistant())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()