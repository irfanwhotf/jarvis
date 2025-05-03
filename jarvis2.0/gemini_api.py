import os
import time
import sys
import google.generativeai as genai
from colorama import Fore, Style

class GeminiAssistant:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print(f"{Fore.RED}Error: GEMINI_API_KEY not found in environment variables{Style.RESET_ALL}")
            sys.exit(1)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 1024,
            }
        )
        self.chat = self.model.start_chat(history=[])

    def get_response(self, user_input):
        try:
            print(f"{Fore.CYAN}Thinking...{Style.RESET_ALL}")
            start_time = time.time()
            response = self.chat.send_message(user_input)
            response_time = time.time() - start_time
            print(f"{Fore.BLUE}Response time: {response_time:.2f}s{Style.RESET_ALL}")
            return response.text
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return f"I'm sorry, I encountered an error: {str(e)}" 