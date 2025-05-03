# Jarvis 2.0 Voice Assistant

A minimal, low-latency AI voice assistant featuring:
- Wake word detection ("jarvis") via pvporcupine
- Speech-to-text via faster_whisper (small.en)
- Gemini 2.0 Flash for LLM responses
- Text-to-speech via OpenAI.fm

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Add your Gemini API key to a `.env` file:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```
3. Run the assistant:
   ```
   python main.py
   ```

Say "jarvis" to activate, then speak your query. Say "exit" or "quit" to stop. 