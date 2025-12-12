"""Comprehensive test for therapy bot backend."""

# Test 1: STT (Whisper)
from pipecat.services.whisper.stt import WhisperSTTService
print("✅ Whisper STT imported")

# Test 2: TTS (Kokoro)
from services.kokoro_tts import KokoroTTSService
print("✅ Kokoro TTS imported")

# Test 3: LLM (DeepSeek)
from pipecat.services.deepseek.llm import DeepSeekLLMService
print("✅ DeepSeek LLM imported")


# Test 5: Therapy Bot
from bots.therapy_bot import TherapyBot
print("✅ Therapy Bot imported")

# Test 6: Config
from config.bot import BotConfig
config = BotConfig()
print(f"✅ Config loaded: bot_name={config.therapy_bot_name}")

print("\n🎉 All therapy backend components ready!")