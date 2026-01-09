"""Bot configuration management module."""

import os
from typing import TypedDict, Literal, NotRequired
from dotenv import load_dotenv

# NOTE: Assuming these imports are available in your environment
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService

load_dotenv()

class LiveKitConfig(TypedDict):
    api_key: str
    api_secret: str
    url: str
class DailyConfig(TypedDict):
    api_key: str
    api_url: str
    room_url: NotRequired[str]


BotType = Literal["simple", "flow", "therapy"]
LLMProvider = Literal["deepseek", "google", "openai"]
TTSProvider = Literal["kokoro", "kokoro_fastapi"]  



class BotConfig:
    def __init__(self):
        # Validate required vars
        required = {
            "DAILY_API_KEY": os.getenv("DAILY_API_KEY"),
            # REMOVED: DEEPGRAM_API_KEY - not used with local Whisper
        }
        
        # LLM API key validation based on provider
        llm_provider_check = os.getenv("LLM_PROVIDER", "deepseek").lower()
        if llm_provider_check == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY is required when using DeepSeek")
        if llm_provider_check == "google" and not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is required when using Google")
        if llm_provider_check == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required when using OpenAI")

        missing = [k for k, v in required.items() if not v]
        if missing:
            # ✅ SAFE: Generic error without listing variable names
            raise ValueError("Missing required environment variables")

        self.daily: DailyConfig = {
            "api_key": str(required["DAILY_API_KEY"]), 
            "api_url": os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        }

        if not os.getenv("LIVEKIT_API_KEY"):
            raise ValueError("LIVEKIT_API_KEY is required")
        if not os.getenv("LIVEKIT_API_SECRET"):
            raise ValueError("LIVEKIT_API_SECRET is required")
        if not os.getenv("LIVEKIT_WS_URL"):
            raise ValueError("LIVEKIT_URL is required")

        # Bot configuration
        self._bot_type: BotType = os.getenv("BOT_TYPE", "therapy")  # type: ignore[assignment]
        if self._bot_type not in ("simple", "flow", "therapy"):
            self._bot_type = "therapy"  # type: ignore[assignment]
    
    def __repr__(self) -> str:
        # ✅ SAFE: Minimal representation without full configuration details
        return f"BotConfig(bot_type={self.bot_type}, bot_name={self.bot_name})"

    def _is_truthy(self, value: str) -> bool:
        return value.lower() in (
            "true", "1", "t", "yes", "y", "on", "enable", "enabled",
        )

    ###########################################################################
    # API keys (omitted for brevity, assume they are correct)
    ###########################################################################

    @property
    def deepseek_api_key(self) -> str:
        return os.getenv("DEEPSEEK_API_KEY", "")

    @property
    def google_api_key(self) -> str:
        return os.getenv("GOOGLE_API_KEY", "")

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    # REMOVED: deepgram_api_key - not used with local Whisper
    
    # Other API keys...

    ###########################################################################
    # Bot configuration
    ###########################################################################

    @property
    def bot_type(self) -> BotType:
        return self._bot_type

    @bot_type.setter
    def bot_type(self, value: BotType):
        self._bot_type = value
        os.environ["BOT_TYPE"] = value

    @property
    def bot_name(self) -> str:
        return os.getenv("BOT_NAME", "Aletheia")  # Changed default to therapy bot name

    @bot_name.setter
    def bot_name(self, value: str):
        os.environ["BOT_NAME"] = value

    @property
    def llm_provider(self) -> LLMProvider:
        return os.getenv("LLM_PROVIDER", "deepseek").lower()  # type: ignore[return-value]

    @llm_provider.setter
    def llm_provider(self, value: str):
        value = value.lower()
        if value not in ("deepseek", "google", "openai"):
            raise ValueError(f"Invalid LLM provider: {value}")

        os.environ["LLM_PROVIDER"] = value

    # --- DeepSeek Configuration ---

    @property
    def deepseek_model(self) -> str:
        return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    @deepseek_model.setter
    def deepseek_model(self, value: str):
        os.environ["DEEPSEEK_MODEL"] = value

    @property
    def deepseek_params(self) -> DeepSeekLLMService.InputParams:
        temp = float(os.getenv("DEEPSEEK_TEMPERATURE", 0.7))
        max_t = int(os.getenv("DEEPSEEK_MAX_TOKENS", 1024))
        return DeepSeekLLMService.InputParams(temperature=temp, max_tokens=max_t)

    @deepseek_params.setter
    def deepseek_params(self, value: DeepSeekLLMService.InputParams):
        os.environ["DEEPSEEK_TEMPERATURE"] = str(value.temperature)
        os.environ["DEEPSEEK_MAX_TOKENS"] = str(value.max_tokens)

    # --- Google Configuration ---

    @property
    def google_model(self) -> str:
        return os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

    @google_model.setter
    def google_model(self, value: str):
        os.environ["GOOGLE_MODEL"] = value

    @property
    def google_params(self) -> GoogleLLMService.InputParams:
        temp = float(os.getenv("GOOGLE_TEMPERATURE", 0.7))
        max_t = int(os.getenv("GOOGLE_MAX_TOKENS", 1024))
        return GoogleLLMService.InputParams(temperature=temp, max_tokens=max_t)

    @google_params.setter
    def google_params(self, value: GoogleLLMService.InputParams):
        os.environ["GOOGLE_TEMPERATURE"] = str(value.temperature)
        os.environ["GOOGLE_MAX_TOKENS"] = str(value.max_tokens)

    # --- OpenAI Configuration ---

    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4o")

    @openai_model.setter
    def openai_model(self, value: str):
        os.environ["OPENAI_MODEL"] = value

    @property
    def openai_params(self) -> OpenAILLMService.InputParams:
        temp = float(os.getenv("OPENAI_TEMPERATURE", 0.7))
        max_t = int(os.getenv("OPENAI_MAX_TOKENS", 1024))
        return OpenAILLMService.InputParams(temperature=temp, max_tokens=max_t)

    @openai_params.setter
    def openai_params(self, value: OpenAILLMService.InputParams):
        os.environ["OPENAI_TEMPERATURE"] = str(value.temperature)
        os.environ["OPENAI_MAX_TOKENS"] = str(value.max_tokens)

    # --- TTS Configuration (Kokoro) ---

    
    @property
    def kokoro_voice(self) -> str:
        return os.getenv("KOKORO_VOICE", "af_bella")

    @kokoro_voice.setter
    def kokoro_voice(self, value: str):
        os.environ["KOKORO_VOICE"] = value

    @property
    def kokoro_speed(self) -> float:
        return float(os.getenv("KOKORO_SPEED", "0.9"))

    @kokoro_speed.setter
    def kokoro_speed(self, value: float):
        os.environ["KOKORO_SPEED"] = str(value)


    # In the TTS section of BotConfig:

    @property
    def tts_provider(self) -> TTSProvider:
        return os.getenv("TTS_PROVIDER", "kokoro_fastapi").lower()  # type: ignore[return-value]

    @tts_provider.setter
    def tts_provider(self, value: str):
        value = value.lower()
        if value not in ("kokoro", "kokoro_fastapi"):
            raise ValueError(f"Invalid TTS provider: {value}. Only 'kokoro' and 'kokoro_fastapi' are supported.")
        os.environ["TTS_PROVIDER"] = value

    # Kokoro FastAPI settings
    @property
    def kokoro_fastapi_url(self) -> str:
        return os.getenv("KOKORO_FASTAPI_URL", "http://localhost:8880")

    @kokoro_fastapi_url.setter
    def kokoro_fastapi_url(self, value: str):
        os.environ["KOKORO_FASTAPI_URL"] = value

    @property
    def kokoro_fastapi_endpoint(self) -> str:
        return os.getenv("KOKORO_FASTAPI_ENDPOINT", "/v1/audio/speech")

    @kokoro_fastapi_endpoint.setter
    def kokoro_fastapi_endpoint(self, value: str):
        os.environ["KOKORO_FASTAPI_ENDPOINT"] = value


    

    # --- STT Configuration (Whisper - Local) ---
    # NOTE: Whisper runs locally on CPU and requires NO API key

    @property
    def whisper_model(self) -> str:
        """Whisper model size: tiny, base, small, medium, large"""
        return os.getenv("WHISPER_MODEL", "tiny")

    @whisper_model.setter
    def whisper_model(self, value: str):
        os.environ["WHISPER_MODEL"] = value

    @property
    def whisper_device(self) -> str:
        """Device to run Whisper: cpu or cuda"""
        return os.getenv("WHISPER_DEVICE", "cpu")

    @whisper_device.setter
    def whisper_device(self, value: str):
        os.environ["WHISPER_DEVICE"] = value

    @property
    def whisper_language(self) -> str:
        """Language code for Whisper"""
        return os.getenv("WHISPER_LANGUAGE", "en")

    @whisper_language.setter
    def whisper_language(self, value: str):
        os.environ["WHISPER_LANGUAGE"] = value

    # --- STT Mute Filter ---

    @property
    def enable_stt_mute_filter(self) -> bool:
        return self._is_truthy(os.getenv("ENABLE_STT_MUTE_FILTER", "true"))

    @enable_stt_mute_filter.setter
    def enable_stt_mute_filter(self, value: bool):
        os.environ["ENABLE_STT_MUTE_FILTER"] = str(value)


    @property
    def livekit_api_key(self) -> str:
        return os.getenv("LIVEKIT_API_KEY", "")

    @property
    def livekit_api_secret(self) -> str:
        return os.getenv("LIVEKIT_API_SECRET", "")

    @property
    def livekit_url(self) -> str:
        return os.getenv("LIVEKIT_URL", "")