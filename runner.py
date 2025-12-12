import argparse
import asyncio
import os
from typing import Type

from config.bot import BotConfig
from loguru import logger



async def run_bot(bot_class: Type, config: BotConfig, room_url: str, token: str) -> None:
    """Universal bot runner handling bot lifecycle.

    Args:
        bot_class: The bot class to instantiate (e.g. TherapyBot or SimpleBot)
        config: The configuration instance to use (with bot_type possibly overridden)
        room_url: The Daily room URL
        token: The Daily room token
    """
    # Instantiate the bot using the provided configuration instance.
    bot = bot_class(config)

    # Set up transport and pipeline.
    await bot.setup_transport(room_url, token)
    bot.create_pipeline()

    # Start the bot.2q
    await bot.start()


def cli() -> None:
    """Parse command-line arguments, override configuration if needed, and start the bot."""
    parser = argparse.ArgumentParser(description="Aletheia Therapy Bot Runner")

    # Required arguments
    parser.add_argument("-u", "--room-url", type=str, required=True, help="Daily room URL")
    parser.add_argument("-t", "--token", type=str, required=True, help="Authentication token")

    # Bot type selection
    parser.add_argument(
        "-b",
        "--bot-type",
        type=str.lower,
        choices=["simple", "therapy"],
        help="Type of bot (overrides BOT_TYPE in configuration)",
    )

    # Bot name configuration
    parser.add_argument(
        "-n",
        "--bot-name",
        type=str,
        help="Override BOT_NAME (default: Aletheia)",
    )

    # LLM configuration
    parser.add_argument(
        "-l",
        "--llm-provider",
        type=str.lower,
        choices=["deepseek", "google", "openai"],
        help="Override LLM_PROVIDER",
    )

    # DeepSeek configuration
    parser.add_argument(
        "--deepseek-model",
        type=str,
        help="Override DEEPSEEK_MODEL (default: deepseek-chat)",
    )

    parser.add_argument(
        "--deepseek-temperature",
        type=float,
        help="Override DEEPSEEK_TEMPERATURE (default: 1.3)",
    )

    # Google configuration
    parser.add_argument(
        "-m",
        "--google-model",
        type=str,
        help="Override GOOGLE_MODEL",
    )

    parser.add_argument(
        "-T",
        "--google-temperature",
        type=float,
        help="Override GOOGLE_TEMPERATURE (default: 1.3)",
    )

    # OpenAI configuration
    parser.add_argument(
        "--openai-model",
        type=str,
        help="Override OPENAI_MODEL (default: gpt-4o)",
    )
    parser.add_argument(
        "--openai-temperature",
        type=float,
        help="Override OPENAI_TEMPERATURE (default: 1.3)",
    )

    # TTS configuration
    parser.add_argument(
        "-p",
        "--tts-provider",
        type=str.lower,
        choices=["kokoro", "kokoro_streaming"],
        help="Override TTS_PROVIDER (default: kokoro_streaming)",
    )

    # Kokoro voice configuration
    parser.add_argument(
        "--kokoro-voice",
        type=str,
        help="Override KOKORO_VOICE (default: af_bella)",
    )

    parser.add_argument(
        "--kokoro-speed",
        type=float,
        help="Override KOKORO_SPEED (default: 0.95)",
    )

    # STT mute filter configuration
    parser.add_argument(
        "--enable-stt-mute-filter",
        type=lambda x: str(x).lower() in ("true", "1", "t", "yes", "y", "on", "enable", "enabled"),
        help="Override ENABLE_STT_MUTE_FILTER (true/false)",
    )

    args = parser.parse_args()

    # Set environment variables based on CLI arguments
    if args.bot_type:
        os.environ["BOT_TYPE"] = args.bot_type
    if args.bot_name:
        os.environ["BOT_NAME"] = args.bot_name
    if args.llm_provider:
        os.environ["LLM_PROVIDER"] = args.llm_provider.lower()
    
    # DeepSeek
    if args.deepseek_model:
        os.environ["DEEPSEEK_MODEL"] = args.deepseek_model
    if args.deepseek_temperature is not None:
        os.environ["DEEPSEEK_TEMPERATURE"] = str(args.deepseek_temperature)
    
    # Google
    if args.google_model:
        os.environ["GOOGLE_MODEL"] = args.google_model
    if args.google_temperature is not None:
        os.environ["GOOGLE_TEMPERATURE"] = str(args.google_temperature)
    
    # OpenAI
    if args.openai_model:
        os.environ["OPENAI_MODEL"] = args.openai_model
    if args.openai_temperature is not None:
        os.environ["OPENAI_TEMPERATURE"] = str(args.openai_temperature)
    
    # TTS
    if args.tts_provider:
        os.environ["TTS_PROVIDER"] = args.tts_provider.lower()
    
    # Kokoro
    if args.kokoro_voice:
        os.environ["KOKORO_VOICE"] = args.kokoro_voice
    if args.kokoro_speed is not None:
        os.environ["KOKORO_SPEED"] = str(args.kokoro_speed)
    
    # STT mute filter
    if args.enable_stt_mute_filter is not None:
        os.environ["ENABLE_STT_MUTE_FILTER"] = "false"
        os.environ["ENABLE_STT_MUTE_FILTER"] = str(args.enable_stt_mute_filter).lower()

    # Instantiate the configuration AFTER setting environment variables
    config = BotConfig()

    # Always use TherapyBot
    from bots.therapy_bot import TherapyBot
    bot_class = TherapyBot

    asyncio.run(run_bot(bot_class, config, room_url=args.room_url, token=args.token))



if __name__ == "__main__":
    cli()