"""Bot runner script for launching Aletheia therapy bots."""

from dotenv import load_dotenv
load_dotenv()  

import sys
from loguru import logger




import argparse
import asyncio
import os
from typing import Type, Optional

from config.bot import BotConfig
from loguru import logger


async def run_bot(
    bot_class: Type, 
    config: BotConfig, 
    room_url: Optional[str] = None, 
    token: Optional[str] = None,
    livekit_url: Optional[str] = None,
    livekit_token: Optional[str] = None,
    room_name: Optional[str] = None,
) -> None:
    """Universal bot runner handling bot lifecycle for both Daily.co and LiveKit."""
    
    # ✅ SAFE: Log that bot is starting without exposing tokens/URLs
    logger.info("Starting bot process")
    
    bot = bot_class(config)

    await bot.setup_transport(
        url=livekit_url or room_url, 
        token=livekit_token or token,
        room_name=room_name
    )
    bot.create_pipeline()
    await bot.start()


def cli() -> None:
    """Parse command-line arguments and start the bot."""
    parser = argparse.ArgumentParser(description="Aletheia Therapy Bot Runner")

    # Legacy Daily.co arguments
    parser.add_argument("-u", "--room-url", type=str, help="Daily room URL (DEPRECATED)")
    parser.add_argument("-t", "--token", type=str, help="Daily room token (DEPRECATED)")

    # LiveKit arguments
    parser.add_argument(
        "-l", "--livekit-url",
        type=str,
        help="LiveKit WebSocket URL (e.g., wss://your-project.livekit.cloud)",
    )
    parser.add_argument(
        "-k", "--livekit-token",
        type=str,
        help="LiveKit JWT token",
    )
    parser.add_argument(
        "-r", "--room-name",
        type=str,
        help="LiveKit room name",
    )
    parser.add_argument(
        "-b", "--bot-type",
        type=str,
        choices=["therapy"],
        help="Type of bot (default: therapy)",
    )

    # Bot configuration
    parser.add_argument("-n", "--bot-name", type=str, help="Override BOT_NAME")
    parser.add_argument("--llm-provider", type=str, choices=["deepseek", "google", "openai"])
    parser.add_argument("--deepseek-temperature", type=float)
    parser.add_argument("--kokoro-voice", type=str)
    parser.add_argument("--kokoro-speed", type=float)

    args = parser.parse_args()

    # Validate: Must provide either LiveKit OR Daily credentials
    using_livekit = bool(args.livekit_url and args.livekit_token and args.room_name)
    using_daily = bool(args.room_url and args.token)
    
    if not (using_livekit or using_daily):
        # ✅ SAFE: Error message doesn't leak args
        parser.error("Must provide LiveKit args OR Daily args")

    # Set environment variables (no logging of these values)
    if args.bot_name: 
        os.environ["BOT_NAME"] = args.bot_name
    if args.llm_provider: 
        os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.deepseek_temperature: 
        os.environ["DEEPSEEK_TEMPERATURE"] = str(args.deepseek_temperature)
    if args.kokoro_voice: 
        os.environ["KOKORO_VOICE"] = args.kokoro_voice
    if args.kokoro_speed: 
        os.environ["KOKORO_SPEED"] = str(args.kokoro_speed)

    # Load config
    config = BotConfig()

    # Always use TherapyBot
    from bots.therapy_bot import TherapyBot
    bot_class = TherapyBot

    # Run bot
    asyncio.run(run_bot(
        bot_class, 
        config, 
        room_url=args.room_url,
        token=args.token,
        livekit_url=args.livekit_url,
        livekit_token=args.livekit_token,
        room_name=args.room_name,
    ))


if __name__ == "__main__":
    cli()