"""Bot runner script for launching Aletheia therapy bots."""

from dotenv import load_dotenv
load_dotenv()

import sys
from loguru import logger

#logger.remove()
#logger.add(sys.stderr, level="WARNING")

import argparse
import asyncio
import os
import json
from typing import Type, Optional

from config.bot import BotConfig


async def run_bot(
    bot_class: Type,
    config: BotConfig,
    livekit_url: Optional[str] = None,
    livekit_token: Optional[str] = None,
    room_name: Optional[str] = None,
    room_url: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    logger.info("Starting bot process")

    bot = bot_class(config)
    await bot.setup_transport(
        url=livekit_url or room_url,
        token=livekit_token or token,
        room_name=room_name,
    )
    bot.create_pipeline()
    await bot.start()


async def idle_wait(pid: int, livekit_url: str) -> dict:
    """
    Pre-warm: import everything, then block until main.py drops
    a .bot_assign_{pid} file with room/token/user_id.
    Polls every 200ms — negligible CPU.
    Times out after 10 minutes to prevent zombie processes.
    """
    server_dir = os.path.dirname(os.path.abspath(__file__))
    signal_path = os.path.join(server_dir, f".bot_assign_{pid}")

    logger.warning(f"Bot {pid} idle and warmed — waiting for room assignment")

    timeout = 600  # 10 minutes max idle
    elapsed = 0

    while elapsed < timeout:
        if os.path.exists(signal_path):
            try:
                with open(signal_path, "r") as f:
                    assignment = json.load(f)
                os.remove(signal_path)
                logger.warning(f"Bot {pid} received room assignment")
                return assignment
            except (json.JSONDecodeError, OSError):
                # File not fully written yet — wait one more tick
                await asyncio.sleep(0.1)
                continue

        await asyncio.sleep(0.2)
        elapsed += 0.2

    logger.warning(f"Bot {pid} idle timeout after {timeout}s — shutting down")
    sys.exit(0)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Lily Therapy Bot Runner")

    # LiveKit arguments
    parser.add_argument("-l", "--livekit-url", type=str)
    parser.add_argument("-k", "--livekit-token", type=str)
    parser.add_argument("-r", "--room-name", type=str)

    # Legacy Daily.co (kept for fallback)
    parser.add_argument("-u", "--room-url", type=str)
    parser.add_argument("-t", "--token", type=str)

    # Bot config overrides
    parser.add_argument("-b", "--bot-type", type=str, choices=["therapy"])
    parser.add_argument("-n", "--bot-name", type=str)
    parser.add_argument("--llm-provider", type=str, choices=["deepseek", "google", "openai", "openrouter"])
    parser.add_argument("--deepseek-temperature", type=float)
    parser.add_argument("--kokoro-voice", type=str)
    parser.add_argument("--kokoro-speed", type=float)

    # ── NEW: idle/pre-warm mode ───────────────────────────────────────────────
    parser.add_argument(
        "--idle",
        action="store_true",
        help="Start in idle mode: pre-warm imports then wait for room assignment",
    )

    args = parser.parse_args()

    # Apply env overrides
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

    # ── Pre-warm: import the heavy stuff before we even check --idle ─────────
    # This is the whole point — these imports are what take 8-15s.
    # They happen at process start regardless of idle vs direct mode.
    config = BotConfig()
    from bots.therapy_bot import TherapyBot
    bot_class = TherapyBot

    # ── IDLE MODE ─────────────────────────────────────────────────────────────
    if args.idle:
        pid = os.getpid()
        livekit_url = args.livekit_url or os.environ.get("LIVEKIT_URL", "")

        assignment = asyncio.run(idle_wait(pid, livekit_url))

        # Unpack assignment from main.py
        room_name = assignment.get("room")
        token = assignment.get("token")
        user_id = assignment.get("user_id")
        livekit_url = assignment.get("livekit_url", livekit_url)

        if not room_name or not token:
            logger.error(f"Bot {pid} got invalid assignment — shutting down")
            sys.exit(1)

        # Inject user_id into env so TherapyBot can read it
        if user_id:
            os.environ["BOT_USER_ID"] = user_id

        asyncio.run(run_bot(
            bot_class,
            config,
            livekit_url=livekit_url,
            livekit_token=token,
            room_name=room_name,
        ))
        return

    # ── DIRECT MODE (existing behaviour, used as fallback) ───────────────────
    using_livekit = bool(args.livekit_url and args.livekit_token and args.room_name)
    using_daily = bool(args.room_url and args.token)

    if not (using_livekit or using_daily):
        parser.error("Must provide --idle OR LiveKit args OR Daily args")

    asyncio.run(run_bot(
        bot_class,
        config,
        livekit_url=args.livekit_url,
        livekit_token=args.livekit_token,
        room_name=args.room_name,
        room_url=args.room_url,
        token=args.token,
    ))


if __name__ == "__main__":
    cli()