"""Single place the app talks to an LLM.

Deliberately thin: one async client, one chat call, one JSON call. Swapping provider
means changing the base_url/model here and nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI

from core.config import settings

_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        _client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
    return _client


async def complete(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.8,
    # Generous because deepseek-v4-* reason before answering and that reasoning is
    # drawn from the same budget. A visible reply is ~40 tokens, but reasoning
    # occasionally runs long — at 700 it sometimes consumed the entire budget and
    # returned nothing, surfacing to the user as "Lily could not reply". Headroom is
    # close to free: you are billed for tokens produced, not for the ceiling.
    max_tokens: int = 1500,
) -> str:
    """Plain text completion."""
    resp = await get_client().chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    content = choice.message.content

    # deepseek-v4-* are reasoning models: hidden reasoning tokens are drawn from the
    # same max_tokens budget as the answer. Too small a budget truncates the reply
    # mid-string, which otherwise surfaces as a confusing "invalid JSON" error.
    if choice.finish_reason == "length":
        details = getattr(resp.usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0
        raise ValueError(
            f"LLM response truncated at max_tokens={max_tokens} "
            f"({reasoning} of {resp.usage.completion_tokens} completion tokens were reasoning)"
        )

    if not content or not content.strip():
        raise ValueError("Empty LLM response")
    return content.strip()


def _strip_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    parts = t.split("```")
    if len(parts) < 2:
        return t
    body = parts[1]
    if body.lstrip().lower().startswith("json"):
        body = body.lstrip()[4:]
    return body.strip()


async def complete_json(
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.3,
    # Matches `complete` for the same reason: this model reasons before it
    # answers and both come out of one budget. At 500 a long transcript could
    # spend the whole allowance thinking and return a truncated object, which
    # fails to parse — and every caller here treats a parse failure as "no
    # result", so the symptom was a generic fallback summary rather than an
    # error. Headroom costs nothing: billing is on tokens produced.
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """Completion that must parse as a JSON object. Models fence it more often than not."""
    raw = await complete(messages, temperature=temperature, max_tokens=max_tokens)
    try:
        parsed = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as e:
        # Length is in the message because truncation and malformed output look
        # identical otherwise, and they have different fixes: one is the token
        # budget, the other is the prompt.
        logger.warning(
            f"LLM returned non-JSON ({len(raw)} chars, max_tokens may be too low): {raw[:200]!r}"
        )
        raise ValueError("LLM did not return valid JSON") from e
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned JSON but not an object")
    return parsed
