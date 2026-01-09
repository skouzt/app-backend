import json
from typing import Sequence, Mapping, Any


def build_summary_prompt(messages: Sequence[Mapping[str, Any]]) -> str:

    transcript = ""
    for msg in messages:
        transcript += f"{msg['role'].capitalize()}: {msg['content']}\n"

    return f"""
You are a senior therapist.

Based on the following conversation transcript:
1. Write a concise session summary (5–7 lines)
2. Give an emotional score from 1 to 10

Transcript:
{transcript}

Respond in JSON:
{{ "summary": "...", "emotion_score": number }}
"""



async def summarize_with_llm(llm_client, prompt: str) -> dict:
    response = await llm_client.complete(prompt)
    
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON")

    if "summary" not in parsed or "session_intensity" not in parsed:
        raise ValueError("LLM response missing required fields")

    return parsed