"""Where to send someone who needs more than a conversation.

Lily was telling everyone to call 988 — the US line — because that is what a
model trained mostly on American text reaches for. Most people using this are in
India, where 988 rings nothing.

Two rules govern this file:

  1. Nothing goes in unverified. A wrong number in a crisis is worse than no
     number, because the person stops looking.
  2. An unknown country gets no number at all. The prompt then asks Lily to help
     them find a local service rather than inventing one, which is the failure
     mode this file exists to prevent.

Adding a country means checking the line is current and national, and that it is
staffed in a language its callers speak.
"""

from __future__ import annotations

from typing import Dict, Optional

# Only zones for countries with a verified entry below. This is deliberately not
# a complete timezone table: a zone that resolves to a country we have no line
# for should fall through to the generic path, not guess a neighbour's number.
_COUNTRY_BY_TIMEZONE: Dict[str, str] = {
    # India. Asia/Calcutta is the tzdata "backward" alias and is still what many
    # Android devices report, so both spellings have to be here.
    "asia/kolkata": "IN",
    "asia/calcutta": "IN",
    # United States. Only the mainland zones people actually report.
    "america/new_york": "US",
    "america/chicago": "US",
    "america/denver": "US",
    "america/phoenix": "US",
    "america/los_angeles": "US",
    "america/anchorage": "US",
    "pacific/honolulu": "US",
    # United Kingdom.
    "europe/london": "GB",
    # Canada.
    "america/toronto": "CA",
    "america/vancouver": "CA",
    "america/edmonton": "CA",
    "america/winnipeg": "CA",
    "america/halifax": "CA",
    # Australia.
    "australia/sydney": "AU",
    "australia/melbourne": "AU",
    "australia/brisbane": "AU",
    "australia/perth": "AU",
    "australia/adelaide": "AU",
}

# Phrased as a sentence Lily can say, not a data structure she has to narrate.
_LINES: Dict[str, str] = {
    "IN": (
        "In India, Tele-MANAS is the government's free mental health helpline: "
        "14416 or 1800-891-4416, open 24/7 and available in 20 languages."
    ),
    "US": (
        "In the US, the Suicide and Crisis Lifeline is 988 — call or text, 24/7."
    ),
    "GB": (
        "In the UK, Samaritans is 116 123, free and open 24 hours. "
        "They can also be reached by text on 85258."
    ),
    "CA": (
        "In Canada, the Suicide Crisis Helpline is 988 — call or text, 24/7."
    ),
    "AU": (
        "In Australia, Lifeline is 13 11 14, open 24 hours."
    ),
}


def country_for_timezone(timezone: Optional[str]) -> Optional[str]:
    if not timezone:
        return None
    return _COUNTRY_BY_TIMEZONE.get(timezone.strip().lower())


def crisis_block(timezone: Optional[str]) -> str:
    """The help-finding section of the system prompt.

    Always returns something. Where the country is unknown the instruction is to
    ask, because a person in Kenya being told to call 988 is worse off than one
    who is asked where they are.
    """
    line = _LINES.get(country_for_timezone(timezone) or "")

    if line:
        return (
            "WHERE TO SEND THEM\n"
            f"{line}\n"
            "Offer it when someone is in real distress or asks how to get help. "
            "Give it plainly, once, without turning it into a script or a "
            "disclaimer. Do not offer numbers for other countries."
        )

    return (
        "WHERE TO SEND THEM\n"
        "You do not know which country they are in, so you do not know which "
        "helpline reaches them. If they need one, ask where they are first and "
        "then help them find a local service. Never name a number you are not "
        "certain serves them — a line that does not answer is worse than none, "
        "because they stop looking."
    )
