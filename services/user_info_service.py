from typing import Optional, Dict, Any, cast
from db.supabase import supabase


def fetch_user_info(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch complete user background info from user_info table
    used for LLM context.
    """

    result = (
        supabase
        .table("user_info")
        .select("*")  # Fetch all columns
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not result.data or not isinstance(result.data[0], dict):
        return None

    # Cast to satisfy type checker - we know it's a dict
    return cast(Dict[str, Any], result.data[0])