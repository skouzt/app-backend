"""DEPRECATED: SessionStore - All storage now handled by Supabase.

This module is kept for API compatibility only.
All actual data storage happens in Supabase.
"""

from typing import Dict, Any, Optional
from loguru import logger


class SessionStore:
    """DEPRECATED: Supabase handles all session storage.
    
    This class exists only for API compatibility.
    All methods are no-ops that safely return without storing data.
    """
    
    def __init__(self, db_path: str = None, encryption_key: str = None): # type: ignore
        # ✅ SAFE: Log at DEBUG level to avoid production noise
        logger.debug("SessionStore is deprecated - using Supabase for storage")
        # Silently initialize without database connection
    
    def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """No-op: Metadata is stored in Supabase."""
        # ✅ SAFE: Do not log metadata content
        session_id = metadata.get('session_id', 'unknown')[:8] if metadata else 'unknown'
        logger.debug(f"Session metadata would be saved (to Supabase) for session: {session_id}...")
        return True  # Return success to avoid breaking callers
    
    def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """No-op: Metadata is retrieved from Supabase."""
        logger.debug(f"Session metadata would be retrieved from Supabase for session: {session_id[:8] if session_id else 'unknown'}...")
        return None  # Return None to indicate no local data
    
    def start_session(self, session_id: str, user_id: str, bot_name: str) -> bool:
        """No-op: Session start is handled by Supabase."""
        # ✅ SAFE: Log only anonymized identifiers
        logger.debug(f"Session would be started in Supabase for user: {user_id[:8] if user_id else 'unknown'}...")
        return True  # Return success to avoid breaking callers
    
    def get_session_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """No-op: Statistics are calculated in Supabase."""
        logger.debug("Session statistics would be retrieved from Supabase")
        return {"total_sessions": 0, "note": "data_in_supabase"}  # Safe default
    
    def cleanup_old_sessions(self, days: int = 30):
        """No-op: Data retention is managed by Supabase."""
        logger.debug(f"Old session cleanup would be handled by Supabase (retention: {days} days)")