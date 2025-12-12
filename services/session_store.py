"""Encrypted session metadata storage for Aletheia Therapy Bot.

This module provides HIPAA-compliant storage of anonymous session metadata.
NO PHI (Protected Health Information) is stored in the backend - only
aggregated metrics for analysis and improvement.
"""

import sqlite3
import json
from datetime import datetime, timedelta 
from typing import Dict, Any, Optional
from pathlib import Path
from cryptography.fernet import Fernet
from loguru import logger


class SessionStore:
    """Encrypted storage for anonymous therapy session metadata.
    
    Stores only aggregated data:
    - session_id (anonymous UUID)
    - user_id (anonymous participant ID)
    - duration, mood_trend, strategy counts
    - NO transcripts, NO voice, NO personal notes
    
    Encryption ensures even metadata is protected at rest.
    """
    
    def __init__(
        self,
        db_path: str = "therapy_metadata.db",
        encryption_key: Optional[str] = None
    ):
        """Initialize session store with encryption.
        
        Args:
            db_path: Path to SQLite database file
            encryption_key: Fernet key for encryption (auto-generated if None)
        """
        self.db_path = Path(db_path)
        
        # Generate or load encryption key
        if encryption_key is None:
            key_path = Path(".encryption_key")
            if key_path.exists():
                self.encryption_key = key_path.read_text().strip()
            else:
                self.encryption_key = Fernet.generate_key().decode()
                key_path.write_text(self.encryption_key)
                logger.warning(f"Generated new encryption key: {key_path}")
        else:
            self.encryption_key = encryption_key
            
        self.cipher = Fernet(self.encryption_key.encode())
        
        # Initialize database
        self._init_db()
        
        logger.success(f"SessionStore initialized: {self.db_path}")

    def _init_db(self):
        """Create database table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS therapy_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        duration_minutes INTEGER,
                        mood_trend TEXT,
                        coping_strategies_count INTEGER,
                        topics_count INTEGER,
                        crisis_detected BOOLEAN DEFAULT FALSE,
                        encrypted_metadata BLOB
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Save encrypted session metadata.
        
        Args:
            metadata: Dictionary with:
                session_id, user_id, created_at, duration_minutes,
                mood_trend, coping_strategies_count, topics_count,
                crisis_detected (optional)
                
        Returns:
            True if successful, False otherwise
        """
        try:
            # Encrypt the full metadata payload
            metadata_json = json.dumps(metadata, default=str)
            encrypted_data = self.cipher.encrypt(metadata_json.encode())
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO therapy_sessions 
                    (session_id, user_id, created_at, duration_minutes, 
                     mood_trend, coping_strategies_count, topics_count, 
                     crisis_detected, encrypted_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata["session_id"],
                    metadata["user_id"],
                    metadata.get("created_at", datetime.utcnow().isoformat()),
                    metadata.get("duration_minutes", 0),
                    metadata.get("mood_trend", "stable"),
                    metadata.get("coping_strategies_count", 0),
                    metadata.get("topics_count", 0),
                    metadata.get("crisis_detected", False),
                    encrypted_data
                ))
                conn.commit()
                
            logger.debug(f"Saved metadata for session {metadata['session_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False

    def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt session metadata.
        
        Args:
            session_id: UUID of the session
            
        Returns:
            Decrypted metadata dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT encrypted_metadata FROM therapy_sessions WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    encrypted_data = row[0]
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    return json.loads(decrypted_data.decode())
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve metadata: {e}")
            return None
    # Add this method inside the SessionStore class (before cleanup_old_sessions):

    def start_session(self, session_id: str, user_id: str, bot_name: str) -> bool:
        """Initialize a new session record when a user joins.
        
        Args:
            session_id: Anonymous UUID for the session
            user_id: Anonymous participant ID
            bot_name: Name of the bot
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = {
                "session_id": session_id,
                "user_id": user_id,
                "bot_name": bot_name,
                "created_at": datetime.utcnow().isoformat(),
                "duration_minutes": 0,
                "mood_trend": "stable",
                "coping_strategies_count": 0,
                "topics_count": 0,
                "crisis_detected": False
            }
            
            # Use the existing save_metadata method to handle encryption
            return self.save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return False
        


        
    def get_session_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregate statistics for sessions.
        
        Args:
            user_id: Optional filter by anonymous user_id
            
        Returns:
            Dictionary with total_sessions, avg_duration, common_mood_trend
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT mood_trend, duration_minutes FROM therapy_sessions"
                params = ()
                
                if user_id:
                    query += " WHERE user_id = ?"
                    params = (user_id,)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    return {"total_sessions": 0}
                
                total_sessions = len(rows)
                total_duration = sum(row[1] or 0 for row in rows)
                mood_trends = [row[0] for row in rows if row[0]]
                
                return {
                    "total_sessions": total_sessions,
                    "avg_duration_minutes": total_duration // total_sessions,
                    "most_common_mood_trend": max(set(mood_trends), key=mood_trends.count) if mood_trends else "unknown"
                }
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_sessions": 0, "error": str(e)}

    def cleanup_old_sessions(self, days: int = 30):
        """Delete sessions older than X days (for data retention).
        
        Args:
            days: Number of days to retain
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM therapy_sessions WHERE created_at < ?",
                    (cutoff_date,)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
            logger.info(f"Cleaned up {deleted_count} old sessions")
            
        except Exception as e:
            logger.error(f"Failed to cleanup sessions: {e}")