import logging
import sqlite3
from datetime import datetime

class AuditLogger:
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.conn = sqlite3.connect(self.db_uri)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    user_id TEXT,
                    details TEXT
                )
            """)

    def log_event(self, event_type: str, user_id: str, details: str):
        ts = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                "INSERT INTO audit_logs (timestamp, event_type, user_id, details) VALUES (?, ?, ?, ?)",
                (ts, event_type, user_id, details)
            )
        logging.info(f"[AUDIT] {event_type} by {user_id}: {details}")
