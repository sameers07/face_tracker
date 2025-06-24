import sqlite3
from datetime import datetime, timedelta
import json
import logging
import time
import numpy as np
import uuid
from pathlib import Path

def resolve_path(path_str):
    """Resolve relative paths to absolute paths"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    
    # Get project root (face_tracker directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Handle paths starting with ../
    if path_str.startswith("../"):
        return project_root / path_str[3:]
    
    return project_root / path_str

class FaceDatabase:
    def __init__(self, config):
        self.logger = logging.getLogger('FaceDatabase')
        db_config = config['database']
        
        # Resolve DB path
        db_path = resolve_path(db_config['path'])
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_cleanup_days = db_config.get('auto_cleanup_days', 30)
        self.current_session = str(uuid.uuid4())
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id INTEGER PRIMARY KEY,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT DEFAULT '{}'
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                session_id TEXT,
                metadata TEXT,
                FOREIGN KEY(face_id) REFERENCES faces(face_id)
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                visitor_count INTEGER DEFAULT 0
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_last_seen ON faces(last_seen)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)')
            
            conn.commit()
        
        # Start new session
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO sessions (session_id, start_time)
            VALUES (?, ?)
            ''', (self.current_session, datetime.now().isoformat()))
            conn.commit()
        
        self.logger.info(f"Database initialized at {self.db_path}. Session ID: {self.current_session}")

    def register_face(self, embedding, metadata=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute('''
                INSERT INTO faces (first_seen, last_seen, embedding, metadata)
                VALUES (?, ?, ?, ?)
                ''', (now, now, json.dumps(embedding.tolist()), json.dumps(metadata or {})))
                
                face_id = cursor.lastrowid
                conn.commit()
                
                # Update session visitor count
                cursor.execute('''
                UPDATE sessions 
                SET visitor_count = visitor_count + 1
                WHERE session_id = ?
                ''', (self.current_session,))
                conn.commit()
                
                return face_id
        except Exception as e:
            self.logger.error(f"Failed to register face: {str(e)}")
            return None

    def log_event(self, face_id, event_type, image_path=None, metadata=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                timestamp = datetime.now().isoformat()
                
                # Insert event
                cursor.execute('''
                INSERT INTO events (face_id, event_type, timestamp, image_path, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (face_id, event_type, timestamp, str(image_path) if image_path else None, 
                      self.current_session, json.dumps(metadata or {})))
                
                # Update face last_seen
                if event_type in ['entry', 're-entry']:
                    cursor.execute('''
                    UPDATE faces SET last_seen = ? WHERE face_id = ?
                    ''', (timestamp, face_id))
                
                conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Failed to log event: {str(e)}")
            return False

    def find_similar_face(self, embedding, threshold):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT face_id, embedding FROM faces')
                
                best_match = None
                best_sim = threshold
                
                for row in cursor.fetchall():
                    face_id, emb_json = row
                    db_embedding = np.array(json.loads(emb_json))
                    
                    # Calculate cosine similarity
                    similarity = np.dot(embedding, db_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
                    
                    if similarity > best_sim:
                        best_sim = similarity
                        best_match = face_id
                        # Early exit for perfect match
                        if best_sim > 0.999:
                            break
                
                return best_match, best_sim
        except Exception as e:
            self.logger.error(f"Similarity search failed: {str(e)}")
            return None, 0.0
            
    def get_unique_visitor_count(self):
        """Return unique visitor count for current session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(DISTINCT face_id) 
                    FROM events 
                    WHERE session_id = ? AND event_type = 'entry'
                ''', (self.current_session,))
                count = cursor.fetchone()
                return count[0] if count else 0
        except Exception as e:
            self.logger.error(f"Visitor count query failed: {str(e)}")
            return 0
            
    def end_session(self):
        """Mark session as ended"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions 
                    SET end_time = ?
                    WHERE session_id = ?
                ''', (datetime.now().isoformat(), self.current_session))
                conn.commit()
            self.logger.info(f"Session {self.current_session} ended")
        except Exception as e:
            self.logger.error(f"Failed to end session: {str(e)}")
