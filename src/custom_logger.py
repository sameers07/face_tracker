import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import cv2
import json
from pathlib import Path
import shutil

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

class FaceLogger:
    def __init__(self, config):
        self.config = config['logging']
        
        # Resolve log directory
        log_dir = resolve_path(self.config.get('log_dir', 'logs'))
        
        self.log_dir = log_dir
        self._setup_logging()
        self._clean_lock = False

    def _setup_logging(self):
        """Initialize logging system"""
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger('face_tracker')
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler with rotation
        log_file = self.log_dir / 'events.log'
        max_bytes = self.config.get('max_log_size_mb', 100) * 1024 * 1024
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # Console handler
        if self.config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
        
        self.logger.info("Logging system initialized")

    def _clean_old_logs(self):
        """Remove old log entries"""
        if self._clean_lock:
            return
            
        self._clean_lock = True
        try:
            retention = self.config.get('retention_days', 30)
            cutoff = datetime.now() - timedelta(days=retention)
            
            # Clean entry and exit directories
            for log_type in ['entries', 'exits']:
                log_dir = self.log_dir / log_type
                if log_dir.exists():
                    for date_dir in log_dir.iterdir():
                        if date_dir.is_dir():
                            try:
                                dir_date = datetime.strptime(date_dir.name, '%Y-%m-%d')
                                if dir_date < cutoff:
                                    shutil.rmtree(date_dir)
                            except ValueError:
                                continue
            
            self.logger.info(f"Cleaned logs older than {cutoff.date()}")
        except Exception as e:
            self.logger.error(f"Log cleanup failed: {str(e)}")
        finally:
            self._clean_lock = False

    def log_event(self, face_id, event_type, face_image=None, metadata=None):
        try:
            # Clean logs periodically
            if datetime.now().hour == 3:  # Daily at 3 AM
                self._clean_old_logs()
        
            timestamp = datetime.now()
            date_folder = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H-%M-%S')  # Hours-Minutes-Seconds
        
            # Determine directory based on event type
            if event_type in ['entry', 're-entry']:
                log_type = 'entries'
            elif event_type == 'exit':
                log_type = 'exits'
            else:
                log_type = 'events'
            
            # Create date directory
            date_dir = self.log_dir / log_type / date_folder
            date_dir.mkdir(parents=True, exist_ok=True)
        
            # Save face image
            img_path = None
            if face_image is not None and face_image.size > 0:
                img_filename = f"face_id_{face_id:03d}_{event_type}_{time_str}.jpg"
                img_path = date_dir / img_filename
                success = cv2.imwrite(
                    str(img_path), 
                    face_image,
                )
                if not success:
                    self.logger.warning(f"Failed to save image: {img_path}")
                    img_path = None
        
            # Prepare log entry
            log_entry = {
                'face_id': face_id,
                'event_type': event_type,
                'timestamp': timestamp.isoformat(),
                'image_path': str(img_path) if img_path else None,
                'metadata': metadata or {}
            }
        
            # Log to file
            self.logger.info(json.dumps(log_entry, ensure_ascii=False))
            
            return img_path

        except Exception as e:
            self.logger.error(f"Event logging failed: {str(e)}")
            return None
