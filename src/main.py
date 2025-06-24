import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import sys
import uuid
from detection import FaceDetector
from recognition import FaceRecognizer
from tracking import FaceTracker
from database import FaceDatabase
from custom_logger import FaceLogger

def resolve_path(path_str):
    """Resolve relative paths to absolute paths"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    
    project_root = Path(__file__).resolve().parent.parent
    if path_str.startswith("../"):
        return project_root / path_str[3:]
    return project_root / path_str

class FaceTrackingSystem:
    def __init__(self, config_path='../config/config.json'):
        # Load configuration
        config_file = resolve_path(config_path)
        try:
            with open(config_file) as f:
                self.config = json.load(f)
        except Exception as e:
            logging.error(f"Config loading failed: {str(e)}")
            sys.exit(1)
        
        # Initialize logging
        self.logger = self._init_logging()
        
        # Initialize modules
        self.detector = FaceDetector(self.config)
        self.recognizer = FaceRecognizer(self.config['recognition'])
        self.tracker = FaceTracker(self.config)
        self.database = FaceDatabase(self.config)
        self.face_logger = FaceLogger(self.config)
        
        # State management
        self.frame_count = 0
        self.active_visitors = {}
        self.unique_visitors = set()
        self.entry_logged = set()
        self.exit_logged = set()
        self.last_processed = time.time()
        self.current_skip = self.config['detection'].get('skip_frames', 0)
        self.session_id = str(uuid.uuid4())
        
        self.logger.info(f"System initialized. Session ID: {self.session_id}")

    def _init_logging(self):
        """Initialize logging system"""
        logging_config = self.config['logging']
        logger = logging.getLogger('FaceTrackingSystem')
        logger.setLevel(logging_config.get('log_level', 'INFO'))
        
        # Create handlers
        c_handler = logging.StreamHandler()
        log_dir = resolve_path(logging_config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        f_handler = logging.FileHandler(
            log_dir / 'system.log',
            encoding='utf-8'
        )
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger

    def process_frame(self, frame):
        """Process a single video frame"""
        start_time = time.time()
        self.frame_count += 1
        
        if self._should_skip_frame():
            return frame
        
        # Detect and process faces
        face_boxes = self.detector.detect(frame)
        face_images = []
        embeddings = []
        valid_boxes = []

        for box in face_boxes:
            x1, y1, x2, y2 = map(int, box)
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0 or min(face_img.shape[:2]) < 10:
                continue
                
            embedding = self.recognizer.get_embedding(face_img)
            if embedding is not None:
                face_images.append(face_img)
                embeddings.append(embedding)
                valid_boxes.append((x1, y1, x2, y2))
        
        # Update tracker
        tracks = self.tracker.update(valid_boxes, embeddings)
        current_time = time.time()
        current_tracks = set()
        
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = map(int, track['position'])
            face_img = frame[y1:y2, x1:x2]
            
            if track_id not in self.active_visitors:
                db_face_id, is_new = self._register_or_recognize_face(track['embedding'])
                if db_face_id is None:
                    continue
                    
                self.active_visitors[track_id] = (db_face_id, current_time)
                self.unique_visitors.add(db_face_id)
                
                event_type = 'entry' if is_new else 're-entry'
                self._log_face_event(db_face_id, event_type, face_img)
            else:
                db_face_id, last_seen = self.active_visitors[track_id]
                self.active_visitors[track_id] = (db_face_id, current_time)
                
                if (current_time - last_seen) > 30:
                    self._log_face_event(db_face_id, 'update', face_img)
            
            current_tracks.add(track_id)
            
            # Visualization
            color = (0, 255, 0) if db_face_id in self.entry_logged else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {db_face_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Handle exits
        exited_tracks = set(self.active_visitors.keys()) - current_tracks
        for track_id in exited_tracks:
            db_face_id, last_seen = self.active_visitors[track_id]
            if (current_time - last_seen) < self.config['tracking']['max_age']:
                self._log_face_event(db_face_id, 'exit')
                self.exit_logged.add(db_face_id)
            del self.active_visitors[track_id]
        
        # Display info
        cv2.putText(frame, f"Visitors: {len(self.unique_visitors)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        fps = 1 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        self.last_processed = time.time()
        return frame

    def _should_skip_frame(self):
        """Adaptive frame skipping"""
        base_skip = self.config['detection'].get('skip_frames', 0)
        if base_skip == 0:
            return False
            
        processing_time = time.time() - self.last_processed
        expected_time = 1.0 / self.config['system'].get('target_fps', 25)
        
        if processing_time > expected_time * 1.5:
            self.current_skip = min(self.current_skip + 1, 10)
        elif self.current_skip > base_skip:
            self.current_skip = max(self.current_skip - 1, base_skip)
            
        return self.frame_count % self.current_skip != 0

    def _register_or_recognize_face(self, embedding):
        """Handle face registration/recognition"""
        if embedding is None:
            return None, False

        db_face_id, similarity = self.database.find_similar_face(
            embedding,
            self.config['recognition']['similarity_threshold']
        )

        if db_face_id is None:
            try:
                db_face_id = self.database.register_face(embedding)
                if db_face_id:
                    return db_face_id, True
            except Exception as e:
                self.logger.error(f"Face registration failed: {str(e)}")
                return None, False

        return db_face_id, False

    def _log_face_event(self, face_id, event_type, face_image=None):
        """Log face events"""
        metadata = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type
        }
        
        img_path = None
        if face_image is not None and event_type != 'exit':
            try:
                img_path = self.face_logger.log_event(
                    face_id, event_type, face_image, metadata
                )
            except Exception as e:
                self.logger.error(f"Failed to save face image: {str(e)}")
        
        self.database.log_event(face_id, event_type, img_path, metadata)
        self.logger.info(f"Logged {event_type} event for face {face_id}")

    def process_video(self, video_path=None):
        """Process video file"""
        video_config = self.config.get('video', {})
        input_path = resolve_path(video_path or video_config.get('input_path'))
        output_path = resolve_path(video_config.get('output_path'))
        
        if not input_path.exists():
            self.logger.error(f"Input video not found: {input_path}")
            return
            
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {input_path}")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            writer = None
            if output_path:
                codec = video_config.get('output_codec', 'mp4v')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame = self.process_frame(frame)
                
                if writer:
                    writer.write(processed_frame)
                
                cv2.imshow('Face Tracker', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.logger.info(f"Session completed. Unique visitors: {len(self.unique_visitors)}")
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.database.end_session()

if __name__ == '__main__':
    system = FaceTrackingSystem()
    
    if "input_path" in system.config["video"] and system.config["video"]["input_path"]:
        system.process_video()
    
    if system.config["rtsp"].get("enabled", False):
        system.process_rtsp()
