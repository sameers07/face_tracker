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
    
    # Get project root (face_tracker directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Handle paths starting with ../
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
            print(f"Config loading failed: {str(e)}")
            sys.exit(1)
        
        # Initialize logging first
        self.logger = self._init_logging()
        
        # Initialize modules
        self.detector = FaceDetector(self.config)
        self.recognizer = FaceRecognizer(self.config['recognition'])
        self.tracker = FaceTracker(self.config)
        self.database = FaceDatabase(self.config)
        self.face_logger = FaceLogger(self.config)
        
        # State management
        self.frame_count = 0
        self.active_visitors = {}  # {track_id: (db_face_id, last_seen)}
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
        
        # Create formatters and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger

    def process_frame(self, frame):
        """Process a single video frame"""
        start_time = time.time()
        self.frame_count += 1
        
        # Skip frames if configured
        if self._should_skip_frame():
            return frame
        
        # Detect faces
        face_boxes = self.detector.detect(frame)
        face_images = []
        embeddings = []
        valid_boxes = []
        
        # Process each detection
        for box in face_boxes:
            if len(box) == 5:
               _, x1, y1, x2, y2 = map(int, box)
            else:
               x1, y1, x2, y2 = map(int, box)
            face_img = frame[y1:y2, x1:x2]
            
            # Validate face crop
            if face_img.size == 0 or min(face_img.shape[:2]) < 10:
                continue
                
            # Get embedding
            embedding = self.recognizer.get_embedding(face_img)
            if embedding is not None:
                face_images.append(face_img)
                embeddings.append(embedding)
                valid_boxes.append((x1, y1, x2, y2))
        
        # Update tracker
        tracks = self.tracker.update(valid_boxes, embeddings)
        
        # Process tracks
        current_time = time.time()
        current_tracks = set()
        
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = map(int, track['position'])
            face_img = frame[y1:y2, x1:x2]
            
            # Get or assign face ID
            if track_id not in self.active_visitors:
                # New track - process entry
                db_face_id, is_new = self._register_or_recognize_face(track['embedding'])
                self.active_visitors[track_id] = (db_face_id, current_time)
                self.unique_visitors.add(db_face_id)
                
                # Log entry if not already logged
                if db_face_id not in self.entry_logged:
                    self._log_face_event(db_face_id, 'entry', face_img)
                    self.entry_logged.add(db_face_id)
            else:
                # Existing track - update last seen
                db_face_id, _ = self.active_visitors[track_id]
                self.active_visitors[track_id] = (db_face_id, current_time)
            
            current_tracks.add(track_id)
            
            # Visualize
            color = (0, 255, 0)  # Green for recognized
            if db_face_id not in self.entry_logged:
                color = (0, 0, 255)  # Red for new
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {db_face_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Check for exits
        exited_tracks = set(self.active_visitors.keys()) - current_tracks
        for track_id in exited_tracks:
            db_face_id, last_seen = self.active_visitors[track_id]
            
            # Only log exit if not already logged and face was visible recently
            if (db_face_id not in self.exit_logged and 
                (current_time - last_seen) < self.config['tracking']['max_age']):
                self._log_face_event(db_face_id, 'exit')
                self.exit_logged.add(db_face_id)
            
            del self.active_visitors[track_id]
        
        # Display visitor count
        cv2.putText(frame, f"Unique Visitors: {len(self.unique_visitors)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Log performance
        proc_time = time.time() - start_time
        if proc_time > 0:
            fps = 1 / proc_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        self.last_processed = time.time()
        return frame

    def _should_skip_frame(self):
        """Adaptive frame skipping based on system load"""
        base_skip = self.config['detection'].get('skip_frames', 0)
        if base_skip == 0:
            return False
            
        # Increase skip frames if processing is falling behind
        processing_time = time.time() - self.last_processed
        expected_time = 1.0 / self.config['system'].get('target_fps', 25)
        
        if processing_time > expected_time * 1.5:
            self.current_skip = min(self.current_skip + 1, 10)
        elif self.current_skip > base_skip:
            self.current_skip = max(self.current_skip - 1, base_skip)
            
        return self.frame_count % self.current_skip != 0

    def _register_or_recognize_face(self, embedding):
        """Find similar face or register new one"""
        # Check database for matching face
        db_face_id, similarity = self.database.find_similar_face(
            embedding,
            self.config['recognition']['similarity_threshold']
        )
        
        # Register new face if no match found
        if db_face_id is None:
            db_face_id = self.database.register_face(embedding)
            return db_face_id, True
        
        return db_face_id, False

    def _log_face_event(self, face_id, event_type, face_image=None):
        """Log face event with proper handling"""
        metadata = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # For exit events, we don't have a current face image
        img_path = None
        if event_type == 'entry' and face_image is not None:
            img_path = self.face_logger.log_event(
                face_id, event_type, face_image, metadata
            )
        else:
            self.face_logger.log_event(
                face_id, event_type, metadata=metadata
            )
        
        # Save to database
        self.database.log_event(face_id, event_type, img_path)
        
        self.logger.info(f"Logged {event_type} event for face {face_id}")

    def process_video(self, video_path=None):
        """Process video file or stream"""
        video_config = self.config.get('video', {})
        input_path = resolve_path(video_path or video_config.get('input_path'))
        output_path = resolve_path(video_config.get('output_path'))
        max_duration = video_config.get('max_duration', 300)
        
        if not input_path.exists():
            self.logger.error(f"Input video not found: {input_path}")
            return
            
        try:
            # Open video source
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video source: {input_path}")
                return
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup output writer
            writer = None
            if output_path:
                codec = video_config.get('output_codec', 'mp4v')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
	                self.logger.error(f"Failed to open video writer with codec {codec}")
	                writer = None
            start_time = time.time()
            frame_count = 0
            
            # Processing loop
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = self.process_frame(frame)
                frame_count += 1
                
                # Write output
                if writer:
                    writer.write(processed_frame)
                
                # Display
                cv2.imshow('Face Tracker', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                # Check duration limit
                if time.time() - start_time > max_duration:
                    self.logger.info("Reached maximum processing duration")
                    break
            
            # Log final visitor count
            self.logger.info(f"Session completed. Total unique visitors: {len(self.unique_visitors)}")
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.database.end_session()

    def process_rtsp(self, rtsp_url=None):
        """Process RTSP stream with robust reconnection"""
        rtsp_config = self.config.get('rtsp', {})
        url = rtsp_url or rtsp_config.get('url')
        max_retries = rtsp_config.get('max_retries', 5)
        timeout = rtsp_config.get('timeout', 10)
        
        if not url:
            self.logger.error("No RTSP URL specified")
            return
            
        retry_count = 0
        retry_delay = 5
        
        while retry_count < max_retries:
            try:
                self.logger.info(f"Connecting to RTSP stream: {url}")
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
                
                if not cap.isOpened():
                    raise ConnectionError("Failed to open stream")
                    
                self.logger.info("RTSP stream connected")
                retry_count = 0
                retry_delay = 5
                
                # Processing loop
                while cap.isOpened():
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning("Frame read error, reconnecting...")
                        break
                        
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Display
                    cv2.imshow('Face Tracker', processed_frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    
                    # Maintain FPS
                    proc_time = time.time() - start_time
                    wait_time = max(1, int((1/30 - proc_time)*1000))
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
                
                cap.release()
                
            except Exception as e:
                self.logger.error(f"RTSP error: {str(e)}")
            
            # Exponential backoff for reconnection
            retry_count += 1
            self.logger.info(f"Retrying in {retry_delay} seconds ({retry_count}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds
            
        self.logger.error("Max retries exceeded, giving up")
        self.database.end_session()

if __name__ == '__main__':
    system = FaceTrackingSystem()
    
    # Process video if configured
    if "input_path" in system.config["video"] and system.config["video"]["input_path"]:
        system.process_video()
    
    # Process RTSP if enabled
    if system.config["rtsp"].get("enabled", False):
        system.process_rtsp()
