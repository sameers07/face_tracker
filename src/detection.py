from ultralytics import YOLO
import cv2
import numpy as np
import logging
from pathlib import Path
import torch

# PyTorch 2.6+ compatibility fix
if torch.__version__ >= "2.6.0":
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])

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

class FaceDetector:
    def __init__(self, config):
        self.logger = logging.getLogger('FaceDetector')
        self.config = config['detection']
        self.skip_frames = self.config['skip_frames']
        self._frame_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Resolve model path
        model_path = resolve_path(self.config['model'])
        
        self.logger.info(f"Loading detection model from: {model_path}")
        
        # Verify model exists
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = YOLO(str(model_path))
            self.logger.info(f"Loaded detection model: {model_path}")
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.logger.debug(f"Model loaded on device: {self.device}")
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def detect(self, frame):
        """Detect faces in frame with frame skipping"""
        self._frame_counter += 1

        if self.skip_frames > 0 and self._frame_counter % self.skip_frames != 0:
            return []

        if frame is None or frame.size == 0:
            self.logger.warning("Received empty frame for detection")
            return []

        try:
            # Convert to RGB format
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
                self.logger.warning("Frame is not 3-channel, skipping BGR2RGB conversion")
            
            # Resize to make dimensions divisible by 32 (model stride)
            height, width = frame_rgb.shape[:2]
            new_width = (width // 32) * 32
            new_height = (height // 32) * 32
            
            # Ensure minimum size
            new_width = max(new_width, 320)
            new_height = max(new_height, 320)
            
            # Resize frame if needed
            if new_width != width or new_height != height:
                self.logger.debug(f"Resizing frame from {width}x{height} to {new_width}x{new_height}")
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Run detection directly on numpy array
            results = self.model.track(
                frame_rgb,
                imgsz=(new_height, new_width),  # Match resized dimensions
                verbose=False,
                conf=self.config['confidence_threshold'],
                iou=self.config['iou_threshold'],
                persist=True  # Enable tracking between frames
            )

            boxes = []
            
            # Process each result
            for result in results:
                # Check if boxes exist
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                # Extract boxes and IDs
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
                
                # Calculate scale factors for original size
                scale_x = width / new_width
                scale_y = height / new_height
                
                for i in range(len(boxes_xyxy)):
                    x1, y1, x2, y2 = boxes_xyxy[i]
                    conf = confidences[i]
                    
                    # Scale coordinates back to original size
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    if conf >= self.config['confidence_threshold']:
                        if track_ids is not None:
                            track_id = int(track_ids[i])
                            boxes.append((track_id, x1, y1, x2, y2))
                        else:
                            boxes.append((x1, y1, x2, y2))

            return boxes

        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}", exc_info=True)
            return []
