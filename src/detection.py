from ultralytics import YOLO
import cv2
import numpy as np
import logging
from pathlib import Path
import torch

def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parent.parent
    if path_str.startswith("../"):
        return project_root / path_str[3:]
    return project_root / path_str

class FaceDetector:
    def __init__(self, config):
        self.logger = logging.getLogger('FaceDetector')
        self.config = config.get('detection', {})
        
        # Get model path with fallback
        model_path = self.config.get('model', '../models/yolov8n-face.pt')
        model_path = resolve_path(model_path)
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.logger.info(f"Loading detector from: {model_path}")
        
        try:
            self.model = YOLO(str(model_path))
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            # Get configuration with defaults
            self.skip_frames = self.config.get('skip_frames', 2)
            self.conf_thresh = self.config.get('confidence_threshold', 0.3)
            self.iou_thresh = self.config.get('iou_threshold', 0.5)
            self.min_face_size = self.config.get('min_face_size', 64)
            
            self.logger.info("Detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Detector initialization failed: {str(e)}")
            raise

    def detect(self, frame):
        """Detect faces with size filtering"""
        if frame is None or frame.size == 0:
            self.logger.warning("Empty frame received")
            return []

        try:
            # Convert to RGB if needed
            if frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            # Run detection
            results = self.model.track(
                frame_rgb,
                imgsz=640,
                verbose=False,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                persist=True
            )

            boxes = []
            for result in results:
                if result.boxes is None:
                    continue
                    
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                for box in boxes_xyxy:
                    x1, y1, x2, y2 = box
                    w, h = x2-x1, y2-y1
                    
                    if w >= self.min_face_size and h >= self.min_face_size:
                        boxes.append((x1, y1, x2, y2))
                    else:
                        self.logger.debug(f"Filtered small face: {int(w)}x{int(h)}px")

            self.logger.debug(f"Found {len(boxes)} valid faces")
            return boxes
            
        except Exception as e:
            self.logger.error(f"Detection failed: {str(e)}")
            return []
