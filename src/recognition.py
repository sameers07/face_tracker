import insightface
import numpy as np
import logging
from pathlib import Path
import warnings
import cv2

warnings.filterwarnings("ignore", message="model ignore: .*")

def resolve_path(path_str):
    """Resolve relative paths to absolute paths"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    
    project_root = Path(__file__).resolve().parent.parent
    if path_str.startswith("../"):
        return project_root / path_str[3:]
    return project_root / path_str

class FaceRecognizer:
    def __init__(self, config):
        self.logger = logging.getLogger('FaceRecognizer')
        self.config = config
        self.min_face_size = config.get('min_face_size', 64)
        
        try:
            model_root = resolve_path(config['model_path'])
            self.logger.info(f"Loading model from: {model_root}/{config['model']}")
            
            self.model = insightface.app.FaceAnalysis(
                name=config['model'],
                root=str(model_root),
                allowed_modules=['detection', 'recognition'],
                providers=['CPUExecutionProvider']
            )
            
            gpu_id = config.get('gpu_id', -1)
            det_size = tuple(config.get('input_size', [640, 640]))
            self.model.prepare(ctx_id=gpu_id, det_size=det_size)
            
            self.logger.info(f"Recognizer ready (min face size: {self.min_face_size}px)")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding with robust preprocessing"""
        if face_image is None or face_image.size == 0:
            self.logger.warning("Empty face image received")
            return None

        # Convert color spaces
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        elif face_image.shape[2] == 4:
            face_image = face_image[:, :, :3]

        # Validate size
        h, w = face_image.shape[:2]
        if w < self.min_face_size or h < self.min_face_size:
            self.logger.debug(f"Face too small: {w}x{h} (min {self.min_face_size}px)")
            return None

        try:
            # Resize large faces
            max_size = 640
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                face_image = cv2.resize(face_image, (0,0), fx=scale, fy=scale)
                self.logger.debug(f"Resized from {w}x{h}")

            # Extract embedding
            faces = self.model.get(face_image)
            if not faces:
                self.logger.debug("No faces found in processed crop")
                return None

            embedding = faces[0].embedding.astype(np.float32)
            norm = np.linalg.norm(embedding) + 1e-10
            normalized = embedding / norm
            
            self.logger.debug(f"Embedding extracted (norm: {norm:.2f})")
            return normalized
            
        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {str(e)}")
            return None
