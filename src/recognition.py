import insightface
import numpy as np
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="model ignore: .*")

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

class FaceRecognizer:
    def __init__(self, config):
        self.logger = logging.getLogger('FaceRecognizer')
        self.config = config
        
        # Configure model path
        model_path = resolve_path(self.config['model_path'])
        
        # Initialize model
        try:
            self.model = insightface.app.FaceAnalysis(
                name=self.config['model'],
                root=str(model_path.parent),
                allowed_modules=['detection', 'recognition'],
                providers=['CPUExecutionProvider']
            )
            
            # Configure GPU/CPU
            gpu_id = self.config.get('gpu_id', 0)
            if gpu_id >= 0:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self.logger.warning("CUDA not available, using CPU")
                        gpu_id = -1
                except ImportError:
                    gpu_id = -1
            
            # Prepare model
            det_size = tuple(self.config.get('input_size', [640, 640]))
            self.model.prepare(ctx_id=gpu_id, det_size=det_size)
            
            self.logger.info(f"FaceRecognizer initialized with model: {model_path}")
            if gpu_id < 0:
                self.logger.info("Using CPU for face recognition")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract normalized face embedding"""
        if face_image is None or face_image.size == 0:
            self.logger.warning("Empty face image received")
            return None
            
        # Validate input
        if len(face_image.shape) != 3 or face_image.shape[2] != 3:
            self.logger.warning("Face image must be RGB format")
            return None
            
        min_size = self.config.get('min_face_size', 20)
        if face_image.shape[0] < min_size or face_image.shape[1] < min_size:
            self.logger.debug(f"Face image too small: {face_image.shape}")
            return None
            
        try:
            # Process face
            faces = self.model.get(face_image)
            if not faces:
                self.logger.debug("No faces detected in crop")
                return None
                
            # Get primary face embedding
            embedding = faces[0].embedding
            norm = np.linalg.norm(embedding)
            if norm < 1e-6:
                self.logger.warning("Zero-norm embedding detected")
                return None
                
            return embedding / norm
        except Exception as e:
            self.logger.error(f"Embedding extraction failed: {str(e)}")
            return None
