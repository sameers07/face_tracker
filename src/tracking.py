import numpy as np
from collections import OrderedDict
import time
import logging
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional

class KalmanFilter:
    """Kalman filter for bounding box tracking"""
    def __init__(self, box, config):
        # Validate input box
        if not isinstance(box, (list, tuple, np.ndarray)) or len(box) < 4:
            raise ValueError("Invalid box format for KalmanFilter")
            
        # State: [x_center, y_center, width, height, vx, vy]
        self.state = np.array([
            (box[0] + box[2]) / 2,  # x_center
            (box[1] + box[3]) / 2,  # y_center
            box[2] - box[0],        # width
            box[3] - box[1],        # height
            0, 0                     # velocity
        ], dtype=np.float32)
        
        # State covariance matrix
        self.P = np.eye(6, dtype=np.float32)
        
        # Process noise
        self.Q = np.eye(6) * config.get('process_noise', 0.1)
        
        # Measurement noise
        self.R = np.eye(4) * config.get('measurement_noise', 1.0)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ], dtype=np.float32)

    def predict(self):
        """Predict next state"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state_to_box()

    def update(self, measurement):
        """Update with new measurement"""
        # Validate measurement
        if not isinstance(measurement, (list, tuple, np.ndarray)) or len(measurement) < 4:
            return self.state_to_box()
            
        # Convert measurement to state format
        z = np.array([
            (measurement[0] + measurement[2]) / 2,
            (measurement[1] + measurement[3]) / 2,
            measurement[2] - measurement[0],
            measurement[3] - measurement[1]
        ])
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.state_to_box()
        
        # Update state
        y = z - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.state_to_box()

    def state_to_box(self):
        """Convert state to bounding box format"""
        x, y, w, h = self.state[0:4]
        return (
            float(x - w/2),
            float(y - h/2),
            float(x + w/2),
            float(y + h/2)
        )

class FaceTracker:
    def __init__(self, config):
        self.logger = logging.getLogger('FaceTracker')
        tracking_cfg = config['tracking']
        self.tracks: Dict[int, Dict] = OrderedDict()
        self.next_id = 1
        self.max_age = tracking_cfg.get('max_age', 30)
        self.sim_thresh = config['recognition']['similarity_threshold']
        self.appearance_weight = tracking_cfg.get('appearance_weight', 0.7)
        self.position_weight = 1.0 - self.appearance_weight
        self.reid_threshold = tracking_cfg.get('reid_threshold', 0.5)
        self.reid_buffer = OrderedDict()  # For re-identification
        self.kf_enabled = tracking_cfg.get('kalman_filter', {}).get('enabled', True)
        self.kf_config = tracking_cfg.get('kalman_filter', {})
        self.logger.info(f"Tracker initialized with appearance weight: {self.appearance_weight}")

    def update(self, 
              detections: List[Tuple[float, float, float, float]], 
              embeddings: List[np.ndarray]) -> Dict[int, Dict]:
        # Validate inputs
        if len(detections) != len(embeddings):
            self.logger.warning("Detections and embeddings count mismatch")
            return self.tracks
            
        if not detections:
            self._update_missing_detections()
            return self.tracks
            
        # Convert to numpy arrays
        try:
            detections = np.array(detections, dtype=np.float32)
            embeddings = np.array(embeddings)
        except Exception as e:
            self.logger.error(f"Error converting detections/embeddings: {e}")
            return self.tracks
            
        # Validate array shapes
        if detections.size == 0:
            self._update_missing_detections()
            return self.tracks
            
        if detections.ndim != 2 or detections.shape[1] != 4:
            self.logger.error(f"Invalid detection shape: {detections.shape}")
            return self.tracks
            
        if embeddings.ndim != 2:
            self.logger.error(f"Invalid embedding shape: {embeddings.shape}")
            return self.tracks
            
        # Create cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        # Calculate similarities
        for i, (track_id, track) in enumerate(self.tracks.items()):
            # Position similarity
            ious = np.array([self._iou(track['position'], det) for det in detections])
            
            # Appearance similarity
            norms = np.linalg.norm(embeddings, axis=1)
            valid_mask = norms > 0
            cos_sims = np.zeros(len(detections))
            
            if np.any(valid_mask):
                valid_embeddings = embeddings[valid_mask]
                cos_sims[valid_mask] = np.dot(
                    track['embedding'], 
                    valid_embeddings.T
                ) / (np.linalg.norm(track['embedding']) * norms[valid_mask])
            
            # Combined score
            cost_matrix[i] = self.appearance_weight * cos_sims + self.position_weight * ious
        
        # Hungarian algorithm for optimal assignment
        matches = []
        if cost_matrix.size > 0:
            try:
                row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # Maximize
                matches = [
                    (r, c) for r, c in zip(row_ind, col_ind) 
                    if cost_matrix[r, c] > self.sim_thresh
                ]
            except Exception as e:
                self.logger.error(f"Hungarian algorithm failed: {e}")
        
        # Update matched tracks
        updated_tracks = OrderedDict()
        matched_det_indices = set()
        
        for i, j in matches:
            track_id = list(self.tracks.keys())[i]
            track = self.tracks[track_id]
            
            # Get detection box
            detection_box = detections[j, :4]
            
            # Update Kalman filter if enabled
            if self.kf_enabled and 'kf' in track:
                try:
                    predicted_box = track['kf'].predict()
                    updated_box = track['kf'].update(detection_box)
                except Exception as e:
                    self.logger.error(f"Kalman update failed: {e}")
                    updated_box = detection_box
            else:
                updated_box = detection_box
            
            # Update track
            updated_tracks[track_id] = {
                'position': updated_box,
                'embedding': embeddings[j],
                'last_seen': time.time(),
                'hits': track.get('hits', 0) + 1,
                'kf': track.get('kf')
            }
            
            # Initialize Kalman filter if needed
            if self.kf_enabled and 'kf' not in track:
                try:
                    updated_tracks[track_id]['kf'] = KalmanFilter(updated_box, self.kf_config)
                except Exception as e:
                    self.logger.error(f"Kalman init failed: {e}")
            
            matched_det_indices.add(j)
        
        # Create new tracks for unmatched detections
        for j in range(len(detections)):
            if j not in matched_det_indices:
                detection_box = detections[j, :4]
                new_track = {
                    'position': detection_box,
                    'embedding': embeddings[j],
                    'last_seen': time.time(),
                    'hits': 1
                }
                if self.kf_enabled:
                    try:
                        new_track['kf'] = KalmanFilter(detection_box, self.kf_config)
                    except Exception as e:
                        self.logger.error(f"Kalman init failed for new track: {e}")
                updated_tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Remove stale tracks and move to re-id buffer
        current_time = time.time()
        active_tracks = OrderedDict()
        for tid, track in updated_tracks.items():
            if current_time - track['last_seen'] < self.max_age:
                active_tracks[tid] = track
            else:
                # Move to re-identification buffer
                self.reid_buffer[tid] = track
                self.logger.debug(f"Moved track {tid} to re-id buffer")
        
        # Attempt re-identification of lost tracks
        self._reidentify_lost_tracks(detections, embeddings)
        
        # Cleanup re-id buffer
        for tid in list(self.reid_buffer.keys()):
            if current_time - self.reid_buffer[tid]['last_seen'] > self.max_age * 2:
                del self.reid_buffer[tid]
                self.logger.debug(f"Removed track {tid} from re-id buffer")
        
        self.tracks = active_tracks
        return self.tracks

    def _reidentify_lost_tracks(self, detections, embeddings):
        """Attempt to re-identify lost tracks in the buffer"""
        # Validate inputs
        if not self.reid_buffer or detections.size == 0:
            return
            
        # Create cost matrix between re-id buffer and current detections
        cost_matrix = np.zeros((len(self.reid_buffer), len(detections)))
        for i, (track_id, track) in enumerate(self.reid_buffer.items()):
            # Only use appearance for re-id
            norms = np.linalg.norm(embeddings, axis=1)
            valid_mask = norms > 0
            cos_sims = np.zeros(len(detections))
            
            if np.any(valid_mask):
                valid_embeddings = embeddings[valid_mask]
                cos_sims[valid_mask] = np.dot(
                    track['embedding'], 
                    valid_embeddings.T
                ) / (np.linalg.norm(track['embedding']) * norms[valid_mask])
            
            cost_matrix[i] = cos_sims
        
        # Find matches above threshold
        matches = []
        if cost_matrix.size > 0:
            try:
                row_ind, col_ind = linear_sum_assignment(-cost_matrix)
                matches = [
                    (r, c) for r, c in zip(row_ind, col_ind)
                    if cost_matrix[r, c] > self.reid_threshold
                ]
            except Exception as e:
                self.logger.error(f"Re-id Hungarian failed: {e}")
        
        # Re-identify tracks
        for i, j in matches:
            track_id = list(self.reid_buffer.keys())[i]
            track = self.reid_buffer[track_id]
            detection_box = detections[j, :4]
            
            # Update with new detection
            self.tracks[track_id] = {
                'position': detection_box,
                'embedding': embeddings[j],
                'last_seen': time.time(),
                'hits': track['hits'] + 1,
                'kf': track.get('kf')
            }
            
            # Remove from buffer
            del self.reid_buffer[track_id]
            self.logger.info(f"Re-identified lost track {track_id}")

    def _update_missing_detections(self):
        """Update tracks when no detections are present"""
        current_time = time.time()
        for track_id, track in list(self.tracks.items()):
            # Predict next position with Kalman filter
            if self.kf_enabled and 'kf' in track:
                try:
                    track['position'] = track['kf'].predict()
                    track['last_seen'] = current_time
                except Exception as e:
                    self.logger.error(f"Kalman predict failed: {e}")
        
        # Remove expired tracks and move to buffer
        active_tracks = OrderedDict()
        for tid, track in self.tracks.items():
            if current_time - track['last_seen'] < self.max_age:
                active_tracks[tid] = track
            else:
                self.reid_buffer[tid] = track
        self.tracks = active_tracks

    def _iou(self, box1, box2):
        """Calculate Intersection over Union"""
        # Ensure boxes are in proper format
        if not isinstance(box1, (list, tuple, np.ndarray)) or len(box1) < 4:
            return 0.0
        if not isinstance(box2, (list, tuple, np.ndarray)) or len(box2) < 4:
            return 0.0
            
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)
