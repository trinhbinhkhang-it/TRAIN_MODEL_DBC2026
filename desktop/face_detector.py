"""
Face Detection and Tracking Module
Supports MediaPipe Face Detection and OpenCV alternatives
"""

import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    keypoints: Optional[Dict[str, Tuple[int, int]]] = None
    track_id: Optional[int] = None


class FaceDetector(ABC):
    """Abstract base class for face detectors"""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in frame"""
        pass
    
    @abstractmethod
    def close(self):
        """Release resources"""
        pass


class MediaPipeFaceDetector(FaceDetector):
    """MediaPipe Face Detection"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_selection: int = 0):
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
            
            self.min_detection_confidence = min_detection_confidence
            self.logger = logging.getLogger(__name__)
            self.logger.info("MediaPipe Face Detector initialized")
            
        except ImportError:
            raise RuntimeError("MediaPipe not installed. Install with: pip install mediapipe")
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process
        results = self.detector.process(rgb_frame)
        
        detections = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                # Get bounding box
                bbox_c = detection.location_data.relative_bounding_box
                x = int(bbox_c.xmin * w)
                y = int(bbox_c.ymin * h)
                bw = int(bbox_c.width * w)
                bh = int(bbox_c.height * h)
                
                # Ensure valid bbox
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                
                if bw > 0 and bh > 0:
                    # Get keypoints
                    keypoints = {}
                    for keypoint in detection.location_data.relative_keypoints:
                        kx = int(keypoint.x * w)
                        ky = int(keypoint.y * h)
                        keypoints[f'{keypoint.label}'] = (kx, ky)
                    
                    detections.append(FaceDetection(
                        bbox=(x, y, bw, bh),
                        confidence=detection.score[0],
                        keypoints=keypoints
                    ))
        
        return detections
    
    def close(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'detector'):
            self.detector.close()


class OpenCVHaarFaceDetector(FaceDetector):
    """OpenCV Haar Cascade Face Detector (fast, lightweight)"""
    
    def __init__(self, 
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30)):
        # Load Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.logger = logging.getLogger(__name__)
        self.logger.info("OpenCV Haar Face Detector initialized")
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append(FaceDetection(
                bbox=(x, y, w, h),
                confidence=0.9,  # Haar doesn't provide confidence
                keypoints=None
            ))
        
        return detections
    
    def close(self):
        pass


class OpenCVDNNFaceDetector(FaceDetector):
    """OpenCV DNN Face Detector (more accurate than Haar)"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 model_path: str = None,
                 config_path: str = None):
        # Use OpenCV's built-in face detector model
        if model_path is None:
            # Download or use built-in
            model_path = "models/opencv_face_detector_uint8.pb"
            config_path = "models/opencv_face_detector.pbtxt"
        
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        except:
            # Fallback to Caffe model
            model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            config_path = "models/deploy.prototxt"
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self.logger.info("OpenCV DNN Face Detector initialized")
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using DNN"""
        h, w = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], False, False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        face_detections = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_detections.append(FaceDetection(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        confidence=float(confidence),
                        keypoints=None
                    ))
        
        return face_detections
    
    def close(self):
        pass


class FaceTracker:
    """Simple face tracker using IoU matching"""
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_disappeared: int = 10):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.next_track_id = 0
        self.tracks = {}  # track_id -> {bbox, disappeared_count, last_detection}
        self.logger = logging.getLogger(__name__)
    
    def _iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to x1, y1, x2, y2
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Intersection
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """Update tracks with new detections"""
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                
                iou = self._iou(track['bbox'], det.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                # Update track
                det = detections[best_det_idx]
                track['bbox'] = det.bbox
                track['disappeared'] = 0
                track['last_detection'] = det
                det.track_id = track_id
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
            else:
                # Track disappeared
                track['disappeared'] += 1
        
        # Remove old tracks
        to_remove = [tid for tid, track in self.tracks.items() 
                     if track['disappeared'] > self.max_disappeared]
        for tid in to_remove:
            del self.tracks[tid]
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    'bbox': det.bbox,
                    'disappeared': 0,
                    'last_detection': det
                }
                det.track_id = track_id
        
        # Return detections with track IDs
        result = []
        for det in detections:
            if det.track_id is not None:
                result.append(det)
        
        return result
    
    def reset(self):
        """Reset tracker"""
        self.tracks = {}
        self.next_track_id = 0


def create_face_detector(detector_type: str, **kwargs) -> FaceDetector:
    """Factory function to create face detector"""
    detectors = {
        'mediapipe': MediaPipeFaceDetector,
        'opencv_haar': OpenCVHaarFaceDetector,
        'opencv_dnn': OpenCVDNNFaceDetector
    }
    
    if detector_type not in detectors:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    return detectors[detector_type](**kwargs)


if __name__ == "__main__":
    # Test face detector
    import sys
    logging.basicConfig(level=logging.INFO)
    
    cap = cv2.VideoCapture(0)
    detector = create_face_detector('mediapipe', min_detection_confidence=0.7)
    tracker = FaceTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        
        for det in tracked:
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{det.track_id} {det.confidence:.2f}",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()