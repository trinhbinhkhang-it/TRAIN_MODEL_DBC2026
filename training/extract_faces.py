"""
Extract faces from Celeb-DF-v2 videos using OpenCV DNN Face Detector
No MediaPipe/TensorFlow dependency conflicts
"""

import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


def extract_faces_from_dataset(
    splits_dir: str = "data/splits",
    video_root: str = "data/Celeb-DF-v2",
    output_root: str = "data",
    max_frames_per_video: int = 8,
    min_face_size: int = 80,
    confidence: float = 0.7
):
    """
    Extract faces from videos based on JSON splits using OpenCV DNN
    """
    # Load OpenCV DNN face detector (built-in, no extra download)
    # Model files are included in opencv_contrib_python
    model_file = "models/opencv_face_detector_uint8.pb"
    config_file = "models/opencv_face_detector.pbtxt"
    
    # Check if model files exist, if not use alternative
    if not (Path(model_file).exists() and Path(config_file).exists()):
        # Use Haar cascade as fallback (built into OpenCV)
        print("Using Haar Cascade face detector (built-in)")
        haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not Path(haar_path).exists():
            # Try alternative location
            import site
            for site_path in site.getsitepackages():
                alt = Path(site_path) / 'cv2' / 'data' / 'haarcascade_frontalface_default.xml'
                if alt.exists():
                    haar_path = str(alt)
                    break
        face_cascade = cv2.CascadeClassifier(haar_path)
        if face_cascade.empty():
            raise RuntimeError(f"Cannot load Haar cascade from {haar_path}")
        use_dnn = False
    else:
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        use_dnn = True
    
    # Load splits
    for split_name in ['train', 'val', 'test']:
        json_path = Path(splits_dir) / f"video_disjoint_{split_name}.json"
        if not json_path.exists():
            print(f"Split not found: {json_path}")
            continue
            
        with open(json_path, 'r') as f:
            videos = json.load(f)
        
        print(f"\nProcessing {split_name}: {len(videos)} videos")
        
        for video_info in tqdm(videos, desc=f"Extract {split_name}"):
            video_path = video_info['path']
            label = video_info['label']  # 0=real, 1=fake
            video_name = video_info['video_name']
            label_dir = 'real' if label == 0 else 'fake'
            
            # Output directory: data/train/real/video_name/
            out_dir = Path(output_root) / split_name / label_dir / video_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Skip if already processed
            existing_frames = list(out_dir.glob("*.jpg"))
            if len(existing_frames) >= max_frames_per_video:
                continue
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Cannot open: {video_path}")
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                continue
            
            # Sample frame indices evenly
            frame_indices = np.linspace(0, total_frames - 1, max_frames_per_video, dtype=int)
            
            saved_count = 0
            for idx, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                faces = []
                
                if use_dnn:
                    # DNN detection
                    h, w = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                    net.setInput(blob)
                    detections = net.forward()
                    
                    for i in range(detections.shape[2]):
                        conf = detections[0, 0, i, 2]
                        if conf > confidence:
                            x1 = int(detections[0, 0, i, 3] * w)
                            y1 = int(detections[0, 0, i, 4] * h)
                            x2 = int(detections[0, 0, i, 5] * w)
                            y2 = int(detections[0, 0, i, 6] * h)
                            faces.append((x1, y1, x2 - x1, y2 - y1, conf))
                else:
                    # Haar cascade detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    haar_faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                    )
                    for (x, y, w, h) in haar_faces:
                        faces.append((x, y, w, h, 0.9))
                
                if faces:
                    # Get largest face
                    x, y, w, h, conf = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Add margin
                    margin = 0.3
                    mx = int(w * margin)
                    my = int(h * margin)
                    x1 = max(0, x - mx)
                    y1 = max(0, y - my)
                    x2 = min(frame.shape[1], x + w + mx)
                    y2 = min(frame.shape[0], y + h + my)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.shape[0] >= min_face_size and face_crop.shape[1] >= min_face_size:
                        # Save face
                        out_path = out_dir / f"frame_{idx:04d}.jpg"
                        cv2.imwrite(str(out_path), face_crop)
                        saved_count += 1
            
            cap.release()
            
            # Remove empty directories
            if saved_count == 0:
                try:
                    out_dir.rmdir()
                except:
                    pass
    
    print("\nFace extraction completed!")


if __name__ == "__main__":
    extract_faces_from_dataset(
        splits_dir="data/splits",
        video_root="data/Celeb-DF-v2",
        output_root="data",
        max_frames_per_video=8,
        min_face_size=80,
        confidence=0.7
    )