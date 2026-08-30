"""
ONNX Runtime Inference Module for SwapFace Detection
"""

import os
import numpy as np
import cv2
import logging
import time
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Inference result"""
    fake_score: float  # Probability of fake (0-1)
    prediction: int    # 0: Real, 1: Fake
    logits: np.ndarray # Raw logits [2]
    inference_time_ms: float
    quality: str       # 'GOOD', 'LOW_QUALITY', 'TOO_SMALL', etc.
    face_bbox: Tuple[int, int, int, int]


class SwapFaceInference:
    """ONNX Runtime inference for SwapFace detection"""
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (224, 224),
                 input_name: str = 'input',
                 output_name: str = 'output',
                 providers: list = None,
                 threshold: float = 0.5):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to ONNX model
            input_size: Model input size (H, W)
            input_name: ONNX input tensor name
            output_name: ONNX output tensor name
            providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            threshold: Classification threshold
        """
        self.model_path = model_path
        self.input_size = input_size
        self.input_name = input_name
        self.output_name = output_name
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Normalization (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Initialize ONNX Runtime session
        self._init_session(providers)
        
        # Warmup
        self._warmup()
    
    def _init_session(self, providers: list):
        """Initialize ONNX Runtime session"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("ONNX Runtime not installed. Install with: pip install onnxruntime")
        
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Verify input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.logger.info(f"ONNX Runtime session initialized")
        self.logger.info(f"  Input: {self.input_name} {self.session.get_inputs()[0].shape}")
        self.logger.info(f"  Output: {self.output_name} {self.session.get_outputs()[0].shape}")
        self.logger.info(f"  Providers: {self.session.get_providers()}")
    
    def _warmup(self):
        """Warmup inference"""
        dummy_input = np.random.randn(1, 3, self.input_size[0], self.input_size[1]).astype(np.float32)
        for _ in range(3):
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
    
    def preprocess(self, face_img: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Preprocess face image for model input
        
        Args:
            face_img: Face crop in BGR format (H, W, 3)
            
        Returns:
            Preprocessed tensor (1, 3, H, W) and quality status
        """
        # Check face size
        h, w = face_img.shape[:2]
        min_size = min(h, w)
        
        if min_size < 64:
            return None, 'TOO_SMALL'
        
        # Check blur
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:
            return None, 'BLURRY'
        
        # Check brightness
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 220:
            return None, 'POOR_LIGHTING'
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize
        rgb = cv2.resize(rgb, (self.input_size[1], self.input_size[0]), 
                        interpolation=cv2.INTER_CUBIC)
        
        # Normalize
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        
        # HWC to CHW
        rgb = rgb.transpose(2, 0, 1)
        
        # Add batch dimension
        tensor = np.expand_dims(rgb, axis=0)
        
        return tensor, 'GOOD'
    
    def infer(self, face_img: np.ndarray) -> InferenceResult:
        """
        Run inference on face image
        
        Args:
            face_img: Face crop in BGR format
            
        Returns:
            InferenceResult with fake score and metadata
        """
        start_time = time.perf_counter()
        
        # Preprocess
        tensor, quality = self.preprocess(face_img)
        
        if tensor is None:
            return InferenceResult(
                fake_score=0.0,
                prediction=-1,
                logits=np.array([0.0, 0.0]),
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                quality=quality,
                face_bbox=(0, 0, 0, 0)
            )
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        logits = outputs[0][0]  # [2]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        fake_score = float(probs[1])
        prediction = 1 if fake_score >= self.threshold else 0
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return InferenceResult(
            fake_score=fake_score,
            prediction=prediction,
            logits=logits,
            inference_time_ms=inference_time,
            quality=quality,
            face_bbox=(0, 0, 0, 0)  # Will be set by caller
        )
    
    def set_threshold(self, threshold: float):
        """Update classification threshold"""
        self.threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Threshold updated to {self.threshold:.2f}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'input_size': self.input_size,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'threshold': self.threshold,
            'providers': self.session.get_providers()
        }


def create_inference_engine(config: dict) -> SwapFaceInference:
    """Create inference engine from config"""
    model_config = config.get('model', {})
    inference_config = config.get('inference', {})
    
    return SwapFaceInference(
        model_path=model_config.get('path', 'model_files/swapface_detector_fp32.onnx'),
        input_size=tuple(model_config.get('input_size', [224, 224])),
        input_name=model_config.get('input_name', 'input'),
        output_name=model_config.get('output_name', 'output'),
        providers=model_config.get('providers', ['CPUExecutionProvider']),
        threshold=inference_config.get('threshold', 0.5)
    )


if __name__ == "__main__":
    # Test inference
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy model for testing
    # In real usage, provide path to actual ONNX model
    model_path = 'model_files/swapface_detector_fp32.onnx'
    
    if os.path.exists(model_path):
        engine = create_inference_engine({
            'model': {'path': model_path, 'input_size': [224, 224]},
            'inference': {'threshold': 0.5}
        })
        
        # Test with random image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = engine.infer(test_img)
        print(f"Fake Score: {result.fake_score:.4f}")
        print(f"Prediction: {result.prediction}")
        print(f"Inference Time: {result.inference_time_ms:.2f} ms")
        print(f"Quality: {result.quality}")
    else:
        print(f"Model not found at {model_path}")
        print("Run export_onnx.py first to generate ONNX model")