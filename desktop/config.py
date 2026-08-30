"""
Desktop Application Configuration
"""

import yaml
from pathlib import Path


class DesktopConfig:
    """Configuration for desktop realtime application"""
    
    def __init__(self, config_path: str = None):
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
    
    def _default_config(self):
        return {
            'camera': {
                'index': 0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'face_detector': {
                'type': 'mediapipe',  # 'mediapipe', 'opencv_haar', 'opencv_dnn'
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.5,
                'model_selection': 0  # 0: short-range, 1: full-range
            },
            'model': {
                'path': 'model_files/swapface_detector_fp32.onnx',
                'input_size': [224, 224],
                'input_name': 'input',
                'output_name': 'output',
                'providers': ['CPUExecutionProvider']  # or ['CUDAExecutionProvider', 'CPUExecutionProvider']
            },
            'inference': {
                'target_fps': 10,  # AI inference FPS (lower than camera FPS)
                'threshold': 0.5,
                'hysteresis_high': 0.7,
                'hysteresis_low': 0.3
            },
            'face_processing': {
                'min_face_size': 80,
                'max_blur': 100.0,
                'min_brightness': 30,
                'max_brightness': 220,
                'crop_margin': 0.3
            },
            'temporal_smoothing': {
                'method': 'ema',  # 'ema', 'sliding_window', 'voting'
                'alpha': 0.3,
                'window_size': 5,
                'min_persistence_frames': 3
            },
            'ui': {
                'window_title': 'SwapFace Detector - Realtime',
                'show_fps': True,
                'show_fake_score': True,
                'show_face_quality': True,
                'show_threshold_slider': True,
                'bounding_box_thickness': 2,
                'font_scale': 0.6,
                'colors': {
                    'real': [0, 255, 0],      # Green
                    'suspicious': [0, 165, 255],  # Orange
                    'fake': [0, 0, 255],      # Red
                    'unknown': [0, 255, 255], # Yellow
                    'text': [255, 255, 255],  # White
                    'bg': [0, 0, 0]           # Black
                }
            },
            'quality_check': {
                'enabled': True,
                'check_blur': True,
                'check_brightness': True,
                'check_size': True
            }
        }
    
    def get(self, key: str, default=None):
        """Get nested config value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value):
        """Set nested config value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, config_path: str):
        """Save config to file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


# Global config instance
_config = None

def get_config(config_path: str = None) -> DesktopConfig:
    global _config
    if _config is None:
        default_path = Path(__file__).parent / 'config.yaml'
        _config = DesktopConfig(config_path or str(default_path))
    return _config


if __name__ == "__main__":
    config = get_config()
    print("Desktop Config:")
    print(yaml.dump(config.config, default_flow_style=False))