"""
Desktop Realtime SwapFace Detection Application
PyQt5 GUI with webcam capture, face detection, and realtime inference
"""

import sys
import cv2
import numpy as np
import logging
import time
from typing import Optional, List
from dataclasses import dataclass

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QComboBox, QCheckBox, QGroupBox, QGridLayout,
                             QMessageBox, QStatusBar)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen

from desktop.config import get_config
from desktop.face_detector import (create_face_detector, FaceDetector, 
                                    FaceDetection, FaceTracker)
from desktop.inference import create_inference_engine, SwapFaceInference, InferenceResult
from desktop.temporal_filter import (create_multi_track_filter, MultiTrackTemporalFilter,
                                      SignalState)


@dataclass
class FrameData:
    """Data for a processed frame"""
    frame: np.ndarray
    detections: List[FaceDetection]
    inference_results: List[InferenceResult]
    fps: float
    inference_fps: float


class VideoCaptureThread(QThread):
    """Thread for video capture"""
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self.cap = None
    
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera {self.camera_index}")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.error_occurred.emit("Failed to read frame")
                break
            
            self.frame_ready.emit(frame)
            self.msleep(1)  # Small sleep to prevent CPU overload
        
        if self.cap:
            self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()


class ProcessingThread(QThread):
    """Thread for face detection and inference"""
    result_ready = pyqtSignal(object)  # FrameData
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.frame_queue = []
        self.max_queue_size = 2
        
        # Initialize components
        self.face_detector = create_face_detector(
            config.get('face_detector.type', 'mediapipe'),
            min_detection_confidence=config.get('face_detector.min_detection_confidence', 0.7),
            min_tracking_confidence=config.get('face_detector.min_tracking_confidence', 0.5),
            model_selection=config.get('face_detector.model_selection', 0)
        )
        
        self.face_tracker = FaceTracker(
            iou_threshold=0.3,
            max_disappeared=10
        )
        
        self.inference_engine = create_inference_engine(config.config)
        
        self.temporal_filter = create_multi_track_filter(config.config)
        
        self.logger = logging.getLogger(__name__)
    
    def process_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        if len(self.frame_queue) < self.max_queue_size:
            self.frame_queue.append(frame.copy())
    
    def run(self):
        self.running = True
        last_inference_time = 0
        inference_interval = 1.0 / self.config.get('inference.target_fps', 10)
        
        while self.running:
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
                current_time = time.time()
                
                # Face detection (run every frame)
                detections = self.face_detector.detect(frame)
                tracked_detections = self.face_tracker.update(detections)
                
                # Inference (run at target FPS)
                inference_results = []
                run_inference = (current_time - last_inference_time) >= inference_interval
                
                if run_inference:
                    last_inference_time = current_time
                    
                    for det in tracked_detections:
                        x, y, w, h = det.bbox
                        
                        # Apply crop margin
                        margin = self.config.get('face_processing.crop_margin', 0.3)
                        mx = int(w * margin)
                        my = int(h * margin)
                        
                        x1 = max(0, x - mx)
                        y1 = max(0, y - my)
                        x2 = min(frame.shape[1], x + w + mx)
                        y2 = min(frame.shape[0], y + h + my)
                        
                        if x2 > x1 and y2 > y1:
                            face_crop = frame[y1:y2, x1:x2]
                            
                            # Run inference
                            result = self.inference_engine.infer(face_crop)
                            result.face_bbox = det.bbox
                            
                            # Temporal filtering
                            smoothed_score, state = self.temporal_filter.update(
                                det.track_id, result.fake_score, result.quality)
                            
                            # Update result with smoothed score
                            result.fake_score = smoothed_score
                            result.prediction = 1 if state in [SignalState.FAKE, SignalState.SUSPICIOUS] else 0
                            
                            inference_results.append((det, result, state))
                
                # Calculate FPS
                fps = self._calculate_fps()
                inference_fps = self._calculate_inference_fps()
                
                # Emit result
                self.result_ready.emit(FrameData(
                    frame=frame,
                    detections=tracked_detections,
                    inference_results=inference_results,
                    fps=fps,
                    inference_fps=inference_fps
                ))
            
            self.msleep(1)
    
    def _calculate_fps(self) -> float:
        # Placeholder - implement actual FPS calculation
        return 30.0
    
    def _calculate_inference_fps(self) -> float:
        return self.config.get('inference.target_fps', 10)
    
    def update_threshold(self, threshold: float):
        self.inference_engine.set_threshold(threshold)
    
    def stop(self):
        self.running = False
        self.wait()
        self.face_detector.close()


class VideoWidget(QLabel):
    """Widget for displaying video with overlays"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        
        self.current_frame = None
        self.detections = []
        self.inference_results = []
        self.fps = 0
        self.inference_fps = 0
        self.show_fps = True
        self.show_fake_score = True
        self.show_quality = True
        
        # Colors
        self.colors = {
            'real': QColor(0, 255, 0),
            'suspicious': QColor(0, 165, 255),
            'fake': QColor(255, 0, 0),
            'unknown': QColor(255, 255, 0),
            'text': QColor(255, 255, 255),
            'bg': QColor(0, 0, 0, 180)
        }
    
    def update_frame(self, frame_data: FrameData):
        self.current_frame = frame_data.frame
        self.detections = frame_data.detections
        self.inference_results = frame_data.inference_results
        self.fps = frame_data.fps
        self.inference_fps = frame_data.inference_fps
        self.update()
    
    def set_display_options(self, show_fps: bool, show_fake_score: bool, show_quality: bool):
        self.show_fps = show_fps
        self.show_fake_score = show_fake_score
        self.show_quality = show_quality
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.current_frame is None:
            return
        
        painter = QPainter(self)
        
        # Convert frame to QImage
        h, w, ch = self.current_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(self.current_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to widget size while maintaining aspect ratio
        widget_rect = self.rect()
        scaled_img = q_img.scaled(widget_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Center the image
        x = (widget_rect.width() - scaled_img.width()) // 2
        y = (widget_rect.height() - scaled_img.height()) // 2
        
        painter.drawImage(x, y, scaled_img)
        
        # Calculate scale factors
        scale_x = scaled_img.width() / w
        scale_y = scaled_img.height() / h
        
        # Draw detections and inference results
        for det, result, state in self.inference_results:
            bbox = det.bbox
            x1 = int((bbox[0]) * scale_x) + x
            y1 = int((bbox[1]) * scale_y) + y
            w = int(bbox[2] * scale_x)
            h = int(bbox[3] * scale_y)
            
            # Color based on state
            if state == SignalState.REAL:
                color = self.colors['real']
            elif state == SignalState.SUSPICIOUS:
                color = self.colors['suspicious']
            elif state == SignalState.FAKE:
                color = self.colors['fake']
            else:
                color = self.colors['unknown']
            
            # Draw bounding box
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(x1, y1, w, h)
            
            # Draw label background
            label_text = f"ID:{det.track_id}"
            if self.show_fake_score:
                label_text += f" | Fake: {result.fake_score:.1%}"
            if self.show_quality:
                label_text += f" | {result.quality}"
            
            font = QFont("Arial", 9)
            painter.setFont(font)
            
            text_rect = painter.fontMetrics().boundingRect(label_text)
            text_rect.moveTopLeft(x1, y1 - text_rect.height() - 4)
            
            painter.fillRect(text_rect.adjusted(-4, -2, 4, 2), self.colors['bg'])
            painter.setPen(self.colors['text'])
            painter.drawText(text_rect, Qt.AlignCenter, label_text)
        
        # Draw info overlay
        if self.show_fps:
            info_text = f"FPS: {self.fps:.1f} | Inf FPS: {self.inference_fps:.1f}"
            painter.setPen(self.colors['text'])
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, info_text)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle(self.config.get('ui.window_title', 'SwapFace Detector'))
        self.resize(1000, 700)
        
        # Initialize UI
        self._init_ui()
        
        # Initialize threads
        self._init_threads()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # FPS tracking
        self.frame_times = []
        self.last_frame_time = time.time()
    
    def _init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        self.video_widget = VideoWidget()
        main_layout.addWidget(self.video_widget, 3)
        
        # Right panel - Controls
        controls_panel = self._create_controls_panel()
        main_layout.addWidget(controls_panel, 1)
    
    def _create_controls_panel(self) -> QWidget:
        """Create controls panel"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        
        # Camera controls
        cam_group = QGroupBox("Camera")
        cam_layout = QGridLayout(cam_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Camera {i}" for i in range(4)])
        self.camera_combo.setCurrentIndex(self.config.get('camera.index', 0))
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        cam_layout.addWidget(QLabel("Camera:"), 0, 0)
        cam_layout.addWidget(self.camera_combo, 0, 1)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._toggle_camera)
        cam_layout.addWidget(self.start_btn, 1, 0, 1, 2)
        
        layout.addWidget(cam_group)
        
        # Detection controls
        det_group = QGroupBox("Face Detection")
        det_layout = QGridLayout(det_group)
        
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["MediaPipe", "OpenCV Haar", "OpenCV DNN"])
        self.detector_combo.setCurrentText(
            self.config.get('face_detector.type', 'mediapipe').replace('_', ' ').title()
        )
        det_layout.addWidget(QLabel("Detector:"), 0, 0)
        det_layout.addWidget(self.detector_combo, 0, 1)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(int(self.config.get('face_detector.min_detection_confidence', 0.7) * 100))
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)
        det_layout.addWidget(QLabel("Confidence:"), 1, 0)
        det_layout.addWidget(self.conf_slider, 1, 1)
        
        self.conf_label = QLabel(f"{self.conf_slider.value() / 100:.2f}")
        det_layout.addWidget(self.conf_label, 1, 2)
        
        layout.addWidget(det_group)
        
        # Inference controls
        inf_group = QGroupBox("Inference")
        inf_layout = QGridLayout(inf_group)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(int(self.config.get('inference.threshold', 0.5) * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        inf_layout.addWidget(QLabel("Threshold:"), 0, 0)
        inf_layout.addWidget(self.threshold_slider, 0, 1)
        
        self.threshold_label = QLabel(f"{self.threshold_slider.value() / 100:.2f}")
        inf_layout.addWidget(self.threshold_label, 0, 2)
        
        self.hyst_high_slider = QSlider(Qt.Horizontal)
        self.hyst_high_slider.setRange(50, 95)
        self.hyst_high_slider.setValue(int(self.config.get('inference.hysteresis_high', 0.7) * 100))
        inf_layout.addWidget(QLabel("Hyst. High:"), 1, 0)
        inf_layout.addWidget(self.hyst_high_slider, 1, 1)
        
        self.hyst_high_label = QLabel(f"{self.hyst_high_slider.value() / 100:.2f}")
        inf_layout.addWidget(self.hyst_high_label, 1, 2)
        
        self.hyst_low_slider = QSlider(Qt.Horizontal)
        self.hyst_low_slider.setRange(10, 50)
        self.hyst_low_slider.setValue(int(self.config.get('inference.hysteresis_low', 0.3) * 100))
        inf_layout.addWidget(QLabel("Hyst. Low:"), 2, 0)
        inf_layout.addWidget(self.hyst_low_slider, 2, 1)
        
        self.hyst_low_label = QLabel(f"{self.hyst_low_slider.value() / 100:.2f}")
        inf_layout.addWidget(self.hyst_low_label, 2, 2)
        
        layout.addWidget(inf_group)
        
        # Display options
        disp_group = QGroupBox("Display")
        disp_layout = QVBoxLayout(disp_group)
        
        self.show_fps_cb = QCheckBox("Show FPS")
        self.show_fps_cb.setChecked(self.config.get('ui.show_fps', True))
        self.show_fps_cb.toggled.connect(self._on_display_option_changed)
        disp_layout.addWidget(self.show_fps_cb)
        
        self.show_score_cb = QCheckBox("Show Fake Score")
        self.show_score_cb.setChecked(self.config.get('ui.show_fake_score', True))
        self.show_score_cb.toggled.connect(self._on_display_option_changed)
        disp_layout.addWidget(self.show_score_cb)
        
        self.show_quality_cb = QCheckBox("Show Face Quality")
        self.show_quality_cb.setChecked(self.config.get('ui.show_face_quality', True))
        self.show_quality_cb.toggled.connect(self._on_display_option_changed)
        disp_layout.addWidget(self.show_quality_cb)
        
        layout.addWidget(disp_group)
        
        # Status info
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Camera: Stopped")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: --")
        status_layout.addWidget(self.fps_label)
        
        self.inf_fps_label = QLabel("Inf FPS: --")
        status_layout.addWidget(self.inf_fps_label)
        
        layout.addWidget(status_group)
        
        layout.addStretch()
        
        return panel
    
    def _init_threads(self):
        """Initialize processing threads"""
        self.capture_thread = VideoCaptureThread(
            camera_index=self.config.get('camera.index', 0),
            width=self.config.get('camera.width', 640),
            height=self.config.get('camera.height', 480),
            fps=self.config.get('camera.fps', 30)
        )
        self.capture_thread.frame_ready.connect(self._on_frame_ready)
        self.capture_thread.error_occurred.connect(self._on_camera_error)
        
        self.processing_thread = ProcessingThread(self.config)
        self.processing_thread.result_ready.connect(self._on_result_ready)
    
    def _on_frame_ready(self, frame: np.ndarray):
        """Handle new frame from capture thread"""
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Track FPS
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]
        fps = len(self.frame_times)
        
        # Send to processing thread
        self.processing_thread.process_frame(frame)
        
        # Update status
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def _on_result_ready(self, frame_data: FrameData):
        """Handle processed frame"""
        self.video_widget.update_frame(frame_data)
        self.inf_fps_label.setText(f"Inf FPS: {frame_data.inference_fps:.1f}")
        
        # Update status with detection info
        if frame_data.inference_results:
            status_parts = []
            for _, result, state in frame_data.inference_results:
                status_parts.append(f"{state.value}: {result.fake_score:.1%}")
            self.status_label.setText(" | ".join(status_parts))
        else:
            self.status_label.setText("No faces detected")
    
    def _on_camera_changed(self, index: int):
        self.config.set('camera.index', index)
        self._restart_camera()
    
    def _on_confidence_changed(self, value: int):
        conf = value / 100.0
        self.conf_label.setText(f"{conf:.2f}")
        self.config.set('face_detector.min_detection_confidence', conf)
        # Would need to recreate detector - simplified for now
    
    def _on_threshold_changed(self, value: int):
        thresh = value / 100.0
        self.threshold_label.setText(f"{thresh:.2f}")
        self.config.set('inference.threshold', thresh)
        self.processing_thread.update_threshold(thresh)
    
    def _on_display_option_changed(self):
        self.video_widget.set_display_options(
            self.show_fps_cb.isChecked(),
            self.show_score_cb.isChecked(),
            self.show_quality_cb.isChecked()
        )
    
    def _toggle_camera(self):
        if self.capture_thread.isRunning():
            self._stop_camera()
        else:
            self._start_camera()
    
    def _start_camera(self):
        self.capture_thread.start()
        self.processing_thread.start()
        self.start_btn.setText("Stop")
        self.status_bar.showMessage("Camera started")
    
    def _stop_camera(self):
        self.capture_thread.stop()
        self.processing_thread.stop()
        self.start_btn.setText("Start")
        self.status_bar.showMessage("Camera stopped")
        self.video_widget.current_frame = None
        self.video_widget.update()
    
    def _restart_camera(self):
        was_running = self.capture_thread.isRunning()
        if was_running:
            self._stop_camera()
        if was_running:
            self._start_camera()
    
    def _on_camera_error(self, error: str):
        self.logger.error(f"Camera error: {error}")
        QMessageBox.critical(self, "Camera Error", error)
        self._stop_camera()
    
    def closeEvent(self, event):
        self._stop_camera()
        event.accept()


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()