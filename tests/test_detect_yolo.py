# tests/test_detect_yolo.py
"""
Unit tests for YOLODetector class.
Tests model loading, detection output format, confidence filtering, and device selection.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.detect_yolo import YOLODetector


class TestYOLODetector:
    """Test suite for YOLODetector class."""
    
    def test_model_loading_valid_path(self):
        """Test that model loads successfully with valid path."""
        with patch('src.detect_yolo.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.model.names = {0: "person", 1: "box"}
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.3)
            
            assert detector.model_path == "yolov8n.pt"
            assert detector.conf_threshold == 0.3
            mock_yolo.assert_called_once_with("yolov8n.pt")
    
    def test_model_loading_invalid_path(self):
        """Test that appropriate error is raised with invalid model path."""
        with patch('src.detect_yolo.YOLO') as mock_yolo:
            mock_yolo.side_effect = FileNotFoundError("Model not found")
            
            with pytest.raises(FileNotFoundError):
                detector = YOLODetector(model_path="invalid_path.pt")
    
    def test_detection_output_format(self):
        """Test that detection output has correct structure."""
        with patch('src.detect_yolo.YOLO') as mock_yolo:
            # Setup mock model and results
            mock_model = Mock()
            mock_model.model.names = {0: "person", 1: "box"}
            
            # Create mock detection box
            mock_box = Mock()
            mock_box.conf = Mock()
            mock_box.conf.cpu.return_value.numpy.return_value = 0.85
            mock_box.cls = Mock()
            mock_box.cls.cpu.return_value.numpy.return_value = 0
            mock_box.xyxy = [[np.array([100, 50, 300, 400])]]
            
            mock_result = Mock()
            mock_result.boxes = [mock_box]
            
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.3)
            
            # Create dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)
            
            # Verify output structure
            assert isinstance(detections, dict)
            assert "person" in detections
            assert isinstance(detections["person"], list)
            assert len(detections["person"]) == 1
            
            det = detections["person"][0]
            assert "class" in det
            assert "cls_idx" in det
            assert "bbox" in det
            assert "confidence" in det
            assert det["class"] == "person"
            assert len(det["bbox"]) == 4
            assert 0 <= det["confidence"] <= 1
    
    def test_confidence_filtering(self):
        """Test that detections below confidence threshold are filtered out."""
        with patch('src.detect_yolo.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.model.names = {0: "person"}
            
            # Create two mock boxes: one above threshold, one below
            mock_box_high = Mock()
            mock_box_high.conf = Mock()
            mock_box_high.conf.cpu.return_value.numpy.return_value = 0.85
            mock_box_high.cls = Mock()
            mock_box_high.cls.cpu.return_value.numpy.return_value = 0
            mock_box_high.xyxy = [[np.array([100, 50, 300, 400])]]
            
            mock_box_low = Mock()
            mock_box_low.conf = Mock()
            mock_box_low.conf.cpu.return_value.numpy.return_value = 0.15
            mock_box_low.cls = Mock()
            mock_box_low.cls.cpu.return_value.numpy.return_value = 0
            mock_box_low.xyxy = [[np.array([200, 100, 400, 500])]]
            
            mock_result = Mock()
            mock_result.boxes = [mock_box_high, mock_box_low]
            
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.3)
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)
            
            # Only high confidence detection should be included
            assert len(detections["person"]) == 1
            assert detections["person"][0]["confidence"] == 0.85
    
    def test_empty_frame_detection(self):
        """Test detection on frame with no objects."""
        with patch('src.detect_yolo.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.model.names = {0: "person"}
            
            mock_result = Mock()
            mock_result.boxes = []
            
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.3)
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)
            
            # Should return empty dict or dict with empty lists
            assert isinstance(detections, dict)
            assert len(detections) == 0 or all(len(v) == 0 for v in detections.values())
    
    def test_multiple_class_detection(self):
        """Test detection of multiple object classes."""
        with patch('src.detect_yolo.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.model.names = {0: "person", 1: "box", 2: "lid"}
            
            # Create mock boxes for different classes
            boxes = []
            for cls_idx, cls_name in [(0, "person"), (1, "box"), (2, "lid")]:
                mock_box = Mock()
                mock_box.conf = Mock()
                mock_box.conf.cpu.return_value.numpy.return_value = 0.8
                mock_box.cls = Mock()
                mock_box.cls.cpu.return_value.numpy.return_value = cls_idx
                mock_box.xyxy = [[np.array([100, 50, 300, 400])]]
                boxes.append(mock_box)
            
            mock_result = Mock()
            mock_result.boxes = boxes
            
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.3)
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = detector.detect(frame)
            
            # Should have all three classes
            assert "person" in detections
            assert "box" in detections
            assert "lid" in detections
            assert len(detections["person"]) == 1
            assert len(detections["box"]) == 1
            assert len(detections["lid"]) == 1
    
    def test_class_names_fallback(self):
        """Test fallback class names when model metadata unavailable."""
        with patch('src.detect_yolo.YOLO') as mock_yolo:
            mock_model = Mock()
            # Simulate missing model.names
            mock_model.model.names = None
            mock_yolo.return_value = mock_model
            
            detector = YOLODetector(model_path="yolov8n.pt")
            
            # Should use fallback class names
            assert 0 in detector.class_names
            assert detector.class_names[0] == "person"
            assert detector.class_names[1] == "box"
            assert detector.class_names[2] == "lid"
            assert detector.class_names[3] == "tool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
