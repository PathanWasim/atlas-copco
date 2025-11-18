# src/detect_yolo.py
"""
YOLOv8 object detection module for identifying persons, boxes, lids, and tools.
Uses ultralytics YOLOv8 for real-time object detection.
"""
import os
from typing import List, Dict, Any
import numpy as np

from ultralytics import YOLO

class YOLODetector:
    """
    Wrapper around ultralytics YOLOv8 model for object detection.
    
    This class loads a YOLOv8 model and provides a simple interface for
    detecting objects in video frames. Detections are grouped by class name
    and filtered by confidence threshold.
    
    Attributes:
        model_path: Path to the YOLOv8 model file (.pt)
        device: Device for inference ('auto', 'cpu', or 'cuda')
        conf_threshold: Minimum confidence score for detections (0-1)
        model: Loaded YOLO model instance
        class_names: Dictionary mapping class indices to names
    
    Example:
        >>> detector = YOLODetector("yolov8n.pt", conf_threshold=0.3)
        >>> detections = detector.detect(frame)
        >>> person_boxes = detections.get("person", [])
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "auto", conf_threshold: float = 0.3):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLOv8 model file (e.g., 'yolov8n.pt')
            device: Device for inference - 'auto' selects GPU if available
            conf_threshold: Minimum confidence threshold for filtering detections
        
        Raises:
            FileNotFoundError: If model_path does not exist
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load YOLO model - ultralytics will auto-select device if available
        self.model = YOLO(model_path)
        
        # Get class names from model metadata or use fallback
        self.class_names = self._get_class_names()
    
    def _get_class_names(self) -> Dict[int, str]:
        """
        Extract class names from model metadata.
        
        Returns:
            Dictionary mapping class indices to class names
        """
        # Try to get class names from model metadata
        try:
            names = self.model.model.names
            if names:
                return names
        except Exception:
            pass
        
        # Fallback class mapping for custom trained models
        return {0: "person", 1: "box", 2: "lid", 3: "tool"}
    
    def detect(self, frame: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run object detection on a single frame.
        
        Performs YOLOv8 inference on the input frame and returns detections
        grouped by class name. Only detections above the confidence threshold
        are included.
        
        Args:
            frame: Input image as numpy array (H, W, C) in BGR format
        
        Returns:
            Dictionary with class names as keys and lists of detections as values.
            Each detection contains:
                - class: Class name (str)
                - cls_idx: Class index (int)
                - bbox: Bounding box [x1, y1, x2, y2] in pixels (List[int])
                - confidence: Detection confidence score 0-1 (float)
        
        Example:
            >>> detections = detector.detect(frame)
            >>> {
            ...     'person': [{'class': 'person', 'bbox': [100, 50, 300, 400], 'confidence': 0.92}],
            ...     'box': [{'class': 'box', 'bbox': [200, 300, 350, 450], 'confidence': 0.85}]
            ... }
        """
        # Run YOLO inference - returns Results object
        results = self.model(frame)
        
        if len(results) == 0:
            return {}
        
        res = results[0]
        out = {}
        
        # Process each detected box
        for box in res.boxes:
            # Extract confidence score
            conf = float(box.conf.cpu().numpy()) if hasattr(box.conf, 'cpu') else float(box.conf)
            
            # Filter by confidence threshold
            if conf < self.conf_threshold:
                continue
            
            # Extract class index and name
            cls_idx = int(box.cls.cpu().numpy()) if hasattr(box.cls, 'cpu') else int(box.cls)
            cls_name = self.class_names.get(cls_idx, str(cls_idx))
            
            # Extract bounding box coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0].numpy()
            x1, y1, x2, y2 = [int(x) for x in xyxy.tolist()]
            
            # Create detection dictionary
            det = {
                "class": cls_name,
                "cls_idx": cls_idx,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            }
            
            # Group by class name
            out.setdefault(cls_name, []).append(det)
        
        return out
