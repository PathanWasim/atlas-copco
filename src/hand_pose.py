# src/hand_pose.py
"""
Hand pose estimation module using Mediapipe Pose for wrist keypoint extraction.
Extracts left and right wrist positions from detected person regions.
"""
import cv2
from typing import Dict, Optional, Tuple, List
import numpy as np
import mediapipe as mp

class HandPoseEstimator:
    """
    Extracts wrist keypoints from person bounding boxes using Mediapipe Pose.
    
    This class uses Google's Mediapipe Pose model to detect body landmarks
    and extract wrist positions. It processes cropped person regions and
    returns wrist coordinates in the full frame coordinate system.
    
    Attributes:
        mp_pose: Mediapipe pose module
        pose: Mediapipe Pose model instance
    
    Example:
        >>> estimator = HandPoseEstimator()
        >>> wrists = estimator.extract_wrists(frame, person_bbox)
        >>> left_wrist = wrists['left_wrist']  # (x, y) or None
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize the Mediapipe Pose estimator.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection (0-1)
            min_tracking_confidence: Minimum confidence for landmark tracking (0-1)
        """
        self.mp_pose = mp.solutions.pose
        
        # Initialize Mediapipe Pose in static image mode
        # Static mode is better for individual frames vs video tracking
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_wrists(self, frame: np.ndarray, person_bbox: List[int]) -> Dict[str, Optional[Tuple[int, int]]]:
        """
        Extract left and right wrist coordinates from a person bounding box.
        
        This method crops the person region from the frame, runs Mediapipe Pose
        estimation, and extracts wrist landmarks. Coordinates are converted from
        normalized (0-1) to absolute pixel coordinates in the full frame.
        
        Args:
            frame: Full video frame as numpy array (H, W, 3) in BGR format
            person_bbox: Person bounding box [x1, y1, x2, y2] in pixel coordinates
        
        Returns:
            Dictionary with keys 'left_wrist' and 'right_wrist'.
            Values are (x, y) tuples in pixel coordinates, or None if not detected.
        
        Example:
            >>> wrists = estimator.extract_wrists(frame, [100, 50, 300, 400])
            >>> if wrists['left_wrist']:
            ...     x, y = wrists['left_wrist']
            ...     print(f"Left wrist at ({x}, {y})")
        
        Note:
            - Returns None for wrists if person bbox is invalid or pose not detected
            - Handles occlusion and partial visibility gracefully
            - Coordinates are clipped to frame boundaries
        """
        x1, y1, x2, y2 = person_bbox
        h, w = frame.shape[:2]
        
        # Clip bounding box to frame boundaries
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w-1, x2), min(h-1, y2)
        
        # Validate bbox dimensions
        if x2c <= x1c or y2c <= y1c:
            return {"left_wrist": None, "right_wrist": None}
        
        # Crop person region from frame
        crop = frame[y1c:y2c, x1c:x2c]
        
        # Convert BGR to RGB (Mediapipe expects RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Run Mediapipe Pose estimation
        res = self.pose.process(crop_rgb)
        
        # Default to None if no landmarks detected
        left, right = None, None
        
        if res.pose_landmarks:
            # Mediapipe Pose landmark indices:
            # LEFT_WRIST = 15, RIGHT_WRIST = 16
            lm = res.pose_landmarks.landmark
            
            try:
                # Get wrist landmarks
                lw = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
                rw = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                
                # Convert normalized coordinates (0-1) to absolute pixel coordinates
                # Mediapipe returns coordinates relative to crop, so we add crop offset
                lw_x = int(x1c + lw.x * (x2c - x1c))
                lw_y = int(y1c + lw.y * (y2c - y1c))
                rw_x = int(x1c + rw.x * (x2c - x1c))
                rw_y = int(y1c + rw.y * (y2c - y1c))
                
                left = (lw_x, lw_y)
                right = (rw_x, rw_y)
            except Exception:
                # Handle cases where landmarks are not available
                left, right = None, None
        
        return {"left_wrist": left, "right_wrist": right}
