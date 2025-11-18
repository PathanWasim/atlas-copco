# tests/test_hand_pose.py
"""
Unit tests for HandPoseEstimator class.
Tests wrist extraction, coordinate conversion, and handling of edge cases.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from src.hand_pose import HandPoseEstimator


class TestHandPoseEstimator:
    """Test suite for HandPoseEstimator class."""
    
    def test_initialization(self):
        """Test that HandPoseEstimator initializes correctly."""
        with patch('src.hand_pose.mp.solutions.pose') as mock_pose:
            mock_pose_instance = Mock()
            mock_pose.Pose.return_value = mock_pose_instance
            
            estimator = HandPoseEstimator(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.7
            )
            
            mock_pose.Pose.assert_called_once()
            call_kwargs = mock_pose.Pose.call_args[1]
            assert call_kwargs['static_image_mode'] == True
            assert call_kwargs['min_detection_confidence'] == 0.6
            assert call_kwargs['min_tracking_confidence'] == 0.7
    
    def test_wrist_extraction_successful(self):
        """Test successful wrist extraction from person bbox."""
        with patch('src.hand_pose.mp.solutions.pose') as mock_pose_module:
            # Setup mock pose estimator
            mock_pose_instance = Mock()
            mock_pose_module.Pose.return_value = mock_pose_instance
            
            # Create mock landmarks
            mock_left_wrist = Mock()
            mock_left_wrist.x = 0.3  # Normalized coordinates
            mock_left_wrist.y = 0.5
            
            mock_right_wrist = Mock()
            mock_right_wrist.x = 0.7
            mock_right_wrist.y = 0.5
            
            mock_landmarks = [None] * 33  # Mediapipe has 33 landmarks
            mock_landmarks[15] = mock_left_wrist  # LEFT_WRIST index
            mock_landmarks[16] = mock_right_wrist  # RIGHT_WRIST index
            
            mock_result = Mock()
            mock_result.pose_landmarks = Mock()
            mock_result.pose_landmarks.landmark = mock_landmarks
            
            mock_pose_instance.process.return_value = mock_result
            
            # Setup PoseLandmark enum
            mock_pose_module.PoseLandmark = Mock()
            mock_pose_module.PoseLandmark.LEFT_WRIST = 15
            mock_pose_module.PoseLandmark.RIGHT_WRIST = 16
            
            estimator = HandPoseEstimator()
            
            # Create test frame and person bbox
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            person_bbox = [100, 50, 300, 400]  # x1, y1, x2, y2
            
            wrists = estimator.extract_wrists(frame, person_bbox)
            
            # Verify wrist positions
            assert wrists is not None
            assert "left_wrist" in wrists
            assert "right_wrist" in wrists
            assert wrists["left_wrist"] is not None
            assert wrists["right_wrist"] is not None
            
            # Check coordinate conversion (normalized to absolute)
            left_x, left_y = wrists["left_wrist"]
            right_x, right_y = wrists["right_wrist"]
            
            # Coordinates should be within person bbox
            assert 100 <= left_x <= 300
            assert 50 <= left_y <= 400
            assert 100 <= right_x <= 300
            assert 50 <= right_y <= 400
    
    def test_wrist_extraction_no_landmarks(self):
        """Test handling when no pose landmarks are detected."""
        with patch('src.hand_pose.mp.solutions.pose') as mock_pose_module:
            mock_pose_instance = Mock()
            mock_pose_module.Pose.return_value = mock_pose_instance
            
            # No landmarks detected
            mock_result = Mock()
            mock_result.pose_landmarks = None
            mock_pose_instance.process.return_value = mock_result
            
            estimator = HandPoseEstimator()
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            person_bbox = [100, 50, 300, 400]
            
            wrists = estimator.extract_wrists(frame, person_bbox)
            
            # Should return None for both wrists
            assert wrists["left_wrist"] is None
            assert wrists["right_wrist"] is None
    
    def test_invalid_person_bbox(self):
        """Test handling of invalid person bounding box."""
        with patch('src.hand_pose.mp.solutions.pose') as mock_pose_module:
            mock_pose_instance = Mock()
            mock_pose_module.Pose.return_value = mock_pose_instance
            
            estimator = HandPoseEstimator()
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Test with inverted bbox (x2 < x1)
            invalid_bbox = [300, 50, 100, 400]
            wrists = estimator.extract_wrists(frame, invalid_bbox)
            
            assert wrists["left_wrist"] is None
            assert wrists["right_wrist"] is None
            
            # Test with zero-width bbox
            zero_width_bbox = [100, 50, 100, 400]
            wrists = estimator.extract_wrists(frame, zero_width_bbox)
            
            assert wrists["left_wrist"] is None
            assert wrists["right_wrist"] is None
    
    def test_bbox_clipping_to_frame(self):
        """Test that person bbox is clipped to frame boundaries."""
        with patch('src.hand_pose.mp.solutions.pose') as mock_pose_module:
            mock_pose_instance = Mock()
            mock_pose_module.Pose.return_value = mock_pose_instance
            
            # Mock successful detection
            mock_left_wrist = Mock()
            mock_left_wrist.x = 0.5
            mock_left_wrist.y = 0.5
            
            mock_right_wrist = Mock()
            mock_right_wrist.x = 0.5
            mock_right_wrist.y = 0.5
            
            mock_landmarks = [None] * 33
            mock_landmarks[15] = mock_left_wrist
            mock_landmarks[16] = mock_right_wrist
            
            mock_result = Mock()
            mock_result.pose_landmarks = Mock()
            mock_result.pose_landmarks.landmark = mock_landmarks
            mock_pose_instance.process.return_value = mock_result
            
            mock_pose_module.PoseLandmark = Mock()
            mock_pose_module.PoseLandmark.LEFT_WRIST = 15
            mock_pose_module.PoseLandmark.RIGHT_WRIST = 16
            
            estimator = HandPoseEstimator()
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Bbox extends beyond frame boundaries
            person_bbox = [-50, -50, 700, 500]
            
            wrists = estimator.extract_wrists(frame, person_bbox)
            
            # Should still work with clipped bbox
            assert wrists["left_wrist"] is not None
            assert wrists["right_wrist"] is not None
            
            # Coordinates should be within frame
            left_x, left_y = wrists["left_wrist"]
            right_x, right_y = wrists["right_wrist"]
            
            assert 0 <= left_x < 640
            assert 0 <= left_y < 480
            assert 0 <= right_x < 640
            assert 0 <= right_y < 480
    
    def test_coordinate_conversion_accuracy(self):
        """Test that normalized coordinates are correctly converted to pixels."""
        with patch('src.hand_pose.mp.solutions.pose') as mock_pose_module:
            mock_pose_instance = Mock()
            mock_pose_module.Pose.return_value = mock_pose_instance
            
            # Mock wrist at specific normalized position
            mock_left_wrist = Mock()
            mock_left_wrist.x = 0.0  # Left edge of crop
            mock_left_wrist.y = 0.0  # Top edge of crop
            
            mock_right_wrist = Mock()
            mock_right_wrist.x = 1.0  # Right edge of crop
            mock_right_wrist.y = 1.0  # Bottom edge of crop
            
            mock_landmarks = [None] * 33
            mock_landmarks[15] = mock_left_wrist
            mock_landmarks[16] = mock_right_wrist
            
            mock_result = Mock()
            mock_result.pose_landmarks = Mock()
            mock_result.pose_landmarks.landmark = mock_landmarks
            mock_pose_instance.process.return_value = mock_result
            
            mock_pose_module.PoseLandmark = Mock()
            mock_pose_module.PoseLandmark.LEFT_WRIST = 15
            mock_pose_module.PoseLandmark.RIGHT_WRIST = 16
            
            estimator = HandPoseEstimator()
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            person_bbox = [100, 50, 300, 250]  # 200x200 crop
            
            wrists = estimator.extract_wrists(frame, person_bbox)
            
            # Left wrist at (0, 0) normalized should be at bbox top-left
            left_x, left_y = wrists["left_wrist"]
            assert left_x == 100  # x1
            assert left_y == 50   # y1
            
            # Right wrist at (1, 1) normalized should be at bbox bottom-right
            right_x, right_y = wrists["right_wrist"]
            assert right_x == 299  # x2 - 1 (due to int conversion)
            assert right_y == 249  # y2 - 1
    
    def test_exception_handling(self):
        """Test graceful handling of exceptions during landmark extraction."""
        with patch('src.hand_pose.mp.solutions.pose') as mock_pose_module:
            mock_pose_instance = Mock()
            mock_pose_module.Pose.return_value = mock_pose_instance
            
            # Mock result that raises exception when accessing landmarks
            mock_result = Mock()
            mock_result.pose_landmarks = Mock()
            mock_result.pose_landmarks.landmark = Mock()
            mock_result.pose_landmarks.landmark.__getitem__ = Mock(
                side_effect=Exception("Landmark access error")
            )
            mock_pose_instance.process.return_value = mock_result
            
            estimator = HandPoseEstimator()
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            person_bbox = [100, 50, 300, 400]
            
            # Should handle exception and return None
            wrists = estimator.extract_wrists(frame, person_bbox)
            
            assert wrists["left_wrist"] is None
            assert wrists["right_wrist"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
