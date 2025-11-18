# tests/test_opening_logic.py
"""
Unit tests for box-opening detection logic.
Tests hand-in-box detection, lid movement detection, and confidence scoring.
"""
import pytest
from src.opening_logic import point_in_bbox, bbox_area, is_opening_box


class TestPointInBbox:
    """Test suite for point_in_bbox function."""
    
    def test_point_inside_bbox(self):
        """Test point clearly inside bounding box."""
        bbox = [100, 100, 200, 200]
        point = (150, 150)
        assert point_in_bbox(point, bbox) == True
    
    def test_point_outside_bbox(self):
        """Test point clearly outside bounding box."""
        bbox = [100, 100, 200, 200]
        point = (250, 250)
        assert point_in_bbox(point, bbox) == False
    
    def test_point_on_bbox_edge(self):
        """Test point exactly on bounding box edge."""
        bbox = [100, 100, 200, 200]
        point = (100, 150)  # On left edge
        assert point_in_bbox(point, bbox) == True
        
        point = (200, 150)  # On right edge
        assert point_in_bbox(point, bbox) == True
    
    def test_point_with_margin(self):
        """Test point outside bbox but within margin."""
        bbox = [100, 100, 200, 200]
        point = (95, 150)  # 5 pixels left of bbox
        
        # Without margin, should be outside
        assert point_in_bbox(point, bbox, margin=0.0) == False
        
        # With 10% margin (10 pixels), should be inside
        assert point_in_bbox(point, bbox, margin=0.1) == True
    
    def test_none_point(self):
        """Test handling of None point."""
        bbox = [100, 100, 200, 200]
        assert point_in_bbox(None, bbox) == False


class TestBboxArea:
    """Test suite for bbox_area function."""
    
    def test_normal_bbox(self):
        """Test area calculation for normal bounding box."""
        bbox = [100, 100, 200, 300]
        area = bbox_area(bbox)
        assert area == 100 * 200  # width * height
        assert area == 20000
    
    def test_square_bbox(self):
        """Test area calculation for square bounding box."""
        bbox = [0, 0, 100, 100]
        area = bbox_area(bbox)
        assert area == 10000
    
    def test_zero_area_bbox(self):
        """Test bounding box with zero area."""
        bbox = [100, 100, 100, 100]  # Point, not box
        area = bbox_area(bbox)
        assert area == 0
    
    def test_inverted_bbox(self):
        """Test handling of inverted bounding box coordinates."""
        bbox = [200, 200, 100, 100]  # x2 < x1, y2 < y1
        area = bbox_area(bbox)
        assert area == 0  # Should return 0, not negative


class TestIsOpeningBox:
    """Test suite for is_opening_box function."""
    
    def test_hand_in_box_left_wrist(self):
        """Test detection when left wrist is inside box."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (150, 200),  # Inside box
            "right_wrist": (50, 100)   # Outside box
        }
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions
        )
        
        # Hand in box but no lid movement, so should be False
        assert is_opening == False
        # But confidence should be > 0 due to hand proximity
        assert confidence > 0
    
    def test_hand_in_box_right_wrist(self):
        """Test detection when right wrist is inside box."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (50, 100),   # Outside box
            "right_wrist": (200, 250)  # Inside box
        }
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions
        )
        
        assert is_opening == False  # No lid movement
        assert confidence > 0
    
    def test_hand_outside_box(self):
        """Test when both wrists are outside box."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (50, 100),   # Outside
            "right_wrist": (50, 120)   # Outside
        }
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions
        )
        
        assert is_opening == False
        # Confidence should be low
        assert confidence < 0.5
    
    def test_lid_upward_movement(self):
        """Test detection of lid moving upward."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (150, 200),  # Inside box
            "right_wrist": None
        }
        
        prev_lid_bbox = [120, 160, 280, 180]  # Lid at y=160
        curr_lid_bbox = [120, 150, 280, 170]  # Lid moved up to y=150
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox
        )
        
        # Hand in box AND lid moving up = opening detected
        assert is_opening == True
        assert confidence > 0.5
    
    def test_lid_area_increase(self):
        """Test detection of lid area increasing."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (150, 200),
            "right_wrist": None
        }
        
        prev_lid_bbox = [120, 160, 180, 180]  # Small lid (60x20 = 1200)
        curr_lid_bbox = [120, 160, 220, 180]  # Larger lid (100x20 = 2000)
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox
        )
        
        # Area increased by >10%, should detect opening
        assert is_opening == True
        assert confidence > 0.5
    
    def test_no_lid_movement(self):
        """Test when lid is not moving."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (150, 200),
            "right_wrist": None
        }
        
        # Same lid position in both frames
        prev_lid_bbox = [120, 160, 280, 180]
        curr_lid_bbox = [120, 160, 280, 180]
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox
        )
        
        # Hand in box but no lid movement = no opening
        assert is_opening == False
    
    def test_missing_detections(self):
        """Test handling of missing detections (None values)."""
        # No person, no box, no wrists
        is_opening, confidence = is_opening_box(
            None, None, {"left_wrist": None, "right_wrist": None}
        )
        
        assert is_opening == False
        assert confidence >= 0  # Should not crash
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation with various inputs."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (150, 200),
            "right_wrist": None
        }
        
        prev_lid_bbox = [120, 160, 280, 180]
        curr_lid_bbox = [120, 150, 280, 170]  # Moved up
        
        # High detection confidences
        high_confs = {"person": 0.95, "box": 0.90, "lid": 0.85}
        is_opening, high_conf = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox,
            detection_confidences=high_confs
        )
        
        # Low detection confidences
        low_confs = {"person": 0.4, "box": 0.4, "lid": 0.3}
        is_opening, low_conf = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox,
            detection_confidences=low_confs
        )
        
        # Both should detect opening
        assert is_opening == True
        
        # High confidence detections should yield higher overall confidence
        assert high_conf > low_conf
        
        # Confidence should be in valid range
        assert 0 <= high_conf <= 1
        assert 0 <= low_conf <= 1
    
    def test_hand_with_margin(self):
        """Test hand-in-box detection with 15% margin."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]  # 200x200 box
        
        # Wrist just outside box but within 15% margin (30 pixels)
        wrist_positions = {
            "left_wrist": (85, 200),  # 15 pixels left of box
            "right_wrist": None
        }
        
        prev_lid_bbox = [120, 160, 280, 180]
        curr_lid_bbox = [120, 150, 280, 170]
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox
        )
        
        # Should detect opening due to margin
        assert is_opening == True
    
    def test_small_lid_movement_ignored(self):
        """Test that small lid movements (<2 pixels) are ignored."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (150, 200),
            "right_wrist": None
        }
        
        prev_lid_bbox = [120, 160, 280, 180]
        curr_lid_bbox = [120, 159, 280, 179]  # Only 1 pixel movement
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox
        )
        
        # Small movement should be ignored (noise tolerance)
        assert is_opening == False
    
    def test_small_area_increase_ignored(self):
        """Test that small area increases (<10%) are ignored."""
        person_bbox = [50, 50, 250, 400]
        box_bbox = [100, 150, 300, 350]
        wrist_positions = {
            "left_wrist": (150, 200),
            "right_wrist": None
        }
        
        prev_lid_bbox = [120, 160, 220, 180]  # 100x20 = 2000
        curr_lid_bbox = [120, 160, 225, 180]  # 105x20 = 2100 (5% increase)
        
        is_opening, confidence = is_opening_box(
            person_bbox, box_bbox, wrist_positions,
            prev_lid_bbox, curr_lid_bbox
        )
        
        # 5% increase should be ignored
        assert is_opening == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
