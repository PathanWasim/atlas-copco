# src/opening_logic.py
"""
Box-opening detection logic using rule-based heuristics.
Determines if a box-opening event is occurring based on hand positions and lid movement.
"""
from typing import Optional, Tuple, Dict, List
import math

def point_in_bbox(point: Tuple[int, int], bbox: List[int], margin: float = 0.0) -> bool:
    """
    Check if a point is inside a bounding box with optional margin.
    
    Args:
        point: (x, y) coordinates or None
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Margin as fraction of bbox size (e.g., 0.15 = 15% margin)
    
    Returns:
        True if point is inside bbox (with margin), False otherwise
    
    Example:
        >>> point_in_bbox((150, 200), [100, 100, 200, 300], margin=0.1)
        True
    """
    if point is None:
        return False
    
    x, y = point
    x1, y1, x2, y2 = bbox
    
    # Calculate margin in pixels
    w = x2 - x1
    h = y2 - y1
    mx = margin * w
    my = margin * h
    
    # Check if point is within expanded bbox
    return (x1 - mx) <= x <= (x2 + mx) and (y1 - my) <= y <= (y2 + my)

def bbox_area(bbox: List[int]) -> int:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Area in pixels (non-negative)
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def is_opening_box(
    person_bbox: Optional[List[int]],
    box_bbox: Optional[List[int]],
    wrist_positions: Dict[str, Optional[Tuple[int, int]]],
    prev_lid_bbox: Optional[List[int]] = None,
    curr_lid_bbox: Optional[List[int]] = None,
    detection_confidences: Dict[str, float] = None
) -> Tuple[bool, float]:
    """
    Detect if person is interacting with box (opening/closing).
    Simple logic: person detected + box detected + hands near box = interaction
    """
    if detection_confidences is None:
        detection_confidences = {"person": 0.5, "box": 0.5}
    
    # Check if hands near box
    left = wrist_positions.get("left_wrist")
    right = wrist_positions.get("right_wrist")
    
    hand_near_box = False
    if box_bbox:
        if left and point_in_bbox(left, box_bbox, margin=0.3):
            hand_near_box = True
        if right and point_in_bbox(right, box_bbox, margin=0.3):
            hand_near_box = True
    
    # Detect lid movement across frames
    lid_moving = False
    if prev_lid_bbox and curr_lid_bbox:
        # Detect upward movement by comparing y1 (top) position
        # In image coordinates, smaller y means higher position
        prev_y1 = prev_lid_bbox[1]
        curr_y1 = curr_lid_bbox[1]
        
        # Upward movement: current y1 < previous y1 (with 2px tolerance)
        if curr_y1 < prev_y1 - 2:
            lid_moving = True
        
        # Detect area increase (lid opening makes it more visible)
        prev_area = bbox_area(prev_lid_bbox)
        curr_area = bbox_area(curr_lid_bbox)
        
        # Area increased by more than 10%
        if prev_area > 0 and curr_area > prev_area * 1.1:
            lid_moving = True
    
    # Optional: hand movement toward box check
    # This would require tracking previous wrist positions
    hand_moving_toward_box = False
    
    # Simple: person + box + hands near box = interaction
    is_opening = person_bbox is not None and box_bbox is not None and hand_near_box
    
    # Calculate confidence
    conf_person = detection_confidences.get("person", 0.5)
    conf_box = detection_confidences.get("box", 0.5)
    
    score = 0.0
    if person_bbox and box_bbox:
        score += 0.3 * conf_person
        score += 0.3 * conf_box
    if hand_near_box:
        score += 0.4
    
    score = min(1.0, score)
    return bool(is_opening), float(score)
