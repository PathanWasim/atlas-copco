# src/utils.py
"""
Utility functions for visualization and video processing.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

def draw_bounding_box(frame: np.ndarray, bbox: List[int], label: Optional[str] = None, 
                     color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
    """
    Draw a bounding box on the frame.
    
    Args:
        frame: Image frame (H, W, C)
        bbox: Bounding box [x1, y1, x2, y2]
        label: Optional text label to display
        color: BGR color tuple
        thickness: Line thickness in pixels
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, label, (x1, max(10, y1-6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_bbox(frame, bbox, label=None, color=(0,255,0), thickness=2):
    """Alias for draw_bounding_box for backward compatibility."""
    draw_bounding_box(frame, bbox, label, color, thickness)

def draw_keypoint(frame: np.ndarray, point: Optional[Tuple[int, int]], 
                 label: Optional[str] = None, color: Tuple[int, int, int] = (0, 0, 255),
                 radius: int = 4) -> None:
    """
    Draw a keypoint (circle) on the frame.
    
    Args:
        frame: Image frame (H, W, C)
        point: (x, y) coordinates or None
        label: Optional text label to display
        color: BGR color tuple
        radius: Circle radius in pixels
    """
    if point is None:
        return
    x, y = point
    cv2.circle(frame, (x, y), radius, color, -1)
    if label:
        cv2.putText(frame, label, (x+6, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def draw_point(frame, point, label=None, color=(0,0,255)):
    """Alias for draw_keypoint for backward compatibility."""
    draw_keypoint(frame, point, label, color)

def bbox_area(bbox: List[int]) -> int:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)

def point_in_bbox(point: Optional[Tuple[int, int]], bbox: List[int]) -> bool:
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates or None
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        True if point is inside bbox, False otherwise
    """
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def bbox_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def create_annotated_video(frames: List[np.ndarray], output_path: str, 
                          fps: float = 25.0) -> None:
    """
    Create a video from a list of annotated frames.
    
    Args:
        frames: List of image frames (H, W, C)
        output_path: Path to save the output video
        fps: Frames per second for the output video
    """
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
