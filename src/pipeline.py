# src/pipeline.py
"""
Main pipeline orchestrator for box-opening detection.
Coordinates YOLO detection, pose estimation, and opening logic.
"""
import os
import cv2
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from .detect_yolo import YOLODetector
from .hand_pose import HandPoseEstimator
from .opening_logic import is_opening_box
from .utils import draw_bbox, draw_point

class BoxOpeningPipeline:
    """
    Main pipeline for processing videos and detecting box-opening events.
    
    Attributes:
        detector: YOLODetector instance for object detection
        pose: HandPoseEstimator instance for wrist extraction
        output_dir: Directory for saving results
        visualize: Whether to save annotated frames
        prev_lid_bbox: Previous frame's lid bounding box for temporal tracking
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", output_dir: str = "output", 
                 visualize: bool = True, conf_threshold: float = 0.3):
        """
        Initialize the pipeline with all component modules.
        
        Args:
            model_path: Path to YOLOv8 model file
            output_dir: Directory to save output files
            visualize: Whether to save annotated frames
            conf_threshold: Confidence threshold for YOLO detections
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "vis_frames"), exist_ok=True)
        
        self.detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
        self.pose = HandPoseEstimator()
        self.output_dir = output_dir
        self.visualize = visualize
        self.prev_lid_bbox = None
    
    def process_video(self, video_path: str, frame_skip: int = 1) -> str:
        """
        Process a video file and detect box-opening events.
        
        Args:
            video_path: Path to input video file
            frame_skip: Process every Nth frame (1 = all frames)
        
        Returns:
            Path to the output JSON file
        
        Raises:
            RuntimeError: If video cannot be opened
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_idx = 0
        results = []
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Frame skip: {frame_skip}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            frame_res = self.process_frame(frame, frame_idx)
            results.append(frame_res)
            
            # Visualization
            if self.visualize:
                vis = frame.copy()
                if frame_res.get("person_bbox"):
                    draw_bbox(vis, frame_res["person_bbox"], label="person", color=(0, 255, 0))
                if frame_res.get("box_bbox"):
                    draw_bbox(vis, frame_res["box_bbox"], label="box", color=(255, 0, 0))
                if frame_res.get("left_wrist"):
                    draw_point(vis, frame_res["left_wrist"], label="L", color=(0, 0, 255))
                if frame_res.get("right_wrist"):
                    draw_point(vis, frame_res["right_wrist"], label="R", color=(255, 0, 255))
                
                # Add opening status text
                if frame_res.get("box_opening"):
                    cv2.putText(vis, "BOX OPENING DETECTED", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                out_path = os.path.join(self.output_dir, "vis_frames", f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, vis)
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")
            
            frame_idx += 1
        
        cap.release()
        print(f"Processed {frame_idx} total frames")
        
        # Write outputs
        self._write_results(results)
        
        json_path = os.path.join(self.output_dir, "results.json")
        return json_path
    
    def process_frame(self, frame, frame_id: int) -> Dict[str, Any]:
        """
        Process a single frame through the detection pipeline.
        
        Args:
            frame: Image frame (H, W, C)
            frame_id: Sequential frame number
        
        Returns:
            Dictionary containing detection results and classification
        """
        detections = self.detector.detect(frame)
        
        # Choose the top person and top box (simple heuristic)
        person_bbox = detections.get("person", [{}])[0].get("bbox") if detections.get("person") else None
        box_bbox = detections.get("box", [{}])[0].get("bbox") if detections.get("box") else None
        
        # Extract confidences
        confs = {
            "person": detections.get("person", [{}])[0].get("confidence", 0.0) if detections.get("person") else 0.0,
            "box": detections.get("box", [{}])[0].get("confidence", 0.0) if detections.get("box") else 0.0,
            "lid": detections.get("lid", [{}])[0].get("confidence", 0.0) if detections.get("lid") else 0.0
        }
        
        # Extract wrist positions
        wrists = {"left_wrist": None, "right_wrist": None}
        if person_bbox:
            wrists = self.pose.extract_wrists(frame, person_bbox)
        
        # Get current lid bbox if present
        curr_lid_bbox = detections.get("lid", [{}])[0].get("bbox") if detections.get("lid") else None
        
        # Determine if box is being opened
        is_open, score = is_opening_box(
            person_bbox, box_bbox, wrists, 
            self.prev_lid_bbox, curr_lid_bbox, 
            detection_confidences=confs
        )
        
        # Update temporal state
        self.prev_lid_bbox = curr_lid_bbox
        
        result = {
            "frame_id": frame_id,
            "box_opening": bool(is_open),
            "confidence_score": float(score),
            "person_bbox": person_bbox,
            "box_bbox": box_bbox,
            "left_wrist": wrists.get("left_wrist"),
            "right_wrist": wrists.get("right_wrist"),
            "detections": detections
        }
        return result
    
    def _write_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Write results to JSON and CSV files.
        
        Args:
            results: List of frame result dictionaries
        """
        # Write JSON
        json_path = os.path.join(self.output_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON results to {json_path}")
        
        # Write CSV
        csv_data = []
        for res in results:
            csv_data.append({
                "frame_id": res["frame_id"],
                "box_opening": res["box_opening"],
                "confidence": res["confidence_score"]
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV results to {csv_path}")
        
        # Print summary
        opening_count = sum(1 for r in results if r["box_opening"])
        print(f"\nSummary:")
        print(f"  Total frames: {len(results)}")
        print(f"  Box-opening events: {opening_count}")
        print(f"  Percentage: {100 * opening_count / len(results):.1f}%")
