#!/usr/bin/env python3
"""
Auto-annotate frames using pretrained YOLOv8 model.
Generates initial annotations that can be reviewed and corrected.
"""
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def auto_annotate_frames(frames_dir, output_dir, model_path="yolov8n.pt", conf_threshold=0.3):
    """
    Auto-annotate frames using pretrained YOLO model.
    Detects 'person' class and saves in YOLO format.
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get all image files
    frames = list(Path(frames_dir).glob("*.jpg")) + list(Path(frames_dir).glob("*.png"))
    print(f"Found {len(frames)} frames to annotate")
    
    os.makedirs(output_dir, exist_ok=True)
    
    annotated_count = 0
    
    for frame_path in tqdm(frames, desc="Auto-annotating"):
        # Run detection
        results = model(str(frame_path), verbose=False)
        
        # Get image dimensions
        img = cv2.imread(str(frame_path))
        h, w = img.shape[:2]
        
        # Create label file
        label_path = os.path.join(output_dir, frame_path.stem + ".txt")
        
        with open(label_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class and confidence
                    cls = int(box.cls.cpu().numpy())
                    conf = float(box.conf.cpu().numpy())
                    
                    # Only keep person class (0 in COCO) with high confidence
                    if cls == 0 and conf > conf_threshold:  # person class
                        # Get bbox in xyxy format
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        
                        # Convert to YOLO format (normalized xywh)
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        # Write in YOLO format: class x_center y_center width height
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        annotated_count += 1
    
    print(f"\n✓ Auto-annotated {annotated_count} persons in {len(frames)} frames")
    print(f"✓ Labels saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review and correct annotations in Roboflow/CVAT")
    print("2. Add 'box' annotations manually")
    print("3. Export and organize into train/val splits")
    print("4. Run: python train_yolo.py --data data/dataset.yaml")

def main():
    # Auto-annotate OP-1 frames
    print("=" * 60)
    print("Auto-Annotation - OP-1")
    print("=" * 60)
    auto_annotate_frames(
        frames_dir="data/frames/op-1",
        output_dir="data/labels/op-1",
        conf_threshold=0.5
    )
    
    print("\n" + "=" * 60)
    print("Auto-Annotation - OP-2")
    print("=" * 60)
    auto_annotate_frames(
        frames_dir="data/frames/op-2",
        output_dir="data/labels/op-2",
        conf_threshold=0.5
    )
    
    print("\n" + "=" * 60)
    print("Auto-annotation complete!")
    print("=" * 60)
    print("\nPerson detections have been auto-annotated.")
    print("Now you need to:")
    print("1. Upload frames + labels to Roboflow")
    print("2. Add 'box' class annotations manually")
    print("3. Review and fix person annotations")
    print("4. Export in YOLOv8 format")
    print("5. Fine-tune: python train_yolo.py --data data/dataset.yaml")

if __name__ == "__main__":
    main()
