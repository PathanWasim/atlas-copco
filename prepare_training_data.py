#!/usr/bin/env python3
"""
Prepare training data from Atlas Copco Station-03 videos.
Extracts frames for annotation and model training.
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm

def extract_frames_for_annotation(video_path, output_dir, frame_skip=30, max_frames=300):
    """
    Extract frames from video for annotation.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_skip: Extract every Nth frame (30 = ~1 frame per second at 30fps)
        max_frames: Maximum frames to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Frame skip: {frame_skip}")
    print(f"  Expected output: ~{min(total_frames // frame_skip, max_frames)} frames")
    
    frame_idx = 0
    saved_count = 0
    
    with tqdm(total=min(total_frames // frame_skip, max_frames), desc="Extracting") as pbar:
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                pbar.update(1)
            
            frame_idx += 1
    
    cap.release()
    print(f"✓ Extracted {saved_count} frames to {output_dir}")
    return saved_count

def main():
    print("=" * 60)
    print("Training Data Preparation - Atlas Copco Station-03")
    print("=" * 60)
    
    # Video paths
    videos = {
        "OP-1": "OneDrive_1_11-11-2025/Station-03 Videos/OP-1-MA-Rushikesh/IMG_1436.MOV",
        "OP-2": "OneDrive_1_11-11-2025/Station-03 Videos/OP-2-MA-Shubham/VID20250801085523[1].mp4"
    }
    
    # Configuration
    frame_skip = 30  # Extract ~1 frame per second (assuming 30fps)
    max_frames_per_video = 200  # 200 frames per video = 400 total
    
    print("\nConfiguration:")
    print(f"  Frame skip: {frame_skip} (extracts ~1 frame/second)")
    print(f"  Max frames per video: {max_frames_per_video}")
    print(f"  Total expected: ~{len(videos) * max_frames_per_video} frames")
    print()
    
    total_extracted = 0
    
    for name, video_path in videos.items():
        if not os.path.exists(video_path):
            print(f"✗ Video not found: {video_path}")
            continue
        
        output_dir = f"data/frames/{name.lower().replace(' ', '_')}"
        
        try:
            count = extract_frames_for_annotation(
                video_path=video_path,
                output_dir=output_dir,
                frame_skip=frame_skip,
                max_frames=max_frames_per_video
            )
            total_extracted += count
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")
    
    print()
    print("=" * 60)
    print("Frame Extraction Complete!")
    print("=" * 60)
    print(f"\nTotal frames extracted: {total_extracted}")
    print("\nNext steps:")
    print("1. Review extracted frames in data/frames/")
    print("2. Annotate frames using CVAT, Roboflow, or Label Studio")
    print("   - Classes: person, box, lid, tool")
    print("   - Format: YOLO (normalized coordinates)")
    print("3. Split into train/val sets (80/20)")
    print("4. Update data/dataset.yaml with correct paths")
    print("5. Train model: python train_yolo.py --data data/dataset.yaml")
    print()
    print("Annotation tips:")
    print("  - Focus on frames with clear box-opening actions")
    print("  - Annotate person bounding boxes (full body)")
    print("  - Annotate box and lid separately when visible")
    print("  - Include tool annotations if tools are used")
    print("  - Aim for 200-300 annotated frames minimum")
    print()

if __name__ == "__main__":
    main()
