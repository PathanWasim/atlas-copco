#!/usr/bin/env python3
"""
Script to extract frames from video files for annotation and training.
Supports frame skipping and quality control.
"""
import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_skip: int = 1,
    max_frames: int = None,
    quality: int = 95
):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_skip: Extract every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to extract (None = all)
        quality: JPEG quality (0-100)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frame skip: {frame_skip}")
    
    # Calculate expected output frames
    expected_frames = total_frames // frame_skip
    if max_frames:
        expected_frames = min(expected_frames, max_frames)
    
    print(f"  Expected output: {expected_frames} frames")
    print(f"\nExtracting frames to: {output_dir}")
    
    frame_idx = 0
    saved_count = 0
    
    with tqdm(total=expected_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we should save this frame
            if frame_idx % frame_skip == 0:
                # Generate output filename
                output_path = os.path.join(
                    output_dir,
                    f"frame_{saved_count:06d}.jpg"
                )
                
                # Save frame
                cv2.imwrite(
                    output_path,
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                
                saved_count += 1
                pbar.update(1)
                
                # Check max frames limit
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_idx += 1
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  Saved {saved_count} frames")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video for annotation and training"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: all)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95)"
    )
    
    args = parser.parse_args()
    
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_skip=args.skip,
        max_frames=args.max_frames,
        quality=args.quality
    )


if __name__ == "__main__":
    main()
