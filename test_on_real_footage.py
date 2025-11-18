#!/usr/bin/env python3
"""
Test the box-opening detection system on real Atlas Copco footage.
This script processes the uploaded Station-03 videos.
"""
import os
import sys
from pathlib import Path
from src.pipeline import BoxOpeningPipeline

def main():
    print("=" * 60)
    print("Box-Opening Detection - Real Footage Test")
    print("=" * 60)
    print()
    
    # Video paths
    videos = {
        "OP-1": "OneDrive_1_11-11-2025/Station-03 Videos/OP-1-MA-Rushikesh/IMG_1436.MOV",
        "OP-2": "OneDrive_1_11-11-2025/Station-03 Videos/OP-2-MA-Shubham/VID20250801085523[1].mp4"
    }
    
    # Check which videos exist
    available_videos = {}
    for name, path in videos.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            available_videos[name] = path
            print(f"✓ Found {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ Not found: {path}")
    
    if not available_videos:
        print("\nNo videos found! Please check the paths.")
        return
    
    print()
    print("Select video to process:")
    for i, (name, path) in enumerate(available_videos.items(), 1):
        print(f"  {i}. {name}")
    print(f"  {len(available_videos) + 1}. Process all videos")
    print()
    
    try:
        choice = input("Enter choice (or press Enter for all): ").strip()
        
        if not choice or choice == str(len(available_videos) + 1):
            selected_videos = available_videos
        else:
            idx = int(choice) - 1
            selected_name = list(available_videos.keys())[idx]
            selected_videos = {selected_name: available_videos[selected_name]}
    except (ValueError, IndexError):
        print("Invalid choice, processing all videos...")
        selected_videos = available_videos
    
    print()
    print("Configuration:")
    print("  Model: yolov8n.pt (pretrained COCO)")
    print("  Frame skip: 5 (process every 5th frame for speed)")
    print("  Confidence threshold: 0.3")
    print("  Visualization: Enabled")
    print()
    
    # Process each selected video
    for name, video_path in selected_videos.items():
        print("=" * 60)
        print(f"Processing {name}: {video_path}")
        print("=" * 60)
        
        # Create output directory for this video
        output_dir = f"output/{name.lower().replace(' ', '_')}"
        
        try:
            # Initialize pipeline
            pipeline = BoxOpeningPipeline(
                model_path="yolov8n.pt",
                output_dir=output_dir,
                visualize=True,
                conf_threshold=0.3
            )
            
            # Process video (skip every 5 frames for faster processing)
            results_path = pipeline.process_video(video_path, frame_skip=5)
            
            print()
            print(f"✓ Processing complete for {name}!")
            print(f"  Results: {results_path}")
            print(f"  CSV: {output_dir}/results.csv")
            print(f"  Visualizations: {output_dir}/vis_frames/")
            print()
            
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("All processing complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Review results in output/ directories")
    print("2. Check CSV files for box-opening events")
    print("3. View annotated frames in vis_frames/ folders")
    print("4. Annotate frames for custom model training")
    print()

if __name__ == "__main__":
    main()
