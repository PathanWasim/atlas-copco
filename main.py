# main.py
import argparse
from src.pipeline import BoxOpeningPipeline

def main():
    parser = argparse.ArgumentParser(description="Box Opening Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video (mp4)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path (yolov8n.pt or trained model)")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--visualize", action="store_true", help="Save visualization frames")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (1 = all frames)")
    args = parser.parse_args()
    
    pipeline = BoxOpeningPipeline(model_path=args.model, output_dir=args.output, visualize=args.visualize, conf_threshold=args.conf_threshold)
    out_json = pipeline.process_video(args.video, frame_skip=args.frame_skip)
    print(f"Results written to {out_json}")

if __name__ == "__main__":
    main()
