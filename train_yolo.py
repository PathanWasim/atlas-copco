#!/usr/bin/env python3
"""
Script for training custom YOLOv8 model on box-opening dataset.
Trains model to detect: person, box, lid, tool classes.
"""
import argparse
from ultralytics import YOLO


def train_model(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto"
):
    """
    Fine-tune YOLOv8 model on custom dataset.
    Uses transfer learning from pretrained COCO weights.
    """
    # Load pretrained COCO model for fine-tuning
    model_name = f"yolov8{model_size}.pt"
    print(f"Fine-tuning pretrained model: {model_name}")
    print("Using transfer learning from COCO weights")
    model = YOLO(model_name)
    
    # Fine-tune the model
    print(f"Starting fine-tuning for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/detect",
        name="box_finetune",
        patience=15,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
        pretrained=True,  # Use pretrained weights
        freeze=10  # Freeze first 10 layers for fine-tuning
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: runs/detect/box_opening_train/weights/best.pt")
    print(f"Last model saved to: runs/detect/box_opening_train/weights/last.pt")
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for box-opening detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file (e.g., data/dataset.yaml)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="Model size: n (nano), s (small), m (medium), l (large), x (xlarge)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (reduce if out of memory)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training: auto, cpu, or cuda"
    )
    
    args = parser.parse_args()
    
    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == "__main__":
    main()
