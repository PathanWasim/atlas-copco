# Box Detection System

Detects boxes and person interactions in industrial videos using YOLOv8 + Mediapipe.

## What It Does

- Detects **persons** and **boxes** in video
- Tracks when hands are near boxes
- Identifies box interaction events

## Setup (Already Done!)

Dependencies installed with GPU support for RTX 4050.

## Run Detection

```bash
python main.py --video your_video.mp4 --model yolov8n.pt --visualize
```

## Fine-Tune Model

**Person annotations already done!** (382 persons auto-annotated)

**1. Add box annotations** (Roboflow):
- Upload `data/frames/` + `data/labels/` to Roboflow
- Person boxes already there - just add **box** class
- Export in YOLOv8 format to `data/`

**2. Fine-tune** (30-45 min on RTX 4050):
```bash
python train_yolo.py --data data/dataset.yaml --model n --epochs 50 --batch 32
```

**3. Test**:
```bash
python main.py --video your_video.mp4 --model runs/detect/box_finetune/weights/best.pt --visualize
```

## Project Structure

```
.
├── models/              # Trained YOLO models
├── data/
│   ├── frames/          # Extracted video frames
│   └── annotations/     # YOLO label files
├── output/
│   ├── vis_frames/      # Annotated frames
│   └── results.json     # Detection results
├── src/
│   ├── detect_yolo.py   # YOLOv8 detector
│   ├── hand_pose.py     # Mediapipe pose estimator
│   ├── opening_logic.py # Opening detection logic
│   ├── pipeline.py      # Main orchestrator
│   └── utils.py         # Visualization helpers
├── main.py              # CLI entry point
└── requirements.txt
```

## Output Format

### JSON Output
```json
[
  {
    "frame_id": 123,
    "box_opening": true,
    "confidence_score": 0.87,
    "person_bbox": [120, 50, 380, 450],
    "box_bbox": [200, 300, 350, 420],
    "left_wrist": [180, 280],
    "right_wrist": [220, 290]
  }
]
```

### CSV Output
```csv
frame_id,box_opening,confidence
0,False,0.0
1,False,0.0
2,True,0.87
```

## Training Custom YOLOv8 Model

### 1. Prepare Dataset

Annotate frames with classes: `person`, `box`, `lid`, `tool`

Use CVAT, Roboflow, or Label Studio in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

### 2. Create dataset.yaml

```yaml
path: ./data
train: images/train
val: images/val

nc: 4
names: ['person', 'box', 'lid', 'tool']
```

### 3. Train Model

```bash
yolo train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

### 4. Use Trained Model

```bash
python main.py --video test.mp4 --model runs/detect/train/weights/best.pt --visualize
```

## Detection Logic

The system classifies a frame as containing a box-opening event when:

1. **Hand-in-Box**: Either wrist is within the box bounding box (with 15% margin)
2. **Lid Movement**: Lid moves upward or increases in area by >10%

**Formula**: `box_opening = hand_in_box AND (lid_moving OR hand_moving_toward_box)`

Confidence score is calculated from:
- Detection confidences (person, box, lid)
- Strength of hand-box interaction
- Magnitude of lid movement

## Hardware Requirements

- **Minimum**: CPU with 4GB RAM
- **Recommended**: NVIDIA GPU with 4GB VRAM, 8GB RAM
- **Storage**: 500MB for models and dependencies

## Performance

- **YOLOv8n on GPU**: ~100-150 FPS (1080p)
- **YOLOv8n on CPU**: ~10-20 FPS (1080p)
- **Overall Pipeline**: ~15-30 FPS on GPU, ~5-10 FPS on CPU

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_opening_logic.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

The test suite includes:
- **Unit tests** for YOLO detector, pose estimator, and opening logic
- **Integration tests** for end-to-end pipeline
- **Output format validation** for JSON and CSV

View coverage report:
```bash
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

## Troubleshooting

### CUDA not available
The system automatically falls back to CPU. To use GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model not found
Download pre-trained YOLOv8:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Low detection accuracy
- Increase `--conf-threshold` to reduce false positives
- Train custom model on your specific environment
- Annotate 200-500 frames for better results

### Import errors
Ensure virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

## Development

### Project Structure

```
.
├── src/                 # Source code
│   ├── detect_yolo.py   # YOLO detection
│   ├── hand_pose.py     # Pose estimation
│   ├── opening_logic.py # Detection logic
│   ├── pipeline.py      # Main pipeline
│   └── utils.py         # Utilities
├── tests/               # Test suite
├── scripts/             # Utility scripts
├── data/                # Dataset files
├── models/              # Trained models
├── main.py              # CLI entry point
├── train_yolo.py        # Training script
└── demo.ipynb           # Jupyter demo
```

### Adding New Features

1. Implement feature in appropriate module
2. Add unit tests in `tests/`
3. Update documentation
4. Run full test suite

## Next Steps

1. Annotate 150-300 frames from your industrial footage
2. Train custom YOLOv8 model with your annotations
3. Evaluate on held-out test set
4. Tune confidence thresholds and heuristic parameters
5. Add temporal smoothing for more stable detections

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT
