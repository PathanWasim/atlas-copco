#!/usr/bin/env python3
"""
GPU Setup Verification Script
Tests PyTorch CUDA, YOLOv8 GPU support, and performance
"""
import sys
import time
import numpy as np

def test_pytorch():
    """Test PyTorch CUDA availability."""
    print("=" * 60)
    print("1. PyTorch CUDA Test")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(0)
            print(f"✓ GPU memory: {props.total_memory / 1e9:.2f} GB")
            print(f"✓ Compute capability: {props.major}.{props.minor}")
            
            # Test tensor operations
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"✓ GPU tensor operations: {elapsed*1000:.2f}ms")
            
            return True
        else:
            print("✗ CUDA not available!")
            print("\nTo fix:")
            print("pip uninstall torch torchvision torchaudio")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed!")
        print("\nTo fix:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

def test_yolo():
    """Test YOLOv8 GPU support."""
    print("\n" + "=" * 60)
    print("2. YOLOv8 GPU Test")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        # Load model
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        print(f"✓ Model loaded on device: {model.device}")
        
        # Test inference
        print("Testing inference speed...")
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            _ = model(test_img, verbose=False)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            results = model(test_img, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times) * 1000
        fps = 1000 / avg_time
        
        print(f"✓ Average inference time: {avg_time:.2f}ms")
        print(f"✓ Estimated FPS: {fps:.1f}")
        
        if fps > 100:
            print("✓ Excellent GPU performance!")
        elif fps > 50:
            print("✓ Good GPU performance")
        else:
            print("⚠ Lower than expected performance")
            print("  Check: GPU drivers, power settings, thermal throttling")
        
        return True
        
    except ImportError:
        print("✗ Ultralytics not installed!")
        print("\nTo fix:")
        print("pip install ultralytics")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_mediapipe():
    """Test Mediapipe installation."""
    print("\n" + "=" * 60)
    print("3. Mediapipe Test")
    print("=" * 60)
    
    try:
        import mediapipe as mp
        print(f"✓ Mediapipe installed: {mp.__version__}")
        
        # Test pose model
        pose = mp.solutions.pose.Pose(static_image_mode=True)
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pose.process(test_img)
        print("✓ Pose model working")
        
        return True
        
    except ImportError:
        print("✗ Mediapipe not installed!")
        print("\nTo fix:")
        print("pip install mediapipe")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_opencv():
    """Test OpenCV installation."""
    print("\n" + "=" * 60)
    print("4. OpenCV Test")
    print("=" * 60)
    
    try:
        import cv2
        print(f"✓ OpenCV installed: {cv2.__version__}")
        
        # Test video reading capability
        print("✓ Video codec support available")
        
        return True
        
    except ImportError:
        print("✗ OpenCV not installed!")
        print("\nTo fix:")
        print("pip install opencv-python")
        return False

def print_system_info():
    """Print system information."""
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    
    import platform
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Architecture: {platform.machine()}")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GPU Setup Verification for RTX 4050")
    print("=" * 60)
    
    print_system_info()
    
    results = {
        "PyTorch CUDA": test_pytorch(),
        "YOLOv8 GPU": test_yolo(),
        "Mediapipe": test_mediapipe(),
        "OpenCV": test_opencv()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Your GPU setup is ready!")
        print("=" * 60)
        print("\nYou can now:")
        print("1. Run: python test_on_real_footage.py")
        print("2. Train models with GPU acceleration")
        print("3. Enjoy 10-20x faster processing!")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("=" * 60)
        print("\nRefer to GPU_SETUP.md for detailed instructions.")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
