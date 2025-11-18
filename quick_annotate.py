"""
Quick manual annotation tool - draw boxes with mouse
Press 'b' to mark box, 'n' for next frame, 'q' to quit
"""
import cv2
import os
import glob

drawing = False
ix, iy = -1, -1
boxes = []
current_box = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_box, boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_box = [x, y, x, y]
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box[2], current_box[3] = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box[2], current_box[3] = x, y
        boxes.append(current_box[:])
        current_box = None

def annotate_frames(frames_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(frames)} frames")
    print("\nControls:")
    print("  - Draw box with mouse")
    print("  - Press 'n' for next frame")
    print("  - Press 'u' to undo last box")
    print("  - Press 'q' to quit")
    
    cv2.namedWindow('Annotate')
    cv2.setMouseCallback('Annotate', draw_rectangle)
    
    idx = 0
    while idx < len(frames):
        global boxes
        boxes = []
        
        frame_path = frames[idx]
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]
        
        # Load existing annotations if any
        label_path = os.path.join(output_dir, os.path.basename(frame_path).replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = map(float, parts)
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        boxes.append([x1, y1, x2, y2])
        
        while True:
            display = frame.copy()
            
            # Draw saved boxes
            for box in boxes:
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw current box being drawn
            if current_box:
                cv2.rectangle(display, (current_box[0], current_box[1]), 
                            (current_box[2], current_box[3]), (0, 0, 255), 2)
            
            cv2.putText(display, f"Frame {idx+1}/{len(frames)} - {len(boxes)} boxes", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Annotate', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):  # next
                break
            elif key == ord('u'):  # undo
                if boxes:
                    boxes.pop()
            elif key == ord('q'):  # quit
                cv2.destroyAllWindows()
                return
        
        # Save annotations in YOLO format
        if boxes:
            with open(label_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"1 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")  # class 1 = box
            print(f"Saved {len(boxes)} boxes for frame {idx+1}")
        
        idx += 1
    
    cv2.destroyAllWindows()
    print(f"\nAnnotation complete! Labels saved to {output_dir}")

if __name__ == "__main__":
    annotate_frames("data/frames/box_opening_sample", "data/labels/box_opening_sample")
