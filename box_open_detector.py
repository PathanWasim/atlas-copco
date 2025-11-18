"""
box_open_detector.py

Usage:
  python box_open_detector.py --video path/to/video.mp4 \
      --out output_dir \
      [--model path/to/yolo_box_model.pt] \
      [--frame-skip 2] [--min-open-frames 3]

Notes:
- If you provide --model that recognizes 'box' class (or classes named 'box', 'box_open', 'box_closed'),
  the script will use YOLO for box detection. Otherwise it uses an OpenCV contour rectangle heuristic.
- For human-hand detection it prefers mediapipe (wrists). If mediapipe isn't installed, it will use YOLO 'person' bbox proximity.
"""

import os
import argparse
import json
import math
from collections import deque, defaultdict
import cv2
import numpy as np
import time

# optional imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

def find_rectangles_contours(img, min_area=1500):
    """Return list of bbox (x,y,w,h,area) for rectangular-ish contours."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) >= 4:  # allow >=4, sometimes noisy
            x,y,w,h = cv2.boundingRect(approx)
            aspect = float(w)/float(max(h,1))
            if 0.25 < aspect < 4.0:
                rects.append((x,y,w,h,area))
    rects = sorted(rects, key=lambda r: -r[4])
    return rects

def rect_iou(a,b):
    ax1,ay1,aw,ah = a
    bx1,by1,bw,bh = b
    ax2,ay2 = ax1+aw, ay1+ah
    bx2,by2 = bx1+bw, by1+bh
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter/union

def point_in_rect(pt, rect, margin=0.15):
    x,y = pt
    rx,ry,rw,rh = rect
    mx = margin*rw; my = margin*rh
    return (rx-mx) <= x <= (rx+rw+mx) and (ry-my) <= y <= (ry+rh+my)

class WristEstimator:
    def __init__(self):
        self.available = MP_AVAILABLE
        if not self.available:
            print("[WristEstimator] Mediapipe not available, falling back to bbox proximity.")
            return
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    def estimate(self, frame, person_bbox=None):
        """
        Returns dict {'left':(x,y),'right':(x,y)} in pixel coords if found else None entries.
        If person_bbox provided, process cropped region for speed.
        """
        if not self.available:
            return {'left': None, 'right': None}
        h,w = frame.shape[:2]
        if person_bbox:
            x,y,ww,hh = person_bbox
            crop = frame[y:y+hh, x:x+ww]
            res = self.pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                return {'left':None,'right':None}
            lm = res.pose_landmarks.landmark
            lw = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            rw = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left = (int(x + lw.x * ww), int(y + lw.y * hh)) if lw.visibility>0.2 else None
            right = (int(x + rw.x * ww), int(y + rw.y * hh)) if rw.visibility>0.2 else None
            return {'left': left, 'right': right}
        else:
            res = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                return {'left':None,'right':None}
            lm = res.pose_landmarks.landmark
            lw = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            rw = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left = (int(lw.x * w), int(lw.y * h)) if lw.visibility>0.2 else None
            right = (int(rw.x * w), int(rw.y * h)) if rw.visibility>0.2 else None
            return {'left': left, 'right': right}

def bbox_center(b):
    x,y,w,h = b
    return (int(x+w/2), int(y+h/2))

def run_detector(args):
    video_path = args.video
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis_frames"); os.makedirs(vis_dir, exist_ok=True)

    # model setup
    yolo = None
    use_yolo = False
    yolo_has_box_class = False
    if args.model and YOLO_AVAILABLE:
        print("[INFO] Loading YOLO model:", args.model)
        yolo = YOLO(args.model)
        use_yolo = True
        try:
            names = yolo.model.names
            print("[INFO] YOLO classes:", names)
            for idx,name in names.items():
                if 'box' in name.lower():
                    yolo_has_box_class = True
                    print(f"[INFO] YOLO has box-like class '{name}' (index {idx})")
                    break
        except Exception:
            print("[WARN] Could not read YOLO class names; proceeding anyway.")
    else:
        if args.model and not YOLO_AVAILABLE:
            print("[WARN] --model provided but ultralytics.YOLO not available; skipping YOLO.")
        else:
            print("[INFO] No YOLO model provided - using contour heuristic for boxes.")

    # wrist estimator
    wrist_est = WristEstimator()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Video FPS={fps}, frames={frame_count}")

    results = []
    lid_prev_bboxes = {}
    open_flags = deque(maxlen=args.min_open_frames)
    frame_id = 0
    last_time = time.time()

    tracked_boxes = {}
    next_box_id = 1

    hand_margin = args.hand_margin
    lid_movement_pixels = args.lid_movement_pixels
    min_box_area = args.min_box_area

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % args.frame_skip != 0:
            continue

        h,w = frame.shape[:2]
        detections = {'person': [], 'box': [], 'lid': []}

        # YOLO detection
        if use_yolo:
            try:
                preds = yolo(frame, imgsz=args.imgsz, conf=args.conf_threshold, verbose=False)[0]
                boxes = preds.boxes
                for b in boxes:
                    cls = int(b.cls.cpu().numpy()[0])
                    conf = float(b.conf.cpu().numpy()[0])
                    xyxy = b.xyxy.cpu().numpy()[0]
                    x1,y1,x2,y2 = map(int, xyxy)
                    bbox = (x1, y1, x2-x1, y2-y1)
                    name = yolo.model.names.get(cls, str(cls))
                    lname = name.lower()
                    if 'person' in lname:
                        detections['person'].append({'bbox':bbox,'conf':conf})
                    elif 'box' in lname:
                        detections['box'].append({'bbox':bbox,'conf':conf})
                    elif 'lid' in lname:
                        detections['lid'].append({'bbox':bbox,'conf':conf})
            except Exception as e:
                print(f"[WARN] YOLO error frame {frame_id}:", e)

        # fallback contour detection
        if (not use_yolo) or (use_yolo and not yolo_has_box_class):
            rects = find_rectangles_contours(frame, min_area=min_box_area)
            for x,y,ww,hh,area in rects[:6]:
                detections['box'].append({'bbox':(x,y,ww,hh),'conf':0.4})

        # NMS for boxes
        final_boxes = []
        used = [False]*len(detections['box'])
        for i,b in enumerate(detections['box']):
            if used[i]: continue
            xi,yi,wi,hi = b['bbox']
            best = b
            for j in range(i+1, len(detections['box'])):
                if used[j]: continue
                xj,yj,wj,hj = detections['box'][j]['bbox']
                iou = rect_iou((xi,yi,wi,hi),(xj,yj,wj,hj))
                if iou>0.4:
                    if wj*hj > wi*hi:
                        best = detections['box'][j]
                        xi,yi,wi,hi = best['bbox']
                    used[j] = True
            final_boxes.append(best)
        detections['box'] = final_boxes

        # lid detection heuristic
        lids = []
        for b in detections['box']:
            bx,by,bw,bh = b['bbox']
            y0 = max(0, by - int(bh*0.6))
            band = frame[y0:by+int(bh*0.2), bx:bx+bw]
            if band.size==0: continue
            band_rects = find_rectangles_contours(band, min_area=200)
            for rx,ry,rw,rh,area in band_rects[:2]:
                lids.append({'bbox':(bx+rx, y0+ry, rw, rh), 'area':area})
        detections['lid'] = lids

        # tracking
        new_tracked = {}
        for box in detections['box']:
            bb = box['bbox']
            assigned = None
            best_iou = 0.0
            for tid, tb in tracked_boxes.items():
                iou = rect_iou(bb, tb)
                if iou>best_iou and iou>0.25:
                    best_iou = iou; assigned = tid
            if assigned:
                new_tracked[assigned] = bb
            else:
                new_tracked[next_box_id] = bb; next_box_id += 1
        tracked_boxes = new_tracked

        # hand estimation
        person_bbox = detections['person'][0]['bbox'] if detections['person'] else None
        wrist = wrist_est.estimate(frame, person_bbox=person_bbox) if wrist_est.available else {'left':None,'right':None}

        # interaction detection
        frame_events = []
        any_open_now = False
        for tid, bb in tracked_boxes.items():
            bx,by,bw,bh = bb
            box_center = bbox_center(bb)
            
            matched_lid = None
            best_l_iou = 0.0
            for lid in detections['lid']:
                iou = rect_iou(bb, (lid['bbox'][0],lid['bbox'][1],lid['bbox'][2],lid['bbox'][3]))
                if iou > best_l_iou:
                    best_l_iou = iou; matched_lid = lid
            
            lid_moving = False
            if matched_lid:
                prev = lid_prev_bboxes.get(tid)
                curr = matched_lid['bbox']
                if prev:
                    py = prev[1]; cy = curr[1]
                    if (py - cy) > lid_movement_pixels:
                        lid_moving = True
                    parea = prev[2]*prev[3]; carea = curr[2]*curr[3]
                    if carea > parea * 1.10: lid_moving = True
                lid_prev_bboxes[tid] = curr

            # human interaction
            hand_near = False
            for side in ('left','right'):
                pt = wrist.get(side)
                if pt and point_in_rect(pt, bb, margin=hand_margin):
                    hand_near = True
            
            if not hand_near and detections['person']:
                person_bb = detections['person'][0]['bbox']
                pc = bbox_center(person_bb)
                dist = math.hypot(pc[0]-box_center[0], pc[1]-box_center[1])
                if dist < max(bw,bh)*0.9:
                    hand_near = True

            is_opening = lid_moving or (hand_near and matched_lid is not None)
            if is_opening:
                any_open_now = True
            
            frame_events.append({
                'frame_id': frame_id,
                'box_id': tid,
                'box_bbox': bb,
                'lid_bbox': matched_lid['bbox'] if matched_lid else None,
                'lid_moving': lid_moving,
                'hand_near': hand_near,
                'is_opening': is_opening
            })

        # temporal smoothing
        open_flags.append(1 if any_open_now else 0)
        smoothed_open = (sum(open_flags) >= args.min_open_frames)
        
        results.append({
            'frame': frame_id,
            'time': cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0,
            'events': frame_events,
            'smoothed_open': bool(smoothed_open)
        })

        # visualization
        vis = frame.copy()
        for ev in frame_events:
            bx,by,bw,bh = ev['box_bbox']
            color = (0,255,0) if (ev['is_opening'] or ev['hand_near']) else (255,0,0)
            cv2.rectangle(vis, (bx,by), (bx+bw, by+bh), color, 2)
            if ev['lid_bbox']:
                lx,ly,lw,lh = ev['lid_bbox']
                cv2.rectangle(vis, (lx,ly), (lx+lw, ly+lh), (0,200,200), 2)
            txt = f"ID{ev['box_id']} H:{int(ev['hand_near'])} L:{int(ev['lid_moving'])}"
            cv2.putText(vis, txt, (bx, by-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if wrist.get('left'):
            cv2.circle(vis, wrist['left'], 5, (0,255,255), -1)
        if wrist.get('right'):
            cv2.circle(vis, wrist['right'], 5, (0,255,255), -1)

        stamp = f"F:{frame_id} Open:{int(smoothed_open)}"
        cv2.putText(vis, stamp, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        outp = os.path.join(vis_dir, f"f{frame_id:06d}.jpg")
        cv2.imwrite(outp, vis)

        if frame_id % (args.frame_skip*50) == 0:
            now = time.time(); dt = now-last_time; last_time = now
            print(f"[proc] frame {frame_id} | boxes:{len(detections['box'])} lids:{len(detections['lid'])} open:{int(smoothed_open)} | {dt:.2f}s")

    cap.release()
    
    # write results
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    import csv
    with open(os.path.join(out_dir, "results.csv"), "w", newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['frame','timestamp','smoothed_open','num_boxes'])
        for r in results:
            writer.writerow([r['frame'], r['time'], int(r['smoothed_open']), len(r['events'])])
    
    print("[DONE] Output written to", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="output_box_detect")
    p.add_argument("--model", default=None)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--conf-threshold", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--hand-margin", type=float, default=0.25)
    p.add_argument("--lid-movement-pixels", type=int, default=8)
    p.add_argument("--min-open-frames", type=int, default=3)
    p.add_argument("--min-box-area", type=int, default=1500)
    args = p.parse_args()
    run_detector(args)
