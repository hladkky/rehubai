#!/usr/bin/env python3
# Pure real-time landmark detection (no validation logic) with maximum accuracy on Apple Silicon.
# Uses Ultralytics YOLO Pose (v8/11) with optional CoreML export for maximum throughput on M1/M2/M3.
#
# Quick start:
#   pip install ultralytics opencv-python numpy torch --upgrade
#   python pose_realtime.py --show                       # webcam, default large model
#
# For even better performance on M3:
#   from ultralytics import YOLO
#   YOLO('yolo11l-pose.pt').export(format='coreml')     # produces .mlpackage
#   python pose_realtime.py --weights yolov8n-pose.mlpackage --engine coreml --imgsz 960 --show
#
# Notes:
# - Default model is "yolo11l-pose.pt" (large, higher accuracy). If FPS is too low, try yolo11m/s/n-pose.
# - Set --imgsz 960 (high accuracy) or 640 (balanced) or 1280 (very high but slower).
# - Multi-person supported via --max-det N.

import argparse, time, sys
from collections import deque
import numpy as np
import cv2

try:
    import torch
    from ultralytics import YOLO
except Exception as e:
    print("Missing deps. Install with: pip install ultralytics opencv-python numpy torch --upgrade")
    raise

# COCO keypoint order:
KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# Pairs for simple skeleton drawing
PAIRS = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
]

def best_device(engine: str):
    if engine == "coreml":
        # Ultralytics uses CoreML backend; device arg is ignored
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def draw_skeleton(img, pts, conf, thr=0.3):
    for i,(x,y) in enumerate(pts):
        if i >= len(conf) or conf[i] < thr or np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x),int(y)), 3, (255,255,255), -1)
        cv2.circle(img, (int(x),int(y)), 3, (0,0,0), 1)
    for a,b in PAIRS:
        if a < len(conf) and b < len(conf) and conf[a] >= thr and conf[b] >= thr:
            xa,ya = pts[a]; xb,yb = pts[b]
            if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
                cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), (255,255,255), 2)
                cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), (0,0,0), 1)

def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def parse_args():
    ap = argparse.ArgumentParser(description="Real-time accurate landmark detection (no validation)")
    ap.add_argument("--source", default="0", help="webcam index (e.g., 0) or path to video/image")
    ap.add_argument("--weights", default="yolo11l-pose.pt", help="YOLO pose weights (.pt or .mlpackage)")
    ap.add_argument("--engine", default="auto", choices=["auto","pytorch","coreml"], help="Backend runtime")
    ap.add_argument("--imgsz", type=int, default=960, help="Inference size (e.g., 640/960/1280)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--max-det", type=int, default=4, help="Max people to detect")
    ap.add_argument("--kp-thr", type=float, default=0.3, help="Keypoint visibility threshold for drawing")
    ap.add_argument("--show", action="store_true", help="Show annotated window")
    ap.add_argument("--save", default="", help="Save annotated video to this path (mp4)")
    ap.add_argument("--jsonl", default="", help="Optional: dump keypoints per frame to JSON Lines file")
    ap.add_argument("--fps-smooth", type=int, default=30, help="Smoothing window for FPS display")
    return ap.parse_args()

def results_to_keypoints(res):
    """
    Convert Ultralytics result to a list of (pts, confs, score) per person:
      - pts: np.ndarray [K,2] xy
      - confs: np.ndarray [K] per-kp confidence
      - score: float person score (bbox conf if present; else mean kp conf)
    """
    out = []
    kp = getattr(res, 'keypoints', None)
    boxes = getattr(res, 'boxes', None)
    if kp is None or kp.data is None or kp.data.shape[1] == 0:
        return out
    kpd = kp.data.detach().cpu().numpy()   # [N,K,3] x,y,conf
    N = kpd.shape[0]
    bconfs = None
    if boxes is not None and boxes.conf is not None:
        bconfs = boxes.conf.detach().cpu().numpy().reshape(-1)
        if bconfs.shape[0] != N:
            bconfs = None
    for i in range(N):
        xy = kpd[i,:,:2]
        kc = kpd[i,:,2]
        score = float(bconfs[i]) if bconfs is not None else float(np.nanmean(kc))
        out.append((xy, kc, score))
    # sort by person score desc
    out.sort(key=lambda t: t[2], reverse=True)
    return out

def main():
    args = parse_args()

    engine = args.engine
    if engine == "auto":
        # pick coreml automatically if weights is an mlpackage; otherwise pytorch
        engine = "coreml" if args.weights.endswith(".mlpackage") else "pytorch"
    device = best_device(engine)

    model = YOLO(args.weights)

    # Open source
    src = 0 if (args.source.isdigit() and len(args.source) == 1) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source: {args.source}")
        return 1

    writer = None
    fps_buf = deque(maxlen=args.fps_smooth)
    jsonl_f = open(args.jsonl, "w") if args.jsonl else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        # Inference (single stage, high-accuracy settings)
        res = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            device=device
        )
        # res is a list with one Result
        if not res or len(res) == 0:
            annotated = frame
            people = []
        else:
            people = results_to_keypoints(res[0])
            annotated = frame.copy()
            for i, (xy, kc, _) in enumerate(people):
                draw_skeleton(annotated, xy, kc, thr=args.kp_thr)
                # draw id
                if xy.shape[0] > 0:
                    x0,y0 = xy[0]
                    if not (np.isnan(x0) or np.isnan(y0)):
                        cv2.rectangle(annotated, (int(x0)-8,int(y0)-24), (int(x0)+30,int(y0)-4), (40,40,40), -1)
                        put_text(annotated, f"#{i}", (int(x0)-6, int(y0)-8), scale=0.5, thickness=1)

        # FPS overlay
        t1 = time.time()
        fps = 1.0 / max(1e-6, t1 - t0)
        fps_buf.append(fps)
        put_text(annotated, f"FPS: {np.mean(fps_buf):.1f} ({device}/{engine})  img={args.imgsz}", (12, annotated.shape[0]-16))

        # Initialize writer when size known
        if writer is None and args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, 30.0, (annotated.shape[1], annotated.shape[0]))
        if writer is not None:
            writer.write(annotated)

        # Optional JSONL dump (per frame)
        if jsonl_f is not None:
            # Structure: {people: [{kps:[[x,y,conf],...], score:...}, ...]}
            import json
            frame_dump = {
                "people": [
                    {"kps": [[float(x), float(y), float(c)] for (x,y), c in zip(xy, kc)], "score": float(sc)}
                    for (xy, kc, sc) in people
                ]
            }
            jsonl_f.write(json.dumps(frame_dump) + "\n")

        if args.show:
            cv2.imshow("Pose Realtime (landmarks only)", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    if writer is not None:
        writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    if jsonl_f is not None:
        jsonl_f.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
