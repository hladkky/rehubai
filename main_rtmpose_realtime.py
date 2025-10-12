#!/usr/bin/env python3
# rtmpose_realtime.py
# From-scratch, minimal RTMPose (MMPose) demo that detects the MAIN COCO skeleton (17 keypoints).
# - Works with webcam or a video file.
# - Uses RTMDet for person detection (top-down pipeline).
# - Robust to minor output structure differences across MMPose versions.

import argparse
import sys
import time
from collections import deque

import numpy as np
import cv2

# MMPose 1.x
from mmpose.apis import MMPoseInferencer


COCO_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee",
    "right_knee","left_ankle","right_ankle"
]

# Edges to draw (COCO-17 main skeleton)
PAIRS = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)
]


def parse_args():
    p = argparse.ArgumentParser(description="RTMPose realtime main-skeleton (COCO-17)")
    p.add_argument("--source", default="0", help="Webcam index like '0' or a path/URL to video/image")
    p.add_argument("--pose2d", default="rtmpose-m_8xb256-420e_coco-256x192",
                   help="RTMPose model alias/config (s/m/l, coco-256x192 variants are fast & accurate)")
    p.add_argument("--det", default="rtmdet_tiny_8xb32-300e_coco",
                   help="Detector model alias/config (rtmdet_tiny is a good default)")
    p.add_argument("--max-people", type=int, default=1, help="Max persons to visualize per frame")
    p.add_argument("--kp-thr", type=float, default=0.3, help="Keypoint confidence threshold for drawing")
    p.add_argument("--show", action="store_true", help="Show annotated window")
    p.add_argument("--save", default="", help="Optional: save annotated video (e.g., out.mp4)")
    p.add_argument("--fps-smooth", type=int, default=30, help="FPS smoothing window")
    return p.parse_args()


def draw_skeleton(img, kps_xy, kps_conf, thr=0.3):
    # joints
    for i, (x, y) in enumerate(kps_xy):
        if i >= len(kps_conf): continue
        if kps_conf[i] < thr or np.isnan(x) or np.isnan(y): continue
        cv2.circle(img, (int(x), int(y)), 3, (255,255,255), -1)
        cv2.circle(img, (int(x), int(y)), 3, (0,0,0), 1)
    # bones
    for a, b in PAIRS:
        if a >= len(kps_conf) or b >= len(kps_conf): continue
        if kps_conf[a] < thr or kps_conf[b] < thr: continue
        xa, ya = kps_xy[a]; xb, yb = kps_xy[b]
        if any(map(np.isnan, [xa, ya, xb, yb])): continue
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255,255,255), 2)
        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0,0,0), 1)


def extract_persons_keypoints(mm_result, max_people=1):
    """
    Return a list of up to N persons as (keypoints[K,2], scores[K]).
    Handles both common MMPoseInferencer output layouts.
    """
    persons = []
    preds = mm_result.get("predictions", [])
    if not preds:
        return persons

    frame_pred = preds[0]  # one frame per iteration

    candidates = []

    # Case A: list of per-instance dicts
    if isinstance(frame_pred, list):
        for inst in frame_pred:
            kps = np.array(inst.get("keypoints", []), dtype=float)  # [K,2] or [1,K,2]
            scs = np.array(inst.get("keypoint_scores", []), dtype=float)  # [K] or [1,K]
            if kps.ndim == 3: kps = kps[0]
            if scs.ndim == 2: scs = scs[0]
            if kps.size == 0: continue
            avg = float(np.nanmean(scs)) if scs.size else 0.0
            candidates.append((avg, kps, scs))

    # Case B: dict with stacked arrays
    elif isinstance(frame_pred, dict):
        kps_stack = frame_pred.get("keypoints", None)
        scs_stack = frame_pred.get("keypoint_scores", None)
        if kps_stack is not None:
            kps_stack = np.array(kps_stack, dtype=float)  # [N,K,2]
            if scs_stack is not None:
                scs_stack = np.array(scs_stack, dtype=float)  # [N,K]
            else:
                scs_stack = np.zeros((kps_stack.shape[0], kps_stack.shape[1]), dtype=float)
            for i in range(kps_stack.shape[0]):
                kps = kps_stack[i]
                scs = scs_stack[i] if scs_stack.ndim == 2 else np.zeros(kps.shape[0], dtype=float)
                avg = float(np.nanmean(scs)) if scs.size else 0.0
                candidates.append((avg, kps, scs))
        else:
            # Some variants store under "instances"
            insts = frame_pred.get("instances", [])
            for inst in insts:
                kps = np.array(inst.get("keypoints", []), dtype=float)
                scs = np.array(inst.get("keypoint_scores", []), dtype=float)
                if kps.ndim == 3: kps = kps[0]
                if scs.ndim == 2: scs = scs[0]
                if kps.size == 0: continue
                avg = float(np.nanmean(scs)) if scs.size else 0.0
                candidates.append((avg, kps, scs))

    # sort by avg kp score (desc) and clip
    candidates.sort(key=lambda t: t[0], reverse=True)
    for _, kps, scs in candidates[:max_people]:
        persons.append((kps, scs))
    return persons


def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)


def main():
    args = parse_args()

    # Device: 'cuda' if available else 'cpu'
    # (You can pass 'mps' manually on macOS with PyTorch MPS if desired.)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    inferencer = MMPoseInferencer(
        pose2d=args.pose2d,
        det_model=args.det,
        device=device,
    )

    source = args.source
    if source.isdigit():
        source = str(int(source))  # MMPose webcam indices are strings

    stream = inferencer(
        source,
        show=False,          # we draw manually (for custom overlay)
        draw_bbox=True,
        return_vis=True,     # get annotated frame from MMPose for free drawing background
        pred_out_dir=None,   # no JSON dumps
    )

    writer = None
    fps_buf = deque(maxlen=args.fps_smooth)

    for result in stream:
        t0 = time.time()

        vis = result.get("visualization", [None])[0]
        if vis is None:
            continue

        persons = extract_persons_keypoints(result, max_people=args.max_people)

        annotated = vis  # already includes boxes & skeleton from MMPose; we’ll redraw our simple skeleton too
        for (kps, scs) in persons:
            draw_skeleton(annotated, kps, scs, thr=args.kp_thr)

        # FPS
        fps = 1.0 / max(1e-6, time.time() - t0)
        fps_buf.append(fps)
        put_text(annotated, f"FPS: {np.mean(fps_buf):.1f} | device={device}", (12, annotated.shape[0]-16))

        # Lazy-init writer when size known
        if writer is None and args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, 30.0, (annotated.shape[1], annotated.shape[0]))

        if args.show:
            cv2.imshow("RTMPose (COCO-17)", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        if writer is not None:
            writer.write(annotated)

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    """
    Quick start:

    # 1) Install (CUDA optional)
    #    pip install "mmpose>=1.3.0" "mmengine>=0.10.0" "mmcv>=2.0.0" "mmdet>=3.2.0" opencv-python

    # 2) Webcam demo
    #    python rtmpose_realtime.py --show

    # 3) Video file
    #    python main_rtmpose_realtime.py --source sources/videos/dead_bug.MOV --show --save outputs/videos/rtmpose_realtime.mp4

    # 4) Try other models
    #    --pose2d rtmpose-s_8xb256-420e_coco-256x192
    #    --pose2d rtmpose-l_8xb256-420e_coco-256x192
    #    --det    rtmdet_s_8xb32-300e_coco   (or rtmdet_tiny_8xb32-300e_coco)
    """
    sys.exit(main())
