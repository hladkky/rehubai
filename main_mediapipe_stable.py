#!/usr/bin/env python3
# mediapipe_min_skeleton_plus.py
# Minimal MediaPipe Pose drawing with full arm/leg lines:
# - 1 head point (nose)
# - Shoulders, elbows, hips, knees
# - 1 point for each HAND (left/right)  -> wrist
# - 1 point for each FOOT (left/right)  -> ankle (or foot tip with --feet-tip)
#
# Usage:
#   pip install mediapipe==0.10.14 opencv-python numpy
#   # download a model (e.g., pose_landmarker_full.task) and point --model to it
#   python mediapipe_min_skeleton_plus.py --source 0 --model pose_landmarker_full.task --show
#   python mediapipe_min_skeleton_plus.py --source path/to/video.mp4 --model pose_landmarker_heavy.task --show --save out.mp4
#
# Notes:
# - Use --feet-tip if you prefer foot tip points (foot_index) instead of ankles.
# - Use --sample-every 2 to process every other frame for higher FPS.

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---- BlazePose landmark indices ----
HEAD = 0  # nose

# shoulders / elbows
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14

# wrists (used as single "hand" endpoints)
L_WR, R_WR = 15, 16

# hips / knees
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

# ankles (stable) and optional foot tips
L_ANK, R_ANK = 27, 28
L_FTIP, R_FTIP = 31, 32


def parse_args():
    ap = argparse.ArgumentParser(description="MediaPipe minimal skeleton with single hand/foot endpoints")
    ap.add_argument("--source", default="0", help="webcam index (e.g. 0) or path to video file")
    ap.add_argument("--model", default="configs/mediapipe/pose_landmarker_full.task", help="path to MediaPipe Pose .task (lite/full/heavy)")
    ap.add_argument("--kp-thr", type=float, default=0.25, help="visibility/presence threshold for drawing")
    ap.add_argument("--feet-tip", action="store_true",
                    help="use foot tip (foot_index) instead of ankle for foot points")
    ap.add_argument("--sample-every", type=int, default=1, help="process every Nth frame (>=2 boosts FPS)")
    ap.add_argument("--show", action="store_true", help="show window")
    ap.add_argument("--save", default="", help="save annotated video to this path (mp4)")
    ap.add_argument("--fps-smooth", type=int, default=30, help="FPS smoothing window")
    return ap.parse_args()


def draw_point(img, x, y, r=4):
    cv2.circle(img, (int(x), int(y)), r + 1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), r, (255, 255, 255), -1, cv2.LINE_AA)


def draw_line(img, a, b, thick=2):
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    cv2.line(img, (ax, ay), (bx, by), (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.line(img, (ax, ay), (bx, by), (255, 255, 255), thick, cv2.LINE_AA)


def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def get_xyc(lm, W, H):
    x = float(lm.x * W)
    y = float(lm.y * H)
    c = float(getattr(lm, "presence", getattr(lm, "visibility", 1.0)))
    return x, y, c


def main():
    args = parse_args()

    # Open source
    src = 0 if (args.source.isdigit() and len(args.source) == 1) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source: {args.source}")
        return 1

    # Landmarker
    base = python.BaseOptions(model_asset_path=args.model)
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    # Point sets
    FOOT_L, FOOT_R = (L_FTIP, R_FTIP) if args.feet_tip else (L_ANK, R_ANK)

    # We will draw these points
    MAJOR_POINTS = [
        HEAD,
        L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
        L_HIP, R_HIP, L_KNEE, R_KNEE, FOOT_L, FOOT_R
    ]

    # Bones to draw for full arm/leg lines (no fingers; simple chains)
    BONES = [
        (L_SH, R_SH),                       # shoulders
        (L_HIP, R_HIP),                     # hips
        (L_SH, L_EL), (L_EL, L_WR),         # left arm to hand point
        (R_SH, R_EL), (R_EL, R_WR),         # right arm to hand point
        (L_HIP, L_KNEE), (L_KNEE, FOOT_L),  # left leg to foot point
        (R_HIP, R_KNEE), (R_KNEE, FOOT_R),  # right leg to foot point
        (L_SH, L_HIP), (R_SH, R_HIP),       # torso sides
    ]

    writer = None
    fps_buf = deque(maxlen=args.fps_smooth)

    frame_idx = -1
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ts_ms = 0.0
    step_ms = 1000.0 / fps_video

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        ts_ms += step_ms

        if args.sample_every > 1 and (frame_idx % args.sample_every != 0):
            annotated = frame_bgr
        else:
            t0 = time.time()

            H, W = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, int(ts_ms))
            annotated = frame_bgr.copy()

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]  # first person
                pts = {i: get_xyc(lms[i], W, H) for i in MAJOR_POINTS}

                # Bones
                for a, b in BONES:
                    xa, ya, ca = pts[a]; xb, yb, cb = pts[b]
                    if ca >= args.kp_thr and cb >= args.kp_thr:
                        draw_line(annotated, (xa, ya), (xb, yb))

                # Head (nose only)
                xh, yh, ch = pts[HEAD]
                if ch >= args.kp_thr:
                    draw_point(annotated, xh, yh, r=6)
                    # head → shoulder midpoint (optional nice visual)
                    xs, ys, cs = pts[L_SH]
                    xd, yd, cd = pts[R_SH]
                    if cs >= args.kp_thr and cd >= args.kp_thr:
                        draw_line(annotated, (xh, yh), ((xs + xd) / 2.0, (ys + yd) / 2.0))

                # Draw remaining joints as dots (shoulders, elbows, hips, knees, hands, feet)
                for idx in MAJOR_POINTS:
                    if idx == HEAD:
                        continue
                    x, y, c = pts[idx]
                    if c >= args.kp_thr:
                        # hands/feet a bit larger
                        r = 5 if idx in (L_WR, R_WR, FOOT_L, FOOT_R) else 4
                        draw_point(annotated, x, y, r=r)

            dt = time.time() - t0
            if dt > 0:
                fps_buf.append(1.0 / dt)

        if fps_buf:
            put_text(annotated, f"FPS: {np.mean(fps_buf):.1f}", (12, annotated.shape[0] - 14))

        # Writer
        if writer is None and args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, fps_video, (annotated.shape[1], annotated.shape[0]))
        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow("MediaPipe Minimal Skeleton (hands/feet endpoints)", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    if writer is not None:
        writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
