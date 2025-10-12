#!/usr/bin/env python3
# Real-time posture detection with MMPose RTM models (2D keypoints on COCO skeleton).
# - Uses MMPoseInferencer with RTMPose for pose and RTMDet for person detection.
# - Works with webcam (default) or a video file.
# - Draws skeleton and prints landmarks + a couple of example posture angles.

import argparse
import sys
import time
from collections import deque
import numpy as np

# MMPose 1.x
from mmpose.apis import MMPoseInferencer

# Optional: nice display using OpenCV
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Real-time posture with MMPose RTMPose")
    p.add_argument(
        "--source",
        default="0",
        help="Input source. '0' for default webcam, or path/URL to video/image.",
    )
    p.add_argument(
        "--pose2d",
        default="rtmpose-m_8xb256-420e_coco-256x192",
        help="Pose model alias or config name (e.g., rtmpose-s/m/l; wholebody variants also work).",
    )
    p.add_argument(
        "--det",
        default="rtmdet_tiny_8xb32-300e_coco",
        help="Detector model alias or config (e.g., rtmdet-nano/tiny/s).",
    )
    p.add_argument(
        "--show", action="store_true", help="Open a window and show annotated frames."
    )
    p.add_argument(
        "--save",
        default="",
        help="Optional path to save an annotated video (e.g., out.mp4).",
    )
    p.add_argument(
        "--max-people",
        type=int,
        default=1,
        help="Track and analyze up to N people per frame (draws all).",
    )
    p.add_argument(
        "--fps-smooth",
        type=int,
        default=30,
        help="Window size for FPS smoothing.",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="How often to print keypoints/angles (every N frames).",
    )
    return p.parse_args()


def angle_3pts(a, b, c):
    """
    Returns angle ABC (in degrees) given 2D points a, b, c as (x, y).
    Robust to zero-length vectors.
    """
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return np.nan
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def coco_names():
    # COCO-17 order used by RTMPose (topdown):
    # 0-nose,1-eye_l,2-eye_r,3-ear_l,4-ear_r,5-shoulder_l,6-shoulder_r,
    # 7-elbow_l,8-elbow_r,9-wrist_l,10-wrist_r,11-hip_l,12-hip_r,13-knee_l,
    # 14-knee_r,15-ankle_l,16-ankle_r
    return [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]


def extract_persons_keypoints(mm_result, max_people=1):
    """
    Robustly extracts [(keypoints[K,2], scores[K]), ...] for up to max_people persons
    from MMPoseInferencer's per-frame result, handling both common output layouts.
    """
    persons = []
    preds = mm_result.get("predictions", [])
    if not preds:
        return persons

    frame_pred = preds[0]  # single frame per iteration

    # Case A: list of per-instance dicts
    if isinstance(frame_pred, list):
        instances = frame_pred
        candidates = []
        for inst in instances:
            kps = np.array(inst.get("keypoints", []), dtype=float)  # [K,2] or [1,K,2]
            scs = np.array(inst.get("keypoint_scores", []), dtype=float)  # [K] or [1,K]
            if kps.ndim == 3:  # [1,K,2] -> [K,2]
                kps = kps[0]
            if scs.ndim == 2:  # [1,K] -> [K]
                scs = scs[0]
            if kps.size == 0:
                continue
            avg = float(np.nanmean(scs)) if scs.size else 0.0
            candidates.append((avg, kps, scs))

    # Case B: dict with stacked arrays { "keypoints": [N,K,2], "keypoint_scores": [N,K] }
    elif isinstance(frame_pred, dict):
        kps_stack = frame_pred.get("keypoints", None)
        scs_stack = frame_pred.get("keypoint_scores", None)

        candidates = []
        if kps_stack is not None:
            kps_stack = np.array(kps_stack, dtype=float)  # [N,K,2]
            if scs_stack is not None:
                scs_stack = np.array(scs_stack, dtype=float)  # [N,K]
            else:
                scs_stack = np.zeros((kps_stack.shape[0], kps_stack.shape[1]), dtype=float)

            N = kps_stack.shape[0]
            for i in range(N):
                kps = kps_stack[i]           # [K,2]
                scs = scs_stack[i] if scs_stack.ndim == 2 else np.zeros(kps.shape[0], dtype=float)
                avg = float(np.nanmean(scs)) if scs.size else 0.0
                candidates.append((avg, kps, scs))
        else:
            # Some variants store under "instances"
            insts = frame_pred.get("instances", [])
            candidates = []
            for inst in insts:
                kps = np.array(inst.get("keypoints", []), dtype=float)
                scs = np.array(inst.get("keypoint_scores", []), dtype=float)
                if kps.ndim == 3:
                    kps = kps[0]
                if scs.ndim == 2:
                    scs = scs[0]
                if kps.size == 0:
                    continue
                avg = float(np.nanmean(scs)) if scs.size else 0.0
                candidates.append((avg, kps, scs))

    else:
        return persons  # unknown structure

    # Sort by average score (desc) and clip to max_people
    candidates.sort(key=lambda t: t[0], reverse=True)
    for _, kps, scs in candidates[:max_people]:
        persons.append((kps, scs))
    return persons



def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    args = parse_args()

    # Build inferencer. Models/weights will be auto-downloaded on first run.
    inferencer = MMPoseInferencer(
        pose2d=args.pose2d,
        det_model=args.det,
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
    )

    # Input: webcam if "0", otherwise pass through.
    source = args.source
    if source.isdigit():
        source = str(int(source))  # MMPose uses string for webcam indices ('0', '1', ...)

    # Prepare video writer if saving
    writer = None
    fps_buf = deque(maxlen=args.fps_smooth)
    frame_idx = 0
    names = coco_names()

    # Run streaming inference (generator)
    stream = inferencer(
        source,
        show=False,             # we draw/visualize manually to add FPS/angles overlays
        draw_bbox=True,         # draw person boxes
        return_vis=True,        # return annotated image
        pred_out_dir=None,      # no JSON dumps
    )

    for result in stream:
        t0 = time.time()

        # Get visualization frame (numpy BGR)
        vis = result.get("visualization", [None])[0]
        if vis is None:
            # If no frame available, try next
            continue

        # Extract keypoints for top-N persons
        persons = extract_persons_keypoints(result, max_people=args.max_people)

        # Example angles for the first person (simple posture cues)
        if persons:
            kps, scs = persons[0]
            # Safe index helper:
            def pt(idx):
                if 0 <= idx < len(kps):
                    return (float(kps[idx][0]), float(kps[idx][1]))
                return (np.nan, np.nan)

            L_SH, R_SH = pt(5), pt(6)
            L_HIP, R_HIP = pt(11), pt(12)
            NOSE = pt(0)

            # Trunk lean (angle at right hip using (shoulder-mid, hip, knee-mid) proxy)
            shoulder_mid = ((L_SH[0] + R_SH[0]) / 2.0, (L_SH[1] + R_SH[1]) / 2.0)
            hip_mid = ((L_HIP[0] + R_HIP[0]) / 2.0, (L_HIP[1] + R_HIP[1]) / 2.0)
            trunk_angle = angle_3pts(shoulder_mid, hip_mid, (hip_mid[0], hip_mid[1] + 100))  # vs. vertical line

            # Shoulder line tilt (degrees): 0 = level
            dx = R_SH[0] - L_SH[0]
            dy = R_SH[1] - L_SH[1]
            shoulder_tilt = float(np.degrees(np.arctan2(dy, dx))) if not (np.isnan(dx) or np.isnan(dy)) else np.nan

            # Head forward (nose vs. shoulder-mid horizontally)
            head_forward = NOSE[0] - shoulder_mid[0]

            # Overlay text
            put_text(vis, f"Trunk angle: {trunk_angle:.1f} deg", (12, 28))
            put_text(vis, f"Shoulder tilt: {shoulder_tilt:.1f} deg", (12, 56))
            put_text(vis, f"Head fwd (px): {head_forward:.0f}", (12, 84))

            # Optionally: print keypoints every N frames
            if frame_idx % max(1, args.print_every) == 0:
                # Show first person's keypoints as name:(x,y,score)
                triplets = []
                for i, (xy, sc) in enumerate(zip(kps, scs)):
                    triplets.append(f"{names[i] if i < len(names) else i}:({xy[0]:.1f},{xy[1]:.1f},{sc:.2f})")
                print(f"[Frame {frame_idx}] Persons: {len(persons)}")
                print("  First person keypoints:")
                print("  " + " | ".join(triplets))

        # FPS
        t1 = time.time()
        fps = 1.0 / max(1e-6, (t1 - t0))
        fps_buf.append(fps)
        smoothed = np.mean(fps_buf) if fps_buf else fps
        put_text(vis, f"FPS: {smoothed:5.1f}", (12, vis.shape[0] - 16))

        # Init writer lazily when frame size known
        if writer is None and args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, 30.0, (vis.shape[1], vis.shape[0]))

        if args.show:
            cv2.imshow("MMPose RTM Posture", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        if writer is not None:
            writer.write(vis)

        frame_idx += 1

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Quick start:

    # 1) Install (CUDA optional but recommended)
    #    pip install "mmpose>=1.3.0" "mmengine>=0.10.0" "mmcv>=2.0.0" opencv-python
    #    pip install "mmdet>=3.2.0"

    # 2) Run on webcam:
    #    python main_rtmpose.py --show

    # 3) Run on a video and save annotated output:
    #    python main_rtmpose.py --source path/to/video.mp4 --show --save out.mp4

    # 4) Try different models:
    #    --pose2d rtmpose-s_8xb256-420e_coco-256x192
    #    --pose2d rtmpose-l_8xb256-420e_coco-256x192
    #    --pose2d rtmpose-m_8xb256-420e_coco-wholebody-256x192   (more KPs; slower)
    #    --det rtmdet-nano_8xb32-300e_coco
    """
    sys.exit(main())
