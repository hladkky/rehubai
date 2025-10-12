#!/usr/bin/env python3
# main_realtime_onnx.py
# Cross-platform real-time pose landmarks (YOLO-Pose ONNX) with robust output decoding.
# - Works with ONNX Runtime EPs (TensorRT/CUDA/CoreML/OpenVINO/DirectML/CPU)
# - Handles output shapes [N, D] or [1, N, D], multiple outputs, etc.
# - Optional flip-TTA (averages original + mirrored predictions)
# - No rotation: keeps your original orientation

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

# COCO-like pairs for drawing (edit if your schema differs)
PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Left/right index pairs to swap when flipping horizontally (COCO-17)
LR_SWAP = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]


def put_text(img, text, org, scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_skeleton(img, pts, conf, thr=0.3):
    # joints
    for i, (x, y) in enumerate(pts):
        if i >= len(conf) or conf[i] < thr or np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), 1)
    # bones
    for a, b in PAIRS:
        if a < len(conf) and b < len(conf) and conf[a] >= thr and conf[b] >= thr:
            xa, ya = pts[a]; xb, yb = pts[b]
            if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), 2)
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0, 0, 0), 1)


def letterbox(img, new_size=640, pad_color=114):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas = np.full((new_size, new_size, 3), pad_color, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = im
    return canvas, r, (left, top)


def pick_providers(disable_coreml=False):
    avail = ort.get_available_providers()
    order = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    picked = []
    for p in order:
        if p == "CoreMLExecutionProvider" and disable_coreml:
            continue
        if p in avail:
            picked.append(p)
    if not picked:
        picked = ["CPUExecutionProvider"]
    return picked


def pick_pose_output(onnx_outputs):
    """
    Normalize ONNX outputs to a single 2D array [N, D] where D = 6 + 3*K.
    Handles:
      - single output [N, D]
      - single output [1, N, D]
      - multiple outputs (choose the one with D >= 36, i.e., >= 10 keypoints * 3 + 6).
    Returns None if nothing matches.
    """
    cands = []
    for arr in onnx_outputs:
        a = np.array(arr)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]  # squeeze batch
        if a.ndim == 2:
            D = a.shape[1]
            if D >= 6 + 3 * 10:  # tolerate >=10 keypoints
                cands.append(a)
    if not cands:
        return None
    # Prefer more columns (more keypoints), then more rows
    cands.sort(key=lambda x: (x.shape[1], x.shape[0]), reverse=True)
    return cands[0]


def hflip_keypoints(kps_xy, img_w):
    """Flip x across image width and swap left/right joints."""
    out = kps_xy.copy()
    out[:, 0] = img_w - 1 - out[:, 0]
    for a, b in LR_SWAP:
        out[[a, b]] = out[[b, a]]
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Real-time ONNX Pose (robust, no-rotation)")
    ap.add_argument("--weights", required=True, help="ONNX model exported from Ultralytics (pose)")
    ap.add_argument("--source", default="0", help="webcam index (e.g., 0) or path to video")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size (must match export)")
    ap.add_argument("--max-det", type=int, default=1, help="max detections per frame")
    ap.add_argument("--kp-thr", dest="kp_thr", type=float, default=0.3, help="keypoint draw threshold")
    ap.add_argument("--flip-tta", dest="flip_tta", action="store_true", help="average original + mirrored prediction")
    ap.add_argument("--show", action="store_true", help="show window")
    ap.add_argument("--save", default="", help="save annotated video path (mp4)")
    ap.add_argument("--fps-smooth", type=int, default=30, help="FPS smoothing window")
    ap.add_argument("--no-coreml", action="store_true", help="disable CoreML EP (debug on macOS)")
    ap.add_argument("--print-outputs-once", action="store_true", help="print output shapes once and exit")
    return ap.parse_args()


def main():
    args = parse_args()

    providers = pick_providers(disable_coreml=args.no_coreml)
    sess = ort.InferenceSession(args.weights, providers=providers)

    iname = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    # Optional debug: print ONNX outputs (shapes) and exit
    if args.print_outputs_once:
        dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        blob = dummy[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[None, ...]
        outs = sess.run(out_names, {iname: blob})
        print("Providers:", providers)
        for n, o in zip(out_names, outs):
            a = np.array(o)
            print(n, a.shape, a.dtype)
        return 0

    # Open source
    src = 0 if (args.source.isdigit() and len(args.source) == 1) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source: {args.source}")
        return 1

    writer = None
    fps_hist = deque(maxlen=args.fps_smooth)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()

        # Preprocess
        img, scale, pad = letterbox(frame, args.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[None, ...]  # NCHW

        # Inference
        outs = sess.run(out_names, {iname: blob})
        tbl = pick_pose_output(outs)  # [N, 6+3K] or None

        annotated = frame
        if tbl is not None and tbl.shape[0] > 0:
            # Sort by det conf (col 4) desc
            order = np.argsort(-tbl[:, 4])
            tbl = tbl[order][:args.max_det]

            annotated = frame.copy()
            left, top = pad

            for det in tbl:
                # det = [x1,y1,x2,y2,box_conf,cls, kps...]
                kparr = det[6:]  # K*3
                K = kparr.shape[0] // 3
                kparr = kparr.reshape(K, 3)

                # Map keypoints from letterboxed space back to original frame
                kx = (kparr[:, 0] - left) / scale
                ky = (kparr[:, 1] - top) / scale
                kc = kparr[:, 2]
                kps = np.stack([kx, ky], axis=1)

                # Optional flip-TTA (no rotation)
                if args.flip_tta:
                    img_flipped = cv2.flip(img, 1)
                    blob_f = img_flipped[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                    blob_f = blob_f[None, ...]
                    outs_f = sess.run(out_names, {iname: blob_f})
                    tbl_f = pick_pose_output(outs_f)
                    if tbl_f is not None and tbl_f.shape[0] > 0:
                        det_f = tbl_f[np.argmax(tbl_f[:, 4])]
                        kparr_f = det_f[6:].reshape(K, 3)
                        kx_f = (kparr_f[:, 0] - left) / scale
                        ky_f = (kparr_f[:, 1] - top) / scale
                        kps_f = np.stack([kx_f, ky_f], axis=1)
                        kps = 0.5 * (kps + hflip_keypoints(kps_f, annotated.shape[1]))
                        kc = np.maximum(kc, kparr_f[:, 2])

                draw_skeleton(annotated, kps, kc, thr=args.kp_thr)

        # FPS
        dt = time.time() - t0
        fps_hist.append(1.0 / max(1e-6, dt))
        put_text(annotated, f"FPS: {np.mean(fps_hist):.1f} | EPs: {', '.join(providers)} | img={args.imgsz}",
                 (12, annotated.shape[0] - 16))

        # Writer
        if writer is None and args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save, fourcc, 30.0, (annotated.shape[1], annotated.shape[0]))
        if writer is not None:
            writer.write(annotated)

        # Show
        if args.show:
            cv2.imshow("ONNX Pose (realtime, no-rotation)", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    if writer is not None:
        writer.release()
    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
