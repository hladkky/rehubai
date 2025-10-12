#!/usr/bin/env python3
"""
High-Accuracy Real-Time Dead Bug Pose Estimation using YOLO11-Pose ONNX
Optimized for 30+ fps with Test-Time Augmentation (TTA) for maximum accuracy

Processes video from sources/videos/dead_bug.MOV and saves to outputs/videos/
"""

import os
import sys
import time
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
DEFAULT_WEIGHTS = "weights/yolo/yolo11m-pose.onnx"  # Medium model - good accuracy/speed balance
INPUT_VIDEO = "sources/videos/dead_bug.MOV"
OUTPUT_VIDEO = "outputs/videos/dead_bug_yolo.MOV"

# Model settings
IMGSZ = 896                      # Inference size (higher = more accurate, slower)
MAX_DETECTIONS = 1               # Max persons to track per frame
KEYPOINT_THRESHOLD = 0.3         # Confidence threshold for drawing keypoints

# Rotation for supine pose (person lying on back)
ROTATION_ANGLE = 90              # Rotate 90° clockwise for better detection

# Test-Time Augmentation (TTA) for higher accuracy
USE_FLIP_TTA = True              # Average original + horizontally flipped predictions

# Visualization
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
SHOW_PREVIEW = True              # Show real-time preview window


# ============================================================================
# COCO-17 KEYPOINT DEFINITIONS
# ============================================================================
# YOLO-Pose uses COCO format:
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

LEFT_RIGHT_PAIRS = [
    (1, 2),   # eyes
    (3, 4),   # ears
    (5, 6),   # shoulders
    (7, 8),   # elbows
    (9, 10),  # wrists
    (11, 12), # hips
    (13, 14), # knees
    (15, 16)  # ankles
]

# Skeleton connections for visualization
SKELETON_PAIRS = [
    (5, 6),   # shoulders
    (5, 7),   # left shoulder-elbow
    (7, 9),   # left elbow-wrist
    (6, 8),   # right shoulder-elbow
    (8, 10),  # right elbow-wrist
    (11, 12), # hips
    (5, 11),  # left shoulder-hip
    (6, 12),  # right shoulder-hip
    (11, 13), # left hip-knee
    (13, 15), # left knee-ankle
    (12, 14), # right hip-knee
    (14, 16)  # right knee-ankle
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def letterbox(img, new_size=640, pad_color=114):
    """Resize and pad image to square while maintaining aspect ratio."""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (new_size - nh) // 2
    left = (new_size - nw) // 2

    canvas = np.full((new_size, new_size, 3), pad_color, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = im

    return canvas, r, (left, top)


def rotate_frame(img, angle):
    """Rotate frame by 0, 90, 180, or 270 degrees."""
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def horizontal_flip_keypoints(kps_xy, img_width):
    """Flip keypoints horizontally and swap left/right pairs."""
    flipped = kps_xy.copy()
    flipped[:, 0] = img_width - 1 - flipped[:, 0]

    # Swap left/right keypoint pairs
    for left_idx, right_idx in LEFT_RIGHT_PAIRS:
        flipped[[left_idx, right_idx]] = flipped[[right_idx, left_idx]]

    return flipped


def pick_pose_output(onnx_outputs):
    """Extract pose predictions from ONNX output tensors."""
    candidates = []
    for arr in onnx_outputs:
        a = np.array(arr)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 2 and a.shape[1] >= 6 + 3 * 10:  # bbox(4) + conf(1) + cls(1) + kps(3*K)
            candidates.append(a)

    if not candidates:
        return None

    # Pick the output with most data
    candidates.sort(key=lambda x: (x.shape[1], x.shape[0]), reverse=True)
    return candidates[0]


def pick_providers(disable_coreml=False):
    """Select best available ONNX Runtime execution provider."""
    available = ort.get_available_providers()

    # Ordered by performance preference
    priority_order = [
        "TensorrtExecutionProvider",  # NVIDIA TensorRT (fastest on GPU)
        "CUDAExecutionProvider",      # NVIDIA CUDA
        "CoreMLExecutionProvider",    # Apple Silicon M1/M2
        "OpenVINOExecutionProvider",  # Intel CPUs
        "DmlExecutionProvider",       # DirectML (Windows)
        "CPUExecutionProvider"        # Fallback
    ]

    selected = []
    for provider in priority_order:
        if provider == "CoreMLExecutionProvider" and disable_coreml:
            continue
        if provider in available:
            selected.append(provider)

    return selected if selected else ["CPUExecutionProvider"]


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_text_with_background(img, text, position, font_scale=0.6, thickness=2):
    """Draw text with black outline for better visibility."""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_skeleton(img, keypoints, confidences, threshold=0.3):
    """Draw pose skeleton with keypoints and connections."""
    # Draw skeleton connections
    for idx_a, idx_b in SKELETON_PAIRS:
        if (idx_a < len(confidences) and idx_b < len(confidences) and
            confidences[idx_a] >= threshold and confidences[idx_b] >= threshold):

            xa, ya = keypoints[idx_a]
            xb, yb = keypoints[idx_b]

            if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
                # Draw white line with black border
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)),
                        (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)),
                        (0, 150, 0), 2, cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if i < len(confidences) and confidences[i] >= threshold:
            if not (np.isnan(x) or np.isnan(y)):
                # Green keypoint with white border
                cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1, cv2.LINE_AA)


def draw_hud(img, fps, frame_num, total_frames, tta_enabled):
    """Draw heads-up display with processing info."""
    h, w = img.shape[:2]

    # Semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Display info
    draw_text_with_background(img, f"FPS: {fps:.1f}", (20, 40), 0.7, 2)

    progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
    draw_text_with_background(img, f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)",
                             (20, 70), 0.6, 1)

    draw_text_with_background(img, f"Model: YOLO11-Pose @ {IMGSZ}px", (20, 95), 0.5, 1)

    tta_status = "ON (High Accuracy)" if tta_enabled else "OFF"
    draw_text_with_background(img, f"TTA: {tta_status}", (20, 120), 0.5, 1)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_video():
    """Main video processing pipeline."""

    print("=" * 70)
    print("DEAD BUG POSE ESTIMATION - YOLO11-Pose ONNX")
    print("=" * 70)
    print(f"Input:  {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Weights: {DEFAULT_WEIGHTS}")
    print(f"Image size: {IMGSZ}px")
    print(f"Rotation: {ROTATION_ANGLE}°")
    print(f"TTA (flip augmentation): {'Enabled' if USE_FLIP_TTA else 'Disabled'}")
    print("=" * 70)

    # Check if input exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: Input video not found: {INPUT_VIDEO}")
        return 1

    # Check if weights exist
    if not os.path.exists(DEFAULT_WEIGHTS):
        print(f"ERROR: Model weights not found: {DEFAULT_WEIGHTS}")
        print("Please ensure YOLO11-Pose ONNX model is in weights/yolo/ directory")
        return 1

    # Initialize ONNX Runtime session
    providers = pick_providers()
    print(f"Using execution providers: {providers}")

    session = ort.InferenceSession(DEFAULT_WEIGHTS, providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {INPUT_VIDEO}")
        return 1

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {frame_width}x{frame_height} @ {fps_input:.2f} fps, {total_frames} frames")

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (frame_width, frame_height))

    if not writer.isOpened():
        print(f"ERROR: Cannot create output video: {OUTPUT_VIDEO}")
        cap.release()
        return 1

    print("\nProcessing... (press 'q' to quit early)\n")

    # Processing loop
    fps_buffer = deque(maxlen=30)
    frame_count = 0

    while True:
        ret, frame_original = cap.read()
        if not ret:
            print("\nEnd of video.")
            break

        frame_count += 1
        t_start = time.time()

        # Rotate frame for better pose detection (supine position)
        frame = rotate_frame(frame_original, ROTATION_ANGLE)

        # Preprocess: letterbox resize
        img_preprocessed, scale, pad = letterbox(frame, IMGSZ)
        left, top = pad

        # Convert to ONNX input format: RGB, CHW, float32, normalized
        blob = img_preprocessed[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[None, ...]  # Add batch dimension

        # Run inference
        outputs = session.run(output_names, {input_name: blob})
        detections = pick_pose_output(outputs)

        annotated_frame = frame.copy()

        if detections is not None and detections.shape[0] > 0:
            # Sort by confidence and take top detections
            sorted_indices = np.argsort(-detections[:, 4])
            top_detections = detections[sorted_indices][:MAX_DETECTIONS]

            for detection in top_detections:
                # Extract keypoints: [x, y, conf] for each of 17 keypoints
                keypoint_data = detection[6:]
                num_keypoints = keypoint_data.shape[0] // 3
                keypoint_data = keypoint_data.reshape(num_keypoints, 3)

                # Map from letterbox coordinates back to original frame
                kp_x = (keypoint_data[:, 0] - left) / scale
                kp_y = (keypoint_data[:, 1] - top) / scale
                kp_conf = keypoint_data[:, 2]
                keypoints_original = np.stack([kp_x, kp_y], axis=1)

                # Apply Test-Time Augmentation if enabled
                if USE_FLIP_TTA:
                    # Run inference on horizontally flipped image
                    img_flipped = cv2.flip(img_preprocessed, 1)
                    blob_flipped = img_flipped[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                    outputs_flipped = session.run(output_names, {input_name: blob_flipped[None, ...]})
                    detections_flipped = pick_pose_output(outputs_flipped)

                    if detections_flipped is not None and detections_flipped.shape[0] > 0:
                        # Get best detection from flipped image
                        best_flipped = detections_flipped[np.argmax(detections_flipped[:, 4])]
                        kp_data_flipped = best_flipped[6:].reshape(num_keypoints, 3)

                        # Map back to original coordinates
                        kp_x_f = (kp_data_flipped[:, 0] - left) / scale
                        kp_y_f = (kp_data_flipped[:, 1] - top) / scale
                        keypoints_flipped = np.stack([kp_x_f, kp_y_f], axis=1)

                        # Flip back and swap left/right
                        keypoints_flipped_corrected = horizontal_flip_keypoints(
                            keypoints_flipped, annotated_frame.shape[1]
                        )

                        # Average original and flipped predictions
                        keypoints = 0.5 * (keypoints_original + keypoints_flipped_corrected)
                        kp_conf = np.maximum(kp_conf, kp_data_flipped[:, 2])
                    else:
                        keypoints = keypoints_original
                else:
                    keypoints = keypoints_original

                # Draw skeleton
                draw_skeleton(annotated_frame, keypoints, kp_conf, KEYPOINT_THRESHOLD)

        # Rotate back to original orientation
        annotated_frame = rotate_frame(annotated_frame, (360 - ROTATION_ANGLE) % 360)

        # Calculate FPS
        elapsed = time.time() - t_start
        fps_buffer.append(1.0 / max(elapsed, 1e-6))
        avg_fps = np.mean(fps_buffer)

        # Draw HUD
        draw_hud(annotated_frame, avg_fps, frame_count, total_frames, USE_FLIP_TTA)

        # Write to output
        writer.write(annotated_frame)

        # Display progress
        progress_pct = (frame_count / total_frames) * 100
        print(f"\rProgress: {progress_pct:.1f}% | Frame {frame_count}/{total_frames} | "
              f"FPS: {avg_fps:.1f}", end='', flush=True)

        # Optional: show preview
        if SHOW_PREVIEW:
            preview = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow('Dead Bug Pose Estimation', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nProcessing interrupted by user.")
                break

    # Cleanup
    cap.release()
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"Processing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Average FPS: {np.mean(fps_buffer):.1f}")
    print(f"Output saved to: {OUTPUT_VIDEO}")
    print("=" * 70)

    return 0


def main():
    return process_video()


if __name__ == "__main__":
    sys.exit(main())