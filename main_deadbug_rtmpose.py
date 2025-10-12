#!/usr/bin/env python3
"""
Dead Bug Exercise - Real-Time Pose Detection with RTMPose
Clean, from-scratch implementation with high accuracy and 30+ fps

RTMPose provides MediaPipe-level accuracy with fine-tuning capability.
Detects 17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Video paths
INPUT_VIDEO = "sources/videos/dead_bug.MOV"
OUTPUT_VIDEO = "outputs/videos/dead_bug_rtmpose.MOV"

# RTMPose model selection
# Options: rtmpose-t (tiny), rtmpose-s (small), rtmpose-m (medium), rtmpose-l (large)
POSE_MODEL = "rtmpose-m_8xb256-420e_coco-256x192"  # Medium: best accuracy/speed balance

# Person detector (RTMDet)
DET_MODEL = "rtmdet_m_8xb32-300e_coco"  # Medium detector (fixed naming)

# Processing settings
MAX_PERSONS = 1              # Track single person
SHOW_PREVIEW = True          # Display real-time preview
KEYPOINT_THRESHOLD = 0.3     # Min confidence to draw keypoint

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# ============================================================================
# COCO-17 KEYPOINT SCHEMA
# ============================================================================
# RTMPose uses COCO-17 format:
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Skeleton connections for visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # nose to eyes
    (1, 3), (2, 4),      # eyes to ears
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16)   # right leg
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_keypoints(mm_result, max_persons=1):
    """
    Extract keypoints from MMPoseInferencer result.
    Returns: list of (keypoints_array[17,2], scores_array[17])
    """
    persons = []
    predictions = mm_result.get("predictions", [])

    if not predictions:
        return persons

    frame_pred = predictions[0]  # Single frame per iteration

    # Handle different output formats
    if isinstance(frame_pred, list):
        # List of person instances
        candidates = []
        for instance in frame_pred:
            kps = np.array(instance.get("keypoints", []), dtype=float)
            scores = np.array(instance.get("keypoint_scores", []), dtype=float)

            # Reshape if needed
            if kps.ndim == 3 and kps.shape[0] == 1:
                kps = kps[0]
            if scores.ndim == 2 and scores.shape[0] == 1:
                scores = scores[0]

            if kps.size == 0:
                continue

            avg_score = float(np.nanmean(scores)) if scores.size > 0 else 0.0
            candidates.append((avg_score, kps, scores))

    elif isinstance(frame_pred, dict):
        # Dictionary with stacked arrays
        kps_stack = frame_pred.get("keypoints", None)
        scores_stack = frame_pred.get("keypoint_scores", None)

        candidates = []
        if kps_stack is not None:
            kps_stack = np.array(kps_stack, dtype=float)

            if scores_stack is not None:
                scores_stack = np.array(scores_stack, dtype=float)
            else:
                scores_stack = np.zeros((kps_stack.shape[0], kps_stack.shape[1]), dtype=float)

            for i in range(kps_stack.shape[0]):
                kps = kps_stack[i]
                scores = scores_stack[i] if scores_stack.ndim == 2 else np.zeros(kps.shape[0])
                avg_score = float(np.nanmean(scores)) if scores.size > 0 else 0.0
                candidates.append((avg_score, kps, scores))
    else:
        return persons

    # Sort by confidence and return top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, kps, scores in candidates[:max_persons]:
        persons.append((kps, scores))

    return persons


def draw_skeleton(image, keypoints, scores, threshold=0.3):
    """Draw skeleton with keypoints and connections."""
    h, w = image.shape[:2]

    # Draw connections (bones)
    for idx_a, idx_b in SKELETON_CONNECTIONS:
        if (idx_a < len(scores) and idx_b < len(scores) and
            scores[idx_a] >= threshold and scores[idx_b] >= threshold):

            pt_a = keypoints[idx_a]
            pt_b = keypoints[idx_b]

            if not (np.isnan(pt_a).any() or np.isnan(pt_b).any()):
                x1, y1 = int(pt_a[0]), int(pt_a[1])
                x2, y2 = int(pt_b[0]), int(pt_b[1])

                # Draw bone with outline
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 4, cv2.LINE_AA)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    # Draw keypoints
    for i, (pt, score) in enumerate(zip(keypoints, scores)):
        if score >= threshold and not np.isnan(pt).any():
            x, y = int(pt[0]), int(pt[1])

            # Draw keypoint with outline
            cv2.circle(image, (x, y), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)


def draw_text_with_outline(image, text, position, font_scale=0.6, thickness=2):
    """Draw text with black outline for visibility."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_hud(image, fps, frame_num, total_frames, pose_detected):
    """Draw heads-up display with processing info."""
    h, w = image.shape[:2]

    # Semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Display info
    draw_text_with_outline(image, f"FPS: {fps:.1f}", (20, 40), 0.7, 2)

    progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
    draw_text_with_outline(image, f"Frame: {frame_num}/{total_frames} ({progress:.1f}%)",
                          (20, 70), 0.6, 1)

    status = "DETECTED" if pose_detected else "NOT DETECTED"
    status_color = (0, 255, 0) if pose_detected else (0, 0, 255)
    draw_text_with_outline(image, f"Pose: {status}", (20, 100), 0.6, 1)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_video():
    """Main video processing pipeline."""

    print("=" * 70)
    print("DEAD BUG POSE ESTIMATION - RTMPose")
    print("=" * 70)
    print(f"Input:  {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Pose model: {POSE_MODEL}")
    print(f"Detector: {DET_MODEL}")
    print("=" * 70)

    # Check input exists
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: Input video not found: {INPUT_VIDEO}")
        return 1

    # Detect device
    device = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    print(f"Using device: {device}")

    # Initialize RTMPose inferencer
    print("Loading RTMPose models (may download on first run)...")
    inferencer = MMPoseInferencer(
        pose2d=POSE_MODEL,
        det_model=DET_MODEL,
        device=device
    )
    print("Models loaded successfully!")

    # Open input video
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

    print("\nProcessing video... (press 'q' to quit early)\n")

    # Processing loop
    fps_buffer = deque(maxlen=30)
    frame_count = 0

    # Read all frames first (for batch processing)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    total_frames = len(frames)
    print(f"Loaded {total_frames} frames into memory")

    # Process frames with RTMPose
    print("Running pose estimation...\n")

    for frame_idx, frame_original in enumerate(frames):
        t_start = time.time()

        # Run inference (returns a generator)
        result_gen = inferencer(
            frame_original,
            show=False,
            return_vis=False,  # We'll draw manually
            draw_bbox=False
        )

        # Get first result from generator
        result = next(result_gen)

        # Extract keypoints
        persons = extract_keypoints(result, max_persons=MAX_PERSONS)

        # Create annotated frame
        annotated = frame_original.copy()
        pose_detected = len(persons) > 0

        # Draw skeleton for each detected person
        for keypoints, scores in persons:
            draw_skeleton(annotated, keypoints, scores, KEYPOINT_THRESHOLD)

        # Calculate FPS
        elapsed = time.time() - t_start
        fps_buffer.append(1.0 / max(elapsed, 1e-6))
        avg_fps = np.mean(fps_buffer)

        # Draw HUD
        draw_hud(annotated, avg_fps, frame_idx + 1, total_frames, pose_detected)

        # Write to output
        writer.write(annotated)

        # Display progress
        progress_pct = ((frame_idx + 1) / total_frames) * 100
        print(f"\rProgress: {progress_pct:.1f}% | Frame {frame_idx + 1}/{total_frames} | "
              f"FPS: {avg_fps:.1f} | Persons: {len(persons)}", end='', flush=True)

        # Optional: show preview
        if SHOW_PREVIEW:
            preview = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow('Dead Bug - RTMPose', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nProcessing interrupted by user.")
                break

    # Cleanup
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"Processing complete!")
    print(f"Processed {frame_idx + 1} frames")
    print(f"Average FPS: {np.mean(fps_buffer):.1f}")
    print(f"Output saved to: {OUTPUT_VIDEO}")
    print("=" * 70)

    return 0


def main():
    try:
        return process_video()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    """
    Requirements:
    pip install mmpose mmcv mmengine mmdet opencv-python numpy

    Usage:
    python main_deadbug_rtmpose.py

    Features:
    - RTMPose-M: MediaPipe-level accuracy with training capability
    - 17 COCO keypoints: full body skeleton
    - 30+ fps real-time processing
    - Auto model download on first run
    - GPU support (CUDA) if available
    """
    sys.exit(main())
