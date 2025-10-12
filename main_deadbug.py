#!/usr/bin/env python3
"""
Real-time Dead Bug Exercise Pose Estimation with MediaPipe
Optimized for 30+ fps with high-quality landmark detection

Displays full body pose landmarks in real-time for dead bug exercise analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque


# ============================================================================
# CONFIGURATION
# ============================================================================

# Video configuration
INPUT_VIDEO = "sources/videos/dead_bug.MOV"
OUTPUT_VIDEO = "outputs/videos/dead_bug.MOV"
DISPLAY_WIDTH = 1280                # Display resolution width
DISPLAY_HEIGHT = 720                # Display resolution height

# MediaPipe pose model settings
MODEL_COMPLEXITY = 2                # 0=lite, 1=full, 2=heavy (highest accuracy)
MIN_DETECTION_CONFIDENCE = 0.7      # Higher = more accurate but slower detection
MIN_TRACKING_CONFIDENCE = 0.7       # Higher = smoother tracking




# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_hud(frame: np.ndarray, fps: float, pose_detected: bool):
    """Draw simple HUD with FPS counter."""
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display pose status
    status_text = "Pose: DETECTED" if pose_detected else "Pose: NOT DETECTED"
    status_color = (0, 255, 0) if pose_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)


def draw_pose_landmarks(frame: np.ndarray, landmarks, mp_pose, mp_drawing):
    """Draw high-quality MediaPipe pose skeleton on frame."""
    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=3
        )
    )


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def main():
    """Process video and save output with pose estimation."""
    import os

    print("=" * 60)
    print("DEAD BUG EXERCISE - POSE ESTIMATION")
    print("=" * 60)
    print(f"Input: {INPUT_VIDEO}")
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Model complexity: {MODEL_COMPLEXITY}")
    print(f"Detection confidence: {MIN_DETECTION_CONFIDENCE}")
    print(f"Tracking confidence: {MIN_TRACKING_CONFIDENCE}")
    print(f"Press 'q' to quit early")
    print("=" * 60)

    # Initialize MediaPipe Pose with high-quality settings
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {INPUT_VIDEO}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {frame_width}x{frame_height} @ {fps_input:.2f} fps")
    print(f"Total frames: {total_frames}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps_input, (frame_width, frame_height))

    if not out.isOpened():
        print(f"ERROR: Cannot create output video: {OUTPUT_VIDEO}")
        cap.release()
        return

    print("Processing video...")

    # FPS calculation for processing speed
    fps_buffer = deque(maxlen=30)
    prev_time = cv2.getTickCount()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached.")
            break

        frame_count += 1

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose estimation
        results = pose.process(rgb_frame)

        # Calculate processing FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        fps_buffer.append(fps)
        avg_fps = np.mean(fps_buffer)
        prev_time = current_time

        # Draw pose landmarks if detected
        pose_detected = False
        if results.pose_landmarks:
            pose_detected = True
            draw_pose_landmarks(frame, results.pose_landmarks, mp_pose, mp_drawing)

        # Draw HUD
        draw_hud(frame, avg_fps, pose_detected)

        # Write frame to output
        out.write(frame)

        # Display progress
        progress = (frame_count / total_frames) * 100
        print(f"\rProgress: {progress:.1f}% ({frame_count}/{total_frames}) | Processing FPS: {avg_fps:.1f}", end='')

        # Show frame (optional - can comment out for faster processing)
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow('Dead Bug Exercise - Processing', display_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n\nProcessing interrupted by user.")
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()

    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"Average processing FPS: {np.mean(fps_buffer):.1f}")
    print(f"Output saved to: {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == "__main__":
    main()