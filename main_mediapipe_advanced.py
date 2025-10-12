"""
MediaPipe Pose Detection - Simplified Version

Real-time pose detection and skeleton visualization using MediaPipe Pose + OpenCV.
Detects body landmarks and draws skeleton overlay on video.

Requirements (Python 3.9+ recommended):
    pip install opencv-python mediapipe numpy

Usage:
    python main_mediapipe_advanced.py
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

input_video_path = "sources/videos/dead_bug.MOV"
output_video_path = "outputs/videos/main_mediapipe_advanced.MOV"


def draw_skeleton(frame, results):
    """Draw pose landmarks and connections"""
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    if results.pose_landmarks:
        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())


def main():
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {input_video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup video writer
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize MediaPipe Pose
    pose = mp.solutions.pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Draw skeleton
            draw_skeleton(frame, results)

            # Draw frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw detection status
            if results.pose_landmarks:
                cv2.putText(frame, "Person detected", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write frame
            out.write(frame)

            # Display
            cv2.imshow("MediaPipe Pose Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            # Progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")

    finally:
        pose.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\n✓ Processing complete!")
    print(f"✓ Processed {frame_count} frames")
    print(f"✓ Output: {output_video_path}")


if __name__ == "__main__":
    main()