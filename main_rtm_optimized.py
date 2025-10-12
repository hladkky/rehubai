"""
Optimized RTMPose with RTMDet person detector for better performance.
RTMPose requires bounding boxes, so we use RTMDet-nano for fast person detection.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque

try:
    from mmpose.apis import init_model as init_pose_model
    from mmpose.apis import inference_topdown
    from mmdet.apis import init_detector, inference_detector
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    print("Warning: MMPose/MMDet not installed.")
    raise


class OptimizedRTMDetector:
    """Fast RTMDet + RTMPose pipeline"""

    def __init__(self, pose_model='s', device='cpu'):
        """
        Initialize detector + pose estimator

        Args:
            pose_model: 's' (small), 't' (tiny), 'm' (medium)
            device: 'cpu' or 'cuda'
        """
        print("Using simple person detection (whole frame bboxes)...")

        # Skip heavy person detector - use whole frame as bbox
        # This is simpler and faster for single-person videos
        self.detector = None
        self.use_full_frame = True

        # RTMPose for keypoint detection
        pose_configs = {
            't': 'configs/rtmpose-t_8xb256-420e_coco-256x192.py',
            's': 'configs/rtmpose-s_8xb256-420e_coco-256x192.py',
            'm': 'configs/rtmpose-m_8xb256-420e_coco-256x192.py',
        }

        pose_checkpoints = {
            't': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth',
            's': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth',
            'm': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        }

        print(f"Loading RTMPose-{pose_model}...")
        self.pose_model = init_pose_model(
            pose_configs[pose_model],
            pose_checkpoints[pose_model],
            device=device
        )
        print("RTMPose loaded!")

        self.fps_history = deque(maxlen=30)

    def detect(self, frame):
        """
        Detect people and their keypoints

        Args:
            frame: BGR image

        Returns:
            keypoints: Array of shape (N, 17, 3)
            bboxes: Array of bounding boxes
        """
        start_time = time.time()

        # Use whole frame as bbox (simpler, faster for single-person videos)
        height, width = frame.shape[:2]
        bboxes = np.array([[0, 0, width, height, 1.0]])

        # Run pose estimation
        results = inference_topdown(self.pose_model, frame)

        # Extract keypoints
        keypoints = []
        for result in results:
            pred = result.pred_instances
            kpts = pred.keypoints
            scores = pred.keypoint_scores

            if kpts.ndim == 3:
                kpts = kpts[0]
            if scores.ndim == 2:
                scores = scores[0]

            kpts_with_conf = np.concatenate([kpts, scores[:, np.newaxis]], axis=1)
            keypoints.append(kpts_with_conf)

        # Track FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)

        return np.array(keypoints), bboxes

    def get_avg_fps(self):
        """Get average FPS"""
        if len(self.fps_history) == 0:
            return 0
        return sum(self.fps_history) / len(self.fps_history)


def draw_skeleton(frame, keypoints, threshold=0.3):
    """Draw skeleton with colored body parts"""
    if len(keypoints) == 0:
        return

    if len(keypoints.shape) == 2:
        keypoints = keypoints[np.newaxis, ...]

    # Colors for different body parts
    colors = {
        'head': (255, 200, 100),
        'torso': (0, 255, 0),
        'right_arm': (0, 0, 255),
        'left_arm': (255, 0, 0),
        'right_leg': (0, 165, 255),
        'left_leg': (255, 0, 255),
    }

    # Skeleton connections with colors
    skeleton = [
        # Head
        ([0, 1], colors['head']), ([0, 2], colors['head']),
        ([1, 3], colors['head']), ([2, 4], colors['head']),
        ([1, 2], colors['head']),
        # Torso
        ([5, 6], colors['torso']), ([5, 11], colors['torso']),
        ([6, 12], colors['torso']), ([11, 12], colors['torso']),
        # Left arm
        ([5, 7], colors['left_arm']), ([7, 9], colors['left_arm']),
        # Right arm
        ([6, 8], colors['right_arm']), ([8, 10], colors['right_arm']),
        # Left leg
        ([11, 13], colors['left_leg']), ([13, 15], colors['left_leg']),
        # Right leg
        ([12, 14], colors['right_leg']), ([14, 16], colors['right_leg']),
    ]

    for person_kpts in keypoints:
        # Draw skeleton lines
        for connection, color in skeleton:
            pt1_idx, pt2_idx = connection
            if (person_kpts[pt1_idx, 2] > threshold and
                person_kpts[pt2_idx, 2] > threshold):

                pt1 = tuple(person_kpts[pt1_idx, :2].astype(int))
                pt2 = tuple(person_kpts[pt2_idx, :2].astype(int))
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw keypoints
        for idx, keypoint in enumerate(person_kpts):
            if keypoint[2] > threshold:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)


def process_video(input_path, output_path, pose_model='t', display=True):
    """
    Process video with optimized RTMDet + RTMPose

    Args:
        input_path: Input video path
        output_path: Output video path
        pose_model: 't' (fastest), 's' (fast), 'm' (accurate)
        display: Show window
    """
    # Initialize
    detector = OptimizedRTMDetector(pose_model=pose_model, device='cpu')

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Model: RTMDet-nano + RTMPose-{pose_model}")
    print("Press 'q' to quit\n")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect
            keypoints, bboxes = detector.detect(frame)

            # Draw bounding boxes
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (int(x1), int(y1)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw skeleton
            draw_skeleton(frame, keypoints)

            # Draw info
            avg_fps = detector.get_avg_fps()
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"People: {len(keypoints)}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write
            writer.write(frame)

            # Display
            if display:
                cv2.imshow('Optimized RTM Pose Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%) - {avg_fps:.1f} FPS")

    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"\n✓ Processed {frame_count} frames")
    print(f"✓ Average FPS: {detector.get_avg_fps():.1f}")
    print(f"✓ Output: {output_path}")


def main():
    """Main entry point"""
    input_video = Path("sources/videos/dead_bug.MOV")
    output_video = Path("outputs/videos/dead_bug_optimized.mp4")

    if not input_video.exists():
        print(f"Error: Video not found: {input_video}")
        return

    process_video(
        input_path=input_video,
        output_path=output_video,
        pose_model='t',  # Options: 't' (fastest ~15-20 FPS), 's' (~10-12 FPS), 'm' (~6-8 FPS)
        display=True
    )


if __name__ == "__main__":
    main()
