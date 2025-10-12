"""
RTMPose-based pose estimation for rehabilitation analysis.
Processes video from sources/videos/ and outputs annotated video with skeleton overlay to outputs/videos/
"""

import cv2
import numpy as np
import os
from pathlib import Path

try:
    from mmpose.apis import init_model, inference_topdown
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    print("Warning: MMPose not installed. Install with: pip install openmim && mim install mmcv mmpose mmdet")
    raise


class RTMPoseDetector:
    """RTMPose pose estimation wrapper"""

    def __init__(self, model_name='rtmpose-m', device='cpu'):
        """
        Initialize RTMPose model

        Args:
            model_name: Model variant (rtmpose-s, rtmpose-m, rtmpose-l)
            device: 'cpu' or 'cuda'
        """
        if not MMPOSE_AVAILABLE:
            raise ImportError("MMPose library not available")

        # RTMPose config and checkpoint paths
        # Using RTMPose-m (medium) for COCO body keypoints
        config_file = 'configs/rtmpose-m_8xb256-420e_coco-256x192.py'
        checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'

        self.model = init_model(config_file, checkpoint_file, device=device)

        # COCO keypoint names
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # COCO skeleton connections
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]

    def detect(self, frame):
        """
        Detect pose keypoints in frame

        Args:
            frame: BGR image (numpy array)

        Returns:
            keypoints: Array of shape (N, 17, 3) where N is number of people detected
                      Each keypoint is (x, y, confidence)
        """
        results = inference_topdown(self.model, frame)

        if len(results) == 0:
            return np.array([])

        # Extract keypoints from results
        keypoints = []
        for result in results:
            pred = result.pred_instances
            kpts = pred.keypoints
            scores = pred.keypoint_scores

            # Handle batch dimensions
            if kpts.ndim == 3:
                kpts = kpts[0]  # Remove batch dimension -> (17, 2)
            if scores.ndim == 2:
                scores = scores[0]  # Remove batch dimension -> (17,)

            # Combine into (17, 3) format
            kpts_with_conf = np.concatenate([kpts, scores[:, np.newaxis]], axis=1)
            keypoints.append(kpts_with_conf)

        return np.array(keypoints)


def draw_skeleton(frame, keypoints, confidence_threshold=0.3):
    """
    Draw skeleton keypoints and connections on frame

    Args:
        frame: BGR image
        keypoints: Array of shape (N, 17, 3) or (17, 3)
        confidence_threshold: Minimum confidence to draw point
    """
    if len(keypoints) == 0:
        return

    if len(keypoints.shape) == 2:
        keypoints = keypoints[np.newaxis, ...]

    # Define colors for different body parts
    colors = {
        'face': (255, 200, 100),      # Light blue
        'torso': (0, 255, 0),         # Green
        'right_arm': (0, 0, 255),     # Red
        'left_arm': (255, 0, 0),      # Blue
        'right_leg': (0, 165, 255),   # Orange
        'left_leg': (255, 0, 255),    # Magenta
    }

    # Define which connections belong to which body part
    connection_colors = {
        (0, 1): colors['face'], (0, 2): colors['face'],
        (1, 3): colors['face'], (2, 4): colors['face'],
        (1, 2): colors['face'],
        (5, 6): colors['torso'], (5, 11): colors['torso'],
        (6, 12): colors['torso'], (11, 12): colors['torso'],
        (5, 7): colors['left_arm'], (7, 9): colors['left_arm'],
        (6, 8): colors['right_arm'], (8, 10): colors['right_arm'],
        (11, 13): colors['left_leg'], (13, 15): colors['left_leg'],
        (12, 14): colors['right_leg'], (14, 16): colors['right_leg'],
    }

    for person_kpts in keypoints:
        # Draw skeleton connections
        for connection in [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                          [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                          [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                          [1, 3], [2, 4], [3, 5], [4, 6]]:
            pt1_idx, pt2_idx = connection

            if (person_kpts[pt1_idx, 2] > confidence_threshold and
                person_kpts[pt2_idx, 2] > confidence_threshold):

                pt1 = tuple(person_kpts[pt1_idx, :2].astype(int))
                pt2 = tuple(person_kpts[pt2_idx, :2].astype(int))

                # Get color for this connection
                color = connection_colors.get((pt1_idx, pt2_idx), (255, 255, 255))

                cv2.line(frame, pt1, pt2, color, 2)

        # Draw keypoints
        for idx, keypoint in enumerate(person_kpts):
            x, y, conf = keypoint
            if conf > confidence_threshold:
                # Color keypoints based on body part
                if idx in [0, 1, 2, 3, 4]:  # Face
                    color = colors['face']
                elif idx in [5, 6, 11, 12]:  # Torso
                    color = colors['torso']
                elif idx in [7, 9]:  # Left arm
                    color = colors['left_arm']
                elif idx in [8, 10]:  # Right arm
                    color = colors['right_arm']
                elif idx in [13, 15]:  # Left leg
                    color = colors['left_leg']
                elif idx in [14, 16]:  # Right leg
                    color = colors['right_leg']
                else:
                    color = (255, 255, 255)

                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 0), 1)


def process_video(input_path, output_path):
    """
    Process video with RTMPose detection and skeleton drawing

    Args:
        input_path: Path to input video
        output_path: Path to output video
    """
    # Initialize detector
    print("Initializing RTMPose detector...")
    detector = RTMPoseDetector(model_name='rtmpose-m', device='cpu')

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect keypoints with RTMPose
            keypoints = detector.detect(frame)

            if len(keypoints) > 0:
                # Draw skeleton on frame
                draw_skeleton(frame, keypoints)

                # Draw info
                num_people = len(keypoints)
                cv2.putText(frame, f"People detected: {num_people}",
                           (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write frame
            out.write(frame)

            # Show progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")

    finally:
        cap.release()
        out.release()

    print(f"\nProcessing complete!")
    print(f"Output video: {output_path}")


def main():
    """Main entry point"""
    # Define paths
    input_video = Path("sources/videos/dead_bug.MOV")
    output_video = Path("outputs/videos/dead_bug_rtmpose.mp4")

    # Check if input exists
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return

    # Process video
    process_video(input_video, output_video)


if __name__ == "__main__":
    main()