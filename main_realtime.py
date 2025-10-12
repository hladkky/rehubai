"""
Real-time RTMPose video processing with minimum latency.
Optimized for live video feed with frame skipping and async processing.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
import threading
from queue import Queue

try:
    from mmpose.apis import init_model, inference_topdown
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    print("Warning: MMPose not installed.")
    raise


class RealtimeRTMPoseDetector:
    """Optimized RTMPose detector for real-time processing"""

    def __init__(self, model_size='s', device='cpu', use_onnx=False):
        """
        Initialize real-time RTMPose detector

        Args:
            model_size: 's' (small, fastest), 'm' (medium), 'l' (large, most accurate)
            device: 'cpu' or 'cuda'
            use_onnx: Use ONNX runtime for faster inference (requires conversion)
        """
        if not MMPOSE_AVAILABLE:
            raise ImportError("MMPose not available")

        self.use_onnx = use_onnx

        # Model configurations
        model_configs = {
            's': {
                'config': 'configs/rtmpose-s_8xb256-420e_coco-256x192.py',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth',
                'input_size': (192, 256)
            },
            'm': {
                'config': 'configs/rtmpose-m_8xb256-420e_coco-256x192.py',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
                'input_size': (192, 256)
            },
            't': {  # Tiny - ultra fast
                'config': 'configs/rtmpose-t_8xb256-420e_coco-256x192.py',
                'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.pth',
                'input_size': (192, 256)
            }
        }

        config = model_configs[model_size]
        self.input_size = config['input_size']

        if use_onnx:
            print("ONNX runtime not yet configured. Using PyTorch backend.")
            # TODO: Load ONNX model when available

        print(f"Loading RTMPose-{model_size} model...")
        self.model = init_model(config['config'], config['checkpoint'], device=device)
        print("Model loaded successfully!")

        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

    def detect(self, frame):
        """
        Fast keypoint detection

        Args:
            frame: BGR image

        Returns:
            keypoints: Array of shape (N, 17, 3)
        """
        start_time = time.time()

        results = inference_topdown(self.model, frame)

        # Track FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)

        if len(results) == 0:
            return np.array([]), fps

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

        return np.array(keypoints), fps

    def get_avg_fps(self):
        """Get average FPS over recent frames"""
        if len(self.fps_history) == 0:
            return 0
        return sum(self.fps_history) / len(self.fps_history)


class VideoStreamHandler:
    """Handle video stream with frame skipping for real-time processing"""

    def __init__(self, source=0, target_fps=30):
        """
        Initialize video stream

        Args:
            source: Video source (0 for webcam, or file path)
            target_fps: Target processing FPS (will skip frames to maintain)
        """
        self.cap = cv2.VideoCapture(source)
        self.target_fps = target_fps
        self.frame_skip = 1
        self.running = False
        self.frame_queue = Queue(maxsize=2)
        self.thread = None

    def start(self):
        """Start video stream thread"""
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Update frame queue in background thread"""
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            frame_count += 1

            # Skip frames if needed for target FPS
            if frame_count % self.frame_skip != 0:
                continue

            # Clear old frame and add new one
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def read(self):
        """Read latest frame"""
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        return False, None

    def stop(self):
        """Stop video stream"""
        self.running = False
        if self.thread is not None:
            self.thread.join()
        self.cap.release()

    def get_properties(self):
        """Get video properties"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return width, height, fps


def draw_skeleton_fast(frame, keypoints, threshold=0.3):
    """Fast skeleton drawing with minimal overhead"""
    if len(keypoints) == 0:
        return

    if len(keypoints.shape) == 2:
        keypoints = keypoints[np.newaxis, ...]

    # Simplified colors
    limb_color = (0, 255, 0)
    joint_color = (255, 100, 100)

    # COCO skeleton
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
        [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6]
    ]

    for person_kpts in keypoints:
        # Draw lines (limbs)
        for pt1_idx, pt2_idx in skeleton:
            if (person_kpts[pt1_idx, 2] > threshold and
                person_kpts[pt2_idx, 2] > threshold):

                pt1 = tuple(person_kpts[pt1_idx, :2].astype(int))
                pt2 = tuple(person_kpts[pt2_idx, :2].astype(int))
                cv2.line(frame, pt1, pt2, limb_color, 2)

        # Draw joints (keypoints)
        for keypoint in person_kpts:
            if keypoint[2] > threshold:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(frame, (x, y), 3, joint_color, -1)


def process_realtime(source, output_path=None, model_size='s', display=True):
    """
    Process video in real-time with minimum latency

    Args:
        source: Video source (0 for webcam, or file path)
        output_path: Path to save output video (optional)
        model_size: 's' (fastest), 'm', 't' (ultra-fast)
        display: Show live display window
    """
    # Initialize detector
    detector = RealtimeRTMPoseDetector(model_size=model_size, device='cpu')

    # Open video directly (no threading)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Target model: RTMPose-{model_size}")
    print("Press 'q' to quit")

    # Video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    last_keypoints = np.array([])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect keypoints
            keypoints, current_fps = detector.detect(frame)

            # Draw skeleton
            if len(keypoints) > 0:
                last_keypoints = keypoints
                draw_skeleton_fast(frame, keypoints)
            elif len(last_keypoints) > 0:
                # Use previous keypoints if detection failed
                draw_skeleton_fast(frame, last_keypoints)

            # Draw performance metrics
            avg_fps = detector.get_avg_fps()
            cv2.putText(frame, f"Inference FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"People: {len(keypoints)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write frame
            if writer is not None:
                writer.write(frame)

            # Display
            if display:
                cv2.imshow('Real-time Pose Detection', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%) - FPS: {avg_fps:.1f}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames")
        print(f"Average FPS: {detector.get_avg_fps():.1f}")


def main():
    """Main entry point"""

    # Option 1: Process webcam in real-time
    # process_realtime(source=0, model_size='t', display=True)

    # Option 2: Process video file in real-time
    input_video = Path("sources/videos/dead_bug.MOV")
    output_video = Path("outputs/videos/dead_bug_realtime.mp4")

    if not input_video.exists():
        print(f"Error: Video not found: {input_video}")
        return

    process_realtime(
        source=str(input_video),
        output_path=str(output_video),
        model_size='s',  # Options: 't' (ultra-fast), 's' (fast), 'm' (balanced)
        display=True
    )


if __name__ == "__main__":
    main()
