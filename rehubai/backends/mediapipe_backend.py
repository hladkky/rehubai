"""MediaPipe Pose backend (BlazePose, 33 keypoints).

Wraps the legacy ``mp.solutions.pose`` pipeline used by ``main_deadbug.py`` behind
the common backend API. Two coordinate modes are supported:

* **image (default)** — landmarks in **pixel** coordinates ``(x_px, y_px, z_px)``
  with ``z`` scaled by frame width to stay commensurate with x/y. Keeps
  pixel-based config thresholds (coordinate y-checks, distances) meaningful.
* **world** (``use_world_landmarks=True``) — MediaPipe's metric 3D landmarks in
  **metres**, origin at the mid-hip, so joint angles are computed from a true,
  view-stable 3D skeleton (see ``docs/3d-pose-rationale.md``). ``visibility`` is
  still taken from the image-space landmarks and returned as the confidence, so
  the validator's confidence gate down-weights occluded (far-side) joints.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

SCHEME = "blazepose33"


class MediaPipePoseBackend:
    """Detect 33 BlazePose keypoints from BGR frames."""

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_world_landmarks: bool = False,
        static_image_mode: bool = False,
    ) -> None:
        import mediapipe as mp  # local import: heavy optional dependency

        self.scheme = SCHEME
        self.use_world_landmarks = use_world_landmarks
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(keypoints[33,3], confidences[33])`` for one BGR frame.

        In image mode keypoints are ``(x_px, y_px, z_px)``; in world mode they are
        metric ``(x, y, z)`` in metres. Confidence is the per-landmark
        ``visibility``. If no pose is found, returns zeros with zero confidence so
        the validator's visibility gate pauses.
        """
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._pose.process(rgb)

        kps = np.zeros((33, 3), dtype=float)
        conf = np.zeros(33, dtype=float)
        if res.pose_landmarks is None:
            return kps, conf

        # visibility always comes from the image-space landmarks.
        image_lms = res.pose_landmarks.landmark
        if self.use_world_landmarks and res.pose_world_landmarks is not None:
            world_lms = res.pose_world_landmarks.landmark
            for i, lm in enumerate(world_lms):
                kps[i] = (lm.x, lm.y, lm.z)  # metres, origin at mid-hip
                conf[i] = image_lms[i].visibility
        else:
            for i, lm in enumerate(image_lms):
                kps[i] = (lm.x * w, lm.y * h, lm.z * w)
                conf[i] = lm.visibility
        return kps, conf

    def close(self) -> None:
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()