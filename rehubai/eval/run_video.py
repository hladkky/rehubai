"""End-to-end runner: validate an exercise in a video file.

Reads a video, runs a pose backend frame-by-frame through
:class:`~rehubai.validator.ExerciseValidator`, and prints the aggregate report
plus a sample of the most important corrective feedback. This is the end-to-end
verification path for own recordings (closes TODO#1: "validate dead bug on real
video").

Usage:
    python -m rehubai.eval.run_video \
        --video sources/videos/dead_bug.MOV \
        --config configs/exercises/dead_bug.json \
        --backend mediapipe [--frame-step 2] [--max-frames N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from ..backends import get_backend
from ..config import ExerciseConfig
from ..feedback import interpret_violations
from ..validator import ExerciseValidator


def run_video(
    video_path: str,
    config_path: str,
    backend_name: str = "mediapipe",
    frame_step: int = 1,
    max_frames: int | None = None,
    world_landmarks: bool = False,
) -> dict:
    """Process ``video_path`` and return the validator report dict."""
    import cv2

    cfg = ExerciseConfig.load(config_path)
    backend_kwargs = {}
    if backend_name == "mediapipe" and world_landmarks:
        backend_kwargs["use_world_landmarks"] = True
    backend = get_backend(backend_name, **backend_kwargs)
    scheme = getattr(backend, "scheme", None)
    validator = ExerciseValidator(cfg, scheme=scheme)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    feedback_counter: Counter = Counter()
    frame_idx = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_step == 0:
                kps, conf = backend.detect(frame)
                res = validator.process_frame(kps, conf, frame_idx=frame_idx)
                for msg in interpret_violations(res.violations, top_k=1):
                    feedback_counter[msg] += 1
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
        if hasattr(backend, "close"):
            backend.close()

    report = validator.generate_report()
    report["top_feedback"] = feedback_counter.most_common(5)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate an exercise video end-to-end.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", default="mediapipe")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--world",
        action="store_true",
        help="use MediaPipe metric world landmarks (true 3D) for angle checks",
    )
    args = ap.parse_args()

    report = run_video(
        args.video, args.config, args.backend, args.frame_step, args.max_frames,
        world_landmarks=args.world,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
