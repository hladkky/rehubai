"""Pose-estimation backends with a common ``detect(frame) -> (kps, conf)`` API.

Each backend wraps one of the repository's existing detection prototypes so the
validation core stays independent of the detector. Backends are imported lazily
to avoid importing heavy optional dependencies (mediapipe, mmpose, ultralytics)
unless actually used.
"""

from __future__ import annotations


def get_backend(name: str, **kwargs):
    """Instantiate a backend by name.

    Args:
        name: One of ``"mediapipe"``.
        **kwargs: Forwarded to the backend constructor.
    """
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipePoseBackend
        return MediaPipePoseBackend(**kwargs)
    raise ValueError(f"Unknown backend '{name}'")
