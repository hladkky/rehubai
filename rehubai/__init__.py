"""RehubAI — universal, config-driven validation of rehabilitation exercises.

Public API:

    >>> from rehubai import ExerciseConfig, ExerciseValidator
    >>> cfg = ExerciseConfig.load("configs/exercises/dead_bug.json")
    >>> val = ExerciseValidator(cfg, scheme="blazepose33")
    >>> result = val.process_frame(keypoints, confidences)
    >>> report = val.generate_report()

The algorithm validates three deviation classes simultaneously — geometric
(joint angles), sequential (stage order) and dynamic (transition tempo) — from a
declarative JSON config, without training a model per exercise.
"""

from .angles import calculate_angle, vector_angle
from .config import ExerciseConfig
from .feedback import interpret_violations
from .quality import compute_quality
from .validator import ExerciseValidator, FrameResult

__all__ = [
    "ExerciseConfig",
    "ExerciseValidator",
    "FrameResult",
    "calculate_angle",
    "vector_angle",
    "compute_quality",
    "interpret_violations",
]

__version__ = "0.1.0"
