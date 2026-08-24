"""Open-dataset adapters.

Each adapter converts a public rehabilitation dataset into the unified keypoint
representation consumed by :class:`rehubai.validator.ExerciseValidator`:

    * keypoints as an ``(T, J, D)`` array (frames x joints x 2|3),
    * a *semantic* skeleton scheme registered in :mod:`rehubai.joints`,

so a single exercise config validates the dataset skeletons without any
per-exercise model training. See :mod:`rehubai.datasets.uiprmd`.
"""

from __future__ import annotations

__all__ = ["uiprmd"]
