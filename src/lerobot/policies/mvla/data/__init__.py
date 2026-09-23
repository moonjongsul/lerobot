"""Data layer: subtask runs, derived head targets, and the dataset wrapper.

Three modules, in the order they depend on each other:

    segments.py   reads the recorded annotations -- where each subtask run
                  starts and ends, and whether it failed. Everything else
                  is derived from this, so the definitions live here and
                  nowhere else.

    derive.py     per-frame targets for the auxiliary heads, plus the
                  neutral task prompts. Pure functions of `segments`, so
                  they can be checked against the dataset without a model.

    adapter.py    wraps a LeRobotDataset so each sample carries the
                  assembled prompt and those targets. A no-op when every
                  head and prompt flag is off, which is what makes the
                  stage-1 SmolVLA baseline exact.
"""

from .adapter import MVLADatasetWrapper, describe, wrap_dataset
from .derive import build_neutral_prompts, derive_frame_targets
from .segments import runs_frame, subtask_names

__all__ = [
    "MVLADatasetWrapper",
    "build_neutral_prompts",
    "derive_frame_targets",
    "describe",
    "runs_frame",
    "subtask_names",
    "wrap_dataset",
]
