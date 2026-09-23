"""Dataset wrapper: assembled prompt and head targets on every sample.

Wrapping rather than subclassing `LeRobotDataset`, so the dataset is built
by LeRobot's own factory exactly as it is for SmolVLA and this layer only
rewrites what comes out of `__getitem__`. Attribute access falls through,
so the training loop's `dataset.meta`, `.num_frames` and friends keep
working.

The wrapper is not applied at all when every head and prompt flag is off:
`wrap_dataset` returns the dataset unchanged, so the stage-1 baseline runs
through SmolVLA's path with nothing of MVLA's in it. That is what makes
stage 1 a real control rather than an approximate one.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from ..configuration_mvla import MVLAConfig
from ..modeling_mvla import ELAPSED_KEY, TARGET_KEYS
from ..processor_mvla import PromptBuilder, PromptFields, quality_from_score, speed_bin
from .derive import derive_frame_targets, episode_metadata
from .segments import dataset_root, subtask_sentences

# The keys the heads read, imported rather than repeated: they carry an
# `observation.` prefix without which the preprocessor drops them before
# the policy ever sees them, and two copies of that rule would be one copy
# too many.
SUBTASK_KEY, VALUE_SUBTASK_KEY, VALUE_EPISODE_KEY, STATUS_KEY = TARGET_KEYS


class MVLADatasetWrapper(Dataset):
    """A LeRobotDataset whose samples carry MVLA's prompt and targets."""

    def __init__(self, dataset, config: MVLAConfig, root=None, seed: int = 0):
        self.dataset = dataset
        self.config = config
        self.root = dataset_root(root if root is not None else dataset)
        self.builder = PromptBuilder(config, random.Random(seed))

        targets = derive_frame_targets(self.root)
        # Indexed the way the dataset indexes: a flat frame counter in
        # (episode, frame) order, which is the order `derive_frame_targets`
        # returns and the order LeRobot's global `index` column follows.
        self._subtask = targets["subtask_index"].to_numpy(dtype=np.int64)
        self._value_subtask = targets["value_subtask"].to_numpy(dtype=np.float32)
        self._value_episode = targets["value_episode"].to_numpy(dtype=np.float32)
        self._status = targets["status"].to_numpy(dtype=np.int64)
        self._elapsed = targets["elapsed_subtask"].to_numpy(dtype=np.float32)
        self._mistake = targets["mistake"].to_numpy(dtype=bool)
        self._episode_of = targets["episode_index"].to_numpy(dtype=np.int64)

        meta = episode_metadata(self.root).set_index("episode")
        self._task = meta["task"].to_dict()
        self._neutral = meta["neutral"].to_dict()
        self._quality = {
            ep: quality_from_score(float(s), config.quality_bins)
            for ep, s in meta["score"].items()
        }
        self._speed = {
            ep: speed_bin(int(n), config.speed_bin_frames) for ep, n in meta["length"].items()
        }
        self._subtask_sentence = subtask_sentences(self.root)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name):
        # Only reached for attributes this wrapper does not define, so the
        # training loop's dataset.meta / .num_frames / .episodes still work.
        return getattr(self.__dict__["dataset"], name)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        cfg = self.config

        # The dataset's own global frame counter, which is the row this
        # sample's targets live at. Falling back to `idx` covers a dataset
        # sliced to a subset of episodes, where the two differ.
        row = int(item["index"].item()) if "index" in item else int(idx)
        episode = int(self._episode_of[row])

        if self._prompting_enabled():
            item["task"] = self._build_prompt(row, episode)

        if cfg.use_elapsed_subtask:
            item[ELAPSED_KEY] = torch.tensor(
                float(self._elapsed[row]) / cfg.elapsed_scale, dtype=torch.float32
            )

        if cfg.use_subtask_head:
            item[SUBTASK_KEY] = torch.tensor(int(self._subtask[row]), dtype=torch.int64)
        if cfg.use_value_heads:
            item[VALUE_SUBTASK_KEY] = torch.tensor(
                float(self._value_subtask[row]), dtype=torch.float32
            )
            item[VALUE_EPISODE_KEY] = torch.tensor(
                float(self._value_episode[row]), dtype=torch.float32
            )
        if cfg.use_status_head:
            item[STATUS_KEY] = torch.tensor(int(self._status[row]), dtype=torch.int64)

        return item

    # ───────────────────────────────────────────────────────────── prompt
    def _build_prompt(self, row: int, episode: int) -> str:
        cfg = self.config
        subtask_index = int(self._subtask[row])
        fields = PromptFields(
            task=self._task.get(episode, ""),
            subtask=self._subtask_sentence.get(subtask_index) or None,
            speed_bin=self._speed.get(episode) if cfg.use_metadata_prompt else None,
            quality=self._quality.get(episode) if cfg.use_metadata_prompt else None,
            mistake=bool(self._mistake[row]) if cfg.use_mistake_prompt else None,
        )
        return self.builder.build_training(
            recorded_task=self._task.get(episode, ""),
            neutral_task=self._neutral.get(episode, self._task.get(episode, "")),
            fields=fields,
        )

    def _prompting_enabled(self) -> bool:
        cfg = self.config
        return bool(
            cfg.use_subtask_prompt
            or cfg.use_metadata_prompt
            or cfg.use_advantage_prompt
            or cfg.neutral_prompt_prob > 0.0
        )


def needs_wrapping(config: MVLAConfig) -> bool:
    """Does this config ask for anything the wrapper provides?"""
    return bool(
        config.use_subtask_head
        or config.use_value_heads
        or config.use_status_head
        or config.use_subtask_prompt
        or config.use_metadata_prompt
        or config.use_advantage_prompt
        or config.use_elapsed_subtask
        or config.neutral_prompt_prob > 0.0
    )


def wrap_dataset(dataset, config, root=None):
    """Wrap `dataset` for MVLA, or return it untouched.

    Untouched is the stage-1 case: with every head and prompt flag off
    there is nothing to add, and returning the original dataset keeps that
    baseline byte-for-byte SmolVLA's.
    """
    if not isinstance(config, MVLAConfig) or not needs_wrapping(config):
        return dataset
    return MVLADatasetWrapper(dataset, config, root=root)


def describe(dataset) -> str:
    """One line for the training log saying what the wrapper turned on."""
    if not isinstance(dataset, MVLADatasetWrapper):
        return "MVLA: dataset unwrapped (stage-1 baseline)"
    cfg = dataset.config
    on = [
        name
        for name, flag in (
            ("subtask-head", cfg.use_subtask_head),
            ("value-heads", cfg.use_value_heads),
            ("status-head", cfg.use_status_head),
            ("subtask-prompt", cfg.use_subtask_prompt),
            ("metadata-prompt", cfg.use_metadata_prompt),
            ("mistake-prompt", cfg.use_mistake_prompt),
            ("elapsed-subtask", cfg.use_elapsed_subtask),
            ("advantage-prompt", cfg.use_advantage_prompt),
        )
        if flag
    ]
    return (
        f"MVLA: wrapped {len(dataset)} frames; "
        f"neutral_prompt_prob={cfg.neutral_prompt_prob}; enabled: {', '.join(on) or 'none'}"
    )
