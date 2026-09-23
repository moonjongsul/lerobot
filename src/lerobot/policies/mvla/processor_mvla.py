"""Prompt assembly for MVLA.

The prompt carries everything about *how* a demonstration was performed
that the images and the task sentence do not. Following pi0.7, it is plain
text with per-field dropout:

    Task: kit the black plastic object into the white tray.
    Subtask: flip black plastic object upside down.
    Speed: 1500. Quality: 5. Mistake: false.

Text rather than learned embeddings so each field can be dropped
independently, which is what makes classifier-free guidance possible at
inference -- you cannot guide on a conditioning token you have no
unconditional counterpart for.

The inference values are the *best* values, not the average ones: quality
at its maximum, mistake false, speed at the 15th percentile of episode
length. Training on suboptimal demonstrations then helps rather than hurts,
because the metadata tells the model which ones were suboptimal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .configuration_mvla import MVLAConfig

ADVANTAGE_POSITIVE = "positive"
ADVANTAGE_NEGATIVE = "negative"


@dataclass
class PromptFields:
    """Everything that can go in a prompt, before dropout."""

    task: str
    subtask: str | None = None
    speed_bin: int | None = None
    quality: int | None = None
    mistake: bool | None = None
    advantage: str | None = None


def quality_from_score(score: float, bins: int = 5) -> int:
    """Episode score in [0, 1] -> an integer grade in [1, bins]."""
    return int(max(1, min(bins, round(score * bins))))


def speed_bin(n_frames: int, bin_size: int = 500) -> int:
    """Episode length rounded to a coarse bucket, as pi0.7 does.

    Coarse so that the model learns "fast" versus "slow" rather than
    memorising exact episode lengths.
    """
    return int(round(n_frames / bin_size) * bin_size)


def target_speed_bin(lengths: list[int], percentile: float = 15.0, bin_size: int = 500) -> int:
    """Speed to ask for at inference: brisk but demonstrated.

    The 15th percentile of observed lengths -- fast enough to be worth
    asking for, slow enough that the model has examples of it.
    """
    import numpy as np

    return speed_bin(int(np.percentile(lengths, percentile)), bin_size)


class PromptBuilder:
    """Builds training and inference prompts from a config."""

    def __init__(self, config: MVLAConfig, rng: random.Random | None = None):
        self.config = config
        self.rng = rng or random.Random()

    # ────────────────────────────────────────────────────────── training
    def build_training(
        self,
        recorded_task: str,
        neutral_task: str,
        fields: PromptFields,
        has_subgoal: bool = False,
    ) -> str:
        """Prompt for one training sample, with dropout applied.

        `recorded_task` and `neutral_task` are both passed so the choice is
        made here rather than in the data loader: it is a conditioning
        decision, and keeping it next to the other dropout keeps the policy
        auditable in one place.
        """
        cfg = self.config
        task = (
            neutral_task
            if self.rng.random() < cfg.neutral_prompt_prob
            else recorded_task
        )
        parts = [f"Task: {task}."]

        if cfg.use_subtask_prompt and fields.subtask:
            drop = has_subgoal and self.rng.random() < cfg.subgoal_drop_subtask_prob
            if not drop:
                subtask = fields.subtask
                if self.rng.random() < cfg.subtask_corruption_prob:
                    # Blank rather than wrong: teaches the policy to fall back
                    # on vision, without teaching it that the field lies.
                    subtask = None
                if subtask:
                    parts.append(f"Subtask: {subtask}.")

        if cfg.use_metadata_prompt and self.rng.random() >= cfg.metadata_dropout_prob:
            p = cfg.metadata_field_dropout_prob
            if fields.speed_bin is not None and self.rng.random() >= p:
                parts.append(f"Speed: {fields.speed_bin}.")
            if fields.quality is not None and self.rng.random() >= p:
                parts.append(f"Quality: {fields.quality}.")
            if (
                cfg.use_mistake_prompt
                and fields.mistake is not None
                and self.rng.random() >= p
            ):
                parts.append(f"Mistake: {str(fields.mistake).lower()}.")

        if (
            cfg.use_advantage_prompt
            and fields.advantage is not None
            and self.rng.random() >= cfg.advantage_dropout_prob
        ):
            parts.append(f"Advantage: {fields.advantage}.")

        return " ".join(parts)

    # ───────────────────────────────────────────────────────── inference
    def build_inference(
        self,
        neutral_task: str,
        subtask: str | None = None,
        speed_bin_value: int | None = None,
    ) -> str:
        """Prompt to run with: neutral task, best metadata.

        Always the neutral task -- asking for a flip in the prompt would
        hand the model the decision it is supposed to make from the image.
        """
        cfg = self.config
        parts = [f"Task: {neutral_task}."]
        if cfg.use_subtask_prompt and subtask:
            parts.append(f"Subtask: {subtask}.")
        if cfg.use_metadata_prompt:
            if speed_bin_value is not None:
                parts.append(f"Speed: {speed_bin_value}.")
            parts.append(f"Quality: {cfg.quality_bins}.")
            if cfg.use_mistake_prompt:
                parts.append("Mistake: false.")
        if cfg.use_advantage_prompt:
            parts.append(f"Advantage: {ADVANTAGE_POSITIVE}.")
        return " ".join(parts)

    def build_unconditional(self, neutral_task: str) -> str:
        """Prompt with the guided fields removed, for classifier-free guidance."""
        return f"Task: {neutral_task}."
