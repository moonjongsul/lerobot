"""Auxiliary heads on the VLM prefix: subtask, value, status.

They read the prefix -- images, prompt, state -- and not the action
expert's suffix, so they are available before any action is denoised and
cost one linear layer each.

Two of them replace things SmolVLA does not have:

* **Subtask.** Lets the policy write its own prompt at runtime. Applied one
  chunk late (the head sees the previous step's prefix), which is safe
  because action discontinuity at subtask boundaries is only 1.25x the
  within-segment norm and the demonstrators do not pause there.

* **Value.** The failure detector. Two failure modes need catching and they
  present differently: a bad `place` is a visible wrong outcome, while a
  failed `flip` looks normal throughout and only shows up as progress that
  never arrives. A single distributional value head covers both -- it drops
  sharply on the first and drifts down on the second.

The value heads are distributional (201 bins, cross-entropy) rather than
scalar regressions. Early in a run a successful and a failing attempt are
genuinely indistinguishable, so the correct prediction is bimodal; a
regression head collapses that to a midpoint that describes neither, and
the spread -- which is the early warning -- is lost.

A frame-level binary "mistake" classifier is deliberately absent. Its label
covers a whole failed run including the frames before anything went wrong,
so it fires from the attempt's first frame. That label is used as prompt
conditioning instead, where being retrospective is fine.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_mvla import MVLAConfig

# status head classes
STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE = 0, 1, 2
STATUS_NAMES = ("running", "success", "failure")


def value_bin_centers(n_bins: int, device=None, dtype=torch.float32) -> Tensor:
    """Bin centres spanning [-1, 0], the normalised value range."""
    return torch.linspace(-1.0, 0.0, n_bins, device=device, dtype=dtype)


def discretize_value(value: Tensor, n_bins: int) -> Tensor:
    """Values in [-1, 0] -> bin indices, for the cross-entropy target."""
    v = value.clamp(-1.0, 0.0)
    return ((v + 1.0) * (n_bins - 1)).round().long().clamp(0, n_bins - 1)


class DistributionalValueHead(nn.Module):
    """Linear map to a distribution over value bins."""

    def __init__(self, hidden_size: int, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Linear(hidden_size, n_bins)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features)

    def expected_value(self, logits: Tensor) -> Tensor:
        probs = logits.softmax(-1)
        centers = value_bin_centers(self.n_bins, logits.device, probs.dtype)
        return (probs * centers).sum(-1)

    def spread(self, logits: Tensor) -> Tensor:
        """Std of the predicted distribution.

        High spread means the model cannot yet tell a success from a
        failure. Watching it rise is a different, earlier signal than
        watching the mean fall, and it is the one that flags an attempt
        whose outcome is still genuinely undecided.
        """
        probs = logits.softmax(-1)
        centers = value_bin_centers(self.n_bins, logits.device, probs.dtype)
        mean = (probs * centers).sum(-1, keepdim=True)
        var = (probs * (centers - mean) ** 2).sum(-1)
        return var.clamp_min(0).sqrt()

    def loss(self, logits: Tensor, target_value: Tensor) -> Tensor:
        target = discretize_value(target_value, self.n_bins)
        return F.cross_entropy(logits, target)


class MVLAHeads(nn.Module):
    """All auxiliary heads, built from a config."""

    def __init__(self, config: MVLAConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.subtask = (
            nn.Linear(hidden_size, config.num_subtasks) if config.use_subtask_head else None
        )
        self.value_subtask = (
            DistributionalValueHead(hidden_size, config.value_bins)
            if config.use_value_heads
            else None
        )
        self.value_episode = (
            DistributionalValueHead(hidden_size, config.value_bins)
            if config.use_value_heads
            else None
        )
        self.status = nn.Linear(hidden_size, 3) if config.use_status_head else None
        if self.status is not None:
            self.register_buffer(
                "status_weights", torch.tensor(config.status_class_weights), persistent=False
            )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """`features` is the pooled prefix, (batch, hidden)."""
        out: dict[str, Tensor] = {}
        if self.subtask is not None:
            out["subtask_logits"] = self.subtask(features)
        if self.value_subtask is not None:
            out["value_subtask_logits"] = self.value_subtask(features)
            out["value_episode_logits"] = self.value_episode(features)
        if self.status is not None:
            out["status_logits"] = self.status(features)
        return out

    def losses(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Per-head losses, already scaled by their configured weights.

        Heads whose target is missing from the batch are skipped rather
        than fed zeros, so a partially annotated batch trains the heads it
        can and leaves the others alone.
        """
        cfg = self.config
        losses: dict[str, Tensor] = {}

        if "subtask_logits" in outputs and "subtask_index" in targets:
            target = targets["subtask_index"].long()
            valid = target >= 0
            if valid.any():
                losses["subtask"] = cfg.subtask_loss_weight * F.cross_entropy(
                    outputs["subtask_logits"][valid], target[valid]
                )

        if "value_subtask_logits" in outputs and "value_subtask" in targets:
            losses["value_subtask"] = cfg.value_subtask_loss_weight * self.value_subtask.loss(
                outputs["value_subtask_logits"], targets["value_subtask"]
            )
        if "value_episode_logits" in outputs and "value_episode" in targets:
            losses["value_episode"] = cfg.value_episode_loss_weight * self.value_episode.loss(
                outputs["value_episode_logits"], targets["value_episode"]
            )

        if "status_logits" in outputs and "status" in targets:
            losses["status"] = cfg.status_loss_weight * F.cross_entropy(
                outputs["status_logits"],
                targets["status"].long(),
                weight=self.status_weights.to(outputs["status_logits"].dtype),
            )
        return losses

    @torch.no_grad()
    def interpret(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Head outputs as the quantities the task manager consumes."""
        read: dict[str, Tensor] = {}
        if "subtask_logits" in outputs:
            read["subtask"] = outputs["subtask_logits"].argmax(-1)
        if "value_subtask_logits" in outputs:
            logits = outputs["value_subtask_logits"]
            read["value_subtask"] = self.value_subtask.expected_value(logits)
            read["value_subtask_spread"] = self.value_subtask.spread(logits)
        if "value_episode_logits" in outputs:
            read["value_episode"] = self.value_episode.expected_value(
                outputs["value_episode_logits"]
            )
        if "status_logits" in outputs:
            read["status"] = outputs["status_logits"].argmax(-1)
            read["status_probs"] = outputs["status_logits"].softmax(-1)
        return read
