"""MVLAPolicy: SmolVLA plus auxiliary heads on the VLM prefix.

Subclassing rather than forking. The flow-matching action expert, the
prefix construction and the RTC handling are SmolVLA's and stay its; what
MVLA adds is a pooled read of the prefix feeding a subtask head, two
distributional value heads and a status head, plus their losses.

Keeping the action path untouched matters for staging: stage 1 is a
SmolVLA baseline, and MVLA with every head disabled must reproduce it
exactly. If it does not, later comparisons measure the refactor rather than
the heads.

The heads also replace what Knowledge Insulation does in pi0.6/0.7. There,
the VLM backbone is supervised with discrete FAST-token cross-entropy while
the action expert's gradients are stopped, so the backbone trains on a
stable objective. SmolVLA has no such path and lets flow-matching gradients
into the backbone directly. The subtask and value cross-entropies give the
backbone that stable discrete signal back.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from .configuration_mvla import MVLAConfig
from .heads import MVLAHeads

# Batch keys the auxiliary heads read their targets from. Produced by
# data/derive.py and attached by the dataset pipeline.
TARGET_KEYS = ("subtask_index", "value_subtask", "value_episode", "status")


class MVLAPolicy(SmolVLAPolicy):
    """SmolVLA with subtask, value and status heads."""

    config_class = MVLAConfig
    name = "mvla"

    def __init__(self, config: MVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: MVLAConfig = config
        self.heads = MVLAHeads(config, self._prefix_feature_dim())
        # Cache of the last head read, so the runtime can ask for the value
        # estimate without a second forward pass.
        self._last_read: dict[str, Tensor] = {}

    def _prefix_feature_dim(self) -> int:
        """Width of the pooled prefix the heads read.

        `encode_prefix` pools the last layer's cached value states, which
        are grouped-query attention values -- `num_key_value_heads` of them,
        not `num_attention_heads` -- so the width is smaller than the text
        model's hidden size.
        """
        text_config = self.model.vlm_with_expert.get_vlm_model().config.text_config
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )
        kv_heads = getattr(text_config, "num_key_value_heads", None) or (
            text_config.num_attention_heads
        )
        return int(head_dim * kv_heads)

    # ──────────────────────────────────────────────────────── prefix read
    def encode_prefix(self, batch: dict[str, Tensor]) -> Tensor:
        """Pooled prefix embedding: images + prompt + state.

        Pooled over the unpadded positions only. The prefix is right-padded
        to a fixed length, so a plain mean would dilute short prompts more
        than long ones and make the heads sensitive to prompt length.
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        # `make_att_2d_masks` builds the 2-D mask by multiplying the padding
        # masks together, so an integer mask propagates through and reaches
        # attention as int64, where `torch.where` rejects it. Dataloaders
        # hand these over as long often enough to be worth normalising here.
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(torch.bool)

        embs, pad_masks, att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        pad_masks = pad_masks.to(torch.bool)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # Same call SmolVLA makes to populate its prefix cache. The stack
        # only accepts a None suffix on the cache-filling path, so this is
        # the supported way to run the prefix alone; the cache it returns
        # carries the per-layer keys and values we pool over.
        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        # Last layer's values, (batch, positions, kv_heads, head_dim),
        # flattened to one vector per position.
        last = past_key_values[max(past_key_values)]["value_states"]
        prefix_out = last.flatten(2)

        mask = pad_masks[:, : prefix_out.shape[1]].unsqueeze(-1).to(prefix_out.dtype)
        pooled = (prefix_out * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return pooled.to(torch.float32)

    def head_outputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.heads(self.encode_prefix(batch))

    # ─────────────────────────────────────────────────────────── training
    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """Flow-matching loss plus every auxiliary loss the batch supports."""
        loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)

        if not self._heads_enabled() or not any(k in batch for k in TARGET_KEYS):
            return loss, loss_dict

        outputs = self.head_outputs(batch)
        targets = {k: batch[k] for k in TARGET_KEYS if k in batch}
        aux = self.heads.losses(outputs, targets)
        for name, value in aux.items():
            loss_dict[f"loss_{name}"] = value.item()

        if aux:
            total = sum(aux.values())
            loss_dict["loss_aux"] = total.item()
            # `reduction="none"` returns per-sample losses; broadcasting the
            # scalar auxiliary term keeps that shape intact.
            loss = loss + total
            loss_dict["loss"] = (
                loss.mean().item() if loss.ndim else loss.item()
            )
        return loss, loss_dict

    def _heads_enabled(self) -> bool:
        cfg = self.config
        return cfg.use_subtask_head or cfg.use_value_heads or cfg.use_status_head

    # ────────────────────────────────────────────────────────── inference
    @torch.no_grad()
    def read_state(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Head predictions for the task manager: subtask, value, status.

        Separate from `select_action` so the runtime can poll the value
        estimate at action rate without paying for a denoising pass.
        """
        self._last_read = self.heads.interpret(self.head_outputs(batch))
        return self._last_read

    @property
    def last_read(self) -> dict[str, Tensor]:
        return self._last_read

    def get_optim_params(self) -> dict:
        """Head parameters train alongside the rest, at the same rate.

        No separate group: the heads are single linear layers, so there is
        nothing for a different learning rate to buy, and one group keeps
        checkpoints comparable with the SmolVLA baseline.
        """
        return super().get_optim_params()
