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

import logging

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

# Batch keys the auxiliary heads read their targets from, and the key the
# elapsed-time channel arrives under. Produced by data/derive.py and
# attached by the dataset wrapper.
#
# The `observation.` prefix is load-bearing, not cosmetic. The preprocessor
# turns each batch into an `EnvTransition` via `batch_to_transition`, which
# keeps only keys under that prefix plus a fixed allowlist (task, index,
# episode_index, *_is_pad). Plain names like "subtask_index" are dropped
# there, silently: the heads would then find no targets, contribute no
# loss, and train on nothing while the run looks healthy.
TARGET_KEYS = (
    "observation.subtask_index",
    "observation.value_subtask",
    "observation.value_episode",
    "observation.status",
)
ELAPSED_KEY = "observation.elapsed_subtask"

# Head-loss target names, in the order `TARGET_KEYS` lists them.
TARGET_NAMES = ("subtask_index", "value_subtask", "value_episode", "status")


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

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """SmolVLA's state, with elapsed-time-in-subtask appended.

        Appended here rather than in the dataset because normalisation runs
        between the two: its statistics describe the dataset's own state
        channels, so an extra one added upstream would not line up. This
        channel is already scaled to roughly [0, 1] by the adapter and wants
        no further normalising.

        The result is zero-padded to `max_state_dim` downstream, so adding a
        channel changes no shape the rest of the model sees.
        """
        raw = batch.get(OBS_STATE)
        expected = self.config.robot_state_feature.shape[0]
        if raw is not None and raw.shape[-1] != expected:
            raise ValueError(
                f"observation.state has {raw.shape[-1]} channels but the policy "
                f"declares {expected}. The elapsed-subtask channel belongs in its "
                "own batch key, not appended upstream: normalisation statistics "
                "only cover the recorded channels."
            )

        state = super().prepare_state(batch)
        elapsed = batch.get(ELAPSED_KEY)
        if not self.config.use_elapsed_subtask or elapsed is None:
            return state
        elapsed = elapsed.to(state.dtype).reshape(state.shape[0], 1)
        # `super()` already padded to max_state_dim; write the channel into
        # the first padding slot so the real state keeps its layout.
        width = self.config.robot_state_feature.shape[0]
        if width < state.shape[-1]:
            state = state.clone()
            state[:, width : width + 1] = elapsed
            return state
        return torch.cat([state, elapsed], dim=-1)

    def _prefix_feature_dim(self) -> int:
        """Width of the pooled prefix the heads read.

        The prefix's final hidden states, so the text model's hidden size.
        Both paths that feed the heads -- `encode_prefix` at inference and
        the capture during training -- pool the same tensor, so there is one
        width and the heads cannot silently see two different features.
        """
        text_config = self.model.vlm_with_expert.get_vlm_model().config.text_config
        return int(text_config.hidden_size)

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
        outputs, _ = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        # The prefix's final hidden states, (batch, positions, hidden).
        # This is the same tensor the training capture takes, so the heads
        # read one feature whichever path produced it.
        prefix_out = outputs[0]

        mask = pad_masks[:, : prefix_out.shape[1]].unsqueeze(-1).to(prefix_out.dtype)
        pooled = (prefix_out * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return pooled.to(torch.float32)

    def head_outputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.heads(self.encode_prefix(batch))

    # ────────────────────────────────────── prefix sharing during training
    def _forward_with_prefix(
        self, batch: dict[str, Tensor], noise, time, reduction: str
    ) -> tuple[Tensor, dict, Tensor | None]:
        """SmolVLA's loss, plus the prefix it already computed on the way.

        `super().forward()` runs the VLM over the prefix and throws the
        result away -- it only wants the suffix. Calling `encode_prefix`
        afterwards would run that same VLM a second time, and since both
        activations have to stay alive for backward, training needs close
        to twice the memory. At batch 90 on an H200 that is the difference
        between fitting and an OOM at step 0.

        So the prefix is captured from the pass that already happens. The
        hook swaps `use_cache` on for the one call, which changes nothing
        about the attention maths -- the flag only decides whether the keys
        and values are kept -- and then pools the cached values exactly as
        `encode_prefix` does, so the heads see the same 320-wide feature
        either way.

        Inference keeps using `encode_prefix`: `read_state` is called on
        its own at ~30 Hz with no action to denoise, so there is no forward
        pass to share and nothing to save.
        """
        captured: dict[str, Tensor] = {}
        model = self.model
        vlm_forward = model.vlm_with_expert.forward
        embed_prefix = model.embed_prefix

        def capturing_embed_prefix(*args, **kwargs):
            embs, pad_masks, att_masks = embed_prefix(*args, **kwargs)
            # The prefix's own padding mask, kept so the pooling below can
            # cut the cache back to prefix positions without recomputing it.
            captured["pad_masks"] = pad_masks
            return embs, pad_masks, att_masks

        def capturing_vlm(*args, **kwargs):
            # The prefix hidden states this pass already produced. SmolVLA's
            # training forward destructures the result as `(_, suffix_out)`
            # and drops them on the floor; taking them here costs nothing.
            #
            # `use_cache` / `fill_kv_cache` are left exactly as SmolVLA set
            # them. They are not independent knobs: `fill_kv_cache` is the
            # first term of the layer-routing condition in
            # `VLMWithExpert.forward`, so forcing it true sends every layer
            # down the self-attention path and feeds the expert VLM-width
            # states, while `use_cache=True` with it false makes the layers
            # try to read a cache that was never filled. Either way training
            # dies at step 0.
            outputs, past_key_values = vlm_forward(*args, **kwargs)
            if outputs and outputs[0] is not None:
                captured["prefix"] = outputs[0]
            return outputs, past_key_values

        model.embed_prefix = capturing_embed_prefix
        model.vlm_with_expert.forward = capturing_vlm
        try:
            loss, loss_dict = super().forward(batch, noise=noise, time=time, reduction=reduction)
        finally:
            model.embed_prefix = embed_prefix
            model.vlm_with_expert.forward = vlm_forward

        pooled = None
        prefix_out = captured.get("prefix")
        pad_masks = captured.get("pad_masks")
        if prefix_out is not None and pad_masks is not None:
            # The cache spans prefix *and* suffix positions, so it is cut
            # back to the prefix first -- pooling over the suffix would mix
            # the action expert's states into a feature that is supposed to
            # describe the observation.
            pad_masks = pad_masks.to(torch.bool)
            prefix_out = prefix_out[:, : pad_masks.shape[1]]
            mask = pad_masks[:, : prefix_out.shape[1]].unsqueeze(-1).to(prefix_out.dtype)
            pooled = ((prefix_out * mask).sum(1) / mask.sum(1).clamp_min(1.0)).to(torch.float32)
        return loss, loss_dict, pooled

    # ─────────────────────────────────────────────────────────── training
    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """Flow-matching loss plus every auxiliary loss the batch supports."""
        wanted = self._heads_enabled() and any(k in batch for k in TARGET_KEYS)
        if not wanted:
            # Nothing to add: take SmolVLA's path untouched, which is what
            # makes the stage-1 baseline exact.
            return super().forward(batch, noise=noise, time=time, reduction=reduction)

        loss, loss_dict, pooled = self._forward_with_prefix(batch, noise, time, reduction)
        if pooled is None:
            # The capture did not fire (a SmolVLA internal changed shape).
            # Falling back keeps training correct at twice the prefix cost,
            # and says so once rather than failing at step 0.
            logging.warning("MVLA: prefix capture missed; recomputing for the heads")
            pooled = self.encode_prefix(batch)

        outputs = self.heads(pooled)
        targets = {
            name: batch[key]
            for key, name in zip(TARGET_KEYS, TARGET_NAMES, strict=True)
            if key in batch
        }
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
