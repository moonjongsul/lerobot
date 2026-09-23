"""MVLA configuration: SmolVLA plus the conditioning and heads it lacks.

Every flag below traces to something measured on xarm7_kitting_260923 or
stated in the pi0.6 / pi0.7 papers; the defaults are the settings those
measurements support.

Naming: the policy is registered as "mvla" so LeRobot's factory can build
it the same way it builds smolvla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("mvla")
@dataclass
class MVLAConfig(SmolVLAConfig):
    # ─────────────────────────────────────────────── prompt composition
    # Fraction of samples that get a task prompt naming the goal but not the
    # method. The recorded prompts say "flip it ... then pick it up", which
    # predicts whether a flip is needed 99.6% of the time, so a policy
    # trained only on them never learns to read the object's pose. Inference
    # always uses the neutral form, so training has to cover it.
    neutral_prompt_prob: float = 0.5

    # Include the current subtask sentence in the prompt (pi0.5 onward).
    use_subtask_prompt: bool = True
    # Replace the subtask with a blank or a wrong one this often, so the
    # policy degrades gracefully when the subtask head is wrong at runtime
    # rather than following a confident mistake off a cliff.
    subtask_corruption_prob: float = 0.2

    # Episode metadata, carried as plain text like pi0.7 rather than learned
    # embeddings -- text is what makes per-component dropout, and therefore
    # classifier-free guidance, possible at inference.
    use_metadata_prompt: bool = True
    metadata_dropout_prob: float = 0.15  # drop the whole block
    metadata_field_dropout_prob: float = 0.05  # drop one field
    quality_bins: int = 5
    speed_bin_frames: int = 500

    # Advantage conditioning (pi0.6 / RECAP). Off until a value function has
    # been fitted and rollouts exist to estimate advantages from; the prompt
    # slot is reserved now so enabling it later is not a retrain-everything
    # change.
    use_advantage_prompt: bool = False
    advantage_dropout_prob: float = 0.10

    # ──────────────────────────────────────────────────── state vector
    # Seconds since the current subtask began, appended to the state.
    # A single frame of "part held in the air" is identical at 3 s and 15 s,
    # so stall -- how flip failures actually present -- is invisible without
    # it. Known at inference from the subtask head, so it leaks nothing.
    use_elapsed_subtask: bool = True
    # Normaliser for that channel, in seconds.
    elapsed_scale: float = 30.0

    # ───────────────────────────────────────────────────── subgoal images
    # Goal images conditioning the policy (pi0.7). Wrist views only: the
    # environment camera is blocked by the arm for most of a manipulation,
    # and "seated in the pocket" is only legible from the wrist.
    use_subgoal_images: bool = False
    subgoal_cameras: tuple[str, ...] = ("wrist_front", "wrist_rear")
    subgoal_prob: float = 0.25  # fraction of samples given a subgoal
    subgoal_end_of_segment_prob: float = 0.40  # else uniform in the window
    subgoal_window_s: float = 2.0  # wrist views move fast; keep it short
    # Drop the subtask text when a subgoal image is present: the image says
    # the same thing in more detail, and dropping it teaches the policy to
    # use whichever one it is given.
    subgoal_drop_subtask_prob: float = 0.30

    # ───────────────────────────────────────────────────── auxiliary heads
    # Predicting the current subtask lets the policy compose its own prompt
    # at runtime. Applied one chunk late, which is safe: action jumps at
    # subtask boundaries are only 1.25x the within-segment norm, and the
    # demonstrators never pause there.
    use_subtask_head: bool = True
    num_subtasks: int = 7  # 6 labels + unlabelled
    subtask_loss_weight: float = 0.1

    # Distributional value heads. Distributional rather than scalar because
    # early in a run a successful and a failing attempt look identical; the
    # honest prediction is bimodal and a regression head would average the
    # modes into a value describing neither. The spread itself is the
    # early-warning signal.
    use_value_heads: bool = True
    value_bins: int = 201
    value_subtask_loss_weight: float = 0.1
    value_episode_loss_weight: float = 0.1

    # Terminal status. Defined per subtask run, not per episode: that is
    # what turns 6 episode-level failures into 94.
    use_status_head: bool = True
    status_loss_weight: float = 0.05
    # 94 failures against 1274 successes; without weighting the head learns
    # to answer "running" always.
    status_class_weights: tuple[float, float, float] = (1.0, 5.0, 15.0)

    # Retrospective per-run failure flag, used as a conditioning input only.
    # Not as a detection target: the label covers the whole run including
    # frames before anything went wrong, so a frame classifier fitted to it
    # fires from the first frame. Detection is the value head's job.
    use_mistake_prompt: bool = True

    optimizer_lr: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.neutral_prompt_prob <= 1.0:
            raise ValueError("neutral_prompt_prob must be in [0, 1]")
        if self.use_advantage_prompt and not self.use_value_heads:
            raise ValueError(
                "advantage conditioning needs a value function: "
                "set use_value_heads=True or use_advantage_prompt=False"
            )
        if self.use_subgoal_images and not self.subgoal_cameras:
            raise ValueError("use_subgoal_images=True but subgoal_cameras is empty")
        if self.value_bins < 2:
            raise ValueError("value_bins must be at least 2")

    @property
    def extra_state_dims(self) -> int:
        """State channels this config adds on top of the dataset's own."""
        return 1 if self.use_elapsed_subtask else 0
