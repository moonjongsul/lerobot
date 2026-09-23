"""Per-frame targets for the auxiliary heads, and the neutral task prompts.

Pure functions of `segments`: nothing here reads a model or a config, so
`tests/test_against_dataset.py` can check the targets against the recorded
demonstrations directly.

The value convention is time-to-go, normalised to [-1, 0]: 0 at the moment
the thing succeeds, -1 at its start (and -1 throughout a run that fails).
Negative-and-up rather than 0-to-1 because it makes "the value is falling"
mean the same thing for both heads, and because the failure floor is then a
constant the head can actually represent -- a discounted-return convention
would put it at a value that depends on the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .segments import (
    dataset_root,
    episodes_frame,
    load_frames,
    runs_frame,
    subtask_names,
    subtask_sentences,
)

# Status classes, mirroring heads.py.
STATUS_RUNNING, STATUS_SUCCESS, STATUS_FAILURE = 0, 1, 2

# Neutral task sentences, keyed by the goal they describe.
#
# The recorded prompts name the method -- "flip it upside down, then pick it
# up" -- which tells the policy the very thing it is supposed to read off
# the image. Measured on this dataset the recorded wording predicts whether
# a flip is needed 99.7% of the time; the neutral wording says only what
# outcome is wanted, so the flip decision has to come from vision.
#
# Keyed by goal rather than by recorded sentence because the planner names
# a goal, not an episode: `rules.plan()` looks its prompt up here. The
# recorded sentences reach these through `TASK_GOALS` below.
NEUTRAL_PROMPTS: dict[str, str] = {
    "kit": "kit the black plastic object into the white tray",
    "pick": "pick up the black plastic object",
    "place": "kit the black plastic object into the white tray",
}

# Recorded task sentence -> the goal it asks for. Unrecognised sentences
# raise in `build_neutral_prompts` rather than silently keeping their leak.
TASK_GOALS: dict[str, str] = {
    "grasp the black plastic object, flip it upside down, then pick it up": "pick",
    "flip the black plastic object upside down then pick it up, move it to the "
    "white tray, and place it in the tray": "kit",
    "place the black plastic object in the white tray": "kit",
    "pick up the black plastic object": "pick",
    "pick up the black plastic object, move it to the white tray, and place it "
    "in the tray": "kit",
}


def neutral_for_task(sentence: str) -> str:
    """Recorded task sentence -> the neutral sentence to train and run with."""
    return NEUTRAL_PROMPTS[TASK_GOALS[sentence]]


@dataclass
class NeutralPromptMapping:
    """Recorded -> neutral sentences, with the leak each wording carries.

    `leak_before` / `leak_after` are the accuracy of the best single rule
    that guesses "does this episode need a flip?" from the task sentence
    alone. The point of the rewording is to drive it down to the base rate,
    and reporting both is what makes that checkable.
    """

    mapping: dict[str, str]
    leak_before: float
    leak_after: float


def build_neutral_prompts(root) -> NeutralPromptMapping:
    """Neutral sentences for this dataset's tasks, with the leak measured."""
    root = dataset_root(root)
    episodes = episodes_frame(root)
    runs = runs_frame(root)
    names = subtask_names(root)

    # Ground truth: did this episode contain a flip?
    flipped = (
        runs.assign(is_flip=runs["subtask"].map(names).eq("flip"))
        .groupby("episode")["is_flip"]
        .any()
    )
    episodes = episodes.join(flipped.rename("needs_flip"), on="episode")
    episodes["needs_flip"] = episodes["needs_flip"].fillna(False)

    missing = sorted(set(episodes["task"]) - set(TASK_GOALS))
    if missing:
        raise KeyError(
            "no goal mapped for task sentence(s): "
            + "; ".join(repr(m) for m in missing)
        )
    episodes["neutral"] = episodes["task"].map(neutral_for_task)

    return NeutralPromptMapping(
        mapping={s: neutral_for_task(s) for s in TASK_GOALS},
        leak_before=_leak(episodes["task"], episodes["needs_flip"]),
        leak_after=_leak(episodes["neutral"], episodes["needs_flip"]),
    )


def _leak(sentences: pd.Series, needs_flip: pd.Series) -> float:
    """Accuracy of the best per-sentence guess at `needs_flip`.

    Each distinct sentence is allowed its own answer -- the strongest rule
    available to something reading only the prompt -- so this is an upper
    bound on what the wording gives away, not an average case.
    """
    frame = pd.DataFrame({"sentence": sentences.to_numpy(), "flip": needs_flip.to_numpy()})
    correct = (
        frame.groupby("sentence")["flip"]
        .apply(lambda g: max(int(g.sum()), int((~g).sum())))
        .sum()
    )
    return float(correct) / max(len(frame), 1)


def derive_frame_targets(root) -> pd.DataFrame:
    """One row per frame, carrying every target the heads read.

    Columns:
        episode_index, frame_index, run_id, subtask_index
        value_subtask   time-to-go within the run, in [-1, 0]
        value_episode   time-to-go within the episode, in [-1, 0]
        status          running / success / failure, terminal frames only
        elapsed_subtask seconds since the run began, scaled by `elapsed_scale`
        mistake         did this frame's run fail (retrospective)
        subtask_score, episode_score, episode_length
    """
    root = dataset_root(root)
    frames = load_frames(
        str(root), ("episode_index", "frame_index", "subtask_index", "subtask_score")
    )
    runs = runs_frame(root)
    fps = _fps(root)

    n = len(frames)
    run_id = np.full(n, -1, dtype=np.int64)
    value_subtask = np.zeros(n, dtype=np.float32)
    value_episode = np.zeros(n, dtype=np.float32)
    status = np.zeros(n, dtype=np.int64)
    elapsed = np.zeros(n, dtype=np.float32)
    mistake = np.zeros(n, dtype=bool)

    # Frames are sorted by (episode, frame), so each episode is one slice
    # and a run's absolute position is that slice's offset plus its start.
    episode_start = (
        frames.groupby("episode_index", sort=True)["frame_index"].count().cumsum().shift(1).fillna(0)
    ).astype(int)
    episode_length = frames.groupby("episode_index", sort=True)["frame_index"].count()

    for run in runs.itertuples():
        base = int(episode_start[run.episode])
        lo, hi = base + run.start, base + run.end + 1
        run_id[lo:hi] = run.run_id
        mistake[lo:hi] = run.failed

        steps = np.arange(run.n_frames, dtype=np.float32)
        elapsed[lo:hi] = steps / fps
        if run.failed:
            # No progress to credit: a failed attempt is worth the floor
            # throughout. Crediting its early frames would teach the value
            # head that the approach was going fine, which is exactly the
            # judgement the head exists to make.
            value_subtask[lo:hi] = -1.0
        else:
            # -1 at the first frame, 0 at the last.
            value_subtask[lo:hi] = -(1.0 - steps / max(run.n_frames - 1, 1))

        status[hi - 1] = STATUS_FAILURE if run.failed else STATUS_SUCCESS

    # Episode value: time-to-go to the end of the episode, monotone by
    # construction. Unlike the subtask value it is not floored on failure --
    # a failed run is retried, and the episode does still finish.
    for episode, length in episode_length.items():
        base = int(episode_start[episode])
        steps = np.arange(length, dtype=np.float32)
        value_episode[base : base + length] = -(1.0 - steps / max(length - 1, 1))

    out = pd.DataFrame(
        {
            "episode_index": frames["episode_index"].to_numpy(),
            "frame_index": frames["frame_index"].to_numpy(),
            "run_id": run_id,
            "subtask_index": frames["subtask_index"].to_numpy(),
            "value_subtask": value_subtask,
            "value_episode": value_episode,
            "status": status,
            "elapsed_subtask": elapsed,
            "mistake": mistake,
            "subtask_score": frames["subtask_score"].to_numpy(),
        }
    )
    return out


def _fps(root) -> float:
    import json

    info = json.loads((dataset_root(root) / "meta" / "info.json").read_text())
    return float(info.get("fps", 30.0))


def episode_metadata(root) -> pd.DataFrame:
    """Per-episode prompt metadata: task, neutral task, quality, speed.

    Quality is the episode's own score, and speed is its length. Both are
    binned by the prompt builder rather than here, so the bin widths stay a
    config decision.
    """
    root = dataset_root(root)
    episodes = episodes_frame(root)
    frames = load_frames(str(root), ("episode_index", "observation.score"))
    score = frames.groupby("episode_index")["observation.score"].min()
    episodes = episodes.join(score.rename("score"), on="episode")
    episodes["neutral"] = [
        neutral_for_task(t) if t in TASK_GOALS else t for t in episodes["task"]
    ]
    failed = runs_frame(root).groupby("episode")["failed"].any()
    episodes = episodes.join(failed.rename("failed"), on="episode")
    episodes["failed"] = episodes["failed"].fillna(False)
    return episodes


def subtask_prompt_table(root) -> dict[int, str]:
    """subtask index -> the sentence that goes in the prompt."""
    return subtask_sentences(root)
