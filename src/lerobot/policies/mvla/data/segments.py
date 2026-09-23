"""Subtask runs, read off the recorded annotations.

A *run* is one contiguous attempt at one subtask. It is the unit everything
downstream is defined on -- values, status, the mistake flag, recognizer
samples -- because it is the unit the operators actually worked in: an
episode with a failed flip contains a failed `flip` run and a successful
retry, and collapsing those into one episode-level outcome throws away the
94 failures this dataset has and leaves the 6 episode-level ones.

Two conventions, fixed here so nothing downstream reinvents them:

* **Boundaries come from the per-frame `subtask_index` column**, not from
  the episode metadata's `subtask_start_frames`. The metadata splits a few
  adjacent runs of the same subtask that the frame column merges (1379
  segments against 1368 runs). The frame column is what the policy sees at
  training time, so it is what the targets are aligned to.

* **A run failed if its `subtask_score` reaches 0 at any point**, not just
  at its last frame. The editor marks the moment the attempt went wrong,
  which is not always the final frame -- scoring only the end misses 7 of
  the 94 failures. Intermediate scores between 0 and 1 are quality, not
  failure, and feed the prompt instead.

Those two give 1368 runs: 1274 successful, 94 failed, over 284562 frames.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

# Score at or below which a run counts as failed. Exact zero in practice;
# a tolerance so float round-trips through parquet cannot flip a label.
FAILURE_SCORE = 1e-6


def dataset_root(root) -> Path:
    """Accept a path, a string, or a LeRobotDataset.

    Tested by type rather than by `hasattr(root, "root")`: `Path` has a
    `.root` of its own ("/"), so duck-typing here silently reads the
    filesystem root instead of the dataset.
    """
    if isinstance(root, (str, Path)):
        return Path(root)
    return Path(root.root)


@functools.lru_cache(maxsize=4)
def _episodes_meta(root_str: str) -> pd.DataFrame:
    root = Path(root_str)
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def episodes_frame(root) -> pd.DataFrame:
    """Per-episode metadata: index, task sentence, length."""
    meta = _episodes_meta(str(dataset_root(root)))
    out = pd.DataFrame(
        {
            "episode": meta["episode_index"].astype(int),
            "length": meta["length"].astype(int),
        }
    )
    # `tasks` is a list with one entry per episode in this dataset.
    out["task"] = [t[0] if len(t) else "" for t in meta["tasks"]]
    return out


@functools.lru_cache(maxsize=4)
def subtask_names(root_str: str | Path) -> dict[int, str]:
    """subtask index -> short name, from `meta/subtasks.parquet`.

    The recorded names are sentences ("approach to black plastic object for
    flip"); the short names the planner and recognizer use are derived from
    them here so the two vocabularies cannot drift apart.
    """
    root = Path(str(root_str))
    table = pd.read_parquet(root / "meta" / "subtasks.parquet")
    out: dict[int, str] = {}
    for sentence, row in table.iterrows():
        out[int(row["subtask_index"])] = _short_name(str(sentence))
    return out


def subtask_sentences(root) -> dict[int, str]:
    """subtask index -> the recorded sentence, for the prompt."""
    root = dataset_root(root)
    table = pd.read_parquet(root / "meta" / "subtasks.parquet")
    return {int(row["subtask_index"]): str(sentence) for sentence, row in table.iterrows()}


def _short_name(sentence: str) -> str:
    """Recorded sentence -> the short label used in code.

    The mapping is on the verb and, for approaches, on which subtask the
    approach is for; that is the only distinction the sentences carry.
    """
    s = sentence.lower().strip()
    if not s:
        return "unlabelled"
    if s.startswith("approach"):
        return "approach_flip" if "flip" in s else "approach_pick"
    for verb in ("flip", "pick", "move", "place"):
        if s.startswith(verb):
            return verb
    return s.split()[0]


@functools.lru_cache(maxsize=4)
def load_frames(root_str: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Per-frame table, sorted by (episode, frame).

    Cached: the adapter, the recognizer and the checks all read it, and it
    is ~285k rows of a handful of columns.
    """
    root = Path(root_str)
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data files under {root / 'data'}")
    # The sort keys are always read, whether or not the caller asked for
    # them: the row order is what every derived target is aligned to, so a
    # projection that dropped them would silently return unordered rows.
    cols = None
    if columns:
        cols = list(dict.fromkeys(["episode_index", "frame_index", *columns]))
    frame = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return frame.sort_values(["episode_index", "frame_index"], ignore_index=True)


def _load_frames(root, columns) -> pd.DataFrame:
    """`load_frames` with the caller's argument shapes made hashable."""
    return load_frames(str(dataset_root(root)), tuple(columns) if columns else None)


@functools.lru_cache(maxsize=4)
def _runs_cached(root_str: str) -> pd.DataFrame:
    frames = load_frames(
        root_str, ("episode_index", "frame_index", "subtask_index", "subtask_score")
    )
    rows = []
    run_id = 0
    for episode, group in frames.groupby("episode_index", sort=True):
        index = group["subtask_index"].to_numpy()
        score = group["subtask_score"].to_numpy(dtype=np.float64)
        if len(index) == 0:
            continue
        # Change points of the subtask column: the run boundaries.
        starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
        ends = np.r_[starts[1:], len(index)]  # exclusive
        for start, stop in zip(starts, ends, strict=True):
            segment = score[start:stop]
            rows.append(
                {
                    "run_id": run_id,
                    "episode": int(episode),
                    "subtask": int(index[start]),
                    "start": int(start),
                    "end": int(stop - 1),  # inclusive, as the checks expect
                    "n_frames": int(stop - start),
                    "score": float(segment[-1]),
                    "min_score": float(segment.min()),
                    "failed": bool(segment.min() <= FAILURE_SCORE),
                }
            )
            run_id += 1
    runs = pd.DataFrame(rows)
    runs["order"] = runs.groupby("episode").cumcount()
    return runs


def runs_frame(root) -> pd.DataFrame:
    """Every subtask run in the dataset, one row each.

    Columns: run_id, episode, subtask, start, end (both inclusive, relative
    to the episode), n_frames, score, min_score, failed, order.
    """
    return _runs_cached(str(dataset_root(root))).copy()


def episode_failed(root) -> pd.Series:
    """Per-episode flag: did any run in it fail?"""
    runs = runs_frame(root)
    return runs.groupby("episode")["failed"].any()
