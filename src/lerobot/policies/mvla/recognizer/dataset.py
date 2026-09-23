"""Recognizer training samples, read straight out of the episode structure.

No hand labelling. The editor already answered both questions when it cut
the episodes into subtask runs:

  mat_state       an operator who reached for a flip saw an object that
                  needed flipping; one who reached straight for a pick saw
                  a pickable one. So the first frames of `approach_flip`
                  and `approach_pick` are labelled `target` / `flipped`
                  respectively, and the frames of `move` / `place`, where
                  the object is in the gripper, are `empty`.

  tray_placement  the `place` run's own score.

Two sampling decisions carry the Task B result and are easy to get wrong:

* **Frames are taken just after the gripper opens, not at the run's end.**
  A place run ends ~3 s after release, by which time the wrist has pulled
  back far enough that the part it just placed drifts into the corner used
  as the no-object control -- which is how that control scored 0.960 and
  hid the confound. Sampling at release+8/18/28 keeps the wrist close, and
  the control drops to 0.718.

* **Samples are grouped by run, never by episode.** 32 episodes contain a
  failed place and its successful retry.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.segments import dataset_root, load_frames, runs_frame, subtask_names

# Offsets from the gripper-opening frame, in frames at 30 fps.
RELEASE_OFFSETS = (8, 18, 28)
# Frames into an approach run, while the arm is still clear of the object.
APPROACH_OFFSETS = (3, 12, 21)

MAT_CLASSES = ("empty", "target", "flipped", "unknown_pose")
TRAY_CLASSES = ("ok", "bad")

CAMERAS = ("wrist_front", "wrist_rear")


@dataclass(frozen=True)
class Sample:
    task: str  # "mat" | "tray"
    label: str
    episode: int
    frame: int
    run_id: int  # grouping unit for scoring
    group: int  # grouping unit for cross-validation (the episode)


def _flip_subtasks(root) -> tuple[set[int], set[int], set[int]]:
    """(approach-for-flip, approach-for-pick, in-gripper) subtask indices."""
    names = subtask_names(root)
    app_flip = {k for k, v in names.items() if v.startswith("approach") and "flip" in v}
    app_pick = {k for k, v in names.items() if v.startswith("approach") and "pick" in v}
    held = {k for k, v in names.items() if v in ("move", "place")}
    return app_flip, app_pick, held


def build_samples(root, max_empty_per_episode: int = 3) -> list[Sample]:
    """Every labelled recognizer sample in the dataset."""
    runs = runs_frame(root)
    app_flip, app_pick, held = _flip_subtasks(root)
    place_subtasks = {
        k for k, v in subtask_names(root).items() if v == "place"
    }
    release = _release_frames(root)

    out: list[Sample] = []
    for r in runs.itertuples():
        if r.subtask in app_flip or r.subtask in app_pick:
            label = "target" if r.subtask in app_flip else "flipped"
            for off in APPROACH_OFFSETS:
                f = r.start + off
                if f <= r.end:
                    out.append(Sample("mat", label, r.episode, f, r.run_id, r.episode))
        elif r.subtask in held:
            # Object is in the gripper, so the mat is empty. Sub-sampled: these
            # frames are plentiful and would otherwise swamp the other classes.
            step = max(r.n_frames // (max_empty_per_episode + 1), 1)
            for k in range(1, max_empty_per_episode + 1):
                f = r.start + k * step
                if f <= r.end:
                    out.append(Sample("mat", "empty", r.episode, f, r.run_id, r.episode))

        if r.subtask in place_subtasks and r.score in (0.0, 1.0):
            rel = release.get(r.run_id)
            if rel is None:
                continue
            label = "bad" if r.score == 0.0 else "ok"
            for off in RELEASE_OFFSETS:
                out.append(
                    Sample("tray", label, r.episode, min(rel + off, r.end), r.run_id, r.episode)
                )
    return out


def _release_frames(root) -> dict[int, int]:
    """Frame at which the gripper opens, per place run.

    Detected as the largest positive jump in the gripper channel of
    `observation.state`. Consistent across the dataset: it lands ~64% of the
    way through the run with ~3 s of retraction after it.
    """
    frames = load_frames(
        root, ["episode_index", "frame_index", "subtask_index", "observation.state"]
    )
    state = np.stack(frames["observation.state"].to_numpy())
    gripper = state[:, 6]
    runs = runs_frame(root)
    ep_start = frames.groupby("episode_index").apply(
        lambda g: g.index[0], include_groups=False
    )
    place = {k for k, v in subtask_names(root).items() if v == "place"}

    out: dict[int, int] = {}
    for r in runs.itertuples():
        if r.subtask not in place:
            continue
        base = int(ep_start[r.episode])
        seg = gripper[base + r.start : base + r.end + 1]
        if len(seg) < 10:
            continue
        out[r.run_id] = r.start + int(np.argmax(np.diff(seg)))
    return out


def samples_frame(root) -> pd.DataFrame:
    df = pd.DataFrame([s.__dict__ for s in build_samples(root)])
    return df


# ─────────────────────────────────────────────────────── frame extraction
def _video_index(root) -> dict[tuple[int, str], tuple[int, float]]:
    """(episode, camera) -> (video file index, start timestamp).

    Episodes are concatenated into shared mp4s, so reading a frame means
    seeking to `from_timestamp + frame / fps` in the right file.
    """
    root = dataset_root(root)
    meta = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    # Column names contain "/", which itertuples() cannot expose as attributes.
    out = {}
    episodes = meta["episode_index"].to_numpy()
    for cam in CAMERAS:
        files = meta[f"videos/observation.images.{cam}/file_index"].to_numpy()
        starts = meta[f"videos/observation.images.{cam}/from_timestamp"].to_numpy()
        for ep, fi, t0 in zip(episodes, files, starts, strict=True):
            out[(int(ep), cam)] = (int(fi), float(t0))
    return out


def extract_frames(
    root, samples: list[Sample], out_dir: Path, fps: float = 30.0, workers: int = 12
) -> int:
    """Decode every (sample, camera) frame to jpg under `out_dir`. Idempotent."""
    from concurrent.futures import ThreadPoolExecutor

    root = dataset_root(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = _video_index(root)

    jobs = []
    for s in samples:
        for cam in CAMERAS:
            dst = out_dir / f"{s.episode:04d}_{s.frame:05d}_{cam}.jpg"
            if dst.exists():
                continue
            file_index, t0 = index[(s.episode, cam)]
            src = root / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{file_index:03d}.mp4"
            jobs.append((str(src), t0 + s.frame / fps, str(dst)))

    def run(job):
        src, t, dst = job
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", src,
             "-vframes", "1", "-q:v", "3", "-y", dst],
            check=False, stderr=subprocess.DEVNULL,
        )

    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(run, jobs))
    return len(jobs)


def frame_path(out_dir: Path, sample: Sample, camera: str) -> Path:
    return Path(out_dir) / f"{sample.episode:04d}_{sample.frame:05d}_{camera}.jpg"
