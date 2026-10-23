"""Build a copy of the cup-hang dataset with an extra `observation.images.hook` video (SAM3 hook-region crops).

Stage 1 (one process per GPU, resumable):
    CUDA_VISIBLE_DEVICES=1 python hook_crop/build_hook_dataset.py crops --shard 0 --num-shards 3
Stage 2 (after all shards finished):
    python hook_crop/build_hook_dataset.py finalize

The hook video of episode e is stored as its own file (chunk e // 1000, file e), so its timestamps start at 0.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from hook_crop import HOOK_IMAGE_KEY, OUT_SIZE, HookCropper, compose, detect_poles  # noqa: E402

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.io_utils import write_info, write_stats  # noqa: E402
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames  # noqa: E402

SRC_REPO = "yunjuyoung64/smolvla_hang_cup_red_black_rot6d_merged"
SRC_REV = "14088f58d3bcf921c216e07ae0a2d4b4ab2dc954"
DST_ROOT = Path("/data/keti/mjs/lerobot/datasets/kcare_hang_cup_red_black_rot6d_hookcrop")
WORK = DST_ROOT.parent / f"{DST_ROOT.name}_work"
CAMS = ("observation.images.head", "observation.images.wrist")
DECODE_CHUNK = 64
DETECT_BATCH = 16
HOOK_FEATURE = {
    "dtype": "video",
    "shape": [OUT_SIZE, OUT_SIZE, 3],
    "names": ["height", "width", "channels"],
}


def hook_video_rel(ep: int) -> Path:
    return Path(f"videos/{HOOK_IMAGE_KEY}/chunk-{ep // 1000:03d}/file-{ep % 1000:03d}.mp4")


def decode_episode(ds: LeRobotDataset, ep: int, cam: str, timestamps: np.ndarray):
    path = ds.root / ds.meta.get_video_file_path(ep, cam)
    offset = ds.meta.episodes[ep][f"videos/{cam}/from_timestamp"]
    for i in range(0, len(timestamps), DECODE_CHUNK):
        ts = (offset + timestamps[i : i + DECODE_CHUNK]).tolist()
        frames = decode_video_frames(path, ts, ds.tolerance_s, ds.video_backend)
        yield from (f.permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy() for f in frames)


def build_episode(ds: LeRobotDataset, cropper: HookCropper, ep: int, timestamps: np.ndarray) -> None:
    head = list(decode_episode(ds, ep, CAMS[0], timestamps))
    wrist = list(decode_episode(ds, ep, CAMS[1], timestamps))
    thr = cropper.params.score_threshold

    # Same semantics as HookCropper.__call__ run frame by frame, but batched:
    # head box = first detection in the episode, wrist box = latest detection (held when missed).
    cropper.reset()
    for i in range(0, len(head), DETECT_BATCH):
        for box in detect_poles(cropper.model, cropper.processor, head[i : i + DETECT_BATCH], thr):
            if box is not None:
                cropper.head_box = box
                break
        if cropper.head_box is not None:
            break
    wrist_boxes = []
    for i in range(0, len(wrist), DETECT_BATCH):
        wrist_boxes += detect_poles(cropper.model, cropper.processor, wrist[i : i + DETECT_BATCH], thr)

    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        tmp = Path(tmp)
        paths, log = [], []
        for k, (h, w, wb) in enumerate(zip(head, wrist, wrist_boxes, strict=True)):
            if wb is not None:
                cropper.wrist_box = wb
            head_rect, wrist_rect = cropper.rects(h.shape[:2], w.shape[:2])
            p = tmp / f"frame-{k:06d}.png"
            Image.fromarray(compose(h, w, head_rect, wrist_rect)).save(p)
            paths.append(str(p))
            log.append({"frame_index": k, "head_rect": head_rect, "wrist_rect": wrist_rect, "wrist_detected": wb is not None})

        video = WORK / hook_video_rel(ep)
        encode_video_frames(tmp, video, ds.fps, overwrite=True)
        stats = compute_episode_stats({HOOK_IMAGE_KEY: paths}, {HOOK_IMAGE_KEY: HOOK_FEATURE})[HOOK_IMAGE_KEY]

    pd.DataFrame(log).to_parquet(WORK / "rects" / f"ep{ep:03d}.parquet")
    np.savez(WORK / "stats" / f"ep{ep:03d}.npz", **stats)


def run_crops(shard: int, num_shards: int) -> None:
    for d in ("rects", "stats"):
        (WORK / d).mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset(SRC_REPO, revision=SRC_REV)
    cropper = HookCropper()
    frames = pd.concat(
        pd.read_parquet(f, columns=["episode_index", "timestamp"]) for f in sorted((ds.root / "data").glob("*/*.parquet"))
    )
    ts_by_ep = frames.groupby("episode_index")["timestamp"].apply(lambda s: s.to_numpy(np.float64))
    episodes = [e for e in range(ds.meta.total_episodes) if e % num_shards == shard]
    for ep in tqdm(episodes, desc=f"shard {shard}"):
        if (WORK / "stats" / f"ep{ep:03d}.npz").exists():
            continue
        build_episode(ds, cropper, ep, ts_by_ep[ep])


def run_finalize() -> None:
    ds = LeRobotDataset(SRC_REPO, revision=SRC_REV)
    n = ds.meta.total_episodes
    missing = [e for e in range(n) if not (WORK / "stats" / f"ep{e:03d}.npz").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} episodes have no hook crops yet, e.g. {missing[:10]}")

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    shutil.copytree(ds.root, DST_ROOT, ignore=shutil.ignore_patterns(".cache"))
    shutil.copytree(WORK / "videos" / HOOK_IMAGE_KEY, DST_ROOT / "videos" / HOOK_IMAGE_KEY)

    ep_stats = [{HOOK_IMAGE_KEY: dict(np.load(WORK / "stats" / f"ep{e:03d}.npz"))} for e in range(n)]

    info = json.loads((DST_ROOT / "meta/info.json").read_text())
    head_info = info["features"][CAMS[0]]["info"]
    info["features"][HOOK_IMAGE_KEY] = {
        **HOOK_FEATURE,
        "info": {**head_info, "video.height": OUT_SIZE, "video.width": OUT_SIZE},
    }
    write_info(info, DST_ROOT)

    stats = json.loads((DST_ROOT / "meta/stats.json").read_text())
    stats = {k: {s: np.asarray(v) for s, v in d.items()} for k, d in stats.items()}
    stats[HOOK_IMAGE_KEY] = aggregate_stats(ep_stats)[HOOK_IMAGE_KEY]
    write_stats(stats, DST_ROOT)

    for f in sorted((DST_ROOT / "meta/episodes").glob("*/*.parquet")):
        df = pd.read_parquet(f)
        eps = df["episode_index"].to_numpy()
        prefix = f"videos/{HOOK_IMAGE_KEY}"
        df[f"{prefix}/chunk_index"] = eps // 1000
        df[f"{prefix}/file_index"] = eps % 1000
        df[f"{prefix}/from_timestamp"] = 0.0
        df[f"{prefix}/to_timestamp"] = df["length"].to_numpy() / ds.fps
        for stat in ep_stats[0][HOOK_IMAGE_KEY]:
            df[f"stats/{HOOK_IMAGE_KEY}/{stat}"] = [ep_stats[e][HOOK_IMAGE_KEY][stat].tolist() for e in eps]
        df.to_parquet(f)

    check = LeRobotDataset("local/hookcrop_check", root=DST_ROOT)
    for idx in (0, len(check) // 2, len(check) - 1):
        item = check[idx]
        print(idx, {k: tuple(item[k].shape) for k in (*CAMS, HOOK_IMAGE_KEY)})
    print("done:", DST_ROOT, "episodes", check.meta.total_episodes, "frames", check.meta.total_frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--shard", type=int, required=True)
    c.add_argument("--num-shards", type=int, required=True)
    sub.add_parser("finalize")
    args = ap.parse_args()
    if args.cmd == "crops":
        run_crops(args.shard, args.num_shards)
    else:
        run_finalize()
