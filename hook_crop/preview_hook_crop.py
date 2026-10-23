"""Visual check of HookCropper on a few episodes: full frames with crop rectangles + the composed hook image."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from hook_crop import HookCropper, crop_rect, detect_poles  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

REPO = "yunjuyoung64/smolvla_hang_cup_red_black_rot6d_merged"
REV = "14088f58d3bcf921c216e07ae0a2d4b4ab2dc954"
OUT = Path(__file__).parent / "preview"
OUT.mkdir(exist_ok=True)
FONT = ImageFont.load_default(size=16)
STRIDE = 5
SHOW = 6

ds = LeRobotDataset(REPO, revision=REV)
ep = ds.meta.episodes
tasks = [t[0] for t in ep["tasks"]]
episodes = [0, 150, 299, 80, 230, 399]
cropper = HookCropper()


def to_np(x):
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


for e in episodes:
    lo, hi = ep["dataset_from_index"][e], ep["dataset_to_index"][e]
    cropper.reset()
    frames, head_boxes, n_miss = [], [], 0
    for k, idx in enumerate(range(lo, hi, STRIDE)):
        item = ds[idx]
        head, wrist = to_np(item["observation.images.head"]), to_np(item["observation.images.wrist"])
        prev_wrist = cropper.wrist_box
        comp = cropper(head, wrist)
        n_miss += int(cropper.wrist_box is prev_wrist and k > 0)
        (hb,) = detect_poles(cropper.model, cropper.processor, [head], 0.5)
        head_boxes.append(hb)
        frames.append((idx, head, wrist, comp, cropper.head_box, cropper.wrist_box))

    hb = np.array([b for b in head_boxes if b is not None])
    jitter = np.abs(hb - np.median(hb, axis=0)).max(axis=0) if len(hb) else None
    print(
        f"ep{e} [{tasks[e]}] frames={len(frames)} head detected {len(hb)}/{len(frames)} "
        f"max |head box - median| px={np.round(jitter, 1) if jitter is not None else '-'} wrist held(miss)={n_miss}"
    )

    picks = np.linspace(0, len(frames) - 1, SHOW).astype(int)
    rows = []
    for i in picks:
        idx, head, wrist, comp, hbox, wbox = frames[i]
        p = cropper.params
        tiles = []
        for img, box, min_h in ((head, hbox, p.head_min_height_frac), (wrist, wbox, p.wrist_min_height_frac)):
            vis = Image.fromarray(img)
            d = ImageDraw.Draw(vis)
            if box is not None:
                d.rectangle(box, outline=(0, 255, 0), width=2)
            d.rectangle(crop_rect(box, *img.shape[:2], min_h, p.height_scale), outline=(255, 255, 0), width=4)
            tiles.append(vis.resize((424, 240)))
        tiles.append(Image.fromarray(comp).resize((240, 240), Image.NEAREST))
        row = Image.new("RGB", (424 * 2 + 240, 240), "white")
        x = 0
        for t in tiles:
            row.paste(t, (x, 0))
            x += t.width
        ImageDraw.Draw(row).text((4, 2), f"ep{e} frame {idx - lo}", fill=(255, 255, 0), font=FONT)
        rows.append(row)
    canvas = Image.new("RGB", (rows[0].width, 240 * len(rows)), "white")
    for i, r in enumerate(rows):
        canvas.paste(r, (0, 240 * i))
    canvas.save(OUT / f"ep{e:03d}.png")
print("saved to", OUT)
