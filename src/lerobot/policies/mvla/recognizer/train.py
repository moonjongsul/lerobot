"""Train and validate the state recognizer.

    python -m lerobot.policies.mvla.recognizer.train \
        --dataset /path/to/xarm7_kitting_260923 \
        --out outputs/mvla_recognizer

Runs end to end: sample the frames the episode structure implies, decode
them, embed them with a frozen backbone, cross-validate under the protocol
in `evaluate.py`, then fit the final probes on everything and save.

Validation is not optional here and not a separate command. A recognizer
that scores well because it memorised the room looks identical to a working
one until it reaches a different room, so the placebo margin is printed
next to every headline number and a thin margin fails the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..data.segments import subtask_names
from . import crops as crop_module
from .dataset import CAMERAS, build_samples, extract_frames, frame_path
from .evaluate import evaluate_task, random_cv_predict, aggregate_by_unit
from .model import FrozenEncoder, LinearProbe, RecognizerConfig, StateRecognizer

# Which crop each task is validated with, and its no-object control.
TASK_CROPS = {"mat": ("mat", "placebo"), "tray": ("tray", "placebo")}
# Label treated as the positive class when reporting AUC / AP.
TASK_POSITIVE = {"mat": "target", "tray": "bad"}
# Below this gap between the real crop and its placebo, the result is not
# evidence about the object.
#
# The mat task's placebo sits higher than the tray's (~0.81 vs ~0.71) and
# that is expected rather than a leak: its `empty` class means the object is
# in the gripper, which puts the arm somewhere the corner crop can see. Over
# the two on-mat classes alone the same control drops to ~0.55. Read the
# margin, not the placebo score on its own.
MIN_PLACEBO_MARGIN = 0.15


def embed(
    samples, cache_dir: Path, crop_name: str, encoder: FrozenEncoder
) -> np.ndarray:
    """(N, D*cameras) features for one crop variant, cameras concatenated."""
    cropper = crop_module.CROPPERS[crop_name]
    per_camera = []
    for camera in CAMERAS:
        images, missed = [], 0
        for sample in samples:
            bgr = cv2.imread(str(frame_path(cache_dir, sample, camera)))
            if bgr is None:
                raise FileNotFoundError(frame_path(cache_dir, sample, camera))
            cropped, found = cropper(bgr, encoder.cfg.image_size)
            missed += not found
            images.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        if missed:
            print(f"    {crop_name}/{camera}: {missed}/{len(samples)} detections missed")
        per_camera.append(encoder(images))
    return np.concatenate(per_camera, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--cache", type=Path, default=None, help="decoded frame cache (default: <out>/frames)"
    )
    parser.add_argument("--backbone", default=RecognizerConfig.backbone)
    parser.add_argument("--image-size", type=int, default=RecognizerConfig.image_size)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-placebo", action="store_true", help="skip the control (not recommended)"
    )
    args = parser.parse_args()

    cfg = RecognizerConfig(
        backbone=args.backbone, image_size=args.image_size, device=args.device
    )
    cache_dir = args.cache or (args.out / "frames")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"dataset   {args.dataset}")
    print(f"backbone  {cfg.backbone}  @{cfg.image_size}px  cameras={list(CAMERAS)}")
    print(f"subtasks  {subtask_names(args.dataset)}")

    samples = build_samples(args.dataset)
    decoded = extract_frames(args.dataset, samples, cache_dir)
    print(f"samples   {len(samples)}  ({decoded} frames decoded, rest cached)")

    encoder = FrozenEncoder(cfg)
    results, probes, report = {}, {}, {}

    for task, (real_crop, placebo_crop) in TASK_CROPS.items():
        subset = [s for s in samples if s.task == task]
        labels = np.array([s.label for s in subset])
        units = np.array([s.run_id for s in subset])
        groups = np.array([s.group for s in subset])
        print(f"\n[{task}] {len(subset)} frames / {len(set(units.tolist()))} runs")
        for name, count in sorted(zip(*np.unique(labels, return_counts=True), strict=True)):
            print(f"    {name:14s} {count}")

        features = embed(subset, cache_dir, real_crop, encoder)
        placebo = (
            None if args.skip_placebo else embed(subset, cache_dir, placebo_crop, encoder)
        )

        result = evaluate_task(
            task=task,
            features=features,
            labels=labels,
            units=units,
            groups=groups,
            positive=TASK_POSITIVE[task],
            placebo_features=placebo,
            n_blocks=args.blocks,
            probe_c=cfg.probe_c,
        )
        results[task] = result
        print(result.summary())

        # For `mat`, also report the control over the two on-mat classes
        # alone. The full placebo is inflated by `empty`, where the object is
        # in the gripper and the arm's position is visible in any crop; this
        # restriction is the honest test of whether the object itself is
        # being read.
        on_mat = None
        if task == "mat" and placebo is not None:
            keep = labels != "empty"
            if keep.any() and len(set(labels[keep].tolist())) > 1:
                on_mat = evaluate_task(
                    task="mat(on-mat only)",
                    features=features[keep],
                    labels=labels[keep],
                    units=units[keep],
                    groups=groups[keep],
                    positive=TASK_POSITIVE[task],
                    placebo_features=placebo[keep],
                    n_blocks=args.blocks,
                    probe_c=cfg.probe_c,
                )
                print("  " + on_mat.summary().replace("\n", "\n  "))

        # Shuffled folds alongside, to keep the size of the leak visible.
        classes = sorted(set(labels.tolist()))
        shuffled = random_cv_predict(features, labels, groups, probe_c=cfg.probe_c)
        keep = ~np.isnan(shuffled).any(axis=1)
        truth, probs = aggregate_by_unit(units[keep], labels[keep], shuffled[keep], classes)
        optimistic = float((np.array(classes)[probs.argmax(1)] == truth).mean())
        print(f"  shuffled-fold accuracy {optimistic * 100:.1f}%  (optimistic, for contrast)")

        if result.operating_points is not None and task == "tray":
            print("  operating points:")
            print(result.operating_points.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

        probes[task] = LinearProbe(cfg, cfg.tasks[task]).fit(features, labels)
        report[task] = {
            "units": result.n_units,
            "positive": result.n_positive,
            "accuracy": result.accuracy,
            "auc": result.auc,
            "average_precision": result.average_precision,
            "placebo_auc": result.placebo_auc,
            "placebo_margin": result.placebo_margin,
            "shuffled_fold_accuracy": optimistic,
            "per_class": result.per_class,
        }
        if on_mat is not None:
            report[task]["on_mat_only"] = {
                "accuracy": on_mat.accuracy,
                "auc": on_mat.auc,
                "placebo_auc": on_mat.placebo_auc,
                "placebo_margin": on_mat.placebo_margin,
            }

    StateRecognizer(cfg, probes).save(args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nsaved to {args.out}")

    suspect = [
        task
        for task, result in results.items()
        if result.placebo_margin is not None and result.placebo_margin < MIN_PLACEBO_MARGIN
    ]
    if suspect:
        print(
            f"\nWARNING: placebo margin below {MIN_PLACEBO_MARGIN} for {suspect}. "
            "The score is not evidence about the object -- check the crop."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
