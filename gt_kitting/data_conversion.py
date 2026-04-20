"""
Convert joint-space LeRobot dataset to EEF task-space representations.

Generates two new branches from the original joint-space dataset:
  - eef_quat:  position(3) + quaternion(4) + gripper(1) = 8D
  - eef_rot6d: position(3) + rot6d(6) + gripper(1) = 10D

FK is computed via pinocchio using the FR3 URDF.
Gripper action is binarized: >= 0.8 -> 1.0, < 0.8 -> 0.0
Gripper state is kept as-is.

Usage:
    python gt_kitting/data_conversion.py --repo-id moonjongsul/manufacturing_kitting_dataset
    python gt_kitting/data_conversion.py --repo-id moonjongsul/manufacturing_kitting_dataset --mode eef_quat
    python gt_kitting/data_conversion.py --repo-id moonjongsul/manufacturing_kitting_dataset --mode eef_rot6d
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin
from huggingface_hub import HfApi, snapshot_download
from scipy.spatial.transform import Rotation
from tqdm import tqdm

URDF_PATH = Path(__file__).parent / "fr3.urdf"
TCP_FRAME = "fr3_link8"
GRIPPER_THRESHOLD = 0.8


class FR3Kinematics:
    """Forward kinematics for Franka FR3 using pinocchio."""

    def __init__(self, urdf_path: str | Path = URDF_PATH):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(TCP_FRAME)

    def forward(self, joint_positions_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FK for 7-DOF joint positions in radians.

        Returns:
            position: (3,) xyz in base frame
            rotation: (3, 3) rotation matrix
        """
        q = pin.neutral(self.model)
        q[:7] = joint_positions_rad[:7]

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        pose = self.data.oMf[self.frame_id]
        return pose.translation.copy(), pose.rotation.copy()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    return Rotation.from_matrix(R).as_quat()


def rotation_matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return np.concatenate([R[:, 0], R[:, 1]])


def binarize_gripper(gripper_value: float) -> float:
    """Binarize gripper action: >= 0.8 -> 1.0, < 0.8 -> 0.0"""
    return 1.0 if gripper_value >= GRIPPER_THRESHOLD else 0.0


def convert_joint_to_eef(
    joint_positions_rad: np.ndarray,
    gripper: float,
    fk: FR3Kinematics,
    mode: str,
    is_action: bool = False,
) -> np.ndarray:
    """
    Convert joint-space data to EEF task-space.

    Args:
        joint_positions_rad: (7,) joint angles in radians
        gripper: gripper value
        fk: FK solver
        mode: 'eef_quat' or 'eef_rot6d'
        is_action: if True, binarize gripper

    Returns:
        eef_quat:  (8,)  [x, y, z, qx, qy, qz, qw, gripper]
        eef_rot6d: (10,) [x, y, z, r1, r2, r3, r4, r5, r6, gripper]
    """
    pos, rot = fk.forward(joint_positions_rad)

    if is_action:
        gripper = binarize_gripper(gripper)

    if mode == "eef_quat":
        quat = rotation_matrix_to_quaternion(rot)
        return np.concatenate([pos, quat, [gripper]]).astype(np.float32)
    elif mode == "eef_rot6d":
        rot6d = rotation_matrix_to_rot6d(rot)
        return np.concatenate([pos, rot6d, [gripper]]).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_feature_info(mode: str) -> tuple[int, dict]:
    """Get output dimension and feature names for the target mode."""
    if mode == "eef_quat":
        dim = 8
        names = {
            "observation.state": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"],
            "action": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        }
    elif mode == "eef_rot6d":
        dim = 10
        names = {
            "observation.state": ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper_width"],
            "action": ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return dim, names


def convert_parquet(pq_path: Path, fk: FR3Kinematics, mode: str) -> int:
    """Convert a single parquet file in-place. Returns number of frames processed."""
    df = pd.read_parquet(pq_path)

    states = np.array(df["observation.state"].tolist())
    actions = np.array(df["action"].tolist())

    new_states = []
    new_actions = []

    for i in range(len(df)):
        eef_state = convert_joint_to_eef(states[i, :7], float(states[i, 7]), fk, mode, is_action=False)
        eef_action = convert_joint_to_eef(actions[i, :7], float(actions[i, 7]), fk, mode, is_action=True)
        new_states.append(eef_state.tolist())
        new_actions.append(eef_action.tolist())

    df["observation.state"] = new_states
    df["action"] = new_actions
    df.to_parquet(pq_path, index=False)

    return len(df)


def convert_dataset(repo_id: str, mode: str, output_dir: str | None = None):
    """
    Download dataset, convert parquet data to EEF space, update metadata,
    and push to a new HuggingFace branch. Videos are unchanged.
    """
    if output_dir is None:
        output_root = Path(f"/tmp/lerobot_conversion/{repo_id.replace('/', '_')}_{mode}")
    else:
        output_root = Path(output_dir) / mode

    # Step 1: Download source
    print(f"[1/4] Downloading source dataset: {repo_id} (main branch)")
    src_dir = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
    ))
    print(f"  Source: {src_dir}")

    # Step 2: Copy to working directory
    dst_dir = output_root / "dataset"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns(".git*"))
    print(f"[2/4] Working copy: {dst_dir}")

    # Step 3: Convert parquet files
    print(f"[3/4] Converting parquet data to {mode}...")
    fk = FR3Kinematics()
    dim, feature_names = get_feature_info(mode)

    parquet_files = sorted(dst_dir.glob("data/**/*.parquet"))
    total_frames = 0
    for pq_path in tqdm(parquet_files, desc="Converting"):
        total_frames += convert_parquet(pq_path, fk, mode)
    print(f"  Converted {total_frames} frames across {len(parquet_files)} files")

    # Update info.json
    info_path = dst_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"]["observation.state"]["shape"] = [dim]
    info["features"]["observation.state"]["names"] = feature_names["observation.state"]
    info["features"]["action"]["shape"] = [dim]
    info["features"]["action"]["names"] = feature_names["action"]

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print("  Updated info.json")

    # Recompute stats.json with correct dimensions
    print("  Recomputing stats.json...")
    all_states = []
    all_actions = []
    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        all_states.extend(df["observation.state"].tolist())
        all_actions.extend(df["action"].tolist())

    states_arr = np.array(all_states, dtype=np.float32)
    actions_arr = np.array(all_actions, dtype=np.float32)

    def _compute_stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }

    stats_path = dst_dir / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["observation.state"] = _compute_stats(states_arr)
    stats["action"] = _compute_stats(actions_arr)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Updated stats.json (state: {dim}D, action: {dim}D)")

    # Step 4: Push to HuggingFace
    print(f"[4/4] Pushing to HuggingFace branch: {mode}")
    api = HfApi()

    try:
        api.create_branch(repo_id=repo_id, branch=mode, repo_type="dataset")
        print(f"  Created branch: {mode}")
    except Exception:
        print(f"  Branch '{mode}' already exists")

    api.upload_folder(
        folder_path=str(dst_dir),
        repo_id=repo_id,
        repo_type="dataset",
        revision=mode,
        commit_message=f"Convert joint-space to {mode} via FK (pinocchio)",
    )

    print(f"\nDone! Dataset: {repo_id} (branch: {mode})")
    print(f"  observation.state [{dim}D]: {feature_names['observation.state']}")
    print(f"  action [{dim}D]: {feature_names['action']}")


def main():
    parser = argparse.ArgumentParser(description="Convert joint-space dataset to EEF task-space")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eef_quat", "eef_rot6d", "both"],
        default="both",
        help="Conversion mode (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local output directory",
    )
    args = parser.parse_args()

    if args.mode == "both":
        for m in ["eef_quat", "eef_rot6d"]:
            convert_dataset(args.repo_id, m, args.output_dir)
    else:
        convert_dataset(args.repo_id, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
