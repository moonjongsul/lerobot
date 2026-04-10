#!/usr/bin/env python3
"""
ROS2 bag → LeRobot 데이터셋 변환 스크립트

토픽 구성:
  action : /gello/joint_states (7-DoF)
           /gripper/gripper_client/target_gripper_width_percent (1-DoF)
  state  : /joint_states (7-DoF)
           /dxl_parallel_gripper/joint_states (1-DoF)
  images : /wrist/front/color/image_raw/compressed
           /wrist/rear/color/image_raw/compressed
           /front_view/color/image_raw/compressed
           /side_view/color/image_raw/compressed

리샘플링: 30 Hz (nearest-neighbor)
이미지  : 원본 해상도 (720×1280, RGB)

Usage:
    python convert_rosbag_to_lerobot.py \\
        --bags-dir  /workspace/datasets/rosbag/pick_kit \\
        --output-dir /workspace/datasets/lerobot/manufacturing_parts_kitting_dataset \\
        --repo-id   moonjongsul/manufacturing_parts_kitting_dataset \\
        --warmup-sec 3.0 \\
        --private
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# lerobot 패키지 경로 추가 (editable install 환경)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_lerobot_dataset_card

# ─── 상수 ────────────────────────────────────────────────────────────────────
FPS = 30
TASK_DESCRIPTION = "Pick part, flip or rotate to correct orientation, and place into kitting tray."
ROBOT_TYPE = "franka_fr3"

TOPIC_ROBOT_ACTION  = "/gello/joint_states"
TOPIC_GRIPPER_ACTION = "/gripper/gripper_client/target_gripper_width_percent"
TOPIC_ROBOT_STATES  = "/joint_states"
TOPIC_GRIPPER_STATE = "/dxl_parallel_gripper/joint_states"

CAMERA_TOPICS = {
    "wrist_front": "/wrist/front/color/image_raw/compressed",
    "wrist_rear":  "/wrist/rear/color/image_raw/compressed",
    "front_view":  "/front_view/color/image_raw/compressed",
    "side_view":   "/side_view/color/image_raw/compressed",
}

ACTION_DIM = 8  # 7 gello joints + 1 gripper width
STATE_DIM  = 8  # 7 joint_states + 1 dxl gripper


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def build_timeline(start_ns: int, end_ns: int, fps: int) -> np.ndarray:
    """fps 주기의 타임스탬프 배열 (ns 단위)을 반환합니다."""
    dt_ns = int(1e9 / fps)
    return np.arange(start_ns, end_ns, dt_ns)


def nearest_index(timestamps: np.ndarray, query: int) -> int:
    """query에 가장 가까운 timestamps 인덱스를 반환합니다."""
    idx = np.searchsorted(timestamps, query)
    if idx == 0:
        return 0
    if idx == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after  = timestamps[idx]
    return idx - 1 if (query - before) <= (after - query) else idx


def collect_messages(reader: Reader, topic: str, typestore):
    """특정 토픽의 (timestamp_ns, message) 목록을 시간순으로 반환합니다."""
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        return [], []
    ts_list, msg_list = [], []
    for conn, ts, rawdata in reader.messages(connections=conns):
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        ts_list.append(ts)
        msg_list.append(msg)
    order = np.argsort(ts_list)
    return [ts_list[i] for i in order], [msg_list[i] for i in order]


def decode_compressed_image(msg) -> np.ndarray:
    """CompressedImage 메시지를 RGB numpy 배열로 디코딩합니다."""
    arr = np.frombuffer(msg.data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─── 에피소드 처리 ──────────────────────────────────────────────────────────

def process_episode(bag_path: Path, typestore, warmup_sec: float) -> list[dict]:
    """
    단일 rosbag 에피소드를 읽어 30Hz 리샘플링된 프레임 목록을 반환합니다.
    - 타임라인 시작: 모든 토픽의 첫 메시지 중 가장 늦은 시점 (싱크 맞춤)
    - warmup_sec: 싱크 시작점 이후 추가로 버릴 시간
    """
    with Reader(str(bag_path)) as reader:
        # ── 모든 토픽 메시지 수집 ─────────────────────────────────────────
        ts_gello,    msgs_gello    = collect_messages(reader, TOPIC_ROBOT_ACTION,   typestore)
        ts_grip_act, msgs_grip_act = collect_messages(reader, TOPIC_GRIPPER_ACTION, typestore)
        ts_joint,    msgs_joint    = collect_messages(reader, TOPIC_ROBOT_STATES,   typestore)
        ts_grip_st,  msgs_grip_st  = collect_messages(reader, TOPIC_GRIPPER_STATE,  typestore)

        cam_data: dict[str, tuple[list, list]] = {}
        for cam_key, cam_topic in CAMERA_TOPICS.items():
            ts_cam, msgs_cam = collect_messages(reader, cam_topic, typestore)
            cam_data[cam_key] = (ts_cam, msgs_cam)

    ts_gello_arr    = np.array(ts_gello)
    ts_grip_act_arr = np.array(ts_grip_act)
    ts_joint_arr    = np.array(ts_joint)
    ts_grip_st_arr  = np.array(ts_grip_st)
    cam_ts_arrs     = {k: np.array(v[0]) for k, v in cam_data.items()}

    # ── 타임라인 범위 결정 ────────────────────────────────────────────────
    # 모든 토픽의 첫 메시지 중 가장 늦은 시점 → 카메라 싱크 기준점
    sync_start_ns = max(
        ts_gello_arr[0],
        ts_grip_act_arr[0],
        ts_joint_arr[0],
        ts_grip_st_arr[0],
        *[cam_ts_arrs[k][0] for k in CAMERA_TOPICS],
    )
    # 모든 토픽의 마지막 메시지 중 가장 이른 시점
    sync_end_ns = min(
        ts_gello_arr[-1],
        ts_grip_act_arr[-1],
        ts_joint_arr[-1],
        ts_grip_st_arr[-1],
        *[cam_ts_arrs[k][-1] for k in CAMERA_TOPICS],
    )

    # warm-up 구간 제거
    warmup_ns = int(warmup_sec * 1e9)
    start_ns  = sync_start_ns + warmup_ns
    end_ns    = sync_end_ns

    if start_ns >= end_ns:
        raise ValueError(
            f"warm-up({warmup_sec}s) 이후 유효 구간이 없습니다. "
            f"유효 bag 길이: {(sync_end_ns - sync_start_ns) / 1e9:.1f}s"
        )

    timeline = build_timeline(start_ns, end_ns, FPS)
    duration = (end_ns - start_ns) / 1e9
    print(f"  sync_start  : +{(sync_start_ns - ts_gello_arr[0]) / 1e6:.1f}ms (토픽 간 최대 지연)")
    print(f"  warmup 제거 : {warmup_sec}s")
    print(f"  유효 구간   : {duration:.1f}s → {len(timeline)} 프레임")

    # ── 프레임 생성 ───────────────────────────────────────────────────────
    frames = []
    for t_ns in timeline:
        # action
        gello_msg    = msgs_gello[nearest_index(ts_gello_arr, t_ns)]
        grip_act_msg = msgs_grip_act[nearest_index(ts_grip_act_arr, t_ns)]
        action = np.concatenate([
            np.array(gello_msg.position, dtype=np.float32),
            np.array([grip_act_msg.data], dtype=np.float32),
        ])  # shape (8,)

        # state
        joint_msg    = msgs_joint[nearest_index(ts_joint_arr, t_ns)]
        grip_st_msg  = msgs_grip_st[nearest_index(ts_grip_st_arr, t_ns)]
        state = np.concatenate([
            np.array(joint_msg.position,    dtype=np.float32),
            np.array(grip_st_msg.position,  dtype=np.float32),
        ])  # shape (8,)

        # images
        images: dict[str, np.ndarray] = {}
        for cam_key, (_, msgs_cam) in cam_data.items():
            idx = nearest_index(cam_ts_arrs[cam_key], t_ns)
            images[f"observation.images.{cam_key}"] = decode_compressed_image(msgs_cam[idx])

        frames.append({
            "action":            action,
            "observation.state": state,
            "task":              TASK_DESCRIPTION,
            **images,
        })

    return frames


# ─── 데이터셋 features 정의 ──────────────────────────────────────────────────

def build_features(cam_shapes: dict[str, tuple[int, int, int]]) -> dict:
    """LeRobot features 딕셔너리를 반환합니다."""
    features = {
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [
                "joint1", "joint2", "joint3", "joint4",
                "joint5", "joint6", "joint7", "gripper_width",
            ],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [
                "joint1", "joint2", "joint3", "joint4",
                "joint5", "joint6", "joint7", "gripper_width",
            ],
        },
    }
    for cam_key, (h, w, c) in cam_shapes.items():
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        }
    return features


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ROS2 bag → LeRobot 데이터셋 변환",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bags-dir",
        type=Path,
        default=Path("/workspace/datasets/rosbag/pick_kit"),
        help="에피소드 bag 폴더들이 있는 상위 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/datasets/lerobot/manufacturing_parts_kitting_dataset"),
        help="LeRobot 데이터셋 저장 경로",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="moonjongsul/manufacturing_parts_kitting_dataset",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=1.5,
        help="에피소드 초반 버릴 warm-up 시간 (초)",
    )
    parser.add_argument(
        "--private",
        default=False,
        action="store_true",
        help="HuggingFace 업로드 시 비공개 레포지토리로 설정",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="기존 데이터셋에 새 에피소드를 추가 (기존 데이터 유지)",
    )
    parser.add_argument(
        "--push-to-hub",
        default=True,
        action="store_true",
        help="변환 완료 후 HuggingFace Hub에 업로드",
    )
    args = parser.parse_args()

    # ── 에피소드 폴더 목록 수집 ───────────────────────────────────────────
    all_episode_dirs = sorted(
        p for p in args.bags_dir.iterdir()
        if p.is_dir() and (p / "metadata.yaml").exists()
    )
    if not all_episode_dirs:
        print(f"[오류] {args.bags_dir} 안에 bag 폴더가 없습니다.")
        sys.exit(1)

    # ROS_DISTRO 환경변수에 따라 typestore 자동 선택
    # import os
    # _ros_distro = os.environ.get("ROS_DISTRO", "humble").lower()
    # _store_map = {
    #     "humble":  Stores.ROS2_HUMBLE,
    #     "iron":    Stores.ROS2_IRON,
    #     "jazzy":   Stores.ROS2_JAZZY,
    #     "rolling": Stores.ROS2_ROLLING,
    # }
    # _store = _store_map.get(_ros_distro, Stores.ROS2_HUMBLE)
    # print(f"ROS_DISTRO={_ros_distro} → typestore: {_store}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # ── 기존 데이터셋 감지 → 자동 append ────────────────────────────────
    info_file = args.output_dir / "meta" / "info.json"
    if info_file.exists():
        with open(info_file) as f:
            existing_info = json.load(f)
        already_done = existing_info.get("total_episodes", 0)
        episode_dirs = all_episode_dirs[already_done:]
        print(f"[append 모드] 기존 에피소드 수: {already_done}")
        print(f"[append 모드] 추가할 에피소드 수: {len(episode_dirs)}")
        if not episode_dirs:
            print("추가할 에피소드가 없습니다.")
            sys.exit(0)
    else:
        episode_dirs = all_episode_dirs
        already_done = 0
        print(f"발견된 에피소드 수: {len(episode_dirs)}")

    for ep in episode_dirs:
        print(f"  {ep.name}")

    # ── 카메라 해상도 확인 ────────────────────────────────────────────────
    print("\n이미지 해상도 확인 중...")
    cam_shapes: dict[str, tuple[int, int, int]] = {}
    with Reader(str(episode_dirs[0])) as reader:
        for cam_key, cam_topic in CAMERA_TOPICS.items():
            conns = [c for c in reader.connections if c.topic == cam_topic]
            if not conns:
                raise RuntimeError(f"토픽 없음: {cam_topic}")
            for _, _, rawdata in reader.messages(connections=conns):
                msg = typestore.deserialize_cdr(rawdata, conns[0].msgtype)
                arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                cam_shapes[cam_key] = img.shape  # (H, W, C)
                break
    for cam_key, shape in cam_shapes.items():
        print(f"  {cam_key}: {shape[0]}×{shape[1]} (H×W)")

    features = build_features(cam_shapes)

    # ── LeRobot 데이터셋 생성 또는 로드 ──────────────────────────────────
    if already_done > 0:
        print(f"\n기존 LeRobot 데이터셋 로드: {args.output_dir}")
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.output_dir,
        )
        dataset.start_image_writer(num_processes=0, num_threads=4)
    else:
        print(f"\nLeRobot 데이터셋 생성: {args.output_dir}")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=FPS,
            features=features,
            root=args.output_dir,
            robot_type=ROBOT_TYPE,
            use_videos=True,
            image_writer_threads=4,
        )

    # ── 에피소드별 변환 ───────────────────────────────────────────────────
    total_new = len(episode_dirs)
    for ep_idx, bag_path in enumerate(episode_dirs):
        global_ep_idx = already_done + ep_idx
        print(f"\n[{ep_idx + 1}/{total_new}] 처리 중: {bag_path.name} (전체 에피소드 #{global_ep_idx})")
        try:
            frames = process_episode(bag_path, typestore, warmup_sec=args.warmup_sec)
        except ValueError as e:
            print(f"  [경고] 스킵: {e}")
            continue

        print(f"  → {len(frames)} 프레임 (≈ {len(frames) / FPS:.1f}초)")

        for frame in frames:
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  → 에피소드 {global_ep_idx} 저장 완료")

    # ── finalize ──────────────────────────────────────────────────────────
    print("\n데이터셋 마무리 중...")
    dataset.finalize()

    # ── Dataset card 생성 ─────────────────────────────────────────────────
    avg_duration = dataset.meta.total_frames // dataset.meta.total_episodes / FPS
    card = create_lerobot_dataset_card(
        tags=["robotics", "manipulation", "kitting", "franka"],
        dataset=dataset,
    )
    card.text += f"""
## Task Description
Pick parts from a supply area, reorient each part by flipping or rotating to the correct orientation,
and place it precisely into the kitting tray.

## Robot Setup
- **Robot**: Franka FR3
- **Gripper**: DXL Parallel Gripper
- **Teleoperation**: GELLO
- **Cameras**:
  - `wrist_front`: Wrist-mounted front camera
  - `wrist_rear`: Wrist-mounted rear camera
  - `front_view`: Front view camera
  - `side_view`: Side view camera

## Data Collection
- **Operator**: Moon Jongsul (KETI)
- **Collection date**: 2026-04
- **Location**: KETI Robotics Lab
- **Episodes**: {dataset.meta.total_episodes}
- **Total frames**: {dataset.meta.total_frames}
- **FPS**: {FPS}
- **Episode duration**: ~{avg_duration:.1f}s avg
- **Warmup removed**: {args.warmup_sec}s per episode

## Action Space
| Index | Name | Description |
|---|---|---|
| 0–6 | `joint_1` ~ `joint_7` | GELLO arm joint angles (rad) |
| 7 | `gripper_width` | Target gripper width (%) |

## Observation Space
| Index | Name | Description |
|---|---|---|
| 0–6 | `joint_1` ~ `joint_7` | Franka FR3 joint angles (rad) |
| 7 | `gripper_width` | DXL gripper joint state |

## Notes
- Parts require reorientation before placement; random initial orientations in supply area
- Kitting tray has fixed slots requiring precise placement orientation
- Timeline resampled at 30 Hz using nearest-neighbor interpolation
- All camera streams synchronized to common time window before resampling
"""
    card.save(args.output_dir / "README.md")
    print("Dataset card 저장 완료: README.md")

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    print(f"\n✅ 완료! 저장 위치: {args.output_dir}")
    print(f"   총 에피소드: {dataset.meta.total_episodes}")
    print(f"   총 프레임  : {dataset.meta.total_frames}")

    # ── HuggingFace 업로드 (옵션) ─────────────────────────────────────────
    if args.push_to_hub:
        from huggingface_hub import HfApi, whoami
        try:
            user = whoami()
            print(f"\nHuggingFace 로그인 확인: {user['name']}")
        except Exception:
            print("\n❌ HuggingFace 로그인 필요:")
            print("   python -c \"from huggingface_hub import login; login()\"")
            sys.exit(1)

        api = HfApi()
        print(f"레포지토리 확인/생성: {args.repo_id}")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=args.private,
        )
        print(f"업로드 중: {args.output_dir} → {args.repo_id}")
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"✅ 업로드 완료: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()