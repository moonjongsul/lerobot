#!/usr/bin/env bash
# SmolVLA on the xarm7 kit dataset with the top-down tray camera, GPU 1장.
#
# 데이터: datasets/xarm_kit_tray_261104 를 convert_to_lerobot.py
# (data_convert_lerobot_tray_config.yaml)로 변환해 Hub에 올린 LeRobot 데이터셋.
# 카메라 4대 (env, wrist_front, wrist_rear, tray), 161 에피소드 / 110,196 프레임.
#
# 기존 체크포인트에는 tray 입력이 없으므로 smolvla_base에서 새로 시작한다.
#
# ── camera4를 base config에 넣는 이유 ─────────────────────────────────
# --policy.path로 시작하면 input_features는 base의 config.json을 그대로
# 쓴다. smolvla_base에는 camera1~3만 있어서, tray를 camera4로 rename해도
# 정책이 모르는 키라 에러 없이 버려진다 (정책 카메라 ⊂ 데이터셋 카메라면
# 검증을 통과한다). SmolVLA는 모든 이미지를 같은 vision encoder에 넣고
# 토큰만 이어붙이므로 카메라별 파라미터가 없다 -- config에 camera4만
# 추가하면 base 가중치를 그대로 쓴다. 그래서 base를 로컬로 받아 config만
# 고친 사본을 --policy.path로 넘긴다.
set -euo pipefail


# Hub의 데이터셋 (upload_to_hub.py로 올린 repo). 첫 실행 때
# $HF_LEROBOT_HOME/<repo_id>로 받아 캐시한다.
DATASET='moonjongsul/xarm7-kit-tray-261104'
TRAIN_NAME='smolvla_xarm7_rot6d+joint_to_eef_velocity_kit_add_tray'
DATE='261104'

SAVE_PATH="$HOME/workspace/lerobot/checkpoints"
BASE_DIR="$HOME/workspace/lerobot/base/smolvla_base_4cam"
WANDB=false

MACHINE='pro6000'
CUDA_DEVICES=0
NUM_WORKER=8
# GPU 메모리에 맞춰 조정. H200에서 카메라 3대 기준 90이 들어갔고, 카메라가
# 4대라 이미지 토큰이 1/3 늘어난다. 대략 24GB: 12~16, 48GB: 32, 80GB: 56.
# OOM이 나면 줄이고, 첫 수백 스텝의 메모리 사용량을 보고 올린다.
BATCH_SIZE=56
# 기존 full 학습 = 약 63 epoch (360 x 50k / 284,562). 같은 epoch 수를 맞추면
# 배치가 작아진 만큼 스텝이 늘어난다: 63 x 110,196 / 56 ≈ 122k 스텝.
# 시간이 부족하면 EPOCHS를 줄인다 (데이터가 적어 과적합 쪽이기도 하다).
DATASET_FRAMES=110196
EPOCHS=50
TOTAL_STEP=$(( EPOCHS * DATASET_FRAMES / BATCH_SIZE ))
# LR 코사인 감쇠가 끝나는 지점. TOTAL_STEP과 같아야 학습 종료 시점에
# decay_lr에 정확히 도달한다.
SCHED_DECAY=${TOTAL_STEP}
SCHED_WARMUP=1000
SAVE_FREQ=5000
LOG_FREQ=100
BEST_LOSS_WARMUP=10000

echo "[launch] SmolVLA tray  GPU=${CUDA_DEVICES}  batch=${BATCH_SIZE}  steps=${TOTAL_STEP} (~${EPOCHS} epoch)"

JOB_NAME="${TRAIN_NAME}_${MACHINE}_b${BATCH_SIZE}_${DATE}"

# ── smolvla_base + camera4 ─────────────────────────────────────────────
python - "${BASE_DIR}" <<'EOF'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download

base = Path(sys.argv[1])
if not (base / "model.safetensors").exists():
    snapshot_download("lerobot/smolvla_base", local_dir=base)
cfg_path = base / "config.json"
cfg = json.loads(cfg_path.read_text())
feats = cfg["input_features"]
if "observation.images.camera1" not in feats:
    sys.exit(f"unexpected smolvla_base input_features: {list(feats)}")
if "observation.images.camera4" not in feats:
    feats["observation.images.camera4"] = dict(feats["observation.images.camera1"])
    cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
print("[base]", base, "images:", [k for k in feats if k.startswith("observation.images.")])
EOF

# 데이터셋 카메라 키 -> camera1~4 슬롯. 1~3은 기존 학습과 같은 배치.
# ── 학습에 쓸 state ────────────────────────────────────────────────
# 정책은 항상 `observation.state` 키를 state로 읽는다. 다른 state를 쓰려면
# 그 키를 `observation.state`로, 원래 `observation.state`는 다른 이름으로
# 맞바꾼다. stats도 lerobot_train에서 같은 rename_map으로 바뀐다.
#   ''          : observation.state           (18, xyz + rot6d + joint + gripper + effort)
#   'eef_rot6d' : observation.state.eef_rot6d (11, xyz + rot6d + gripper + effort)
#   'eef_quat'  : observation.state.eef_quat  (9, xyz + quat + gripper + effort)
STATE_KEY=''
STATE_RENAME=''
[[ -n ${STATE_KEY} ]] && STATE_RENAME="\"observation.state\": \"observation.state.main\", \"observation.state.${STATE_KEY}\": \"observation.state\""
RENAME_MAP="{\"observation.images.env\": \"observation.images.camera1\", \"observation.images.wrist_front\": \"observation.images.camera2\", \"observation.images.wrist_rear\": \"observation.images.camera3\", \"observation.images.tray\": \"observation.images.camera4\"${STATE_RENAME:+, ${STATE_RENAME}}}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --dynamo_backend=no \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --num_workers=${NUM_WORKER} \
  --policy.path="${BASE_DIR}" \
  --policy.device=cuda \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.scheduler_warmup_steps=${SCHED_WARMUP} \
  --policy.scheduler_decay_steps=${SCHED_DECAY} \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET}" \
  --dataset.image_transforms.enable=true \
  --dataset.video_backend=torchcodec \
  --rename_map="${RENAME_MAP}" \
  --steps="${TOTAL_STEP}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --best_loss_warmup_step="${BEST_LOSS_WARMUP}" \
  --wandb.enable=${WANDB} \
  --wandb.project=xarm7-kitting \
  --wandb.disable_artifact=true \
  --batch_size=${BATCH_SIZE} \
  --output_dir="${SAVE_PATH}/outputs/${JOB_NAME}" \
  --job_name="${JOB_NAME}"
