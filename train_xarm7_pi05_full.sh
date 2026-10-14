#!/usr/bin/env bash
# pi0.5 full finetune on the xarm7 kitting dataset.
#
# Hyperparameters follow openpi's `pi05_libero` TrainConfig (PI's own pi0.5
# finetune recipe; LIBERO is ~270k frames, close to our ~285k frames):
# global batch 256, 30k steps, peak LR 5e-5 held constant after a 10k warmup,
# AdamW(0.9, 0.95, wd=1e-10), grad clip 1.0. openpi also keeps an EMA of the
# weights (0.999), which this fork's trainer does not implement.
#
# 전체 데이터 버전: train/val 분할 없이 292개 에피소드 전부로 학습합니다.
# held-out이 없으므로 split 버전과 성능 비교용이 아니라 배포용입니다.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate


DATASET='moonjongsul/xarm7-kitting-260923'
TRAIN_NAME='pi05_xarm7_rot6d+joint_to_eef_velocity_full'
DATE='261014'

SAVE_PATH="$HOME/workspace/lerobot/checkpoints"
WANDB=true

MACHINE='H200'
CUDA_DEVICES=4,5,6,7
NUM_WORKER=25
# openpi pi05_libero는 global batch 256. GPU 4장 기준 64/GPU.
# GPU 수를 바꾸면 BATCH_SIZE * NUM_GPUS = 256이 되도록 맞추세요.
BATCH_SIZE=84
# 256 x 30k = 7.7M 샘플 = 전체 프레임(284,562) 기준 약 27 epoch.
TOTAL_STEP=30000
# openpi pi05_libero: warmup 10k 후 5e-5 상수 (decay_lr = peak_lr).
# DECAY_LR을 PEAK_LR과 같게 두면 코사인 구간이 평탄해져 상수 LR이 됩니다.
# SCHED_DECAY를 TOTAL_STEP보다 크게 주면 lerobot이 warmup까지 같이 축소하므로
# TOTAL_STEP과 같게 둡니다.
PEAK_LR=5e-5
DECAY_LR=5e-5
SCHED_WARMUP=10000
SCHED_DECAY=${TOTAL_STEP}
# openpi AdamW 기본값. lerobot pi05 기본값(1e-2)과 다릅니다.
WEIGHT_DECAY=1e-10
SAVE_FREQ=5000
LOG_FREQ=100
# warmup이 끝난 뒤부터 best 체크포인트를 추적합니다.
BEST_LOSS_WARMUP=${SCHED_WARMUP}
# 데이터셋 30fps 기준 75스텝 = 2.5초. pi05_base 기본값은 50(1.67초).
# n_action_steps는 chunk_size 이하여야 하고, 추론 시 따로 바꿀 수 있습니다.
CHUNK_SIZE=75
N_ACTION_STEPS=${CHUNK_SIZE}

# CUDA_DEVICES 개수에서 프로세스 수를 자동 계산 (GPU 1장당 프로세스 1개)
IFS=',' read -ra _GPU_ARR <<< "${CUDA_DEVICES}"
NUM_GPUS=${#_GPU_ARR[@]}
# --multi_gpu는 2장 이상일 때만 (1장에 넘기면 accelerate가 거부)
MULTI_GPU_FLAG=()
[[ ${NUM_GPUS} -gt 1 ]] && MULTI_GPU_FLAG=(--multi_gpu)
echo "[launch] pi0.5 finetune  GPUs=${CUDA_DEVICES} (n=${NUM_GPUS})  batch/gpu=${BATCH_SIZE}  effective=$((BATCH_SIZE * NUM_GPUS))"

JOB_NAME="${TRAIN_NAME}_${MACHINE}_b${BATCH_SIZE}x${NUM_GPUS}_${DATE}"

# ── pi05_base 가중치 ────────────────────────────────────────────────
# Hub 최신본(2026-06-03 #6 이후)은 preprocessor에 relative_actions_processor
# 단계가 들어 있는데, 이 포크에는 그 단계가 없어 로드가 실패합니다. 그 직전
# 리비전으로 고정해 로컬에 받아 씁니다. model.safetensors는 최신본과 동일(sha 일치).
PI05_BASE_REV='a538eb2732'
PI05_BASE_DIR="${SAVE_PATH}/pretrained/pi05_base_${PI05_BASE_REV}"
python -c "from huggingface_hub import snapshot_download as d;d('lerobot/pi05_base',revision='${PI05_BASE_REV}',local_dir='${PI05_BASE_DIR}')"

# ── 학습에 쓸 state ────────────────────────────────────────────────
# 정책은 항상 `observation.state` 키를 state로 읽습니다. 다른 state를 쓰려면
# 그 키를 `observation.state`로, 원래 `observation.state`(eef velocity)는
# 다른 이름으로 맞바꿉니다(이름이 겹치면 배치 dict 순서에 따라 어느 쪽이
# 남을지 달라짐). stats도 lerobot_train에서 같은 rename_map으로 바뀝니다.
#   ''          : observation.state           (8, eef velocity + gripper + effort)
#   'eef_rot6d' : observation.state.eef_rot6d (11, xyz + rot6d + gripper + effort)
#   'eef_quat'  : observation.state.eef_quat  (9, xyz + quat + gripper + effort)
STATE_KEY=''
STATE_RENAME=''
[[ -n ${STATE_KEY} ]] && STATE_RENAME="\"observation.state\": \"observation.state.eef_velocity\", \"observation.state.${STATE_KEY}\": \"observation.state\""
RENAME_MAP="{${STATE_RENAME}}"

# --policy.type=pi05 + pretrained_path 조합이라 입력 feature는 데이터셋 키
# (env / wrist_front / wrist_rear)를 그대로 씁니다. pi0.5는 카메라별 파라미터가
# 없어 rename_map이 필요 없습니다. 데이터셋 stats에 q01/q99가 있어 기본
# QUANTILES 정규화를 그대로 씁니다.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --dynamo_backend=no \
  "${MULTI_GPU_FLAG[@]}" \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --num_workers=${NUM_WORKER} \
  --policy.type=pi05 \
  --policy.chunk_size=${CHUNK_SIZE} \
  --policy.n_action_steps=${N_ACTION_STEPS} \
  --policy.pretrained_path="${PI05_BASE_DIR}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=${PEAK_LR} \
  --policy.scheduler_decay_lr=${DECAY_LR} \
  --policy.scheduler_warmup_steps=${SCHED_WARMUP} \
  --policy.scheduler_decay_steps=${SCHED_DECAY} \
  --policy.optimizer_weight_decay=${WEIGHT_DECAY} \
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
