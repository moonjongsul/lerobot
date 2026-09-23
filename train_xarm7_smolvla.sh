#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate


DATASET='moonjongsul/xarm7-kitting-260923'
TRAIN_NAME='smolvla_xarm7_eef_velocity'
DATE='260923'

SAVE_PATH="$HOME/workspace/lerobot/checkpoints"
WANDB=true

MACHINE='H200'
CUDA_DEVICES=4,5,6,7
NUM_WORKER=25
BATCH_SIZE=90
TOTAL_STEP=100000
SAVE_FREQ=5000
LOG_FREQ=100
BEST_LOSS_WARMUP=20000

# CUDA_DEVICES 개수에서 프로세스 수를 자동 계산 (GPU 1장당 프로세스 1개)
IFS=',' read -ra _GPU_ARR <<< "${CUDA_DEVICES}"
NUM_GPUS=${#_GPU_ARR[@]}
# --multi_gpu는 2장 이상일 때만 (1장에 넘기면 accelerate가 거부)
MULTI_GPU_FLAG=()
[[ ${NUM_GPUS} -gt 1 ]] && MULTI_GPU_FLAG=(--multi_gpu)
echo "[launch] GPUs=${CUDA_DEVICES} (n=${NUM_GPUS})  batch/gpu=${BATCH_SIZE}  effective_batch=$((BATCH_SIZE * NUM_GPUS))"

JOB_NAME="${TRAIN_NAME}_${MACHINE}_b${BATCH_SIZE}x${NUM_GPUS}_${DATE}"

# 데이터셋 카메라 키 -> smolvla_base가 기대하는 camera1/2/3 슬롯
RENAME_MAP='{"observation.images.env": "observation.images.camera1", "observation.images.wrist_front": "observation.images.camera2", "observation.images.wrist_rear": "observation.images.camera3"}'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --dynamo_backend=no \
  "${MULTI_GPU_FLAG[@]}" \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --num_workers=${NUM_WORKER} \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.scheduler_decay_steps=200000 \
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
