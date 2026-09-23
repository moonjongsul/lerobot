#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

JOB_NAME='smolvla_xarm7_eef_velocity_h200_b90x3_260923'
DATASET='moonjongsul/xarm7-kitting-260923'

SAVE_PATH="$HOME/workspace/lerobot/checkpoints"
WANDB=true

CUDA_DEVICES=5,6,7
BATCH_SIZE=90
TOTAL_STEP=500000
SAVE_FREQ=5000
LOG_FREQ=100
BEST_LOSS_WARMUP=20000

# 데이터셋 카메라 키 -> smolvla_base가 기대하는 camera1/2/3 슬롯
RENAME_MAP='{"observation.images.env": "observation.images.camera1", "observation.images.wrist_front": "observation.images.camera2", "observation.images.wrist_rear": "observation.images.camera3"}'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes=2 \
  --num_machines=1 \
  --dynamo_backend=no \
  --multi_gpu \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --num_workers=50 \
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
