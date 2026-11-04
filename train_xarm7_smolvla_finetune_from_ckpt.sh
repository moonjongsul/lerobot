#!/usr/bin/env bash
# 학습된 SmolVLA 체크포인트를 시작점으로, 새 데이터셋에 파인튜닝합니다.
#
# resume(train_xarm7_smolvla_full_resume.sh)과의 차이:
#  - resume : 같은 run을 이어감. optimizer/scheduler/step(40800)까지 복원하고
#             데이터셋·하이퍼파라미터는 train_config.json 값을 따름.
#  - 이 스크립트 : 체크포인트의 "가중치만" 가져오고 step 0부터 새 run을 시작.
#             optimizer/scheduler는 새로 만들고(warmup부터), 데이터셋은 아래
#             DATASET을 쓰며, 정규화 stats도 새 데이터셋 기준으로 다시 잡힘.
#
# 새 데이터셋은 체크포인트와 같은 형식이어야 합니다:
#  - observation.state는 원래 run과 같은 18차원 구성(rot6d 11 + joint 7), action은 7차원.
#    모델은 state/action을 32차원으로 zero-pad해서 받으므로 차원이 달라도 에러 없이
#    돌아가지만, 각 채널의 의미가 다르면 학습된 가중치가 엉뚱하게 해석합니다.
#    (config.json의 observation.state shape [6]은 smolvla_base에서 물려받은 표기일 뿐
#     실제 차원이 아닙니다.)
#  - 카메라 키는 RENAME_MAP으로 camera1/2/3에 매핑
set -euo pipefail

cd "$(dirname "$0")"
# source .venv/bin/activate

CKPT_DIR='/workspace/m.ax/checkpoints/smolvla_xarm7_rot6d_joint_to_eef_velocity_261014/best_step_040800_loss_0.029999'
POLICY_PATH="${CKPT_DIR}/pretrained_model"
[[ -f ${POLICY_PATH}/model.safetensors ]] || { echo "[error] ${POLICY_PATH}/model.safetensors 없음" >&2; exit 1; }

# ── 새 데이터셋 ──────────────────────────────────────────────────────
DATASET='moonjongsul/xarm7_kitting_flip_261103'
[[ ${DATASET} == *CHANGE_ME* ]] && { echo "[error] DATASET을 새 데이터셋 repo_id로 바꾸세요" >&2; exit 1; }

TRAIN_NAME='smolvla_xarm7_rot6d+joint_to_eef_velocity_flip'
DATE='261103'

SAVE_PATH="/workspace/m.ax/checkpoints"
WANDB=true

MACHINE='THOR'
CUDA_DEVICES=0
NUM_WORKER=10
BATCH_SIZE=48
TOTAL_STEP=100000
# 이미 학습된 가중치에서 시작하므로 원래 run(1e-4)보다 낮게 시작합니다.
PEAK_LR=5e-5
SCHED_DECAY=${TOTAL_STEP}
SCHED_WARMUP=500
SAVE_FREQ=2000
LOG_FREQ=100
BEST_LOSS_WARMUP=2000
# best loss 체크포인트는 최근 N개만 남기고 오래된 것은 삭제 (0 이하면 전부 보관)
KEEP_BEST=2

IFS=',' read -ra _GPU_ARR <<< "${CUDA_DEVICES}"
NUM_GPUS=${#_GPU_ARR[@]}
MULTI_GPU_FLAG=()
[[ ${NUM_GPUS} -gt 1 ]] && MULTI_GPU_FLAG=(--multi_gpu)
echo "[launch] SmolVLA finetune  ckpt=$(basename "${CKPT_DIR}")  dataset=${DATASET}  GPUs=${CUDA_DEVICES} (n=${NUM_GPUS})  effective=$((BATCH_SIZE * NUM_GPUS))"

JOB_NAME="${TRAIN_NAME}_${MACHINE}_b${BATCH_SIZE}x${NUM_GPUS}_${DATE}"

# 데이터셋 카메라 키 -> 체크포인트가 기대하는 camera1/2/3 슬롯 (원래 run과 동일)
RENAME_MAP='{"observation.images.env": "observation.images.camera1", "observation.images.wrist_front": "observation.images.camera2", "observation.images.wrist_rear": "observation.images.camera3"}'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --dynamo_backend=no \
  "${MULTI_GPU_FLAG[@]}" \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --num_workers=${NUM_WORKER} \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=${PEAK_LR} \
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
  --keep_best_checkpoints=${KEEP_BEST} \
  --wandb.enable=${WANDB} \
  --wandb.project=xarm7-kitting \
  --wandb.disable_artifact=true \
  --batch_size=${BATCH_SIZE} \
  --output_dir="${SAVE_PATH}/outputs/${JOB_NAME}" \
  --job_name="${JOB_NAME}"
