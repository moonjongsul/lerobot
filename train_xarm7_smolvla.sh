#!/usr/bin/env bash
# SmolVLA baseline on the xarm7 kitting dataset.
#
# This is the control MVLA is measured against, so every hyperparameter
# below is kept identical to train_xarm7_mvla.sh -- steps, batch, schedule,
# weight decay, logging. The only intentional differences are the policy
# (no MVLA heads or prompt assembly) and the GPU set, so the two can run
# side by side. Change one here, change it there.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate


DATASET='moonjongsul/xarm7-kitting-260923'
TRAIN_NAME='smolvla_xarm7_eef_velocity_split'
DATE='260923'

SAVE_PATH="$HOME/workspace/lerobot/checkpoints"
WANDB=true

MACHINE='H200'
# MVLA가 0-3을 쓰므로 동시 실행을 위해 4-7을 씁니다.
CUDA_DEVICES=4,5,6,7
NUM_WORKER=25
BATCH_SIZE=90
# 유효 배치 = 90 x 4 = 360. 데이터셋이 284,562 프레임이므로
# 50k 스텝이면 약 63 epoch, 100k 스텝이면 약 127 epoch입니다.
# 292 에피소드짜리 단일 셀 데이터에 127 epoch은 과적합 쪽입니다.
TOTAL_STEP=50000
# LR 코사인 감쇠가 끝나는 지점. TOTAL_STEP과 같아야 학습 종료 시점에
# decay_lr(2.5e-6)에 정확히 도달합니다. 크게 잡으면 LeRobot이
# 자동 스케일하면서 warmup까지 같은 비율로 줄여버립니다.
SCHED_DECAY=${TOTAL_STEP}
SCHED_WARMUP=1000
SAVE_FREQ=2000
LOG_FREQ=100
BEST_LOSS_WARMUP=5000

# CUDA_DEVICES 개수에서 프로세스 수를 자동 계산 (GPU 1장당 프로세스 1개)
IFS=',' read -ra _GPU_ARR <<< "${CUDA_DEVICES}"
NUM_GPUS=${#_GPU_ARR[@]}
# --multi_gpu는 2장 이상일 때만 (1장에 넘기면 accelerate가 거부)
MULTI_GPU_FLAG=()
[[ ${NUM_GPUS} -gt 1 ]] && MULTI_GPU_FLAG=(--multi_gpu)
echo "[launch] SmolVLA baseline  GPUs=${CUDA_DEVICES} (n=${NUM_GPUS})  batch/gpu=${BATCH_SIZE}  effective=$((BATCH_SIZE * NUM_GPUS))"

JOB_NAME="${TRAIN_NAME}_${MACHINE}_b${BATCH_SIZE}x${NUM_GPUS}_${DATE}"

# ── train/val 분할 ──────────────────────────────────────────────────
# 학습에는 234개(80%)만 쓰고 나머지 58개는 held-out으로 남긴다. 분할하지
# 않으면 292개 전부를 외운 모델이 나와 "성능이 향상됐다"를 보일 수단이
# 없다. 명단은 make_split.py가 만들고 split_260923.json에 박아 둔다 --
# 매번 계산하면 분할이 조용히 달라져 런끼리 비교가 무의미해진다.
#
# task 블록별로 뒤쪽 20%씩 떼는 방식이다. 전체의 뒤쪽 20%를 자르면 val에
# flip 에피소드가 0개가 되어 MVLA의 핵심 능력을 측정할 수 없다.
SPLIT_FILE="$(dirname "$0")/split_260923.json"
[[ -f ${SPLIT_FILE} ]] || { echo "분할 파일이 없다: ${SPLIT_FILE} (python make_split.py 먼저)" >&2; exit 1; }
TRAIN_EPISODES="$(python -c "import json;print('['+','.join(map(str,json.load(open('${SPLIT_FILE}'))['train']))+']')")"

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
  --policy.scheduler_warmup_steps=${SCHED_WARMUP} \
  --policy.scheduler_decay_steps=${SCHED_DECAY} \
  `# SmolVLA 기본값은 1e-10(사실상 0), pi0/pi05는 1e-2.` \
  `# 292 에피소드 단일 셀 데이터라 정규화가 필요해 중간값을 씁니다.` \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET}" \
  --dataset.episodes="${TRAIN_EPISODES}" \
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
