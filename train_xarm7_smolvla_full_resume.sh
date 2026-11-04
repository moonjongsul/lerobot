#!/usr/bin/env bash
# SmolVLA full-data run 이어서 학습 (resume).
#
# 체크포인트의 train_config.json을 그대로 불러와서 모델 가중치, optimizer,
# LR scheduler, step(40800)을 복원한 뒤 원래 목표 step(50000)까지 계속합니다.
# 하이퍼파라미터(배치, LR 스케줄, weight decay, rename_map, 데이터셋 등)는
# 전부 train_config.json 값을 따르므로 여기서 다시 지정하지 않습니다.
# 이 머신에 맞춰야 하는 값(출력 경로, worker 수, GPU)만 덮어씁니다.
#
# 주의:
#  - 원래 run은 H200 x4 (유효 배치 360)였습니다. GPU 수를 줄이면 유효 배치가
#    줄어들지만 scheduler는 step 기준이라 LR 곡선은 그대로 이어집니다.
#  - `--policy.path`를 넘기면 resume 경로를 타지 않으므로 넣지 마세요.
#  - best loss 기록은 resume 시 초기화되므로 재개 후 첫 로그 구간에서
#    best 체크포인트가 새로 저장됩니다.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

CKPT_DIR='/workspace/m.ax/checkpoints/smolvla_xarm7_rot6d_joint_to_eef_velocity_261014/best_step_040800_loss_0.029999'
CONFIG_PATH="${CKPT_DIR}/pretrained_model/train_config.json"
[[ -f ${CONFIG_PATH} ]] || { echo "[error] ${CONFIG_PATH} 없음" >&2; exit 1; }
[[ -d ${CKPT_DIR}/training_state ]] || { echo "[error] ${CKPT_DIR}/training_state 없음" >&2; exit 1; }

TRAIN_NAME='smolvla_xarm7_rot6d+joint_to_eef_velocity_full_resume'
DATE='261103'

SAVE_PATH="/workspace/m.ax/checkpoints"
WANDB=false
# 원래 run(a2qmpz1q)에 이어서 기록합니다. train_config.json에 저장된 run_id를
# 그대로 쓰며, resume 시 wandb는 resume="must"로 붙으므로 같은 계정/프로젝트로
# 로그인되어 있어야 합니다. 붙을 수 없으면 WANDB=false로 끄세요.

MACHINE='THOR'
CUDA_DEVICES=0
NUM_WORKER=10
# 원래 목표 step. 늘리면 50000 이후로는 decay_lr(2.5e-6)로 고정된 채 학습됩니다.
TOTAL_STEP=100000
# best loss 체크포인트는 최근 N개만 남기고 오래된 것은 삭제 (0 이하면 전부 보관)
KEEP_BEST=3

IFS=',' read -ra _GPU_ARR <<< "${CUDA_DEVICES}"
NUM_GPUS=${#_GPU_ARR[@]}
MULTI_GPU_FLAG=()
[[ ${NUM_GPUS} -gt 1 ]] && MULTI_GPU_FLAG=(--multi_gpu)

START_STEP=$(python -c "import json;print(json.load(open('${CKPT_DIR}/training_state/training_step.json'))['step'])")
echo "[launch] SmolVLA resume  from step ${START_STEP} -> ${TOTAL_STEP}  GPUs=${CUDA_DEVICES} (n=${NUM_GPUS})"

JOB_NAME="${TRAIN_NAME}_${MACHINE}_x${NUM_GPUS}_${DATE}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --dynamo_backend=no \
  "${MULTI_GPU_FLAG[@]}" \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --resume=true \
  --config_path="${CONFIG_PATH}" \
  --num_workers=${NUM_WORKER} \
  --steps="${TOTAL_STEP}" \
  --keep_best_checkpoints=${KEEP_BEST} \
  --wandb.enable=${WANDB} \
  --output_dir="${SAVE_PATH}/outputs/${JOB_NAME}" \
  --job_name="${JOB_NAME}"
