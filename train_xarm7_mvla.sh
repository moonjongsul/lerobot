#!/usr/bin/env bash
# MVLA training. Adapted from train_xarm7_smolvla.sh; the launcher and the
# dataset are the same, what differs is the policy type, the pretrained
# weights path, and the staged head/prompt flags.
#
# Stages are selected with STAGE=<n>. They are cumulative and exist so that
# each addition can be attributed: stage 1 has to reproduce SmolVLA, and
# every later stage changes one thing.
#
#   STAGE=1  baseline    heads off, prompts verbatim  -> must match SmolVLA
#   STAGE=2  subtask     + subtask prompt and head
#   STAGE=3  metadata    + quality/speed/mistake, + neutral prompts
#   STAGE=4  value       + value & status heads, + elapsed-in-subtask
#   STAGE=5  subgoal     + subgoal images
#   STAGE=7  advantage   + advantage conditioning (needs a fitted value fn)
#
#   STAGE=4 ./train_xarm7_mvla.sh
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

STAGE="${STAGE:-4}"

DATASET='moonjongsul/xarm7-kitting-260923'
TRAIN_NAME="mvla_xarm7_eef_velocity_s${STAGE}"
DATE='260923'

SAVE_PATH="$HOME/workspace/lerobot/checkpoints"
WANDB=true

MACHINE='H200'
CUDA_DEVICES=0,1,2,3
NUM_WORKER=25
BATCH_SIZE=90
# 유효 배치 = 90 x 4 = 360. 데이터셋이 284,562 프레임이므로
# 30k 스텝이면 약 38 epoch, 100k 스텝이면 약 127 epoch입니다.
# 292 에피소드짜리 단일 셀 데이터에 127 epoch은 과적합 쪽이라,
# 40k에서 시작해 검증 손실을 보고 늘리는 편이 낫습니다.
TOTAL_STEP=50000
# LR 코사인 감쇠가 끝나는 지점. TOTAL_STEP과 같아야 학습 종료 시점에
# decay_lr(2.5e-6)에 정확히 도달합니다. 아래 SCHED_DECAY 참고.
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

JOB_NAME="${TRAIN_NAME}_${MACHINE}_b${BATCH_SIZE}x${NUM_GPUS}_${DATE}"

# 데이터셋 카메라 키 -> smolvla_base가 기대하는 camera1/2/3 슬롯
RENAME_MAP='{"observation.images.env": "observation.images.camera1", "observation.images.wrist_front": "observation.images.camera2", "observation.images.wrist_rear": "observation.images.camera3"}'

# ── 단계별 플래그 ────────────────────────────────────────────────────
# 전부 명시합니다. 기본값에 의존하면 config 기본값이 바뀔 때
# 단계 정의가 조용히 달라집니다.
case "${STAGE}" in
  1)  # SmolVLA와 동일해야 하는 베이스라인.
      # 모든 헤드를 끄고 프롬프트를 원문 그대로 두면 데이터 래퍼가
      # 아예 적용되지 않습니다(adapter.wrap_dataset이 원본을 반환).
      STAGE_FLAGS=(
        --policy.use_subtask_head=false
        --policy.use_value_heads=false
        --policy.use_status_head=false
        --policy.use_subtask_prompt=false
        --policy.use_metadata_prompt=false
        --policy.use_elapsed_subtask=false
        --policy.neutral_prompt_prob=0.0
      ) ;;
  2)  STAGE_FLAGS=(
        --policy.use_subtask_head=true
        --policy.use_subtask_prompt=true
        --policy.subtask_corruption_prob=0.2
        --policy.use_value_heads=false
        --policy.use_status_head=false
        --policy.use_metadata_prompt=false
        --policy.use_elapsed_subtask=false
        --policy.neutral_prompt_prob=0.0
      ) ;;
  3)  STAGE_FLAGS=(
        --policy.use_subtask_head=true
        --policy.use_subtask_prompt=true
        --policy.use_metadata_prompt=true
        --policy.use_mistake_prompt=true
        --policy.neutral_prompt_prob=0.5
        --policy.metadata_dropout_prob=0.15
        --policy.metadata_field_dropout_prob=0.05
        --policy.use_value_heads=false
        --policy.use_status_head=false
        --policy.use_elapsed_subtask=false
      ) ;;
  4)  STAGE_FLAGS=(
        --policy.use_subtask_head=true
        --policy.use_subtask_prompt=true
        --policy.use_metadata_prompt=true
        --policy.use_mistake_prompt=true
        --policy.neutral_prompt_prob=0.5
        --policy.use_value_heads=true
        --policy.use_status_head=true
        --policy.use_elapsed_subtask=true
        --policy.value_bins=201
        --policy.subtask_loss_weight=0.1
        --policy.value_subtask_loss_weight=0.1
        --policy.value_episode_loss_weight=0.1
        --policy.status_loss_weight=0.05
      ) ;;
  5)  STAGE_FLAGS=(
        --policy.use_subtask_head=true
        --policy.use_subtask_prompt=true
        --policy.use_metadata_prompt=true
        --policy.neutral_prompt_prob=0.5
        --policy.use_value_heads=true
        --policy.use_status_head=true
        --policy.use_elapsed_subtask=true
        --policy.use_subgoal_images=true
        --policy.subgoal_prob=0.25
        --policy.subgoal_window_s=2.0
      ) ;;
  7)  STAGE_FLAGS=(
        --policy.use_subtask_head=true
        --policy.use_subtask_prompt=true
        --policy.use_metadata_prompt=true
        --policy.neutral_prompt_prob=0.5
        --policy.use_value_heads=true
        --policy.use_status_head=true
        --policy.use_elapsed_subtask=true
        --policy.use_advantage_prompt=true
        --policy.advantage_dropout_prob=0.10
      ) ;;
  *)  echo "unknown STAGE='${STAGE}' (expected 1,2,3,4,5,7)" >&2; exit 1 ;;
esac

echo "[launch] MVLA stage=${STAGE}  GPUs=${CUDA_DEVICES} (n=${NUM_GPUS})  batch/gpu=${BATCH_SIZE}  effective=$((BATCH_SIZE * NUM_GPUS))"
printf '[flags]  %s\n' "${STAGE_FLAGS[*]}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes=${NUM_GPUS} \
  --num_machines=1 \
  --dynamo_backend=no \
  "${MULTI_GPU_FLAG[@]}" \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --num_workers=${NUM_WORKER} \
  --policy.type=mvla \
  --policy.pretrained_path=lerobot/smolvla_base \
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
  "${STAGE_FLAGS[@]}" \
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
