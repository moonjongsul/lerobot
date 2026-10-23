# ============================================================
# SmolVLA Cup Hang - 3cups pink upper, rot6d state (260921+260923+260929)
# 260 episodes (67.8K frames, 15fps)
# A6000 x 3
# ============================================================
set -euo pipefail

# 학습 결과 업로드 계정 (dataset은 yunjuyoung64)
HF_USER='moonjongsul'

# A6000 서버 학습 결과 저장 경로
HDD_PATH='/data/keti/mjs/lerobot'

# Hugging Face Dataset
# DB_REPO_ID='yunjuyoung64/smolvla_hang_cup_red_black_261006'
DB_REPO_ID='yunjuyoung64/smolvla_hang_cup_3cups_260921_260923_260929_pink_upper_rot6d'

# 260 episode rot6d 데이터셋 revision 고정
DATASET_REVISION='93542cb50a55a73f7df296e8c73e1e4527348b88'

# 공식 pretrained SmolVLA
PRETRAINED_MODEL='lerobot/smolvla_base'

# full finetune(freeze_vision=false) + GPU 일부 점유 시 32는 OOM 가능 → 필요시 16
BATCH_SIZE=24

JOB_NAME=smolvla_hang_cup_3cups_pink_upper_rot6d_260_b${BATCH_SIZE}x3_261015

STEPS=60000

HF_MODEL_REPO=${HF_USER}/kcare_hang_cup_3cups_pink_upper_rot6d_smolvla_261015

# dataset(head/wrist) -> smolvla_base(camera1/2/3)
# 카메라가 2개라 세 번째 슬롯은 empty_cameras=1 로 채움
RENAME_MAP='{"observation.images.head":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

echo "=== training config ==="
echo "JOB_NAME=${JOB_NAME}"
echo "DB_REPO_ID=${DB_REPO_ID}"
echo "DATASET_REVISION=${DATASET_REVISION}"
echo "RENAME_MAP=${RENAME_MAP}"
echo "empty_cameras=1"
echo "CUDA_VISIBLE_DEVICES=1,2,3"
echo "HF_MODEL_REPO=${HF_MODEL_REPO}"
echo "output_dir=${HDD_PATH}/outputs/${JOB_NAME}"
echo "========================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1,2,3 \
accelerate launch \
  --num_processes=3 \
  --num_machines=1 \
  --dynamo_backend=no \
  --multi_gpu \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  \
  --num_workers=10 \
  \
  --policy.path="${PRETRAINED_MODEL}" \
  --policy.device=cuda \
  --policy.repo_id="${HF_MODEL_REPO}" \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="${STEPS}" \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.empty_cameras=1 \
  --best_loss_warmup_step=5000 \
  \
  --dataset.repo_id="${DB_REPO_ID}" \
  --dataset.revision="${DATASET_REVISION}" \
  --dataset.image_transforms.enable=true \
  \
  --rename_map="${RENAME_MAP}" \
  \
  --steps="${STEPS}" \
  --save_freq=10000 \
  --log_freq=100 \
  \
  --wandb.enable=true \
  --wandb.disable_artifact=true \
  \
  --batch_size="${BATCH_SIZE}" \
  \
  --output_dir="${HDD_PATH}/outputs/${JOB_NAME}" \
  --job_name="${JOB_NAME}"
