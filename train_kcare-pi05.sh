# ============================================================
# Pi0.5 Cup Hang - red/black hook, rot6d state (merged)
# 400 episodes (115K frames, 15fps), 2 tasks (red hook / black hook)
# A6000 x 3
# ============================================================
set -euo pipefail

# 학습 결과 업로드 계정 (dataset은 yunjuyoung64)
HF_USER='moonjongsul'

# A6000 서버 학습 결과 저장 경로
HDD_PATH='/data/keti/mjs/lerobot'

# Hugging Face Dataset
DB_REPO_ID='yunjuyoung64/smolvla_hang_cup_red_black_rot6d_merged'

# 400 episode 데이터셋 revision 고정
DATASET_REVISION='14088f58d3bcf921c216e07ae0a2d4b4ab2dc954'

# pretrained pi05_base
# pinned: a538eb27 (2026-06-01, pre-relative_actions, fork compat)
PRETRAINED_MODEL='/home/mjs/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/a538eb273274eb30f126a118f39dbc0ee212c883'

# full finetune(freeze_vision=false) + fused AdamW + gradient_checkpointing 기준.
# b16은 GPU당 ~42GB라 다른 프로세스가 있는 GPU에서 OOM 위험 → 12
BATCH_SIZE=12

JOB_NAME=pi05_hang_cup_red_black_rot6d_400_aug_b${BATCH_SIZE}x3_261020

STEPS=80000

HF_MODEL_REPO=${HF_USER}/kcare_hang_cup_red_black_rot6d_aug_pi05_261020

# 이미지 증강 (head/wrist 각각 독립 적용, 매 프레임마다 아래 중 정확히 MAX_NUM_TRANSFORMS개를 뽑아 적용)
# - 색으로 컵/고리를 구분하는 태스크라 hue는 제외, saturation은 기본(0.5~1.5)보다 좁힘
# - 손잡이-고리 정렬 정밀도 때문에 기하 변환(crop/affine)은 작게
# - CLI에서는 tfs 전체를 JSON으로만 덮어쓸 수 있음 (기본 6종은 통째로 교체됨)
MAX_NUM_TRANSFORMS=3
IMAGE_TFS='{
  "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
  "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.7, 1.3]}},
  "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.7, 1.3]}},
  "sharpness":  {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
  "blur":       {"weight": 1.0, "type": "GaussianBlur", "kwargs": {"kernel_size": 11, "sigma": [0.1, 2.5]}},
  "crop":       {"weight": 1.0, "type": "RandomResizedCrop", "kwargs": {"size": [480, 848], "scale": [0.9, 1.0], "ratio": [1.75, 1.78]}},
  "affine":     {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-3.0, 3.0], "translate": [0.02, 0.02]}}
}'

# policy.type=pi05 + pretrained_path 이므로 입력 feature는 dataset(head/wrist)을 그대로 사용
# (rename_map 불필요, 부족한 카메라 슬롯은 모델 내부에서 empty image로 채움)

echo "=== training config ==="
echo "JOB_NAME=${JOB_NAME}"
echo "DB_REPO_ID=${DB_REPO_ID}"
echo "DATASET_REVISION=${DATASET_REVISION}"
echo "PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "BATCH_SIZE=${BATCH_SIZE} x 3"
echo "CUDA_VISIBLE_DEVICES=1,2,3"
echo "HF_MODEL_REPO=${HF_MODEL_REPO}"
echo "MAX_NUM_TRANSFORMS=${MAX_NUM_TRANSFORMS}"
echo "IMAGE_TFS=${IMAGE_TFS}"
echo "output_dir=${HDD_PATH}/outputs/${JOB_NAME}"
echo "========================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1,2,3 \
accelerate launch \
  --num_processes=3 \
  --num_machines=1 \
  --dynamo_backend=no \
  --multi_gpu \
  "$(which lerobot-train)" \
  \
  --num_workers=10 \
  \
  --policy.type=pi05 \
  --policy.pretrained_path="${PRETRAINED_MODEL}" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.repo_id="${HF_MODEL_REPO}" \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false \
  --policy.optimizer_fused=true \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps="${STEPS}" \
  --best_loss_warmup_step=5000 \
  \
  --dataset.repo_id="${DB_REPO_ID}" \
  --dataset.revision="${DATASET_REVISION}" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms="${MAX_NUM_TRANSFORMS}" \
  --dataset.image_transforms.tfs="${IMAGE_TFS}" \
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
