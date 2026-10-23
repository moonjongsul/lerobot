# ============================================================
# Pi0.5 Cup Hang - red/black hook, rot6d state (merged) - resume with augmentation
# pi05_hang_cup_red_black_rot6d_400_b12x3_261016 last(80K) 체크포인트에서 이어서 학습
# A6000 x 3
# ============================================================
set -euo pipefail

HF_USER='moonjongsul'

HDD_PATH='/data/keti/mjs/lerobot'

# 이어서 학습할 체크포인트 (optimizer/scheduler/step/wandb run 모두 여기서 복원)
RESUME_JOB=pi05_hang_cup_red_black_rot6d_400_b12x3_261016
RESUME_CKPT=${HDD_PATH}/outputs/${RESUME_JOB}/checkpoints/last

# 새 체크포인트는 원래 폴더와 섞이지 않게 별도 폴더에 저장
JOB_NAME=pi05_hang_cup_red_black_rot6d_400_aug_resume80k_b12x3_261020

# 총 step (80K에서 이어서 +40K)
STEPS=120000

# 원래 cosine decay는 80K에서 끝나 lr=2.5e-6에 고정돼 있음.
# decay 구간을 STEPS까지 늘리면 80K에서 lr이 ~8.1e-6으로 올라갔다가 120K에서 2.5e-6으로 다시 내려감
SCHEDULER_DECAY_STEPS=${STEPS}

# resume 직후엔 증강 때문에 loss가 일시적으로 올라가므로, best 저장은 10K step 적응 후부터
BEST_LOSS_WARMUP_STEP=90000

HF_MODEL_REPO=${HF_USER}/kcare_hang_cup_red_black_rot6d_aug_pi05_261020

# 이미지 증강 (head/wrist 각각 독립 적용, 매 프레임마다 아래 중 정확히 MAX_NUM_TRANSFORMS개를 뽑아 적용)
# - 색으로 컵/고리를 구분하는 태스크라 hue는 제외, saturation은 기본(0.5~1.5)보다 좁힘
# - 손잡이-고리 정렬 정밀도 때문에 기하 변환(crop/affine)은 작게
# - resume 시 tfs JSON은 저장된 설정과 key 단위로 합쳐짐 → 빼려는 항목은 weight 0으로 명시해야 꺼짐
MAX_NUM_TRANSFORMS=3
IMAGE_TFS='{
  "hue":        {"weight": 0.0, "type": "ColorJitter", "kwargs": {"hue": [0.0, 0.0]}},
  "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
  "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.7, 1.3]}},
  "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.7, 1.3]}},
  "sharpness":  {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
  "blur":       {"weight": 1.0, "type": "GaussianBlur", "kwargs": {"kernel_size": 11, "sigma": [0.1, 2.5]}},
  "crop":       {"weight": 1.0, "type": "RandomResizedCrop", "kwargs": {"size": [480, 848], "scale": [0.9, 1.0], "ratio": [1.75, 1.78]}},
  "affine":     {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-3.0, 3.0], "translate": [0.02, 0.02]}}
}'

echo "=== resume config ==="
echo "RESUME_CKPT=${RESUME_CKPT} -> $(readlink "${RESUME_CKPT}")"
echo "JOB_NAME=${JOB_NAME}"
echo "STEPS=${STEPS} (scheduler decay ${SCHEDULER_DECAY_STEPS})"
echo "BEST_LOSS_WARMUP_STEP=${BEST_LOSS_WARMUP_STEP}"
echo "CUDA_VISIBLE_DEVICES=1,2,3"
echo "HF_MODEL_REPO=${HF_MODEL_REPO}"
echo "output_dir=${HDD_PATH}/outputs/${JOB_NAME}"
echo "MAX_NUM_TRANSFORMS=${MAX_NUM_TRANSFORMS}"
echo "IMAGE_TFS=${IMAGE_TFS}"
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
  --config_path="${RESUME_CKPT}/pretrained_model/train_config.json" \
  --resume=true \
  \
  --steps="${STEPS}" \
  --scheduler.num_decay_steps="${SCHEDULER_DECAY_STEPS}" \
  --best_loss_warmup_step="${BEST_LOSS_WARMUP_STEP}" \
  \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms="${MAX_NUM_TRANSFORMS}" \
  --dataset.image_transforms.tfs="${IMAGE_TFS}" \
  \
  --output_dir="${HDD_PATH}/outputs/${JOB_NAME}" \
  --job_name="${JOB_NAME}"
