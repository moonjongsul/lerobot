# ============================================================
# Pi0.5 Cup Hang - red/black hook, rot6d state + SAM3 hook crop camera
# 400 episodes (115K frames, 15fps), 2 tasks (red hook / black hook)
# 원본 yunjuyoung64/smolvla_hang_cup_red_black_rot6d_merged 는 수정하지 않음.
# 학습 데이터는 별도 레포 moonjongsul/kcare_hang_cup_red_black_rot6d_hookcrop.
# pi05_base에서 처음부터 학습. A6000 x 3.
#
# Pi0.5가 실제로 보는 입력 (에피소드마다 매 step, 배치 차원 제외):
#   observation.images.head   float32 (3, 480, 848)  [0,1]  원본 head. 위에 아무것도 그리지 않음
#   observation.images.wrist  float32 (3, 480, 848)  [0,1]  원본 wrist
#   observation.images.hook   float32 (3, 224, 224)  [0,1]  추가 카메라. 왼쪽 112px = head의 기둥 주변 세로 크롭,
#                                                            오른쪽 112px = wrist의 기둥 주변 세로 크롭
#                                                            (hook_crop/hook_crop.py, SAM3 prompt "wooden pole")
#   observation.state         float32 (10,)  px,py,pz, r11,r12,r13, r21,r22,r23, gripper_width
#   task                      "Hang the pink cup on the red hook" 또는 "... black hook"
#   action (학습 타깃)         float32 (7,)  vx,vy,vz, wx,wy,wz, gripper_width
# 데이터셋의 observation.state.eef_* 컬럼은 Pi0.5가 읽지 않는다.
# 세 이미지는 각각 비율을 유지한 채 224x224로 패딩된 뒤 [-1,1]로 들어가고, 빈 카메라 슬롯은 없다.
#
# 추론에서도 같은 입력을 만들어야 한다. 에피소드 시작마다 HookCropper.reset()을 호출하고,
# preprocessor에 넘기기 전에 HookCropper.add_to_observation(obs)로 hook 이미지를 붙인다.
# head 크롭은 에피소드 첫 검출을 유지하고, wrist 크롭은 매 프레임 다시 검출한다(놓치면 이전 박스 유지).
# ============================================================
set -euo pipefail

HF_USER='moonjongsul'

HDD_PATH='/data/keti/mjs/lerobot'

# hook crop 카메라가 추가된 데이터셋 (hook_crop/build_hook_dataset.py 로 생성)
DB_REPO_ID='moonjongsul/kcare_hang_cup_red_black_rot6d_hookcrop'
DATASET_REVISION='4028f5acd42234c34a9f3c0645154b69e88bb71e'

# pretrained pi05_base
# pinned: a538eb27 (2026-06-01, pre-relative_actions, fork compat)
# policy.type + pretrained_path 로 넘기므로 입력 feature는 dataset(head/wrist/hook)을 그대로 사용
PRETRAINED_MODEL='/home/mjs/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/a538eb273274eb30f126a118f39dbc0ee212c883'

# 이미지가 2장 → 3장으로 늘어 b12는 다른 사용자 프로세스가 ~8GB 쓰는 GPU 3에서 OOM → 8
BATCH_SIZE=8

JOB_NAME=pi05_hang_cup_red_black_rot6d_400_hook_b${BATCH_SIZE}x3_261023

STEPS=120000

HF_MODEL_REPO=${HF_USER}/kcare_hang_cup_red_black_rot6d_hook_pi05_261023

# 이미지 증강 (head/wrist/hook 각각 독립 적용, 매 프레임마다 아래 중 정확히 MAX_NUM_TRANSFORMS개를 뽑아 적용)
# - 색으로 컵/고리를 구분하는 태스크라 hue는 제외, saturation은 기본(0.5~1.5)보다 좁힘
# - 같은 설정이 크기가 다른 카메라(848x480, hook 224x224)에 모두 적용되므로 출력 크기를 고정하는
#   RandomResizedCrop 대신 RandomAffine의 scale로 확대, blur도 224 이미지 기준으로 약하게
# - CLI에서는 tfs 전체를 JSON으로만 덮어쓸 수 있음 (기본 6종은 통째로 교체됨)
MAX_NUM_TRANSFORMS=3
IMAGE_TFS='{
  "brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.7, 1.3]}},
  "contrast":   {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.7, 1.3]}},
  "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.7, 1.3]}},
  "sharpness":  {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}},
  "blur":       {"weight": 1.0, "type": "GaussianBlur", "kwargs": {"kernel_size": 7, "sigma": [0.1, 1.5]}},
  "affine":     {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-3.0, 3.0], "translate": [0.02, 0.02], "scale": [1.0, 1.1]}}
}'

echo "=== training config ==="
echo "JOB_NAME=${JOB_NAME}"
echo "DB_REPO_ID=${DB_REPO_ID}"
echo "DATASET_REVISION=${DATASET_REVISION}"
echo "PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "BATCH_SIZE=${BATCH_SIZE} x 3"
echo "STEPS=${STEPS}"
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
