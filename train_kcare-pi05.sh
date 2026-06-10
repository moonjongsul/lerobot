HDD_PATH='/data/keti/mjs/lerobot'
DB_REPO_ID='moonjongsul/open_drawer_merged'
JOB_NAME='pi05_drawer_joint_a6000_b16x2_260610'
STEPS=50000

PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --num_processes=2 \
  --multi_gpu \
  $(which lerobot-train) \
  --num_workers=4 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 \
  --policy.repo_id=${HF_USER}/kcare_open_drawer_pi05 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=${STEPS} \
  --dataset.repo_id=${DB_REPO_ID} \
  --dataset.image_transforms.enable=true \
  --steps=${STEPS} \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --batch_size=16 \
  --output_dir=${HDD_PATH}/outputs/${JOB_NAME} \
  --job_name=${JOB_NAME}