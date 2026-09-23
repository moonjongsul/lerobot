HDD_PATH='/data/keti/mjs/lerobot'
DB_REPO_ID='moonjongsul/open_drawer_merged'
JOB_NAME='smolvla_drawer_open_joint_spark_b128x1_260610'
STEPS=100000

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --num_workers=4 \
  --policy.type=smolvla \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.repo_id=${HF_USER}/kcare_open_drawer_smolvla \
  --policy.train_expert_only=false \
  --policy.scheduler_decay_steps=${STEPS} \
  --policy.optimizer_weight_decay=1e-4 \
  --dataset.repo_id=${DB_REPO_ID} \
  --dataset.image_transforms.enable=true \
  --steps=${STEPS} \
  --save_freq=10000 \
  --log_freq=100 \
  --wandb.enable=true \
  --batch_size=64 \
  --output_dir=${HDD_PATH}/outputs/${JOB_NAME} \
  --job_name=${JOB_NAME}