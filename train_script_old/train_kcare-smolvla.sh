HDD_PATH='/data/keti/mjs/lerobot'
DB_REPO_ID='moonjongsul/open_drawer_merged'
JOB_NAME='smolvla_drawer_open_joint_a6000_b24x4_260610'
STEPS=100000

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --dynamo_backend=no \
  --multi_gpu \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --num_workers=4 \
  --policy.type=smolvla \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
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
  --batch_size=24 \
  --output_dir=${HDD_PATH}/outputs/${JOB_NAME} \
  --job_name=${JOB_NAME}