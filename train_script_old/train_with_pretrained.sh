HDD_PATH='/data/keti/mjs/lerobot'

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
  --num_processes=3 \
  --num_machines=1 \
  --dynamo_backend=no \
  --multi_gpu \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --num_workers=4 \
  --policy.path=${HDD_PATH}/outputs/smolvla_kitting_rot6d_a6000_b24x3_260430_v2/checkpoints/100000/pretrained_model \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.repo_id=${HF_USER}/manufacturing-kitting-smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.scheduler_decay_steps=200000 \
  --policy.optimizer_weight_decay=1e-4 \
  --dataset.repo_id=moonjongsul/manufacturing_kitting_dataset \
  --dataset.image_transforms.enable=true \
  --steps=500000 \
  --save_freq=5000 \
  --log_freq=100 \
  --best_loss_warmup_step=50000 \
  --wandb.enable=true \
  --batch_size=24 \
  --output_dir=${HDD_PATH}/outputs/smolvla_kitting_rot6d_a6000_b24x3_260504 \
  --job_name=smolvla_kitting_rot6d_a6000_b24x3_260504
