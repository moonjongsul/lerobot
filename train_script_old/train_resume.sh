HDD_PATH='/data/keti/mjs/lerobot'
CKPT_DIR=${HDD_PATH}/outputs/smolvla_kitting_rot6d_a6000_b24x3_260504/checkpoints/050000

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
  --num_processes=3 \
  --num_machines=1 \
  --dynamo_backend=no \
  --multi_gpu \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --config_path=${CKPT_DIR}/pretrained_model/train_config.json \
  --resume=true
