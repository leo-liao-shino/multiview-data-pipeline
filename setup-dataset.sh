if [[ -d /workspace/data/all-multiview-datasets ]]; then
  echo "Dataset all-multiview-datasets already exists at /workspace/data/all-multiview-datasets, skipping sync."
  exit 0
else 

  echo "Syncing all-multiview-datasets from R2..."
  mkdir -p /workspace/data

  rclone sync r2:lawrence-multiview-datasets/all-multiview-datasets /workspace/data/all-multiview-datasets \
    --progress \
    --transfers 32 \
    --checkers 16 \
    --fast-list \
    --buffer-size 512M \
    --s3-chunk-size 64M \
    --retries 5 \
    --retries-sleep 2s

  echo "✓ all-multiview-datasets ready at /workspace/data/all-multiview-datasets"
fi

if [[ -d /workspace/data/multiview-qwen-edit ]]; then
  echo "Dataset multiview-qwen-edit already exists at /workspace/data/multiview-qwen-edit, skipping sync."
  exit 0
else
  echo "Syncing multiview-qwen-edit from R2..."
  mkdir -p /workspace/data

  rclone sync r2:lawrence-multiview-datasets/multiview-qwen-edit /workspace/data/multiview-qwen-edit \
    --progress \
    --transfers 32 \
    --checkers 16 \
    --fast-list \
    --buffer-size 512M \
    --s3-chunk-size 64M \
    --retries 5 \
    --retries-sleep 2s

  echo "✓ multiview-qwen-edit ready at /workspace/data/multiview-qwen-edit "
fi