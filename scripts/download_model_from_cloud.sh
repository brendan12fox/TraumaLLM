#!/bin/bash
# Helper script to download trained model from cloud instance
# Usage: ./download_model_from_cloud.sh user@cloud-instance:/path/to/experiment_B_lora_decision_engine/

if [ -z "$1" ]; then
    echo "Usage: $0 user@host:/path/to/experiment_B_lora_decision_engine/"
    echo ""
    echo "Example (RunPod):"
    echo "  $0 root@runpod-xyz.runpod.io:/workspace/experiment_B_lora_decision_engine/"
    echo ""
    echo "Example (with key):"
    echo "  $0 -i ~/.ssh/id_rsa user@host:/path/to/project/"
    exit 1
fi

CLOUD_PATH="$1"
LOCAL_DIR="$(dirname "$0")/../models"

echo "Downloading model from cloud..."
echo "  From: ${CLOUD_PATH}/models/lora_adapter/"
echo "  To:   ${LOCAL_DIR}/lora_adapter/"

mkdir -p "${LOCAL_DIR}"

# Download using rsync (preserves permissions, handles large files well)
rsync -avz --progress "${CLOUD_PATH}/models/lora_adapter/" "${LOCAL_DIR}/lora_adapter/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model downloaded successfully!"
    echo "   Location: ${LOCAL_DIR}/lora_adapter/"
    echo ""
    echo "Next step: Run evaluation"
    echo "   python3 scripts/evaluate_lora.py"
else
    echo ""
    echo "❌ Download failed. Check:"
    echo "   1. SSH connection works: ssh ${CLOUD_PATH%%:*}"
    echo "   2. Path is correct: ${CLOUD_PATH}/models/lora_adapter/"
    echo "   3. Model training completed successfully"
    exit 1
fi
