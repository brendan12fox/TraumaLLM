# Cloud GPU Execution Guide (RunPod / Vast.ai / etc.)

This guide helps you run Experiment B LoRA training on cloud GPU providers.

## Quick Start for RunPod

### 1. Upload Project to Cloud

**Option A: GitHub Clone (Recommended - Easiest!)**

If RunPod is connected to GitHub (via RunPod settings):
```bash
# On cloud instance
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/experiment_B_lora_decision_engine
```

**Benefits:**
- Easy updates: `git pull` to get latest changes
- Version control built-in
- No manual file transfers

See `GITHUB_WORKFLOW.md` for full GitHub setup instructions.

**Option B: Upload via RunPod Web Interface**
1. Upload entire `experiment_B_lora_decision_engine/` folder to RunPod volume via file browser
2. Or use `rclone` / `rsync` to transfer files

### 2. Verify Setup

Run verification script:
```bash
cd experiment_B_lora_decision_engine
python3 scripts/verify_setup.py
```

This checks:
- Required files are present
- Dependencies are installed
- GPU is available

### 3. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

### 4. Re-verify

```bash
python3 scripts/verify_setup.py
```

### 5. Verify Data Files (if verify_setup.py reported issues)

Ensure these files are present:
```bash
ls -la data/OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx
ls -la outputs/instruction_dataset.jsonl
ls -la rules/trauma_triage_rules.json
```

If missing, upload from local machine:
```bash
# From local Mac
scp data/*.xlsx user@cloud-instance:/path/to/experiment_B_lora_decision_engine/data/
scp outputs/*.jsonl user@cloud-instance:/path/to/experiment_B_lora_decision_engine/outputs/
```

### 6. Run Training

```bash
python3 scripts/train_lora.py
```

**Expected runtime**: 1-3 hours depending on GPU (A100 ~1 hour, RTX 3090 ~2-3 hours)

**Outputs**:
- Model will be saved to `models/lora_adapter/`
- Checkpoints saved during training
- Training logs printed to console

### 7. Download Trained Model

**Option A: RunPod Web Interface**
1. Navigate to volumes/storage
2. Download `experiment_B_lora_decision_engine/models/lora_adapter/` folder
3. Extract locally to: `experiment_B_lora_decision_engine/models/lora_adapter/`

**Option B: Using rclone/rsync**
```bash
# From local Mac
rsync -avz user@cloud-instance:/path/to/experiment_B_lora_decision_engine/models/ ./models/
```

**Option C: Using RunPod's Download Feature**
- Right-click `models/lora_adapter/` folder in file browser
- Select "Download"

### 8. Local Evaluation (Back on Mac)

After downloading the model:
```bash
cd experiment_B_lora_decision_engine
python3 scripts/evaluate_lora.py
```

Note: Evaluation can run locally (CPU-only) since it just loads the trained model for inference.

## File Structure on Cloud

```
experiment_B_lora_decision_engine/
├── data/
│   └── OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx
├── rules/
│   └── trauma_triage_rules.json
├── outputs/
│   ├── canonical_states.jsonl
│   ├── instruction_dataset.jsonl
│   └── preflight_report.txt
├── models/                    # Created by training
│   └── lora_adapter/          # Download this folder
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       └── ...
├── scripts/
│   ├── train_lora.py          # Run this on cloud
│   └── ...
├── requirements.txt
└── CLOUD_EXECUTION.md
```

## Cloud Provider Specific Notes

### RunPod
- Use "PyTorch" template
- Recommended: RTX 3090 (24GB) or A100 (40GB)
- Attach persistent volume for data/model storage
- SSH access available

### Vast.ai
- Select instance with CUDA support
- Upload files via SSH
- Use same commands as above

### Google Colab
- Upload project to Google Drive
- Mount drive in Colab
- Install dependencies in Colab cell
- Run training
- Download `models/lora_adapter/` folder

### AWS SageMaker / GCP Vertex AI
- Package project as Docker container or upload to S3/GCS
- Use provided ML training frameworks
- Download model artifacts to local machine

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `train_lora.py`: `BATCH_SIZE = 2`
- Increase gradient accumulation: `GRADIENT_ACCUMULATION_STEPS = 8`

### Model Not Saving
- Check disk space: `df -h`
- Verify write permissions: `ls -la models/`

### Training Stalls
- Check GPU utilization: `nvidia-smi`
- Monitor logs for errors
- Reduce sequence length: `MAX_SEQ_LENGTH = 1024`

## Verification After Training

On cloud instance, verify model was saved:
```bash
ls -lh models/lora_adapter/
# Should see:
# adapter_config.json
# adapter_model.bin (or adapter_model.safetensors)
```

File size should be ~100-500MB (LoRA adapters are small).

## Download Checklist

Before terminating cloud instance, ensure you have:
- [ ] `models/lora_adapter/adapter_config.json`
- [ ] `models/lora_adapter/adapter_model.bin` (or `.safetensors`)
- [ ] Any training logs/summaries you want

You do NOT need:
- Base model weights (will download automatically on local evaluation)
- Checkpoints (only final adapter needed)
