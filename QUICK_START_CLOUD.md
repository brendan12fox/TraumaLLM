# Quick Start: Cloud Training

**TL;DR**: Upload project to cloud GPU → Install deps → Train → Download model

## Step-by-Step

### 1. Prepare Locally (on Mac)

Ensure your project is ready:
```bash
cd experiment_B_lora_decision_engine
python3 scripts/preflight_checks.py  # Should pass
```

### 2. Upload to Cloud

**Option A: GitHub (Recommended - Easiest!)**

If your RunPod is connected to GitHub:
```bash
# On RunPod
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/experiment_B_lora_decision_engine
```

See `GITHUB_WORKFLOW.md` for full setup instructions.

**Option B: Manual Upload (RunPod):**
- Create a RunPod pod (PyTorch template, RTX 3090 or A100)
- Upload `experiment_B_lora_decision_engine/` folder to pod volume via web UI

### 3. On Cloud Instance

```bash
# Navigate to project
cd experiment_B_lora_decision_engine

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Run training
python3 scripts/train_lora.py
```

Training will take 1-3 hours. Monitor with:
```bash
# In another terminal
watch -n 5 nvidia-smi
```

### 4. Download Model

After training completes:

**RunPod Web UI:**
1. Open file browser
2. Navigate to `experiment_B_lora_decision_engine/models/lora_adapter/`
3. Right-click → Download (or zip first)

**Or via SSH:**
```bash
# From local Mac
scp -r user@runpod-instance:/workspace/experiment_B_lora_decision_engine/models/lora_adapter \
      ./models/
```

### 5. Evaluate Locally

```bash
# Back on Mac
cd experiment_B_lora_decision_engine
python3 scripts/evaluate_lora.py
```

## Expected Output Structure

```
models/lora_adapter/
├── adapter_config.json      (~1 KB)
├── adapter_model.bin        (~100-500 MB)
└── tokenizer files...
```

Total size: ~100-500 MB (much smaller than full model)

## Troubleshooting

**"CUDA out of memory"**
- Edit `scripts/train_lora.py`: Set `BATCH_SIZE = 2`

**"File not found"**
- Check `outputs/instruction_dataset.jsonl` exists
- Re-run `python3 scripts/export_training_dataset.py` if needed

**Training seems stuck**
- Normal: First epoch can take 30+ minutes
- Check GPU usage: `nvidia-smi` should show >80% utilization

## What to Download

**Required:**
- `models/lora_adapter/` (entire folder)

**Optional:**
- Training logs (if saved)
- Checkpoint files (only if you want to resume training)

**Not needed:**
- Base model weights (Mistral 7B downloads automatically)
