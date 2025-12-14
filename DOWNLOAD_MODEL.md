# Downloading Trained Model from Cloud

After training completes on cloud GPU, download the model for local evaluation.

## What to Download

**Required:**
- `models/lora_adapter/` folder (contains adapter weights)

**Total size:** ~100-500 MB (much smaller than full 7B model)

## Download Methods

### Method 1: RunPod Web UI (Easiest)

1. Open RunPod pod in browser
2. Navigate to file browser
3. Go to: `experiment_B_lora_decision_engine/models/lora_adapter/`
4. Select all files in folder
5. Right-click → "Download" (or zip first for faster download)

Extract locally to: `experiment_B_lora_decision_engine/models/lora_adapter/`

### Method 2: rsync (Recommended for Large Files)

From your local Mac terminal:

```bash
cd experiment_B_lora_decision_engine

# Replace with your cloud instance details
rsync -avz --progress \
  user@runpod-instance:/workspace/experiment_B_lora_decision_engine/models/lora_adapter/ \
  ./models/lora_adapter/
```

### Method 3: Using Helper Script

```bash
cd experiment_B_lora_decision_engine
./scripts/download_model_from_cloud.sh user@host:/path/to/experiment_B_lora_decision_engine/
```

### Method 4: scp (Simple)

```bash
scp -r user@cloud-instance:/path/to/experiment_B_lora_decision_engine/models/lora_adapter \
      ./models/
```

## Verify Download

After downloading, verify files are present:

```bash
ls -lh models/lora_adapter/
```

Should see:
- `adapter_config.json` (~1 KB)
- `adapter_model.bin` or `adapter_model.safetensors` (~100-500 MB)
- Tokenizer files (if saved)

## Next Steps

Once model is downloaded locally:

```bash
python3 scripts/evaluate_lora.py
```

This will:
- Load the trained LoRA adapter
- Run cross-validation evaluation
- Compare against baselines
- Generate results tables

## Troubleshooting

**"Model directory not found"**
- Check path on cloud: `ls -la models/lora_adapter/`
- Verify training completed successfully

**Download is slow**
- Compress first: `tar -czf lora_adapter.tar.gz models/lora_adapter/`
- Download the tar.gz file
- Extract locally: `tar -xzf lora_adapter.tar.gz`

**"CUDA not available" (during evaluation)**
- This is normal! Evaluation can run on CPU
- It will be slower but works fine for inference
