# Quick Fix: Upload Missing Files on RunPod

## Step 1: Pull Latest Changes

```bash
cd /workspace/TraumaLLM
git pull origin main
```

## Step 2: Upload Two Files via RunPod Web UI

Open RunPod file browser and navigate to `/workspace/TraumaLLM/data/`

### Upload File 1: Excel Dataset
- **Local path on Mac**: `CombinedData/OCH_RCH_2023_2025_Combined_Master_V11.xlsx`
  (or `OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx` if you have a copy)
- **Upload to**: `/workspace/TraumaLLM/data/OCH_RCH_2023_2025_Combined_Master_V11.xlsx`
  (script will accept either `V11.xlsx` or `V11_EXP_B_COPY.xlsx` filename)

### Upload File 2: Cleaned JSON
- **Local path on Mac**: `CombinedData/November 17 complete data set/cleaned_outputs/V11_cleaned_transcripts_gpt5nano.json`
- **Upload to**: `/workspace/TraumaLLM/data/V11_cleaned_transcripts_gpt5nano.json`

## Step 3: Generate Instruction Dataset

After uploading both files, run:

```bash
cd /workspace/TraumaLLM

# Generate canonical states (intermediate step)
python3 scripts/build_canonical_state.py

# Generate instruction dataset (for training)
python3 scripts/export_training_dataset.py

# Verify everything is ready
python3 scripts/verify_setup.py
```

## Step 4: Start Training

Once verification passes:

```bash
python3 scripts/train_lora.py
```

---

## Alternative: Using scp (if you prefer command line)

From your **local Mac terminal**:

```bash
# Get your RunPod SSH info from RunPod dashboard

# Upload Excel (use actual filename you have)
scp "CombinedData/OCH_RCH_2023_2025_Combined_Master_V11.xlsx" \
    root@YOUR_RUNPOD_HOST:/workspace/TraumaLLM/data/

# Upload JSON  
scp "CombinedData/November 17 complete data set/cleaned_outputs/V11_cleaned_transcripts_gpt5nano.json" \
    root@YOUR_RUNPOD_HOST:/workspace/TraumaLLM/data/
```
