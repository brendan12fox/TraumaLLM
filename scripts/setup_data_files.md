# Setting Up Data Files on RunPod

## Quick Upload Guide

### Method 1: RunPod Web UI (Easiest)

1. **Open RunPod Web UI** → Navigate to Jupyter/File Browser
2. **Navigate to** `/workspace/TraumaLLM/data/`
3. **Upload the Excel file**: `OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx`
4. **Create `data/` subfolder for JSON** and upload: `V11_cleaned_transcripts_gpt5nano.json`

### Method 2: Using scp from Mac

From your **local Mac terminal**:

```bash
# Find your RunPod SSH connection info from RunPod dashboard
# Replace USER@HOST with your RunPod SSH details

# Upload Excel file
scp "/Users/brendanfox/Desktop/UB JSMBS/Research/TraumaTriageLLM/CombinedData/OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx" \
    root@YOUR_RUNPOD_HOST:/workspace/TraumaLLM/data/

# Upload cleaned JSON (to data/ folder, script will be updated)
scp "/Users/brendanfox/Desktop/UB JSMBS/Research/TraumaTriageLLM/CombinedData/November 17 complete data set/cleaned_outputs/V11_cleaned_transcripts_gpt5nano.json" \
    root@YOUR_RUNPOD_HOST:/workspace/TraumaLLM/data/
```

### After Uploading Files

Once files are uploaded, run:

```bash
cd /workspace/TraumaLLM
python3 scripts/build_canonical_state.py  # Generate canonical states
python3 scripts/export_training_dataset.py  # Generate instruction dataset
python3 scripts/verify_setup.py  # Verify everything is ready
python3 scripts/train_lora.py  # Start training
```

## File Locations Expected

```
/workspace/TraumaLLM/
├── data/
│   ├── OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx  ← Upload this
│   └── V11_cleaned_transcripts_gpt5nano.json  ← Upload this too
├── outputs/  (will be generated)
└── ...
```
