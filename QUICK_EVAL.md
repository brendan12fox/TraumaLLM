# Quick Evaluation on RunPod

Since your model is already on RunPod from training, evaluation is simple:

## Steps on RunPod:

```bash
# 1. Navigate to project
cd /workspace/TraumaLLM

# 2. Pull latest code changes
git pull origin main

# 3. Verify model is still there
ls -lh models/lora_adapter/

# 4. Run evaluation (GPU accelerated, ~10-30 min)
python3 scripts/evaluate_lora.py
```

That's it! The model is already saved from training, so no need to upload anything.

## After Evaluation:

Download the results from RunPod web UI:
- `outputs/experiment_B_results_table.csv`
- `outputs/experiment_B_summary_report.txt`
