# Evaluate LoRA Model on Cloud GPU

Since the LoRA model is large (7B parameters), evaluation is best run on a GPU instance.

## Option 1: Run Evaluation on Same RunPod Instance (Easiest)

If your training RunPod instance is still running:

```bash
# On RunPod terminal
cd /workspace/TraumaLLM

# The model is already there from training!
ls -lh models/lora_adapter/

# Run evaluation (much faster on GPU)
python3 scripts/evaluate_lora.py
```

The evaluation will:
- Use GPU for fast inference
- Run repeated stratified k-fold CV
- Compare against Experiment A and Zero-shot Conservative baselines
- Generate results tables and summary report

## Option 2: Upload Model to New RunPod Instance

If your training instance is stopped:

1. **Create a new RunPod pod** (GPU instance - RTX 3090 or A100)

2. **Clone the repo:**
```bash
cd /workspace
git clone https://github.com/brendan12fox/TraumaLLM.git
cd TraumaLLM
```

3. **Upload the model folder via RunPod web UI:**
   - Navigate to `/workspace/TraumaLLM/models/`
   - Upload the `lora_adapter/` folder you downloaded

4. **Or use scp from your Mac:**
```bash
# From local Mac terminal
scp -r models/lora_adapter root@RUNPOD_HOST:/workspace/TraumaLLM/models/
```

5. **Run evaluation:**
```bash
cd /workspace/TraumaLLM
pip install -r requirements.txt  # If needed
python3 scripts/evaluate_lora.py
```

## Option 3: Evaluate Directly After Training

If you're still training or just finished:

```bash
# On RunPod, after training completes
cd /workspace/TraumaLLM

# Model is already saved in models/lora_adapter/
# Just run evaluation
python3 scripts/evaluate_lora.py
```

## Expected Runtime

- **On GPU (RTX 3090/A100)**: ~10-30 minutes for full CV evaluation
- **On CPU (Mac)**: Several hours, and may cause memory issues

## Output Files

After evaluation completes, you'll get:
- `outputs/experiment_B_results_table.csv` - Main comparison table
- `outputs/experiment_B_summary_report.txt` - Detailed summary

Download these from RunPod to your Mac.
