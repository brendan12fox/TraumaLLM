# Experiment B: LoRA Decision Policy on Structured State (+ Structured-RAG)

## ⚡ Quick Start: Cloud Execution

**Training requires GPU (CUDA). This project is configured for cloud execution.**

### Recommended: GitHub Workflow (Easiest!)

1. **Setup GitHub**: See [`SETUP_GITHUB.md`](SETUP_GITHUB.md) - Push code to GitHub (5 min)
2. **On RunPod**: Clone repo → Install deps → Train
3. **Download Model**: See [`DOWNLOAD_MODEL.md`](DOWNLOAD_MODEL.md) - After training completes

**Workflow**: Push to GitHub → Clone on cloud → Train → Download model → Evaluate locally

### Alternative: Manual Upload

1. **Quick Guide**: See [`QUICK_START_CLOUD.md`](QUICK_START_CLOUD.md) - 5 minute setup
2. **Full Guide**: See [`CLOUD_EXECUTION.md`](CLOUD_EXECUTION.md) - Detailed instructions

---

## Goal

Train a small local model (e.g., LLaMA/Mistral 7–8B) via LoRA to predict:
- `iss_level` (ISS > 15 vs not)
- Controlled explanations (drivers)

Using only:
- Structured entities (from GPT-5-nano JSON or Excel entities)
- Structured criteria retrieved from a rules store (structured RAG)

**No raw transcript text in training input.**

## Folder Structure

```
experiment_B_lora_decision_engine/
├── data/                    # Dataset copies (never edit master)
├── rules/                   # Trauma triage rules store (JSON)
├── models/                  # Trained LoRA models (to be generated)
├── outputs/                 # Generated datasets and results
├── scripts/                 # Processing scripts
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+
- pandas, numpy, openpyxl (for Excel reading)
- Master dataset: `OCH_RCH_2023_2025_Combined_Master_V11.xlsx`
- Cleaned JSON: `V11_cleaned_transcripts_gpt5nano.json`

## Three Core Components (Completed)

### 1. Canonical State JSON Generator

**Script**: `scripts/build_canonical_state.py`

Builds a single canonical JSON object per case containing:
- **Required fields**: age, hr, sbp, dbp, rr, gcs, mental_status, moi, num_patients, site
- **Context features**: green_wc, percent_complex, hard_terms_ct, reduction_pct
- **Missingness bookkeeping**: 
  - `missing_original`: flags based on original Excel missingness
  - `missing_final`: flags after JSON supplementation
- **Abnormality flags**: hypotension_for_age, gcs_low, resp_distress, ams

Uses the same supplementation logic from Experiment A (fills missing Excel entities from cleaned JSON).

**Output**: `outputs/canonical_states.jsonl` (one JSON object per line)

### 2. Rules Store + Structured-RAG Retrieval

**Rules Store**: `rules/trauma_triage_rules.json`

Contains structured rules as JSON (not text docs):
- Hypotension thresholds (age-adjusted)
- Low GCS criteria
- High-risk MOI patterns
- Altered mental status indicators
- Respiratory distress thresholds
- Insufficient info criteria

**Retrieval Script**: `scripts/retrieve_rules.py`

Deterministic, rule-based retrieval:
- Given canonical state, retrieves relevant rules
- Returns simplified rule representations for model input
- Auditable and deterministic (not semantic search)

**Driver Vocabulary** (constrained):
- `hypotension`
- `low_gcs`
- `ams`
- `high_risk_moi`
- `resp_distress`
- `multi_system_injury_suspected`
- `insufficient_info`

### 3. Instruction Fine-Tuning Dataset Export

**Script**: `scripts/export_training_dataset.py`

Creates training examples in instruction fine-tuning format:

**Input Format**:
```
STATE_JSON: {...}
RETRIEVED_RULES_JSON: {...}
```

**Output Format** (strict JSON schema):
```json
{
  "triage_level": "L1" | "L2",
  "confidence": 0.0-1.0,
  "drivers": ["driver1", "driver2"],  // max 3, from vocabulary
  "notes": "brief description"  // <= 25 words
}
```

**Training Example Structure**:
```json
{
  "instruction": "You are a trauma triage decision engine...",
  "input": "STATE_JSON: {...}\nRETRIEVED_RULES_JSON: {...}",
  "output": "{\"triage_level\":\"L1\",\"confidence\":0.85,\"drivers\":[...],\"notes\":\"...\"}",
  "metadata": {
    "audio_file": "12345",
    "triage_level": "L1"
  }
}
```

**Output**: `outputs/instruction_dataset.jsonl`

## Running the Pipeline

### Step 1: Build Canonical States

```bash
cd experiment_B_lora_decision_engine
python3 scripts/build_canonical_state.py
```

This creates `outputs/canonical_states.jsonl`

### Step 2: Export Training Dataset

```bash
python3 scripts/export_training_dataset.py
```

This creates `outputs/instruction_dataset.jsonl` ready for LoRA fine-tuning.

## Dataset Statistics

- **Total cases**: 351 (after exclusion criteria)
- **L1 cases** (ISS > 15): 46 (13.1%)
- **L2 cases** (ISS ≤ 15): 305 (86.9%)
- **Class imbalance**: ~7:1 (L2:L1)

## Cloud GPU Execution

**Note**: Training requires GPU (CUDA). This project is configured for cloud execution (RunPod, Vast.ai, etc.).

**Quick Start**: See `QUICK_START_CLOUD.md` for step-by-step instructions.

**Full Guide**: See `CLOUD_EXECUTION.md` for detailed cloud provider setup.

### Setup on Cloud

1. Upload project to cloud GPU instance
2. Install: `pip install -r requirements.txt`
3. Verify GPU: Should show CUDA available
4. Run training: `python3 scripts/train_lora.py`
5. Download `models/lora_adapter/` folder back to local machine
6. Evaluate locally: `python3 scripts/evaluate_lora.py`

## Execution Phase (Current)

### Step 1: Preflight Checks (MANDATORY)

Run preflight validation before training:

```bash
python3 scripts/preflight_checks.py
```

This validates:
- Dataset integrity (351 examples, correct class balance)
- Schema enforcement (all inputs/outputs valid)
- Leakage check (no ISS, decisions, or transcripts)
- Rule retrieval sanity

**Output**: `outputs/preflight_report.txt`

**👉 Do not proceed to training unless all checks pass.**

### Step 2: LoRA Training

**Prerequisites**:
- GPU with CUDA support
- Install: `pip install transformers peft datasets bitsandbytes accelerate torch`

**Train model**:

```bash
python3 scripts/train_lora.py
```

**Configuration** (single, no tuning):
- Model: Mistral 7B Instruct
- Method: QLoRA (4-bit)
- LoRA rank: 8
- Class-weighted loss
- Early stopping

**Output**: `models/lora_adapter/`

### Step 3: Evaluation

Evaluate trained model with same CV as Experiment A:

```bash
python3 scripts/evaluate_lora.py
```

**Evaluation**:
- Repeated stratified k-fold CV (5-fold × 10 repeats)
- Same metrics as Experiment A
- Compares against: Structured LR + Zero-shot Conservative LLM

**Outputs**:
- `outputs/experiment_B_results_table.csv` - Main comparison table
- `outputs/experiment_B_summary_report.txt` - Summary and recommendations

## Next Steps (LoRA Training - Legacy Notes)

Once the instruction dataset is ready, use it to fine-tune a small local model:

### Model Recommendations
- Mistral 7B Instruct
- LLaMA 3.1 8B Instruct

### Training Settings
- QLoRA (4-bit) preferred
- Small LoRA rank (8–16)
- Strong dropout
- Early stopping
- Class-weighted loss or balanced sampling

### Training Tools
- Hugging Face Transformers + PEFT
- Axolotl
- Unsloth

## Validation Strategy (Post-Training)

Use the same rigorous validation as Experiment A:
- Repeated stratified k-fold CV (5-fold × 10 repeats)
- Report sensitivity (primary), specificity, precision, PR-AUC, calibration

### Comparisons Required

Compare against:
1. Structured LR baseline (Experiment A)
2. Zero-shot Conservative LLM baseline
3. Best LLM system (91.1% accuracy)

**Comparison must be sensitivity-matched**: Pick minimum sensitivity target and compare specificity/precision at that operating point.

## Success Criteria

LoRA is worth it if it achieves either:
- Higher specificity than LR at the same sensitivity, OR
- Higher sensitivity than conservative LLM without destroying specificity, OR
- Better calibration/stability across low-context cases

If it doesn't beat LR meaningfully, stop and use LR + gating.

## Deployment Shape (If Successful)

If LoRA works, production system becomes:
1. Whisper transcript → GPT-5-nano entity JSON
2. Structured-RAG inject thresholds + rule context
3. Local LoRA model predicts triage + drivers
4. If insufficient_info or low confidence → escalate to GPT-4o

This is auditable and cheap.

## Critical Warnings

1. **Do not fine-tune on raw Whisper text** with n=372. You will overfit and learn artifacts.
2. **Do not allow free-form explanations**. Constrain to driver vocabulary, or you'll get confident nonsense.

## Execution Scripts

- `scripts/preflight_checks.py`: Validates dataset before training (MANDATORY)
- `scripts/train_lora.py`: Trains QLoRA adapter on instruction dataset
- `scripts/evaluate_lora.py`: Evaluates model with CV and compares to baselines

## Data Preparation Scripts

- `scripts/build_canonical_state.py`: Builds canonical state JSON from Excel + JSON entities
- `scripts/retrieve_rules.py`: Deterministic rule retrieval for structured-RAG
- `scripts/export_training_dataset.py`: Exports instruction fine-tuning dataset

## Files and Outputs

- `rules/trauma_triage_rules.json`: Structured rules store
- `outputs/canonical_states.jsonl`: Generated canonical states (intermediate)
- `outputs/instruction_dataset.jsonl`: Final training dataset for LoRA
- `outputs/preflight_report.txt`: Preflight validation results
- `outputs/experiment_B_results_table.csv`: Main comparison table
- `outputs/experiment_B_summary_report.txt`: Evaluation summary
- `models/lora_adapter/`: Trained LoRA model (generated after training)

## Success Criteria & Stop Conditions

LoRA is worth deploying if it achieves **any** of:
1. Higher specificity than LR at the same sensitivity, OR
2. Higher sensitivity than conservative LLM without destroying specificity, OR
3. Better calibration/stability across low-context cases

**If none are true**: Use LR + gating + GPT fallback instead.
