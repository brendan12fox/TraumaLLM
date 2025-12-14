# Project Structure

```
experiment_B_lora_decision_engine/
│
├── 📋 Documentation
│   ├── README.md                    # Main documentation
│   ├── QUICK_START_CLOUD.md         # Quick cloud setup guide
│   ├── CLOUD_EXECUTION.md           # Detailed cloud execution guide
│   ├── DOWNLOAD_MODEL.md            # How to download trained model
│   ├── EXECUTION_SUMMARY.md         # Execution phase status
│   └── PROJECT_STRUCTURE.md         # This file
│
├── 🐍 Python Scripts
│   └── scripts/
│       ├── build_canonical_state.py      # Build canonical state JSON
│       ├── retrieve_rules.py             # Structured-RAG retrieval
│       ├── export_training_dataset.py    # Export instruction dataset
│       ├── preflight_checks.py           # Validate before training
│       ├── verify_setup.py               # Verify cloud setup
│       ├── train_lora.py                 # Train LoRA model (cloud)
│       ├── evaluate_lora.py              # Evaluate model (local)
│       └── download_model_from_cloud.sh  # Helper: download model
│
├── 📊 Data
│   ├── data/
│   │   ├── .gitkeep
│   │   └── OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx
│   │
│   └── outputs/
│       ├── .gitkeep
│       ├── canonical_states.jsonl        # Generated canonical states
│       ├── instruction_dataset.jsonl     # Training dataset (351 examples)
│       ├── preflight_report.txt          # Preflight validation results
│       ├── experiment_B_results_table.csv      # Evaluation results (after eval)
│       └── experiment_B_summary_report.txt     # Summary report (after eval)
│
├── 📐 Rules
│   └── rules/
│       └── trauma_triage_rules.json      # Structured rules store
│
├── 🤖 Models (Generated)
│   └── models/
│       ├── .gitkeep
│       └── lora_adapter/                 # Trained LoRA adapter (after training)
│           ├── adapter_config.json
│           ├── adapter_model.bin
│           └── ...
│
├── ⚙️ Configuration
│   ├── requirements.txt                 # Python dependencies
│   └── .gitignore                       # Git ignore rules
│
└── 📝 Execution Flow
    │
    ├── Phase 1: Data Preparation (Local/Cloud)
    │   └── Run: python3 scripts/export_training_dataset.py
    │
    ├── Phase 2: Validation (Local/Cloud)
    │   └── Run: python3 scripts/preflight_checks.py
    │
    ├── Phase 3: Training (Cloud GPU Only)
    │   └── Run: python3 scripts/train_lora.py
    │   └── Output: models/lora_adapter/
    │
    ├── Phase 4: Download (From Cloud)
    │   └── Download models/lora_adapter/ to local machine
    │
    └── Phase 5: Evaluation (Local)
        └── Run: python3 scripts/evaluate_lora.py
        └── Output: Comparison tables and reports
```

## Key Files

### For Cloud Execution
- `requirements.txt` - Install dependencies
- `scripts/verify_setup.py` - Check setup before training
- `scripts/train_lora.py` - Training script

### For Local Evaluation
- `scripts/evaluate_lora.py` - Evaluation script
- `models/lora_adapter/` - Trained model (download from cloud)

### Data Files
- `outputs/instruction_dataset.jsonl` - Training dataset (351 examples)
- `rules/trauma_triage_rules.json` - Rules for structured-RAG
- `data/OCH_RCH_*.xlsx` - Master dataset copy

## File Sizes (Approximate)

- `instruction_dataset.jsonl`: ~0.5 MB
- `canonical_states.jsonl`: ~0.5 MB
- `trauma_triage_rules.json`: <10 KB
- `lora_adapter/` (trained): ~100-500 MB

Total project size: ~1-2 MB (without trained model)
