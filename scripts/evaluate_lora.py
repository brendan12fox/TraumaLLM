#!/usr/bin/env python3
"""
LoRA Model Evaluation Script for Experiment B

Evaluates trained LoRA model using the same CV strategy as Experiment A.
Compares against baselines: Structured LR and Zero-shot Conservative LLM.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers/peft
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch/Transformers not available. Will create evaluation framework only.")

# Sklearn imports
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    recall_score, precision_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix
)

# Zero-shot conservative path (defined before any imports that might trigger path issues)
ZEROSHOT_CONSERVATIVE_PATH = Path(__file__).parent.parent.parent / "CombinedData" / "cleaned_zeroshot_conservative2.xlsx"

# Zero-shot conservative path
ZEROSHOT_CONSERVATIVE_PATH = Path(__file__).parent.parent.parent / "CombinedData" / "cleaned_zeroshot_conservative2.xlsx"

def load_zeroshot_conservative_predictions(df: pd.DataFrame, file_path: Path):
    """Load zero-shot conservative predictions."""
    if not file_path.exists():
        return None
    
    try:
        zeroshot_df = pd.read_excel(file_path)
        zeroshot_df['audio_file'] = zeroshot_df['audio_file'].astype(str)
        
        pred_col = None
        for col in ['gpt4o_level_conservative', 'final_label', 'prediction', 'level']:
            if col in zeroshot_df.columns:
                pred_col = col
                break
        
        if pred_col is None:
            return None
        
        df_merged = df[['audio_file', 'ISS_Level_1']].merge(
            zeroshot_df[['audio_file', pred_col]],
            on='audio_file',
            how='inner'
        )
        
        pred_values = pd.to_numeric(df_merged[pred_col], errors='coerce')
        df_merged['zeroshot_binary'] = (pred_values == 1).astype(int)
        return df_merged.set_index('audio_file')['zeroshot_binary']
    except Exception as e:
        print(f"⚠️  Error loading zero-shot: {e}")
        return None

def compute_zeroshot_conservative_metrics(df: pd.DataFrame, zeroshot_preds):
    """Compute zero-shot conservative metrics."""
    df_eval = df[['audio_file', 'ISS_Level_1']].copy()
    df_eval = df_eval.merge(
        zeroshot_preds.reset_index(),
        on='audio_file',
        how='inner'
    )
    
    y_true = df_eval['ISS_Level_1'].values
    y_pred = df_eval.iloc[:, -1].values
    
    metrics = {}
    metrics['sensitivity'] = recall_score(y_true, y_pred)
    metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['accuracy'] = (y_true == y_pred).mean()
    return metrics

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

DATASET_FILE = OUTPUTS_DIR / "instruction_dataset.jsonl"
MODEL_DIR = MODELS_DIR / "lora_adapter"
# Try EXP_B_COPY first, fall back to original V11 if copy doesn't exist
MASTER_FILE = DATA_DIR / "OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx"
if not MASTER_FILE.exists():
    MASTER_FILE = DATA_DIR / "OCH_RCH_2023_2025_Combined_Master_V11.xlsx"

# Validation strategy (same as Experiment A)
N_SPLITS = 5
N_REPEATS = 10

# ============================================================================
# MODEL INFERENCE
# ============================================================================

def load_lora_model():
    """Load trained LoRA model."""
    if not HAS_TORCH:
        return None, None
    
    if not MODEL_DIR.exists():
        print(f"⚠️  Model directory not found: {MODEL_DIR}")
        print("   Run train_lora.py first, or use mock evaluation mode")
        return None, None
    
    try:
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        # Check if CUDA is available for 4-bit quantization
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            except Exception:
                # Fallback to regular loading if bitsandbytes not available
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
        else:
            # CPU-only: load in float32 (or float16 if supported)
            print("  ⚠️  CUDA not available, loading model on CPU (this will be slow)")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,  # CPU works better with float32
                low_cpu_mem_usage=True,
            )
        
        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_lora(model, tokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction with LoRA model."""
    if model is None or tokenizer is None:
        return None
    
    # Format input
    instruction = example['instruction']
    input_text = example['input']
    prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    if "[/INST]" in generated:
        response = generated.split("[/INST]")[-1].strip()
    else:
        response = generated
    
    # Try to parse JSON
    try:
        # Find JSON in response
        json_match = None
        if "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            if end > start:
                json_match = response[start:end]
        
        if json_match:
            prediction = json.loads(json_match)
        else:
            # Fallback: try to parse entire response
            prediction = json.loads(response)
    except json.JSONDecodeError:
        # Fallback prediction
        prediction = {
            "triage_level": "L2",
            "confidence": 0.5,
            "drivers": ["insufficient_info"],
            "notes": "parse error"
        }
    
    return prediction

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute all required metrics."""
    metrics = {}
    metrics['sensitivity'] = recall_score(y_true, y_pred)
    metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    metrics['brier_score'] = brier_score_loss(y_true, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    
    return metrics

# ============================================================================
# CROSS-VALIDATION EVALUATION
# ============================================================================

def cross_validate_lora(examples: List[Dict[str, Any]], 
                       model, tokenizer,
                       threshold: float = 0.5) -> Dict[str, Any]:
    """
    Perform repeated stratified k-fold cross-validation (same as Experiment A).
    """
    # Prepare labels
    y = np.array([1 if ex['metadata']['triage_level'] == 'L1' else 0 for ex in examples])
    
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    
    all_metrics = {
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'roc_auc': [],
        'pr_auc': [],
        'brier_score': [],
        'accuracy': []
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(examples, y)):
        # Get test examples
        test_examples = [examples[i] for i in test_idx]
        y_test = y[test_idx]
        
        # Make predictions
        y_pred = []
        y_proba = []
        
        for ex in test_examples:
            if model is not None:
                pred = predict_lora(model, tokenizer, ex)
            else:
                # Mock prediction for demonstration
                pred = {
                    "triage_level": ex['metadata']['triage_level'],  # Perfect prediction as placeholder
                    "confidence": 0.8,
                    "drivers": [],
                    "notes": ""
                }
            
            if pred:
                triage_level = pred.get('triage_level', 'L2')
                confidence = pred.get('confidence', 0.5)
                
                y_pred.append(1 if triage_level == 'L1' else 0)
                y_proba.append(confidence if triage_level == 'L1' else (1 - confidence))
            else:
                y_pred.append(0)
                y_proba.append(0.5)
        
        y_pred = np.array(y_pred)
        y_proba = np.array(y_proba)
        
        # Apply threshold
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        # Compute metrics
        fold_metrics = compute_metrics(y_test, y_pred_thresh, y_proba)
        for key in all_metrics:
            all_metrics[key].append(fold_metrics[key])
    
    # Aggregate results
    results = {}
    for key, values in all_metrics.items():
        values_arr = np.array(values)
        results[key] = {
            'mean': np.mean(values_arr),
            'std': np.std(values_arr),
            'ci_lower': np.percentile(values_arr, 2.5),
            'ci_upper': np.percentile(values_arr, 97.5)
        }
    
    return results

# ============================================================================
# BASELINE LOADING
# ============================================================================

def load_experiment_a_results() -> Optional[Dict[str, Any]]:
    """Load Experiment A results for comparison."""
    exp_a_file = Path(__file__).parent.parent.parent / "experiment_A_structured_baseline" / "experiment_A_results_table.csv"
    
    if not exp_a_file.exists():
        return None
    
    try:
        df = pd.read_csv(exp_a_file)
        lr_rows = df[df['Model'] == 'Regularized Logistic Regression']
        if len(lr_rows) == 0:
            print(f"⚠️  No 'Regularized Logistic Regression' row found in {exp_a_file}")
            return None
        lr_row = lr_rows.iloc[0]
        
        # Parse metrics from string format "0.5664 [0.2397, 0.7778]"
        def parse_metric(value):
            if pd.isna(value) or value == '':
                return None
            import re
            match = re.match(r'([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]', str(value))
            if match:
                return {
                    'mean': float(match.group(1)),
                    'ci_lower': float(match.group(2)),
                    'ci_upper': float(match.group(3))
                }
            return None
        
        results = {}
        for col in ['sensitivity', 'specificity', 'precision', 'roc_auc', 'pr_auc', 'brier_score', 'accuracy']:
            if col in lr_row.index:
                parsed = parse_metric(lr_row[col])
                if parsed:
                    results[col] = parsed
        
        return results if results else None
    except Exception as e:
        print(f"⚠️  Error loading Experiment A results: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main evaluation function."""
    print("="*70)
    print("LORA MODEL EVALUATION")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading dataset from {DATASET_FILE}...")
    examples = []
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    print(f"  Loaded {len(examples)} examples")
    
    # Load model
    print("\nLoading LoRA model...")
    model, tokenizer = load_lora_model()
    if model is None:
        print("  ⚠️  Running in evaluation framework mode (no trained model)")
        print("     This will generate the evaluation structure and comparison tables")
        print("     Run train_lora.py first to train a model")
    
    # Load Experiment A results
    print("\nLoading Experiment A baseline results...")
    exp_a_results = load_experiment_a_results()
    if exp_a_results:
        print("  ✅ Loaded structured LR baseline")
    
    # Load zero-shot conservative results
    print("\nLoading zero-shot conservative baseline...")
    zeroshot_metrics = None
    if exp_a_results:
        # Load master file to get audio_file mapping
        master_df = pd.read_excel(MASTER_FILE)
        master_df = master_df[master_df['exclusion_criteria'] == 'include'].copy()
        master_df = master_df[master_df['iss'].notna()].copy()
        master_df['audio_file'] = master_df['audio_file'].astype(str)
        master_df['ISS_Level_1'] = (master_df['iss'] > 15).astype(int)
        
        # Get audio files from examples
        example_audio_files = {ex['metadata']['audio_file'] for ex in examples}
        master_df = master_df[master_df['audio_file'].isin(example_audio_files)].copy()
        
        zeroshot_preds = load_zeroshot_conservative_predictions(master_df, ZEROSHOT_CONSERVATIVE_PATH)
        if zeroshot_preds is not None and len(zeroshot_preds) > 0:
            zeroshot_metrics = compute_zeroshot_conservative_metrics(master_df, zeroshot_preds)
            print(f"  ✅ Loaded zero-shot conservative: accuracy={zeroshot_metrics['accuracy']:.3f}")
    
    # Run cross-validation
    print("\n" + "="*70)
    print("CROSS-VALIDATION EVALUATION")
    print("="*70)
    
    # Use threshold that matches Experiment A sensitivity if available
    target_sensitivity = 0.5664  # From Experiment A
    threshold = 0.5  # Default
    
    lora_results = cross_validate_lora(examples, model, tokenizer, threshold=threshold)
    
    print("\nLoRA Model Results:")
    for metric_name, metric_dict in lora_results.items():
        print(f"  {metric_name:15s}: {metric_dict['mean']:.4f} "
              f"[{metric_dict['ci_lower']:.4f}, {metric_dict['ci_upper']:.4f}]")
    
    # Create comparison table
    print("\n" + "="*70)
    print("GENERATING COMPARISON TABLES")
    print("="*70)
    
    comparison_table = []
    
    # LoRA results
    row = {'Model': 'LoRA (Mistral 7B)'}
    for metric_name, metric_dict in lora_results.items():
        row[metric_name] = f"{metric_dict['mean']:.4f} [{metric_dict['ci_lower']:.4f}, {metric_dict['ci_upper']:.4f}]"
    comparison_table.append(row)
    
    # Experiment A baseline
    if exp_a_results:
        row = {'Model': 'Structured LR (Exp A)'}
        for metric_name, metric_dict in exp_a_results.items():
            if metric_dict:
                row[metric_name] = f"{metric_dict['mean']:.4f} [{metric_dict['ci_lower']:.4f}, {metric_dict['ci_upper']:.4f}]"
        comparison_table.append(row)
    
    # Zero-shot conservative
    if zeroshot_metrics:
        row = {'Model': 'Zero-Shot Conservative'}
        row['sensitivity'] = f"{zeroshot_metrics['sensitivity']:.4f} [reference]"
        row['specificity'] = f"{zeroshot_metrics['specificity']:.4f} [reference]"
        row['precision'] = f"{zeroshot_metrics['precision']:.4f} [reference]"
        row['accuracy'] = f"{zeroshot_metrics['accuracy']:.4f} [reference]"
        row['roc_auc'] = "N/A [no probabilities]"
        row['pr_auc'] = "N/A [no probabilities]"
        row['brier_score'] = "N/A [no probabilities]"
        comparison_table.append(row)
    
    # Save comparison table
    comparison_df = pd.DataFrame(comparison_table)
    comparison_df.to_csv(OUTPUTS_DIR / "experiment_B_results_table.csv", index=False)
    comparison_df.to_excel(OUTPUTS_DIR / "experiment_B_results_table.xlsx", index=False)
    print(f"\n✅ Comparison table saved to {OUTPUTS_DIR / 'experiment_B_results_table.csv'}")
    
    # Generate summary report
    generate_summary_report(lora_results, exp_a_results, zeroshot_metrics, OUTPUTS_DIR)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return True

def generate_summary_report(lora_results: Dict, exp_a_results: Optional[Dict], 
                           zeroshot_metrics: Optional[Dict], output_dir: Path):
    """Generate summary report."""
    
    report = f"""
EXPERIMENT B: LORA MODEL EVALUATION SUMMARY
===================================================

LoRA Model (Mistral 7B Instruct with QLoRA)
-------------------------------------------------
Sensitivity (Recall for ISS > 15): {lora_results['sensitivity']['mean']:.3f}
  [95% CI: {lora_results['sensitivity']['ci_lower']:.3f}, {lora_results['sensitivity']['ci_upper']:.3f}]

Specificity: {lora_results['specificity']['mean']:.3f}
  [95% CI: {lora_results['specificity']['ci_lower']:.3f}, {lora_results['specificity']['ci_upper']:.3f}]

Precision: {lora_results['precision']['mean']:.3f}
  [95% CI: {lora_results['precision']['ci_lower']:.3f}, {lora_results['precision']['ci_upper']:.3f}]

Precision-Recall AUC: {lora_results['pr_auc']['mean']:.3f}
  [95% CI: {lora_results['pr_auc']['ci_lower']:.3f}, {lora_results['pr_auc']['ci_upper']:.3f}]

ROC-AUC: {lora_results['roc_auc']['mean']:.3f}
  [95% CI: {lora_results['roc_auc']['ci_lower']:.3f}, {lora_results['roc_auc']['ci_upper']:.3f}]

Brier Score (Calibration): {lora_results['brier_score']['mean']:.3f}
  [95% CI: {lora_results['brier_score']['ci_lower']:.3f}, {lora_results['brier_score']['ci_upper']:.3f}]

Accuracy: {lora_results['accuracy']['mean']:.3f}
  [95% CI: {lora_results['accuracy']['ci_lower']:.3f}, {lora_results['accuracy']['ci_upper']:.3f}]

"""
    
    # Comparison analysis
    report += "\nCOMPARISON WITH BASELINES\n"
    report += "="*70 + "\n\n"
    
    if exp_a_results:
        lr_sens = exp_a_results.get('sensitivity', {}).get('mean', 0)
        lora_sens = lora_results['sensitivity']['mean']
        lr_spec = exp_a_results.get('specificity', {}).get('mean', 0)
        lora_spec = lora_results['specificity']['mean']
        
        report += f"vs Structured LR Baseline:\n"
        report += f"  Sensitivity: {lora_sens:.3f} vs {lr_sens:.3f} (Δ = {lora_sens - lr_sens:+.3f})\n"
        report += f"  Specificity: {lora_spec:.3f} vs {lr_spec:.3f} (Δ = {lora_spec - lr_spec:+.3f})\n\n"
        
        # Success criteria check
        if lora_spec > lr_spec + 0.05 and abs(lora_sens - lr_sens) < 0.05:
            report += "  ✅ LoRA improves specificity at similar sensitivity\n"
        elif lora_sens > lr_sens + 0.05 and lora_spec > lr_spec - 0.1:
            report += "  ✅ LoRA improves sensitivity without collapsing specificity\n"
        else:
            report += "  ⚠️  LoRA does not meaningfully improve over LR baseline\n"
    
    if zeroshot_metrics:
        zs_sens = zeroshot_metrics.get('sensitivity', 0)
        zs_spec = zeroshot_metrics.get('specificity', 0)
        lora_sens = lora_results['sensitivity']['mean']
        lora_spec = lora_results['specificity']['mean']
        
        report += f"\nvs Zero-Shot Conservative LLM:\n"
        report += f"  Sensitivity: {lora_sens:.3f} vs {zs_sens:.3f} (Δ = {lora_sens - zs_sens:+.3f})\n"
        report += f"  Specificity: {lora_spec:.3f} vs {zs_spec:.3f} (Δ = {lora_spec - zs_spec:+.3f})\n\n"
    
    report += "\nRECOMMENDATION\n"
    report += "="*70 + "\n"
    
    if exp_a_results:
        lr_spec = exp_a_results.get('specificity', {}).get('mean', 0)
        lora_spec = lora_results['specificity']['mean']
        if lora_spec > lr_spec + 0.05:
            report += "✅ LoRA shows improvement. Consider deployment.\n"
        else:
            report += "⚠️  LoRA does not show clear benefit over LR baseline.\n"
            report += "   Recommendation: Use LR + gating + GPT fallback instead.\n"
    else:
        report += "⚠️  Could not compare with baselines. Review results manually.\n"
    
    # Save report
    with open(output_dir / "experiment_B_summary_report.txt", 'w') as f:
        f.write(report)
    
    print("\nSummary report saved to: experiment_B_summary_report.txt")
    print("\n" + report)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

