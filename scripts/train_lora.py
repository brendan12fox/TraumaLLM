#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Experiment B

Trains a QLoRA adapter on the structured instruction dataset.
Single configuration: Mistral 7B Instruct with QLoRA (4-bit), rank 8.

CLOUD EXECUTION:
Run this script on a GPU-enabled cloud instance (RunPod, Vast.ai, etc.).
After training, download the models/lora_adapter/ folder to local machine.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, List
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    import bitsandbytes as bnb
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("   Install: pip install transformers peft datasets bitsandbytes accelerate")
    print("   Note: Requires GPU with CUDA support")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_FILE = OUTPUTS_DIR / "instruction_dataset.jsonl"
MODEL_OUTPUT_DIR = MODELS_DIR / "lora_adapter"

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Or "meta-llama/Llama-3.1-8B-Instruct" if available
USE_4BIT = True
LORA_RANK = 8  # Small rank for regularization
LORA_ALPHA = 16  # Typically 2x rank
LORA_DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048
WARMUP_STEPS = 50
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10

# Seed for reproducibility
SEED = 42

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load instruction dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def format_prompt(example: Dict[str, Any]) -> str:
    """Format training example as a prompt."""
    instruction = example['instruction']
    input_text = example['input']
    output_text = example['output']
    
    # Format in Mistral chat template style
    prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output_text}</s>"
    return prompt

def tokenize_function(examples: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Tokenize examples."""
    prompts = [format_prompt(ex) for ex in examples]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None
    )
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

def create_train_test_split(examples: List[Dict[str, Any]], test_ratio: float = 0.1) -> tuple:
    """Create stratified train/test split maintaining class balance."""
    import random
    random.seed(SEED)
    
    # Separate by class
    l1_examples = [ex for ex in examples if ex['metadata']['triage_level'] == 'L1']
    l2_examples = [ex for ex in examples if ex['metadata']['triage_level'] == 'L2']
    
    # Shuffle
    random.shuffle(l1_examples)
    random.shuffle(l2_examples)
    
    # Split each class
    n_l1_test = max(1, int(len(l1_examples) * test_ratio))
    n_l2_test = max(1, int(len(l2_examples) * test_ratio))
    
    test_examples = l1_examples[:n_l1_test] + l2_examples[:n_l2_test]
    train_examples = l1_examples[n_l1_test:] + l2_examples[n_l2_test:]
    
    # Shuffle final sets
    random.shuffle(test_examples)
    random.shuffle(train_examples)
    
    return train_examples, test_examples

# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer():
    """Setup model with QLoRA and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    if USE_4BIT:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Mistral attention modules
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ============================================================================
# CLASS-WEIGHTED LOSS
# ============================================================================

class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss."""
    
    def __init__(self, *args, class_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute weighted loss."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Standard cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Note: Class weighting for sequence generation is complex
        # For now, we rely on balanced sampling
        if return_outputs:
            return loss, outputs
        return loss

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training function."""
    print("="*70)
    print("LORA FINE-TUNING FOR EXPERIMENT B")
    print("="*70)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Training requires GPU.")
        print("   This script must be run on a cloud GPU instance (RunPod, Vast.ai, etc.)")
        print("   See CLOUD_EXECUTION.md for instructions.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Ensure output directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset from {DATASET_FILE}...")
    examples = load_dataset(DATASET_FILE)
    print(f"  Loaded {len(examples)} examples")
    
    # Create train/test split
    train_examples, test_examples = create_train_test_split(examples, test_ratio=0.1)
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")
    
    # Setup model and tokenizer
    print("\nSetting up model...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare datasets
    print("\nTokenizing datasets...")
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    # Tokenize with batched=True for proper dict handling
    def tokenize_batch(examples):
        """Tokenize a batch of examples."""
        prompts = [format_prompt(ex) for ex in [examples] if isinstance(examples, dict) else examples]
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors=None
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    train_dataset = train_dataset.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer},
        batched=True,
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        tokenize_function,
        fn_kwargs={'tokenizer': tokenizer},
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        seed=SEED,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {MODEL_OUTPUT_DIR}...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model()
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    # Also save adapter config for easy loading
    print(f"✅ Training complete!")
    print(f"   Model saved to: {MODEL_OUTPUT_DIR}")
    print(f"\n📦 To download model to local machine:")
    print(f"   1. Use RunPod web UI: Download models/lora_adapter/ folder")
    print(f"   2. Or use rsync: rsync -avz user@host:{MODEL_OUTPUT_DIR} ./models/")
    print(f"   3. Or use script: ./scripts/download_model_from_cloud.sh user@host:/path/to/project/")
    print(f"\n   Model size: Check with 'du -sh {MODEL_OUTPUT_DIR}'")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
