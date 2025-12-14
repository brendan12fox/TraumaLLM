#!/usr/bin/env python3
"""
Export Instruction Fine-Tuning Dataset for LoRA Training

Combines canonical states with retrieved rules to create training examples
with strict input/output schema for fine-tuning a small local model.

Format: (input JSON → output JSON) pairs
- Input: STATE_JSON + RETRIEVED_RULES_JSON
- Output: triage_level, confidence, drivers, notes
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_canonical_state import (
    load_cleaned_json_entities, build_canonical_state
)
from retrieve_rules import load_rules_store, retrieve_relevant_rules

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Try EXP_B_COPY first, fall back to original V11 if copy doesn't exist
INPUT_FILE = DATA_DIR / "OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx"
if not INPUT_FILE.exists():
    INPUT_FILE = DATA_DIR / "OCH_RCH_2023_2025_Combined_Master_V11.xlsx"
# Try data/ folder first (for cloud), then fall back to original location
CLEANED_JSON_PATH = DATA_DIR / "V11_cleaned_transcripts_gpt5nano.json"
if not CLEANED_JSON_PATH.exists():
    CLEANED_JSON_PATH = PROJECT_ROOT.parent / "CombinedData" / "November 17 complete data set" / "cleaned_outputs" / "V11_cleaned_transcripts_gpt5nano.json"

CANONICAL_STATES_FILE = OUTPUTS_DIR / "canonical_states.jsonl"
OUTPUT_FILE = OUTPUTS_DIR / "instruction_dataset.jsonl"

# Driver vocabulary (must match rules store)
DRIVER_VOCABULARY = [
    "hypotension",
    "low_gcs",
    "ams",
    "high_risk_moi",
    "resp_distress",
    "multi_system_injury_suspected",
    "insufficient_info"
]

# ============================================================================
# OUTPUT LABEL GENERATION
# ============================================================================

def generate_confidence(drivers: List[str], retrieved_rules: List[Dict[str, Any]]) -> float:
    """
    Generate confidence score based on drivers and rule signal strength.
    
    - 0.8-0.95: Multiple strong drivers present
    - 0.55-0.70: Few drivers or missing info
    """
    if not drivers or 'insufficient_info' in drivers:
        return 0.60
    
    # Count strong signals
    strong_count = 0
    for rule in retrieved_rules:
        if rule.get('signal_strength') == 'high':
            if rule.get('driver_name') in drivers:
                strong_count += 1
    
    if strong_count >= 2:
        return 0.90
    elif strong_count == 1:
        return 0.75
    else:
        return 0.65

def extract_drivers_from_rules(retrieved_rules: List[Dict[str, Any]], 
                                abnormality_flags: Dict[str, Any],
                                state: Dict[str, Any]) -> List[str]:
    """
    Extract driver names from retrieved rules, constrained to vocabulary.
    Returns up to 3 drivers (strongest signals first).
    """
    drivers = []
    
    # Map rules to drivers
    rule_to_driver = {}
    for rule in retrieved_rules:
        driver_name = rule.get('driver_name')
        if driver_name and driver_name in DRIVER_VOCABULARY:
            signal_strength = rule.get('signal_strength', 'low')
            # Prioritize high-strength signals
            priority = {'high': 3, 'medium': 2, 'low': 1}.get(signal_strength, 1)
            rule_to_driver[rule['rule_id']] = (driver_name, priority)
    
    # Sort by priority and extract unique drivers
    seen = set()
    for rule_id, (driver, priority) in sorted(rule_to_driver.items(), 
                                               key=lambda x: x[1][1], 
                                               reverse=True):
        if driver not in seen and len(drivers) < 3:
            drivers.append(driver)
            seen.add(driver)
    
    # Check for insufficient_info if many fields missing
    if len(drivers) == 0 or state.get('missing_final', {}).get('gcs', True) and \
       state.get('missing_final', {}).get('sbp', True):
        if 'insufficient_info' not in drivers:
            drivers.insert(0, 'insufficient_info')
            drivers = drivers[:3]  # Keep max 3
    
    return drivers[:3] if drivers else ['insufficient_info']

def generate_notes(drivers: List[str], state: Dict[str, Any]) -> str:
    """
    Generate brief notes (<=25 words) describing key findings.
    Keep extremely short to avoid model turning into a storyteller.
    """
    notes_parts = []
    
    if 'hypotension' in drivers:
        sbp = state.get('sbp')
        if sbp:
            notes_parts.append(f"SBP {sbp}")
    
    if 'low_gcs' in drivers:
        gcs = state.get('gcs')
        if gcs:
            notes_parts.append(f"GCS {int(gcs)}")
    
    if 'ams' in drivers:
        notes_parts.append("altered MS")
    
    if 'high_risk_moi' in drivers:
        notes_parts.append("high-risk MOI")
    
    if 'resp_distress' in drivers:
        rr = state.get('rr')
        if rr:
            notes_parts.append(f"RR {int(rr)}")
    
    if 'insufficient_info' in drivers:
        notes_parts.append("limited data")
    
    notes = ", ".join(notes_parts[:3])  # Max 3 items
    return notes[:100]  # Hard cap at 100 chars (well under 25 words)

# ============================================================================
# DATASET GENERATION
# ============================================================================

def create_training_example(state: Dict[str, Any], 
                           retrieved_rules: List[Dict[str, Any]],
                           triage_level: str) -> Dict[str, Any]:
    """
    Create a single training example in instruction fine-tuning format.
    
    Input format:
    {
      "instruction": "You are a trauma triage decision engine...",
      "input": "STATE_JSON: {...}\nRETRIEVED_RULES_JSON: {...}",
      "output": "{\"triage_level\": \"L1\", \"confidence\": 0.85, \"drivers\": [...], \"notes\": \"...\"}"
    }
    """
    # Build instruction
    instruction = (
        "You are a trauma triage decision engine. Use only the provided structured data and retrieved rules. "
        "Return ONLY valid JSON with keys: triage_level (\"L1\" or \"L2\"), "
        "confidence (0-1), drivers (array of <=3 strings from allowed list), "
        "notes (<=25 words, optional)."
    )
    
    # Build input (state + rules)
    input_text = f"STATE_JSON: {json.dumps(state, separators=(',', ':'))}\n"
    input_text += f"RETRIEVED_RULES_JSON: {json.dumps(retrieved_rules, separators=(',', ':'))}"
    
    # Extract drivers from rules
    drivers = extract_drivers_from_rules(retrieved_rules, 
                                        state.get('abnormality_flags', {}),
                                        state)
    
    # Generate confidence
    confidence = generate_confidence(drivers, retrieved_rules)
    
    # Generate notes
    notes = generate_notes(drivers, state)
    
    # Build output JSON
    output_json = {
        "triage_level": triage_level,
        "confidence": round(confidence, 2),
        "drivers": drivers,
        "notes": notes
    }
    
    output_text = json.dumps(output_json, separators=(',', ':'))
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": {
            "audio_file": state.get('audio_file'),
            "triage_level": triage_level
        }
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Export instruction fine-tuning dataset."""
    print("="*70)
    print("EXPORTING INSTRUCTION FINE-TUNING DATASET")
    print("="*70)
    
    # Ensure output directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    df = df[df['exclusion_criteria'] == 'include'].copy()
    df = df[df['iss'].notna()].copy()
    df = df[df['iss'] >= 0].copy()
    print(f"  Loaded {len(df)} cases")
    
    # Load JSON entities
    json_entities = load_cleaned_json_entities(CLEANED_JSON_PATH)
    
    # Load rules store
    print(f"\nLoading rules store...")
    rules_store = load_rules_store()
    print(f"  Loaded {len(rules_store.get('rules', []))} rules")
    
    # Build canonical states and training examples
    print(f"\nBuilding training examples...")
    training_examples = []
    
    for idx, row in df.iterrows():
        # Build canonical state
        state = build_canonical_state(row, json_entities)
        
        # Retrieve relevant rules
        retrieved_rules = retrieve_relevant_rules(state, rules_store)
        
        # Determine triage level (L1 if ISS > 15, else L2)
        triage_level = "L1" if row.get('iss', 0) > 15 else "L2"
        
        # Create training example
        example = create_training_example(state, retrieved_rules, triage_level)
        training_examples.append(example)
    
    print(f"  Created {len(training_examples)} training examples")
    
    # Save as JSONL
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Saved {len(training_examples)} examples to {OUTPUT_FILE}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    l1_count = sum(1 for ex in training_examples if ex['metadata']['triage_level'] == 'L1')
    l2_count = sum(1 for ex in training_examples if ex['metadata']['triage_level'] == 'L2')
    print(f"  L1 cases: {l1_count} ({100*l1_count/len(training_examples):.1f}%)")
    print(f"  L2 cases: {l2_count} ({100*l2_count/len(training_examples):.1f}%)")
    
    # Print sample
    if training_examples:
        print(f"\nSample training example:")
        sample = training_examples[0]
        print(f"  Instruction: {sample['instruction'][:80]}...")
        print(f"  Input length: {len(sample['input'])} chars")
        print(f"  Output: {sample['output']}")

if __name__ == "__main__":
    main()
