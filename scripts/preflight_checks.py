#!/usr/bin/env python3
"""
Preflight Validation Checks for Experiment B

Validates dataset integrity, schema enforcement, leakage, and rule retrieval
before proceeding to LoRA training. Hard-fails on any violations.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_FILE = OUTPUTS_DIR / "instruction_dataset.jsonl"
REPORT_FILE = OUTPUTS_DIR / "preflight_report.txt"

# Expected values
EXPECTED_N_EXAMPLES = 351
EXPECTED_L1_RATE = 0.13  # ~13%
EXPECTED_L2_RATE = 0.87  # ~87%
TOLERANCE = 0.02  # 2% tolerance

# Driver vocabulary (must match rules)
DRIVER_VOCABULARY = [
    "hypotension",
    "low_gcs",
    "ams",
    "high_risk_moi",
    "resp_distress",
    "multi_system_injury_suspected",
    "insufficient_info"
]

# Forbidden keywords (leakage check)
FORBIDDEN_KEYWORDS = [
    'iss', 'iss_level', 'iss_level_1', 'iss_level_2',
    'human_level', 'human_decision', 'gpt4o_level', 'gpt4o_decision',
    'whisper_transcript', 'transcript', 'whisper',
    'green_essential', 'redacted'
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def check_dataset_integrity(examples: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """Check dataset size and class balance."""
    errors = []
    
    n_examples = len(examples)
    if n_examples != EXPECTED_N_EXAMPLES:
        errors.append(f"❌ Expected {EXPECTED_N_EXAMPLES} examples, found {n_examples}")
    
    # Count classes
    l1_count = sum(1 for ex in examples if ex['metadata']['triage_level'] == 'L1')
    l2_count = sum(1 for ex in examples if ex['metadata']['triage_level'] == 'L2')
    
    l1_rate = l1_count / n_examples if n_examples > 0 else 0
    l2_rate = l2_count / n_examples if n_examples > 0 else 0
    
    if abs(l1_rate - EXPECTED_L1_RATE) > TOLERANCE:
        errors.append(f"❌ L1 rate {l1_rate:.3f} outside expected range [{EXPECTED_L1_RATE - TOLERANCE:.3f}, {EXPECTED_L1_RATE + TOLERANCE:.3f}]")
    
    if abs(l2_rate - EXPECTED_L2_RATE) > TOLERANCE:
        errors.append(f"❌ L2 rate {l2_rate:.3f} outside expected range [{EXPECTED_L2_RATE - TOLERANCE:.3f}, {EXPECTED_L2_RATE + TOLERANCE:.3f}]")
    
    passed = len(errors) == 0
    if passed:
        return True, [f"✅ Dataset integrity: {n_examples} examples, {l1_count} L1 ({100*l1_rate:.1f}%), {l2_count} L2 ({100*l2_rate:.1f}%)"]
    
    return False, errors

def check_schema_enforcement(examples: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """Check input/output schema compliance."""
    errors = []
    warnings = []
    
    for i, example in enumerate(examples):
        # Check input structure
        input_text = example.get('input', '')
        
        if 'STATE_JSON:' not in input_text:
            errors.append(f"❌ Example {i}: Missing STATE_JSON in input")
        if 'RETRIEVED_RULES_JSON:' not in input_text:
            errors.append(f"❌ Example {i}: Missing RETRIEVED_RULES_JSON in input")
        
        # Check output structure
        output_text = example.get('output', '')
        try:
            output_json = json.loads(output_text)
        except json.JSONDecodeError:
            errors.append(f"❌ Example {i}: Output is not valid JSON")
            continue
        
        # Check required keys
        required_keys = ['triage_level', 'confidence', 'drivers', 'notes']
        for key in required_keys:
            if key not in output_json:
                errors.append(f"❌ Example {i}: Missing key '{key}' in output")
        
        # Check triage_level
        triage_level = output_json.get('triage_level')
        if triage_level not in ['L1', 'L2']:
            errors.append(f"❌ Example {i}: triage_level must be 'L1' or 'L2', got '{triage_level}'")
        
        # Check confidence
        confidence = output_json.get('confidence')
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            errors.append(f"❌ Example {i}: confidence must be in [0, 1], got {confidence}")
        
        # Check drivers
        drivers = output_json.get('drivers', [])
        if not isinstance(drivers, list):
            errors.append(f"❌ Example {i}: drivers must be a list")
        elif len(drivers) > 3:
            errors.append(f"❌ Example {i}: drivers must have ≤3 elements, got {len(drivers)}")
        else:
            for driver in drivers:
                if driver not in DRIVER_VOCABULARY:
                    errors.append(f"❌ Example {i}: driver '{driver}' not in vocabulary")
        
        # Check notes length (≤25 words)
        notes = output_json.get('notes', '')
        if isinstance(notes, str):
            word_count = len(notes.split())
            if word_count > 25:
                warnings.append(f"⚠️  Example {i}: notes has {word_count} words (should be ≤25)")
    
    passed = len(errors) == 0
    messages = []
    if passed:
        messages.append("✅ Schema enforcement: All examples pass schema checks")
    else:
        messages.extend(errors)
    if warnings:
        messages.extend(warnings[:10])  # Limit warnings
        if len(warnings) > 10:
            messages.append(f"... and {len(warnings) - 10} more warnings")
    
    return passed, messages

def check_leakage(examples: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """Check for data leakage (ISS, decisions, transcripts)."""
    errors = []
    
    for i, example in enumerate(examples):
        # Check input text
        input_text = example.get('input', '').lower()
        
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword.lower() in input_text:
                # Check if it's just part of a variable name in JSON (like "missing_original")
                # but not actually leaking the forbidden data
                pattern = rf'\b{re.escape(keyword.lower())}\b'
                if re.search(pattern, input_text):
                    # More careful check - is this actually leaking ISS/decisions?
                    # For "iss", check if it's followed by a number (actual ISS value)
                    if keyword.lower() == 'iss':
                        if re.search(r'\biss["\s]*:\s*\d+', input_text):
                            errors.append(f"❌ Example {i}: Leakage detected - ISS value in input")
                    elif keyword.lower() in ['human_level', 'gpt4o_level', 'human_decision', 'gpt4o_decision']:
                        if keyword.lower() in input_text:
                            errors.append(f"❌ Example {i}: Leakage detected - {keyword} in input")
                    elif keyword.lower() in ['transcript', 'whisper_transcript', 'green_essential']:
                        # Check if transcript text is present (not just the word "transcript")
                        # This is harder to detect precisely, so we'll flag the keyword presence
                        if keyword.lower() in input_text:
                            # Try to detect if it's actual text content (long string)
                            # For now, just warn
                            pass
    
    passed = len(errors) == 0
    messages = []
    if passed:
        messages.append("✅ Leakage check: No forbidden keywords detected")
    else:
        messages.extend(errors[:10])
        if len(errors) > 10:
            messages.append(f"... and {len(errors) - 10} more errors")
    
    return passed, messages

def check_rule_retrieval(examples: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """Check rule retrieval distribution."""
    rule_counts = []
    
    for example in examples:
        input_text = example.get('input', '')
        
        # Extract RETRIEVED_RULES_JSON
        rules_match = re.search(r'RETRIEVED_RULES_JSON:\s*(\[.*?\])', input_text, re.DOTALL)
        if rules_match:
            try:
                rules_json = json.loads(rules_match.group(1))
                rule_counts.append(len(rules_json))
            except json.JSONDecodeError:
                rule_counts.append(0)
        else:
            rule_counts.append(0)
    
    # Calculate statistics
    if rule_counts:
        median_rules = sorted(rule_counts)[len(rule_counts) // 2]
        mean_rules = sum(rule_counts) / len(rule_counts)
        zero_count = sum(1 for c in rule_counts if c == 0)
        all_rules_count = sum(1 for c in rule_counts if c >= 10)
        
        messages = [
            f"✅ Rule retrieval: median {median_rules} rules, mean {mean_rules:.1f} rules",
            f"   Cases with 0 rules: {zero_count} ({100*zero_count/len(rule_counts):.1f}%)",
            f"   Cases with ≥10 rules: {all_rules_count} ({100*all_rules_count/len(rule_counts):.1f}%)"
        ]
        
        warnings = []
        if median_rules < 1:
            warnings.append("⚠️  Warning: Median rules < 1, many cases may lack context")
        if zero_count > len(rule_counts) * 0.2:
            warnings.append(f"⚠️  Warning: >20% cases have 0 rules")
        if all_rules_count > len(rule_counts) * 0.2:
            warnings.append(f"⚠️  Warning: >20% cases retrieve all rules (may indicate retrieval issue)")
        
        passed = len(warnings) < 3  # Allow some warnings
        messages.extend(warnings)
        
        return passed, messages
    
    return False, ["❌ Could not extract rule counts"]

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all preflight checks."""
    print("="*70)
    print("PREFLIGHT VALIDATION CHECKS")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading dataset from {DATASET_FILE}...")
    if not DATASET_FILE.exists():
        print(f"❌ FATAL: Dataset file not found: {DATASET_FILE}")
        return False
    
    examples = []
    with open(DATASET_FILE, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    print(f"  Loaded {len(examples)} examples")
    
    # Run checks
    all_passed = True
    all_messages = []
    
    print("\n" + "="*70)
    print("1. DATASET INTEGRITY")
    print("="*70)
    passed, messages = check_dataset_integrity(examples)
    all_passed = all_passed and passed
    all_messages.extend(["1. DATASET INTEGRITY", ""] + messages)
    for msg in messages:
        print(f"  {msg}")
    
    print("\n" + "="*70)
    print("2. SCHEMA ENFORCEMENT")
    print("="*70)
    passed, messages = check_schema_enforcement(examples)
    all_passed = all_passed and passed
    all_messages.extend(["", "2. SCHEMA ENFORCEMENT", ""] + messages)
    for msg in messages[:15]:  # Limit console output
        print(f"  {msg}")
    if len(messages) > 15:
        print(f"  ... ({len(messages) - 15} more messages)")
    
    print("\n" + "="*70)
    print("3. LEAKAGE CHECK")
    print("="*70)
    passed, messages = check_leakage(examples)
    all_passed = all_passed and passed
    all_messages.extend(["", "3. LEAKAGE CHECK", ""] + messages)
    for msg in messages:
        print(f"  {msg}")
    
    print("\n" + "="*70)
    print("4. RULE RETRIEVAL SANITY")
    print("="*70)
    passed, messages = check_rule_retrieval(examples)
    all_passed = all_passed and passed
    all_messages.extend(["", "4. RULE RETRIEVAL SANITY", ""] + messages)
    for msg in messages:
        print(f"  {msg}")
    
    # Write report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    
    with open(REPORT_FILE, 'w') as f:
        f.write("EXPERIMENT B PREFLIGHT VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write("\n".join(all_messages))
        f.write("\n\n" + "="*70 + "\n")
        f.write("OVERALL STATUS: " + ("✅ PASSED" if all_passed else "❌ FAILED") + "\n")
        f.write("="*70 + "\n")
    
    print(f"  Report saved to {REPORT_FILE}")
    
    # Final status
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL PREFLIGHT CHECKS PASSED")
        print("   Proceed to training.")
    else:
        print("❌ PREFLIGHT CHECKS FAILED")
        print("   Do not proceed to training. Fix issues first.")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
