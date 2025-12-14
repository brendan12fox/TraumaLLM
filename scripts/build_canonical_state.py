#!/usr/bin/env python3
"""
Build Canonical State JSON for Experiment B

For each case, constructs a single canonical JSON object that will be the input
to the LoRA fine-tuned model. This includes:
- Structured entities (with supplementation from cleaned JSON)
- Missingness flags (original and final)
- Abnormality flags
- Context features

This is the prompt input format - no raw transcript text.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RULES_DIR = PROJECT_ROOT / "rules"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INPUT_FILE = DATA_DIR / "OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx"
CLEANED_JSON_PATH = PROJECT_ROOT.parent / "CombinedData" / "November 17 complete data set" / "cleaned_outputs" / "V11_cleaned_transcripts_gpt5nano.json"

OUTPUT_FILE = OUTPUTS_DIR / "canonical_states.jsonl"

# ============================================================================
# ENTITY PARSING (Reused from Experiment A)
# ============================================================================

def parse_list_string(value: Any) -> list:
    """Parse entity field that may be a string representation of a list."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    value_str = str(value).strip()
    if value_str == '' or value_str == '[]' or value_str == 'nan':
        return []
    try:
        import ast
        parsed = ast.literal_eval(value_str)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except:
        return [value_str]

def extract_age(value: Any) -> Optional[float]:
    """Extract numeric age from entity field."""
    items = parse_list_string(value)
    if not items:
        return None
    
    for item in items:
        item_str = str(item).lower()
        age_match = re.search(r'\b(\d{1,2})\s*(?:-?year|yr|y\.o\.)', item_str)
        if age_match:
            age = float(age_match.group(1))
            if 0 <= age <= 120:
                return age
    return None

def extract_heart_rate(value: Any) -> Optional[float]:
    """Extract numeric heart rate from entity field."""
    items = parse_list_string(value)
    if not items:
        return None
    
    for item in items:
        item_str = str(item).lower()
        hr_patterns = [
            r'\b(?:hr|heart\s+rate|pulse)[\s:]+(\d{2,3})\b',
            r'\b(\d{2,3})\s*(?:bpm|/min)',
        ]
        for pattern in hr_patterns:
            match = re.search(pattern, item_str)
            if match:
                hr = float(match.group(1))
                if 30 <= hr <= 250:
                    return hr
        numbers = re.findall(r'\b(\d{2,3})\b', item_str)
        for num_str in numbers:
            num = float(num_str)
            if 40 <= num <= 220:
                return num
    return None

def extract_blood_pressure(value: Any) -> tuple[Optional[float], Optional[float]]:
    """Extract systolic and diastolic BP from entity field."""
    items = parse_list_string(value)
    if not items:
        return None, None
    
    for item in items:
        item_str = str(item).lower()
        bp_patterns = [
            r'(\d{2,3})\s*[/\-]\s*(\d{2,3})',
            r'(\d{2,3})\s+over\s+(\d{2,3})',
            r'systolic[:\s]+(\d{2,3}).*?diastolic[:\s]+(\d{2,3})',
        ]
        for pattern in bp_patterns:
            match = re.search(pattern, item_str)
            if match:
                systolic = float(match.group(1))
                diastolic = float(match.group(2))
                if 50 <= systolic <= 250 and 30 <= diastolic <= 150:
                    return systolic, diastolic
    return None, None

def extract_respiratory_rate(value: Any) -> Optional[float]:
    """Extract numeric respiratory rate from entity field."""
    items = parse_list_string(value)
    if not items:
        return None
    
    for item in items:
        item_str = str(item).lower()
        rr_patterns = [
            r'\b(?:rr|respiratory\s+rate|resp|respirations)[\s:]+(\d{1,2})\b',
            r'\b(\d{1,2})\s*(?:/min|respirations)',
        ]
        for pattern in rr_patterns:
            match = re.search(pattern, item_str)
            if match:
                rr = float(match.group(1))
                if 5 <= rr <= 60:
                    return rr
        numbers = re.findall(r'\b(\d{1,2})\b', item_str)
        for num_str in numbers:
            num = float(num_str)
            if 8 <= num <= 40:
                return num
    return None

def extract_gcs(value: Any) -> Optional[float]:
    """Extract numeric GCS from entity field."""
    items = parse_list_string(value)
    if not items:
        return None
    
    for item in items:
        item_str = str(item).lower()
        gcs_patterns = [
            r'\b(?:gcs|glasgow)[\s:]+(\d{1,2})\b',
            r'\b(\d{1,2})\s*(?:gcs|glasgow)',
        ]
        for pattern in gcs_patterns:
            match = re.search(pattern, item_str)
            if match:
                gcs = float(match.group(1))
                if 3 <= gcs <= 15:
                    return gcs
        numbers = re.findall(r'\b(\d{1,2})\b', item_str)
        for num_str in numbers:
            num = float(num_str)
            if 3 <= num <= 15:
                return num
    return None

def extract_mental_status(value: Any) -> Optional[str]:
    """Extract mental status text."""
    items = parse_list_string(value)
    if not items:
        return None
    for item in items:
        item_str = str(item).strip()
        if item_str and item_str.lower() not in ['', 'nan', '[]']:
            return item_str.lower()
    return None

def extract_mechanism_of_injury(value: Any) -> Optional[str]:
    """Extract mechanism of injury text."""
    items = parse_list_string(value)
    if not items:
        return None
    for item in items:
        item_str = str(item).strip()
        if item_str and item_str.lower() not in ['', 'nan', '[]']:
            return item_str.lower()
    return None

# ============================================================================
# JSON ENTITY SUPPLEMENTATION (Reused from Experiment A)
# ============================================================================

def load_cleaned_json_entities(json_path: Path) -> Dict[str, Dict]:
    """Load cleaned JSON file and extract entities, keyed by audio_file."""
    if not json_path.exists():
        print(f"⚠️  Cleaned JSON not found at {json_path}, skipping entity supplementation")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        entity_map = {}
        for case in json_data:
            audio_file = str(case.get('audio_file', ''))
            if not audio_file:
                continue
            
            entities = case.get('transcripts', {}).get('whisper', {}).get('entities', {})
            if entities:
                entity_map[audio_file] = entities
        
        print(f"✅ Loaded {len(entity_map)} cases with entities from cleaned JSON")
        return entity_map
    except Exception as e:
        print(f"⚠️  Error loading cleaned JSON: {e}, skipping entity supplementation")
        return {}

def supplement_entity_from_json(excel_value: Any, json_value: Any, entity_type: str) -> Any:
    """Supplement missing Excel entity with JSON value if available."""
    # Check if Excel value is missing
    def is_missing(value):
        if pd.isna(value):
            return True
        value_str = str(value).strip()
        return value_str == '' or value_str == '[]' or value_str.lower() == 'nan'
    
    if not is_missing(excel_value):
        return excel_value  # Keep Excel value if present
    
    if json_value is None:
        return None
    
    # Return JSON value (may need formatting)
    return json_value

# ============================================================================
# ABNORMALITY FLAGS
# ============================================================================

def check_hypotension_for_age(sbp: Optional[float], age: Optional[float]) -> Optional[bool]:
    """Check if SBP is hypotensive for age."""
    if sbp is None or age is None:
        return None
    
    try:
        sbp_num = float(sbp)
        age_num = float(age)
        
        # Age-adjusted thresholds
        if age_num < 1:
            threshold = 70
        elif age_num < 10:
            threshold = 70 + 2 * age_num
        else:
            threshold = 90
        
        return sbp_num < threshold
    except (ValueError, TypeError):
        return None

def check_gcs_low(gcs: Optional[float]) -> Optional[bool]:
    """Check if GCS is low (< 13)."""
    if gcs is None:
        return None
    try:
        gcs_num = float(gcs)
        return gcs_num < 13
    except (ValueError, TypeError):
        return None

def check_respiratory_distress(rr: Optional[float], age: Optional[float]) -> Optional[bool]:
    """Check if RR is abnormal for age (indicating distress)."""
    if rr is None:
        return None
    
    try:
        rr_num = float(rr)
        # Simple thresholds (can be age-adjusted if needed)
        if age is not None:
            try:
                age_num = float(age)
                if age_num < 5:
                    # Pediatric thresholds
                    return rr_num < 20 or rr_num > 40
            except (ValueError, TypeError):
                pass
        # Adult thresholds
        return rr_num < 12 or rr_num > 20
    except (ValueError, TypeError):
        return None

def check_ams(mental_status: Optional[str]) -> Optional[bool]:
    """Check if altered mental status is present."""
    if mental_status is None:
        return None
    
    abnormal_keywords = ['altered', 'unconscious', 'unresponsive', 'confused', 
                        'disoriented', 'lethargic', 'combative', 'agitated']
    ms_lower = mental_status.lower()
    return any(kw in ms_lower for kw in abnormal_keywords)

# ============================================================================
# CANONICAL STATE BUILDER
# ============================================================================

def build_canonical_state(row: pd.Series, json_entities: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Build canonical state JSON for a single case.
    
    Returns a structured JSON object ready for model input.
    """
    audio_file = str(row['audio_file'])
    
    # Extract entities from Excel (with original missingness tracking)
    age_excel = extract_age(row.get('gpt_entity_age'))
    hr_excel = extract_heart_rate(row.get('gpt_entity_heart_rate'))
    bp_excel = extract_blood_pressure(row.get('gpt_entity_blood_pressure'))
    rr_excel = extract_respiratory_rate(row.get('gpt_entity_respiratory_rate'))
    gcs_excel = extract_gcs(row.get('gpt_entity_gcs'))
    mental_status_excel = extract_mental_status(row.get('gpt_entity_mental_status'))
    moi_excel = extract_mechanism_of_injury(row.get('gpt_entity_mechanism_of_injury'))
    
    # Track original missingness (before supplementation)
    missing_original = {
        'age': age_excel is None,
        'hr': hr_excel is None,
        'sbp': bp_excel[0] is None if isinstance(bp_excel, tuple) else True,
        'dbp': bp_excel[1] is None if isinstance(bp_excel, tuple) else False,
        'rr': rr_excel is None,
        'gcs': gcs_excel is None,
        'mental_status': mental_status_excel is None,
        'moi': moi_excel is None,
    }
    
    # Supplement from JSON if available
    json_entity = json_entities.get(audio_file, {})
    
    age = age_excel if age_excel is not None else json_entity.get('age')
    hr = hr_excel if hr_excel is not None else json_entity.get('heart_rate')
    
    # Handle BP from JSON (might be string like "134/76")
    sbp, dbp = bp_excel if isinstance(bp_excel, tuple) and bp_excel[0] is not None else (None, None)
    if sbp is None:
        bp_json = json_entity.get('blood_pressure')
        if bp_json:
            if isinstance(bp_json, str):
                bp_match = re.search(r'(\d{2,3})\s*[/\-]\s*(\d{2,3})', bp_json)
                if bp_match:
                    sbp = float(bp_match.group(1))
                    dbp = float(bp_match.group(2))
            elif isinstance(bp_json, dict):
                sbp = bp_json.get('systolic')
                dbp = bp_json.get('diastolic')
    
    rr = rr_excel if rr_excel is not None else json_entity.get('respiratory_rate')
    gcs = gcs_excel if gcs_excel is not None else json_entity.get('gcs')
    mental_status = mental_status_excel if mental_status_excel is not None else json_entity.get('mental_status')
    moi = moi_excel if moi_excel is not None else json_entity.get('mechanism_of_injury')
    
    # Track final missingness (after supplementation)
    missing_final = {
        'age': age is None,
        'hr': hr is None,
        'sbp': sbp is None,
        'dbp': dbp is None,
        'rr': rr is None,
        'gcs': gcs is None,
        'mental_status': mental_status is None,
        'moi': moi is None,
    }
    
    # Build abnormality flags
    abnormality_flags = {
        'hypotension_for_age': check_hypotension_for_age(sbp, age),
        'gcs_low': check_gcs_low(gcs),
        'resp_distress': check_respiratory_distress(rr, age),
        'ams': check_ams(mental_status),
    }
    
    # Context features
    context_features = {
        'green_wc': float(row.get('green_essential_word_count', 0) or 0),
        'percent_complex': float(row.get('percent_complex', 0) or 0),
        'hard_terms_ct': float(row.get('hard_to_transcribe_terms_count', 0) or 0),
        'reduction_pct': float(row.get('word_count_reduction_percentage', 0) or 0),
    }
    
    # Build canonical state
    state = {
        'audio_file': audio_file,
        'age': age,
        'hr': hr,
        'sbp': sbp,
        'dbp': dbp,
        'rr': rr,
        'gcs': gcs,
        'mental_status': mental_status,
        'moi': moi,
        'num_patients': int(pd.to_numeric(row.get('num_patients_gpt'), errors='coerce') or 1) if pd.notna(pd.to_numeric(row.get('num_patients_gpt'), errors='coerce')) else 1,
        'site': str(row.get('site', 'unknown')).lower(),
        'context_features': context_features,
        'missing_original': missing_original,
        'missing_final': missing_final,
        'abnormality_flags': abnormality_flags,
    }
    
    return state

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Build canonical states for all cases."""
    print("="*70)
    print("BUILDING CANONICAL STATE JSON")
    print("="*70)
    
    # Ensure output directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Loaded {len(df)} cases")
    
    # Filter by exclusion criteria
    df = df[df['exclusion_criteria'] == 'include'].copy()
    print(f"  After exclusion filter: {len(df)} cases")
    
    # Filter for valid ISS
    df = df[df['iss'].notna()].copy()
    df = df[df['iss'] >= 0].copy()
    print(f"  After ISS validation: {len(df)} cases")
    
    # Load JSON entities for supplementation
    json_entities = load_cleaned_json_entities(CLEANED_JSON_PATH)
    
    # Build canonical states
    print(f"\nBuilding canonical states...")
    canonical_states = []
    
    for idx, row in df.iterrows():
        state = build_canonical_state(row, json_entities)
        canonical_states.append(state)
    
    print(f"  Built {len(canonical_states)} canonical states")
    
    # Save as JSONL (one JSON object per line)
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for state in canonical_states:
            f.write(json.dumps(state) + '\n')
    
    print(f"✅ Saved {len(canonical_states)} canonical states to {OUTPUT_FILE}")
    
    # Print sample
    if canonical_states:
        print(f"\nSample canonical state (first case):")
        print(json.dumps(canonical_states[0], indent=2))

if __name__ == "__main__":
    main()
