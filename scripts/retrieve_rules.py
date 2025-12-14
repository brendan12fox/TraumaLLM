#!/usr/bin/env python3
"""
Structured-RAG: Deterministic Rule Retrieval

Given a canonical state JSON, retrieves relevant rules from the rules store.
This is "structured RAG" - deterministic, auditable, and rule-based.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RULES_DIR = PROJECT_ROOT / "rules"
RULES_FILE = RULES_DIR / "trauma_triage_rules.json"

# ============================================================================
# RULE RETRIEVAL
# ============================================================================

def load_rules_store() -> Dict[str, Any]:
    """Load the rules store from JSON."""
    with open(RULES_FILE, 'r') as f:
        return json.load(f)

def check_hypotension_rule(state: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    """Check if hypotension rule applies to state."""
    age = state.get('age')
    sbp = state.get('sbp')
    
    if age is None or sbp is None:
        return False
    
    try:
        age_num = float(age)
        sbp_num = float(sbp)
    except (ValueError, TypeError):
        return False
    
    thresholds = rule.get('thresholds', [])
    for threshold in thresholds:
        age_min = threshold.get('age_min', 0)
        age_max = threshold.get('age_max', 999)
        
        if age_min <= age_num < age_max:
            if 'formula' in threshold:
                # Evaluate formula: "70 + 2*age"
                formula = threshold['formula'].replace('age', str(age_num))
                try:
                    sbp_min = eval(formula)
                except:
                    sbp_min = threshold.get('sbp_min', 90)
            else:
                sbp_min = threshold.get('sbp_min', 90)
            
            if sbp_num < sbp_min:
                return True
    
    return False

def check_low_gcs_rule(state: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    """Check if low GCS rule applies to state."""
    gcs = state.get('gcs')
    threshold = rule.get('threshold', 13)
    comparison = rule.get('comparison', '<')
    
    if gcs is None:
        return False
    
    try:
        gcs_num = float(gcs)
        if comparison == '<':
            return gcs_num < threshold
        elif comparison == '<=':
            return gcs_num <= threshold
    except (ValueError, TypeError):
        return False
    
    return False

def check_keyword_rule(state: Dict[str, Any], rule: Dict[str, Any], field: str) -> bool:
    """Check if keyword-based rule applies to state."""
    field_value = state.get(field)
    
    if field_value is None:
        return False
    
    field_str = str(field_value).lower()
    keywords = rule.get('keywords', [])
    
    return any(keyword.lower() in field_str for keyword in keywords)

def check_respiratory_distress_rule(state: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    """Check if respiratory distress rule applies to state."""
    age = state.get('age')
    rr = state.get('rr')
    
    if rr is None:
        return False
    
    try:
        rr_num = float(rr)
        age_num = float(age) if age is not None else None
    except (ValueError, TypeError):
        return False
    
    thresholds = rule.get('thresholds', [])
    for threshold in thresholds:
        age_min = threshold.get('age_min', 0)
        age_max = threshold.get('age_max', 999)
        
        if age_num is None or (age_min <= age_num < age_max):
            rr_min = threshold.get('rr_min', 12)
            rr_max = threshold.get('rr_max', 20)
            
            if rr_num < rr_min or rr_num > rr_max:
                return True
    
    return False

def check_insufficient_info_rule(state: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    """Check if insufficient info rule applies to state."""
    required_fields = rule.get('required_fields', [])
    missing_threshold = rule.get('missing_threshold', 3)
    
    missing_final = state.get('missing_final', {})
    missing_count = sum(1 for field in required_fields if missing_final.get(field, True))
    
    return missing_count >= missing_threshold

def retrieve_relevant_rules(state: Dict[str, Any], rules_store: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieve relevant rules for a given canonical state.
    
    Returns list of rule objects that apply to this state.
    """
    retrieved = []
    rules = rules_store.get('rules', [])
    
    for rule in rules:
        rule_id = rule.get('rule_id')
        applies = False
        
        # Check rule based on category
        if rule_id == 'hypotension_child':
            applies = check_hypotension_rule(state, rule)
        elif rule_id == 'low_gcs':
            applies = check_low_gcs_rule(state, rule)
        elif rule_id in ['high_risk_moi_mvc', 'high_risk_moi_fall', 
                         'high_risk_moi_pedestrian', 'high_risk_moi_motorcycle']:
            applies = check_keyword_rule(state, rule, 'moi')
        elif rule_id == 'altered_mental_status':
            applies = check_keyword_rule(state, rule, 'mental_status')
        elif rule_id == 'respiratory_distress':
            applies = check_respiratory_distress_rule(state, rule)
        elif rule_id == 'insufficient_info':
            applies = check_insufficient_info_rule(state, rule)
        elif rule_id == 'multi_system_suspicion':
            # This would need MOI parsing - for now, skip or use simple heuristic
            moi = state.get('moi', '')
            applies = any(indicator.lower() in str(moi).lower() 
                         for indicator in rule.get('indicators', []))
        
        if applies:
            # Return simplified rule representation for model input
            retrieved.append({
                'rule_id': rule_id,
                'driver_name': rule.get('driver_name'),
                'signal_strength': rule.get('signal_strength'),
                'description': rule.get('description', '')[:100]  # Truncate for brevity
            })
    
    return retrieved

# ============================================================================
# MAIN (for testing)
# ============================================================================

def main():
    """Test rule retrieval with sample state."""
    # Load rules
    rules_store = load_rules_store()
    print(f"Loaded {len(rules_store.get('rules', []))} rules")
    
    # Sample state
    sample_state = {
        'audio_file': '12345',
        'age': 15,
        'sbp': 85,
        'dbp': 55,
        'hr': 110,
        'rr': 24,
        'gcs': 12,
        'mental_status': 'confused and agitated',
        'moi': 'pedestrian struck by vehicle at high speed',
        'num_patients': 1,
        'site': 'rch',
        'context_features': {
            'green_wc': 45,
            'percent_complex': 12.5,
            'hard_terms_ct': 2,
            'reduction_pct': 35.2
        },
        'missing_original': {
            'age': False,
            'sbp': False,
            'dbp': False,
            'hr': False,
            'rr': False,
            'gcs': False,
            'mental_status': False,
            'moi': False
        },
        'missing_final': {
            'age': False,
            'sbp': False,
            'dbp': False,
            'hr': False,
            'rr': False,
            'gcs': False,
            'mental_status': False,
            'moi': False
        },
        'abnormality_flags': {
            'hypotension_for_age': True,
            'gcs_low': True,
            'resp_distress': True,
            'ams': True
        }
    }
    
    # Retrieve relevant rules
    retrieved = retrieve_relevant_rules(sample_state, rules_store)
    
    print(f"\nRetrieved {len(retrieved)} relevant rules:")
    for rule in retrieved:
        print(f"  - {rule['rule_id']}: {rule['driver_name']} ({rule['signal_strength']})")

if __name__ == "__main__":
    main()
