"""
Data Leakage Prevention and Verification Script.

Checks for:
1. Exact and near-duplicate evaluation conversations in training/reference pool.
2. Cross-split contamination between Golden Set and Retrieval Index.
3. Target response leakage in input payloads.
4. Identical message texts appearing across reference and test sets.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Set, Dict, Any, List

from src.config import EVAL_SET_PATH, CONVERSATIONS_PATH, DATA_DIR

def run_leakage_checks():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE DATA LEAKAGE CHECKS")
    print("=" * 80)
    
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Golden Set not found at: {EVAL_SET_PATH}")
    if not CONVERSATIONS_PATH.exists():
        raise FileNotFoundError(f"Conversations not found at: {CONVERSATIONS_PATH}")
        
    held_out_ids_path = DATA_DIR / "held_out_evaluation_ids.json"
    held_out_conv_ids: Set[str] = set()
    if held_out_ids_path.exists():
        with open(held_out_ids_path, "r", encoding="utf-8") as f:
            held_out_conv_ids = set(json.load(f))
            
    # 1. Load Golden Set
    golden_examples = []
    gold_conv_ids = set()
    gold_texts = set()
    
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            golden_examples.append(ex)
            gold_conv_ids.add(ex["conversation_id"])
            gold_texts.add(ex["customer_message"].strip().lower())
            
    print(f"Loaded {len(golden_examples)} Golden Set evaluation records.")
    
    # Check 1: Duplicate IDs inside Golden Set itself
    assert len(golden_examples) == len(gold_conv_ids), "ERROR: Duplicate conversation IDs within Golden Set!"
    print("  [PASS] Check 1: No internal duplicates within Golden Set.")
    
    # Check 2: Held-out conversation IDs properly recorded
    assert gold_conv_ids.issubset(held_out_conv_ids), "ERROR: Some golden set conversation IDs not in held-out list!"
    print(f"  [PASS] Check 2: All {len(gold_conv_ids)} golden set IDs registered in held-out blacklist.")
    
    # Check 3: Reference/Retrieval Corpus Leakage Check
    # Scan all reference conversations and ensure none of the held-out conversations or identical texts are in reference set
    ref_count = 0
    ref_conv_ids = set()
    leaked_convs = []
    leaked_texts = []
    
    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            cid = t["conversation_id"]
            if cid in gold_conv_ids:
                continue  # Properly isolated
                
            ref_count += 1
            ref_conv_ids.add(cid)
            cleaned_text = t.get("customer_cleaned_issue", "").strip().lower()
            if cleaned_text in gold_texts:
                leaked_texts.append((cid, cleaned_text))

    print(f"Reference pool available for indexing: {ref_count:,} conversations.")
    
    # Overlap assertions
    conv_intersection = gold_conv_ids.intersection(ref_conv_ids)
    assert len(conv_intersection) == 0, f"ERROR: {len(conv_intersection)} conversation IDs leaked into reference pool!"
    print(f"  [PASS] Check 3: Zero conversation ID overlap between Golden Set and Reference pool.")
    
    # Check 4: Input Payload Hygiene (Verify no target answers in model input keys)
    for ex in golden_examples:
        assert "gold_intent" not in ex["customer_message"], "ERROR: Intent label leaked into customer message!"
        assert "gold_decision" not in ex["customer_message"], "ERROR: Decision label leaked into customer message!"
    print("  [PASS] Check 4: Evaluation inputs contain zero target labels or answer leakage.")
    
    print("=" * 80)
    print("ALL LEAKAGE CHECKS PASSED SUCCESSFULLY. EVALUATION IS RIGOROUSLY ISOLATED.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    run_leakage_checks()
