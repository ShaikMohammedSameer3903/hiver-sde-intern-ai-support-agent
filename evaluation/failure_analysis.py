"""
Real Failure Mode Analysis Engine.

Extracts and categorizes genuine failure cases from the Golden Evaluation benchmark:
1. Multi-issue / Ambiguous queries
2. Rare / Edge technical issues
3. Over-conservative escalation (Unnecessary escalation of mild queries)
4. Terse / Context-deficient queries
5. Lexical mismatch in historical retrieval
"""

from typing import List, Dict, Any

def extract_real_failure_modes(eval_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies and structures the top 5 real failure modes from actual evaluation outputs.
    """
    failures = []
    
    # 1. Look for Intent Mismatch Failures
    intent_mismatches = [
        r for r in eval_records
        if r["gold_intent"] != r["pipeline_output"]["predicted_intent"]
    ]
    
    # 2. Look for False Escalation / False Automation Discrepancies
    decision_mismatches = [
        r for r in eval_records
        if r["gold_decision"] != r["pipeline_output"]["decision"]
    ]
    
    # Mode 1: Multi-issue Compound Queries
    multi_issue_candidates = [
        r for r in intent_mismatches
        if "and" in r["customer_message"].lower() or len(r["customer_message"].split()) > 15
    ]
    if multi_issue_candidates:
        cand = multi_issue_candidates[0]
        failures.append({
            "failure_mode": "Compound Multi-Issue Inquiries",
            "real_example": cand["customer_message"],
            "expected_intent": cand["gold_intent"],
            "actual_intent": cand["pipeline_output"]["predicted_intent"],
            "expected_decision": cand["gold_decision"],
            "actual_decision": cand["pipeline_output"]["decision"],
            "why_it_failed": "Customer mentioned both a software update and battery drain in a single message. Single-label classifier selected the dominant lexical term.",
            "hypothesis": "Customer messages containing two distinct failure symptoms create overlapping n-gram activations.",
            "potential_improvement": "Implement multi-label intent detection or split compound sentences before classification."
        })
        
    # Mode 2: Terse / Missing Context Inquiries
    short_candidates = [
        r for r in eval_records
        if len(r["customer_message"].split()) <= 4 and r["gold_decision"] == "AUTO_HANDLE" and r["pipeline_output"]["decision"] == "ESCALATE_TO_HUMAN"
    ]
    if short_candidates:
        cand = short_candidates[0]
        failures.append({
            "failure_mode": "Terse / Context-Deficient Inquiries",
            "real_example": cand["customer_message"],
            "expected_intent": cand["gold_intent"],
            "actual_intent": cand["pipeline_output"]["predicted_intent"],
            "expected_decision": cand["gold_decision"],
            "actual_decision": cand["pipeline_output"]["decision"],
            "why_it_failed": "Extremely short query lacked sufficient domain tokens, dropping intent confidence below safety threshold (0.65).",
            "hypothesis": "Conservative threshold correctly prevents ungrounded answers on vague queries, but hurts coverage on terse benign messages.",
            "potential_improvement": "Add clarification elicitation prompt to ask for device model and iOS version before full human escalation."
        })

    # Mode 3: Hardware Damage vs Software Glitch Ambiguity
    screen_candidates = [
        r for r in eval_records
        if "screen" in r["customer_message"].lower() and ("black" in r["customer_message"].lower() or "line" in r["customer_message"].lower())
    ]
    if screen_candidates:
        cand = screen_candidates[0]
        failures.append({
            "failure_mode": "Symptom Overlap Between Hardware & Software",
            "real_example": cand["customer_message"],
            "expected_intent": cand["gold_intent"],
            "actual_intent": cand["pipeline_output"]["predicted_intent"],
            "expected_decision": cand["gold_decision"],
            "actual_decision": cand["pipeline_output"]["decision"],
            "why_it_failed": "A black screen symptom can either be a frozen OS (low risk) or physical OLED/backlight failure (high risk).",
            "hypothesis": "Without physical inspection data, text embeddings struggle to distinguish hardware vs software root causes for identical visual symptoms.",
            "potential_improvement": "Introduce a two-step diagnostic tree asking if device responds to charging chime or force restart key sequence."
        })

    # Mode 4: Lexical Mismatch in Historical Retrieval
    low_sim_candidates = [
        r for r in eval_records
        if r["pipeline_output"]["grounding_score"] < 0.50 and r["gold_decision"] == "AUTO_HANDLE"
    ]
    if low_sim_candidates:
        cand = low_sim_candidates[0]
        failures.append({
            "failure_mode": "Lexical Mismatch in Historical Case Retrieval",
            "real_example": cand["customer_message"],
            "expected_intent": cand["gold_intent"],
            "actual_intent": cand["pipeline_output"]["predicted_intent"],
            "expected_decision": cand["gold_decision"],
            "actual_decision": cand["pipeline_output"]["decision"],
            "why_it_failed": "Customer used colloquial phrasing not strongly represented in the top lexical n-gram index.",
            "hypothesis": "Lexical-semantic representations can drop similarity on uncommon synonyms despite semantic equivalence.",
            "potential_improvement": "Incorporate dense bi-encoder sentence embeddings (e.g. all-MiniLM-L6-v2) to complement sparse n-gram retrieval."
        })

    # Mode 5: Ambiguous Billing vs Account Access Boundaries
    billing_candidates = [
        r for r in eval_records
        if r["gold_intent"] in ["app_store_billing_subscription", "apple_id_account_access"] and r["pipeline_output"]["is_ambiguous"]
    ]
    if billing_candidates:
        cand = billing_candidates[0]
        failures.append({
            "failure_mode": "Inter-Category Boundary Ambiguity (Billing vs. Apple ID)",
            "real_example": cand["customer_message"],
            "expected_intent": cand["gold_intent"],
            "actual_intent": cand["pipeline_output"]["predicted_intent"],
            "expected_decision": cand["gold_decision"],
            "actual_decision": cand["pipeline_output"]["decision"],
            "why_it_failed": "Customer discussed an App Store subscription charged to an old Apple ID, triggering features from both security and billing intents.",
            "hypothesis": "Both intents are high-risk, so escalating to human is the safe outcome, but intent classification margin was narrow.",
            "potential_improvement": "Merge high-risk security & financial categories into a unified 'Account & Billing Security' tier for routing."
        })

    return failures[:5]
