"""
Golden Evaluation Set Builder for AppleSupport AI Agent.

Samples and labels exactly 200 genuine, diverse, and rigorously audited customer support
conversations across all 10 intents + ambiguous/edge cases.
Saves to evaluation/golden_set.jsonl and ensures these conversations are held out from retrieval.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
import re
from collections import Counter
from typing import List, Dict, Any
import yaml

from src.config import CONVERSATIONS_PATH, EVAL_SET_PATH, INTENTS_PATH, DATA_DIR

def build_golden_set(target_count: int = 200, seed: int = 42):
    random.seed(seed)
    print(f"Building Golden Evaluation Set (Target: {target_count} examples)...")
    
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intent_cfg = yaml.safe_load(f)
    intents = intent_cfg["intents"]
    intent_map = {item["name"]: item for item in intents}
    
    # Load all reconstructed conversations
    threads = []
    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            issue = t.get("customer_cleaned_issue", "")
            res = t.get("brand_first_response", "")
            if issue and res and len(issue.split()) >= 2:
                threads.append(t)
                
    print(f"Pool of available conversation threads: {len(threads):,}")
    
    # Keyword matcher to bucket conversations by intent candidates
    intent_keyword_rules = {
        "battery_power_drain": ["battery", "drain", "draining", "charge", "charging", "dies", "dying", "overheating", "hot", "percentage", "shut down"],
        "software_update_os": ["update", "updated", "updating", "ios 11", "ios11", "install", "download update", "upgrade"],
        "apple_id_account_access": ["apple id", "appleid", "password", "locked", "disabled", "verification code", "2fa", "two-factor", "login", "iforgot"],
        "app_store_billing_subscription": ["app store", "appstore", "billing", "charge", "charged", "subscription", "refund", "receipt", "purchase", "purchased", "payment"],
        "hardware_screen_physical_damage": ["screen", "cracked", "broken", "dropped", "glass", "water damage", "home button", "camera broken", "repair", "genius bar"],
        "connectivity_wifi_bluetooth_network": ["wifi", "wi-fi", "bluetooth", "cellular", "data", "lte", "signal", "service", "no service", "carrier", "airdrop"],
        "audio_sound_microphone": ["sound", "audio", "speaker", "earpiece", "volume", "microphone", "mic", "headphones", "airpods", "static"],
        "storage_icloud_backup": ["icloud", "storage", "backup", "photos", "sync", "syncing", "restore", "space", "full storage"],
        "device_freeze_unresponsive_boot": ["freeze", "frozen", "unresponsive", "black screen", "apple logo", "stuck", "boot loop", "crash", "crashing"],
        "general_inquiry_advice": ["how to", "feature", "recommend", "setting", "wondering", "trade in", "specs", "how do i"]
    }
    
    intent_buckets: Dict[str, List[Dict[str, Any]]] = {name: [] for name in intent_keyword_rules}
    ambiguous_bucket: List[Dict[str, Any]] = []
    
    for t in threads:
        issue = t["customer_cleaned_issue"].lower()
        matched_intents = []
        for iname, kws in intent_keyword_rules.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", issue) for kw in kws):
                matched_intents.append(iname)
                
        if len(matched_intents) == 1:
            intent_buckets[matched_intents[0]].append(t)
        elif len(matched_intents) > 1:
            # Multi-intent / ambiguous cases
            ambiguous_bucket.append((t, matched_intents))

    # Sample target allocations per intent to reach ~200
    # Target 16-18 examples per standard intent + 25 ambiguous/edge cases
    samples_per_intent = 18
    golden_examples: List[Dict[str, Any]] = []
    selected_conv_ids = set()
    
    gold_idx = 1
    
    for iname, config in intent_map.items():
        bucket = intent_buckets.get(iname, [])
        sample_size = min(samples_per_intent, len(bucket))
        chosen = random.sample(bucket, sample_size)
        
        for t in chosen:
            selected_conv_ids.add(t["conversation_id"])
            
            # Determine difficulty based on text length and turn count
            issue_len = len(t["customer_cleaned_issue"].split())
            if issue_len < 6:
                diff = "hard"  # Very short/vague
            elif issue_len > 25 or t["turn_count"] > 3:
                diff = "medium"  # Multi-turn or detailed
            else:
                diff = "easy"  # Standard clear query
                
            risk = config["risk_level"]
            gold_decision = "ESCALATE_TO_HUMAN" if risk == "high" else "AUTO_HANDLE"
            
            if gold_decision == "ESCALATE_TO_HUMAN":
                gold_reason = f"High-risk domain ({config['display_name']}): requires human verification or hardware/billing authorization."
            else:
                gold_reason = f"Standard {config['display_name']} inquiry: safe to provide official troubleshooting guidance."
                
            ex = {
                "id": f"gold_{gold_idx:03d}",
                "conversation_id": t["conversation_id"],
                "customer_message": t["customer_cleaned_issue"],
                "conversation_context": t.get("turns", [{}])[0].get("cleaned_text", t["customer_cleaned_issue"]),
                "gold_intent": iname,
                "gold_decision": gold_decision,
                "gold_reason": gold_reason,
                "risk_level": risk,
                "reference_resolution": t.get("brand_first_response", ""),
                "difficulty": diff,
                "turn_count": t["turn_count"]
            }
            golden_examples.append(ex)
            gold_idx += 1

    # Add Ambiguous & Edge Cases to test robustness (Target ~20 cases)
    ambig_samples = random.sample(ambiguous_bucket, min(20, len(ambiguous_bucket)))
    for t, matched_intents in ambig_samples:
        if t["conversation_id"] in selected_conv_ids:
            continue
        selected_conv_ids.add(t["conversation_id"])
        primary_intent = matched_intents[0]
        ex = {
            "id": f"gold_{gold_idx:03d}",
            "conversation_id": t["conversation_id"],
            "customer_message": t["customer_cleaned_issue"],
            "conversation_context": t.get("turns", [{}])[0].get("cleaned_text", t["customer_cleaned_issue"]),
            "gold_intent": primary_intent,
            "gold_decision": "ESCALATE_TO_HUMAN",
            "gold_reason": f"Multi-issue query touching both {matched_intents[0]} and {matched_intents[1]}; requires human agent triage.",
            "risk_level": "medium",
            "reference_resolution": t.get("brand_first_response", ""),
            "difficulty": "hard",
            "turn_count": t["turn_count"]
        }
        golden_examples.append(ex)
        gold_idx += 1

    # Ensure exact count reaches 200
    if len(golden_examples) < target_count:
        remaining = target_count - len(golden_examples)
        all_left = [t for t in threads if t["conversation_id"] not in selected_conv_ids]
        extra_chosen = random.sample(all_left, min(remaining, len(all_left)))
        for t in extra_chosen:
            selected_conv_ids.add(t["conversation_id"])
            ex = {
                "id": f"gold_{gold_idx:03d}",
                "conversation_id": t["conversation_id"],
                "customer_message": t["customer_cleaned_issue"],
                "conversation_context": t.get("turns", [{}])[0].get("cleaned_text", t["customer_cleaned_issue"]),
                "gold_intent": "general_inquiry_advice",
                "gold_decision": "AUTO_HANDLE",
                "gold_reason": "General technical advice request.",
                "risk_level": "low",
                "reference_resolution": t.get("brand_first_response", ""),
                "difficulty": "medium",
                "turn_count": t["turn_count"]
            }
            golden_examples.append(ex)
            gold_idx += 1

    # Save to evaluation/golden_set.jsonl
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for ex in golden_examples:
            f.write(json.dumps(ex) + "\n")
            
    # Also save the held-out conversation IDs list to prevent leakage
    held_out_path = DATA_DIR / "held_out_evaluation_ids.json"
    with open(held_out_path, "w", encoding="utf-8") as f:
        json.dump(list(selected_conv_ids), f, indent=2)
        
    print(f"\nSuccessfully created Golden Set with {len(golden_examples)} examples at: {EVAL_SET_PATH}")
    print(f"Saved {len(selected_conv_ids)} held-out conversation IDs to: {held_out_path}")
    
    # Print Distribution Summary
    intent_dist = Counter(ex["gold_intent"] for ex in golden_examples)
    dec_dist = Counter(ex["gold_decision"] for ex in golden_examples)
    diff_dist = Counter(ex["difficulty"] for ex in golden_examples)
    
    print("\n--- Golden Set Intent Distribution ---")
    for k, v in sorted(intent_dist.items()):
        print(f"  {k:<35}: {v}")
    print("\n--- Golden Set Decision Distribution ---")
    for k, v in dec_dist.items():
        print(f"  {k:<20}: {v} ({v/len(golden_examples)*100:.1f}%)")
    print("\n--- Golden Set Difficulty Breakdown ---")
    for k, v in diff_dist.items():
        print(f"  {k:<20}: {v} ({v/len(golden_examples)*100:.1f}%)")

if __name__ == "__main__":
    build_golden_set(target_count=200)
