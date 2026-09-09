"""
Index builder and loader for Historical Apple Support Cases.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Set, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CONVERSATIONS_PATH, RETRIEVAL_INDEX_DIR, DATA_DIR
from src.retrieval.retriever import HistoricalCase, HistoricalRetriever

def build_retrieval_index(
    conversations_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_cases: int = 45000
) -> HistoricalRetriever:
    """
    Builds an auditable historical case retrieval index while strictly excluding
    held-out golden set evaluation conversation IDs.
    """
    conv_path = conversations_path or CONVERSATIONS_PATH
    out_dir = output_dir or RETRIEVAL_INDEX_DIR
    
    held_out_ids_path = DATA_DIR / "held_out_evaluation_ids.json"
    held_out_ids: Set[str] = set()
    if held_out_ids_path.exists():
        with open(held_out_ids_path, "r", encoding="utf-8") as f:
            held_out_ids = set(json.load(f))
            
    print(f"Loading historical cases from {conv_path} (Held out exclusion: {len(held_out_ids)} convs)...")
    
    core_kws: Dict[str, List[str]] = {
        "battery_power_drain": ["battery", "drain", "charge", "dies", "overheating", "percentage", "shut down"],
        "software_update_os": ["update", "updated", "updating", "ios 11", "ios11", "install", "upgrade"],
        "apple_id_account_access": ["apple id", "appleid", "password", "locked", "disabled", "verification code", "2fa", "iforgot"],
        "app_store_billing_subscription": ["app store", "appstore", "billing", "charge", "subscription", "refund", "receipt", "purchase"],
        "hardware_screen_physical_damage": ["screen", "cracked", "broken", "dropped", "glass", "water damage", "repair", "genius bar"],
        "connectivity_wifi_bluetooth_network": ["wifi", "wi-fi", "bluetooth", "cellular", "data", "lte", "signal", "service", "no service", "airdrop"],
        "audio_sound_microphone": ["sound", "audio", "speaker", "earpiece", "volume", "microphone", "mic", "headphones", "airpods"],
        "storage_icloud_backup": ["icloud", "storage", "backup", "photos", "sync", "syncing", "restore", "space", "full storage"],
        "device_freeze_unresponsive_boot": ["freeze", "frozen", "unresponsive", "black screen", "apple logo", "stuck", "boot loop", "crash"],
        "general_inquiry_advice": ["how to", "feature", "recommend", "setting", "wondering", "trade in", "specs"]
    }
    
    cases: List[HistoricalCase] = []
    indexed_texts: List[str] = []
    
    with open(conv_path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            cid = t["conversation_id"]
            if cid in held_out_ids:
                continue  # STRICT LEAKAGE PREVENTION: Skip all golden set conversations
                
            issue = t.get("customer_cleaned_issue", "")
            res = t.get("brand_first_response", "")
            if not issue or not res or len(issue.split()) < 3 or len(res.split()) < 3:
                continue
                
            if issue.lower() in ["thanks", "thank you", "dm sent", "ok", "yes"]:
                continue
                
            # Infer case intent for evaluation tracking
            matched_intent = "general_inquiry_advice"
            issue_lower = issue.lower()
            for iname, kws in core_kws.items():
                if any(re.search(rf"\b{re.escape(k)}\b", issue_lower) for k in kws):
                    matched_intent = iname
                    break
                    
            case = HistoricalCase(
                case_id=cid,
                customer_issue=issue,
                brand_resolution=res,
                conversation_context=t.get("turns", [{}])[0].get("cleaned_text", issue),
                intent=matched_intent,
                turn_count=t.get("turn_count", 2)
            )
            cases.append(case)
            # Index the customer issue directly for highest semantic match fidelity with incoming queries
            indexed_texts.append(issue)
            
            if len(cases) >= max_cases:
                break

    print(f"Indexing {len(cases):,} verified historical cases...")
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=30000,
        sublinear_tf=True,
        strip_accents='unicode',
        token_pattern=r'(?u)\b[\w\']+\b'
    )
    
    matrix = vectorizer.fit_transform(indexed_texts)
    retriever = HistoricalRetriever(vectorizer=vectorizer, matrix=matrix, cases=cases)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(out_dir)
    print(f"Successfully saved retrieval index with {len(cases):,} cases to: {out_dir}")
    return retriever

def load_retrieval_index(index_dir: Optional[Path] = None) -> HistoricalRetriever:
    target_dir = index_dir or RETRIEVAL_INDEX_DIR
    if not (target_dir / "retriever_data.pkl").exists():
        print(f"Retrieval index not found at {target_dir}. Building now...")
        return build_retrieval_index(output_dir=target_dir)
    return HistoricalRetriever.load(target_dir)

if __name__ == "__main__":
    build_retrieval_index()
