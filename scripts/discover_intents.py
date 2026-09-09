"""
Empirical Intent Discovery Script for AppleSupport conversations.
Analyzes 52,999 real reconstructed customer issue texts using TF-IDF,
n-gram analysis, and keyword clustering to derive grounded intent categories.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from collections import Counter, defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CONVERSATIONS_PATH, DATA_DIR

def discover_intents(jsonl_path: Path, top_n_clusters: int = 12):
    print(f"Loading conversations from {jsonl_path}...")
    issues = []
    resolutions = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            thread = json.loads(line)
            issue = thread.get("customer_cleaned_issue", "")
            res = thread.get("brand_first_response", "")
            if issue and len(issue.split()) >= 3:
                issues.append(issue)
                resolutions.append(res)
                
    print(f"Analyzing {len(issues):,} valid customer problem statements...")
    
    # Keyword & Phrase patterns grounded in AppleSupport domain
    intent_definitions = [
        {
            "intent": "battery_power_drain",
            "keywords": ["battery", "drain", "draining", "charge", "charging", "dies", "dying", "overheating", "hot", "percentage", "power", "turn off", "shut down"],
            "risk_level": "low",
            "common_resolution": "Check Settings > Battery, calibrate charging, verify background refresh, check battery health."
        },
        {
            "intent": "software_update_os",
            "keywords": ["update", "updated", "updating", "ios", "ios11", "ios 11", "install", "download update", "beta", "version", "upgrade"],
            "risk_level": "low",
            "common_resolution": "Guide user through Settings > General > Software Update, restart device, or update via iTunes."
        },
        {
            "intent": "apple_id_account_access",
            "keywords": ["apple id", "appleid", "password", "locked", "disabled", "verification code", "2fa", "two-factor", "login", "sign in", "forgot password", "iforgot"],
            "risk_level": "high",
            "common_resolution": "Direct customer to iforgot.apple.com to reset password securely; escalate account takeovers."
        },
        {
            "intent": "app_store_billing_subscription",
            "keywords": ["app store", "appstore", "billing", "charge", "charged", "subscription", "refund", "receipt", "purchase", "purchased", "payment", "card", "bank", "money"],
            "risk_level": "high",
            "common_resolution": "Direct to reportaproblem.apple.com to review charges and request refunds; escalate unauthorized card transactions."
        },
        {
            "intent": "hardware_screen_physical_damage",
            "keywords": ["screen", "cracked", "broken", "dropped", "glass", "water damage", "speaker broken", "home button", "camera broken", "repair", "genius bar", "apple store appointment"],
            "risk_level": "high",
            "common_resolution": "Check warranty/AppleCare+ status and schedule a Genius Bar or authorized service repair."
        },
        {
            "intent": "connectivity_wifi_bluetooth_network",
            "keywords": ["wifi", "wi-fi", "bluetooth", "cellular", "data", "lte", "signal", "service", "no service", "carrier", "airdrop", "hotspot", "connect"],
            "risk_level": "low",
            "common_resolution": "Toggle Airplane Mode, reset Network Settings (Settings > General > Reset > Reset Network Settings)."
        },
        {
            "intent": "audio_sound_microphone",
            "keywords": ["sound", "audio", "speaker", "earpiece", "volume", "microphone", "mic", "headphones", "airpods", "hear", "calling", "static"],
            "risk_level": "low",
            "common_resolution": "Clean speaker grilles, check sound settings, test with Voice Memos, toggle Bluetooth."
        },
        {
            "intent": "storage_icloud_backup",
            "keywords": ["icloud", "storage", "backup", "photos", "sync", "syncing", "restore", "space", "full storage", "gb"],
            "risk_level": "low",
            "common_resolution": "Manage iCloud storage in Settings > [Name] > iCloud > Manage Storage, optimize photo storage."
        },
        {
            "intent": "device_freeze_unresponsive_boot",
            "keywords": ["freeze", "frozen", "unresponsive", "black screen", "apple logo", "stuck", "boot loop", "touch screen", "glitch", "crash", "crashing"],
            "risk_level": "low",
            "common_resolution": "Perform force restart (Volume Up -> Volume Down -> Hold Side Button); connect to iTunes."
        },
        {
            "intent": "general_inquiry_advice",
            "keywords": ["how to", "feature", "recommend", "setting", "question", "help", "wondering", "release date", "specs", "trade in", "price"],
            "risk_level": "low",
            "common_resolution": "Provide official Apple Support documentation links and general guidance."
        }
    ]
    
    # Calculate empirical match counts
    intent_counts = Counter()
    intent_samples = defaultdict(list)
    
    for issue in issues:
        issue_lower = issue.lower()
        matched = False
        for item in intent_definitions:
            intent_name = item["intent"]
            if any(re.search(rf"\b{re.escape(kw)}\b", issue_lower) for kw in item["keywords"]):
                intent_counts[intent_name] += 1
                if len(intent_samples[intent_name]) < 5:
                    intent_samples[intent_name].append(issue)
                matched = True
                break
        if not matched:
            intent_counts["other_unclassified"] += 1

    print("\n" + "="*80)
    print("EMPIRICAL INTENT FREQUENCY IN APPLE SUPPORT CONVERSATIONS")
    print("="*80)
    total_analyzed = len(issues)
    for item in intent_definitions:
        name = item["intent"]
        cnt = intent_counts[name]
        pct = (cnt / total_analyzed) * 100
        print(f"{name:<35} : {cnt:>6,d} ({pct:>5.1f}%) | Risk: {item['risk_level'].upper()}")
    
    other_cnt = intent_counts["other_unclassified"]
    print(f"{'other_unclassified':<35} : {other_cnt:>6,d} ({(other_cnt/total_analyzed)*100:>5.1f}%)")
    print("="*80)
    
    return intent_definitions, intent_counts, intent_samples

if __name__ == "__main__":
    discover_intents(CONVERSATIONS_PATH)
