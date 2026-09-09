"""
Global Configuration for Hiver Support Agent System
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
REPORT_DIR = PROJECT_ROOT / "report"

# Dataset Paths
RAW_CSV_CANDIDATES = [
    PROJECT_ROOT / "twcs" / "twcs.csv",
    PROJECT_ROOT / "data" / "twcs.csv",
    Path(r"C:\Users\shaik\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10\twcs\twcs.csv")
]

def get_raw_csv_path() -> Path:
    for candidate in RAW_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return RAW_CSV_CANDIDATES[0]

# Selected Brand Configuration
SELECTED_BRAND = "AppleSupport"
BRAND_HANDLE = "@AppleSupport"

# Processed Data Paths
BRAND_RAW_DATA_PATH = DATA_DIR / f"{SELECTED_BRAND.lower()}_raw.csv"
CONVERSATIONS_PATH = DATA_DIR / f"{SELECTED_BRAND.lower()}_conversations.jsonl"
EVAL_SET_PATH = EVALUATION_DIR / "golden_set.jsonl"
INTENTS_PATH = DATA_DIR / "intents.yaml"
RETRIEVAL_INDEX_DIR = DATA_DIR / "retrieval_index"

# Classification & Escalation Configuration
INTENT_CONFIDENCE_THRESHOLD = 0.60  # Calibrated minimum confidence for auto-handle
RETRIEVAL_SIMILARITY_THRESHOLD = 0.30  # Minimum semantic cosine similarity with historical resolution
MAX_AUTOHANDLE_RISK_LEVEL = "low"  # Only low-risk intents allowed to auto-handle without escalation

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
