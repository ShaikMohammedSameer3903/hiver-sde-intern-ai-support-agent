"""
Script to build the historical case retrieval index.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.index import build_retrieval_index

if __name__ == "__main__":
    build_retrieval_index()
