"""
Data loader module for extracting and loading brand-specific Twitter support interactions.
"""

import os
import re
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import (
    get_raw_csv_path,
    SELECTED_BRAND,
    BRAND_HANDLE,
    BRAND_RAW_DATA_PATH,
    DATA_DIR
)
from src.data.cleaner import validate_record

def extract_brand_subset(
    raw_csv_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    brand_name: str = SELECTED_BRAND,
    chunk_size: int = 250000
) -> Path:
    """
    Streams the full Twitter customer support CSV and extracts all interactions
    involving the selected brand (both outbound brand tweets and inbound customer tweets).
    """
    raw_csv = raw_csv_path or get_raw_csv_path()
    out_csv = output_path or BRAND_RAW_DATA_PATH
    
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV not found at: {raw_csv}")
        
    print(f"Extracting interactions for brand '{brand_name}' from {raw_csv}...")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    brand_handle_lower = f"@{brand_name.lower()}"
    brand_name_lower = brand_name.lower()
    
    # Store all matched chunks
    matched_chunks = []
    total_processed = 0
    
    chunks = pd.read_csv(
        raw_csv,
        chunksize=chunk_size,
        dtype={
            'tweet_id': 'int64',
            'author_id': 'string',
            'inbound': 'bool',
            'created_at': 'string',
            'text': 'string',
            'response_tweet_id': 'string',
            'in_response_to_tweet_id': 'float64'
        }
    )
    
    for i, chunk in enumerate(chunks):
        total_processed += len(chunk)
        # 1. Outbound tweets where brand is the author
        outbound_mask = (~chunk['inbound']) & (chunk['author_id'].str.lower() == brand_name_lower)
        
        # 2. Inbound customer tweets mentioning the brand handle
        inbound_mask = chunk['inbound'] & chunk['text'].str.lower().str.contains(brand_handle_lower, na=False, regex=False)
        
        matched = chunk[outbound_mask | inbound_mask]
        if len(matched) > 0:
            matched_chunks.append(matched)
            
        print(f"  Chunk {i+1} processed ({total_processed:,} total rows scanned, {sum(len(c) for c in matched_chunks):,} brand rows matched)...")

    if not matched_chunks:
        raise ValueError(f"No interactions found for brand '{brand_name}'!")
        
    brand_df = pd.concat(matched_chunks, ignore_index=True)
    
    # Drop exact duplicate rows if any
    brand_df = brand_df.drop_duplicates(subset=['tweet_id'])
    
    # Sort chronologically or by tweet_id
    brand_df = brand_df.sort_values(by='tweet_id')
    
    brand_df.to_csv(out_csv, index=False)
    print(f"\nSuccessfully extracted {len(brand_df):,} tweets for '{brand_name}' to {out_csv}!")
    return out_csv

def load_brand_tweets(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the brand-specific tweets CSV into a pandas DataFrame.
    """
    target_path = csv_path or BRAND_RAW_DATA_PATH
    if not target_path.exists():
        print(f"Brand dataset not found at {target_path}. Running extraction now...")
        extract_brand_subset(output_path=target_path)
        
    df = pd.read_csv(
        target_path,
        dtype={
            'tweet_id': 'int64',
            'author_id': 'string',
            'inbound': 'bool',
            'created_at': 'string',
            'text': 'string',
            'response_tweet_id': 'string',
            'in_response_to_tweet_id': 'float64'
        }
    )
    return df

if __name__ == "__main__":
    extract_brand_subset()
