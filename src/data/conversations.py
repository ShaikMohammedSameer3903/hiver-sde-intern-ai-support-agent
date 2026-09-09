"""
Conversation Reconstruction Module for Twitter Customer Support.

Reconstructs multi-turn conversational trees from tweet-level parent/reply links:
  Customer (Root Issue) -> Brand (Reply) -> Customer (Follow-up) -> Brand (Resolution)
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import pandas as pd

from src.config import SELECTED_BRAND, CONVERSATIONS_PATH
from src.data.cleaner import clean_tweet_text, clean_brand_reply_text

def parse_twitter_date(date_str: str) -> Optional[datetime]:
    if not date_str or pd.isna(date_str):
        return None
    try:
        return datetime.strptime(str(date_str), "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None

@dataclass
class ConversationTurn:
    turn_index: int
    tweet_id: int
    author_id: str
    role: str  # 'customer' or 'brand'
    text: str
    cleaned_text: str
    created_at: str
    in_response_to_tweet_id: Optional[int] = None

@dataclass
class ConversationThread:
    conversation_id: str
    root_tweet_id: int
    customer_author_id: str
    turn_count: int
    turns: List[ConversationTurn] = field(default_factory=list)
    customer_initial_issue: str = ""
    customer_cleaned_issue: str = ""
    brand_first_response: Optional[str] = None
    brand_final_resolution: Optional[str] = None
    has_brand_resolution: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
    
    def get_conversation_history_text(self, up_to_turn: Optional[int] = None) -> str:
        """
        Formats conversational turns up to a given index as readable dialog context.
        """
        target_turns = self.turns if up_to_turn is None else self.turns[:up_to_turn]
        history_lines = []
        for t in target_turns:
            speaker = "Customer" if t.role == "customer" else f"Support ({SELECTED_BRAND})"
            history_lines.append(f"{speaker}: {t.cleaned_text}")
        return "\n".join(history_lines)

def reconstruct_conversations(
    df: pd.DataFrame,
    brand_name: str = SELECTED_BRAND,
    output_path: Optional[Path] = None,
    min_turns: int = 2
) -> List[ConversationThread]:
    """
    Reconstructs complete conversation threads from tweet records.
    """
    print(f"Reconstructing conversation threads for '{brand_name}' from {len(df):,} tweets...")
    
    # Map tweet_id to row dict for O(1) lookup
    tweet_map: Dict[int, Dict[str, Any]] = {}
    children_map: Dict[int, List[int]] = {}
    parent_map: Dict[int, int] = {}
    
    brand_lower = brand_name.lower()
    
    for row in df.itertuples(index=False):
        tid = int(row.tweet_id)
        author = str(row.author_id)
        inbound = bool(row.inbound)
        created = str(row.created_at) if not pd.isna(row.created_at) else ""
        text = str(row.text) if not pd.isna(row.text) else ""
        
        parent_id = None
        if not pd.isna(row.in_response_to_tweet_id):
            try:
                parent_id = int(row.in_response_to_tweet_id)
            except (ValueError, TypeError):
                parent_id = None
                
        role = "brand" if (not inbound or author.lower() == brand_lower) else "customer"
        
        tweet_map[tid] = {
            "tweet_id": tid,
            "author_id": author,
            "role": role,
            "created_at": created,
            "text": text,
            "parent_id": parent_id
        }
        
        if parent_id is not None:
            parent_map[tid] = parent_id
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(tid)

    # Identify Root Tweets (tweets that have no parent in this dataset, or whose parent is NaN)
    root_tweet_ids = [tid for tid, info in tweet_map.items() if info["parent_id"] is None or info["parent_id"] not in tweet_map]
    
    print(f"Found {len(root_tweet_ids):,} potential conversation roots. Traversal starting...")
    
    threads: List[ConversationThread] = []
    visited_tweets: Set[int] = set()
    
    for root_id in root_tweet_ids:
        root_info = tweet_map[root_id]
        
        # We focus primarily on customer-initiated support threads
        if root_info["role"] != "customer":
            continue
            
        # Traverse linear reply chain (or primary branch)
        current_id = root_id
        chain: List[Dict[str, Any]] = []
        
        while current_id in tweet_map and current_id not in visited_tweets:
            visited_tweets.add(current_id)
            chain.append(tweet_map[current_id])
            # Follow direct child
            children = children_map.get(current_id, [])
            if not children:
                break
            # Pick first chronological child
            current_id = children[0]
            
        if len(chain) < min_turns:
            continue
            
        # Build turns
        turns: List[ConversationTurn] = []
        brand_replies: List[str] = []
        customer_author = root_info["author_id"]
        
        for idx, t_info in enumerate(chain):
            cleaned = clean_brand_reply_text(t_info["text"]) if t_info["role"] == "brand" else clean_tweet_text(t_info["text"], strip_brand_mentions=True)
            turn = ConversationTurn(
                turn_index=idx + 1,
                tweet_id=t_info["tweet_id"],
                author_id=t_info["author_id"],
                role=t_info["role"],
                text=t_info["text"],
                cleaned_text=cleaned,
                created_at=t_info["created_at"],
                in_response_to_tweet_id=t_info["parent_id"]
            )
            turns.append(turn)
            if t_info["role"] == "brand":
                brand_replies.append(cleaned)
                
        if not brand_replies:
            continue
            
        thread = ConversationThread(
            conversation_id=f"conv_apple_{root_id}",
            root_tweet_id=root_id,
            customer_author_id=customer_author,
            turn_count=len(turns),
            turns=turns,
            customer_initial_issue=root_info["text"],
            customer_cleaned_issue=clean_tweet_text(root_info["text"], strip_brand_mentions=True),
            brand_first_response=brand_replies[0] if brand_replies else None,
            brand_final_resolution=brand_replies[-1] if brand_replies else None,
            has_brand_resolution=len(brand_replies) > 0
        )
        threads.append(thread)

    print(f"Successfully reconstructed {len(threads):,} multi-turn conversation threads!")
    
    out_path = output_path or CONVERSATIONS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(json.dumps(thread.to_dict()) + "\n")
            
    print(f"Saved reconstructed conversations to: {out_path}")
    return threads

if __name__ == "__main__":
    from src.data.loader import load_brand_tweets
    df = load_brand_tweets()
    reconstruct_conversations(df)
