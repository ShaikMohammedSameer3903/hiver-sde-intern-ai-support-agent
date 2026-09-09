"""
Unit tests for conversation tree reconstruction.
"""

import pandas as pd
import pytest
from src.data.conversations import reconstruct_conversations, ConversationThread

def test_reconstruct_conversations(tmp_path):
    records = [
        {
            "tweet_id": 1,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:08:27 +0000 2017",
            "text": "@AppleSupport My phone won't charge",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None
        },
        {
            "tweet_id": 2,
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "Tue Oct 31 22:10:47 +0000 2017",
            "text": "@115712 We are here to help. Have you tried a different lightning cable?",
            "response_tweet_id": "3",
            "in_response_to_tweet_id": 1.0
        },
        {
            "tweet_id": 3,
            "author_id": "115712",
            "inbound": True,
            "created_at": "Tue Oct 31 22:11:45 +0000 2017",
            "text": "@AppleSupport Yes I did, same issue",
            "response_tweet_id": "4",
            "in_response_to_tweet_id": 2.0
        },
        {
            "tweet_id": 4,
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "Tue Oct 31 22:14:00 +0000 2017",
            "text": "@115712 Please inspect the lightning port for lint or visit an Apple Store.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": 3.0
        }
    ]
    df = pd.DataFrame(records)
    test_out = tmp_path / "test_conversations.jsonl"
    threads = reconstruct_conversations(df, brand_name="AppleSupport", output_path=test_out, min_turns=2)
    
    assert len(threads) == 1
    thread = threads[0]
    assert thread.root_tweet_id == 1
    assert thread.turn_count == 4
    assert thread.has_brand_resolution is True
    assert "different lightning cable" in thread.brand_first_response
    assert "inspect the lightning port" in thread.brand_final_resolution
    assert test_out.exists()
