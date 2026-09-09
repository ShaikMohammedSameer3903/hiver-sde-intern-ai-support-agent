"""
Unit tests for data cleaning and validation.
"""

import pytest
from src.data.cleaner import clean_tweet_text, clean_brand_reply_text, validate_record

def test_validate_record():
    valid = {
        "tweet_id": 101,
        "author_id": "115712",
        "text": "@AppleSupport my screen is frozen"
    }
    assert validate_record(valid) is True
    
    # Null text
    assert validate_record({"tweet_id": 101, "author_id": "115712", "text": None}) is False
    # Empty text
    assert validate_record({"tweet_id": 101, "author_id": "115712", "text": "   "}) is False
    # Missing tweet_id
    assert validate_record({"author_id": "115712", "text": "Help"}) is False

def test_clean_tweet_text_preserves_voice():
    raw = "@AppleSupport my iphon 8 cant charge!!! 😡😢 it died @ 20%"
    cleaned = clean_tweet_text(raw, strip_brand_mentions=True)
    # Brand handle stripped
    assert not cleaned.startswith("@AppleSupport")
    # Slang, typos, and emojis preserved
    assert "iphon 8" in cleaned
    assert "cant charge!!!" in cleaned
    assert "😡😢" in cleaned

def test_clean_brand_reply_text():
    raw = "@115712 We would like to help. Please DM us your device model."
    cleaned = clean_brand_reply_text(raw)
    assert not cleaned.startswith("@115712")
    assert "We would like to help." in cleaned
