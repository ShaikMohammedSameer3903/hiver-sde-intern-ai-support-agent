"""
Data cleaning and validation module for Twitter customer support interactions.

Strict Principle: Clean ONLY when justified. Preserve real-world customer voice:
- Preserve spelling mistakes, typos, slang, punctuation, and emojis.
- Remove only malformed records, null/empty texts, and unprintable byte corruption.
"""

import re
from typing import Optional, Dict, Any

# Regex to detect unprintable control characters or corrupt byte sequences
CORRUPT_BYTES_REGEX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]')
# Regex for excess whitespace
MULTI_WHITESPACE_REGEX = re.compile(r'\s+')

def validate_record(record: Dict[str, Any]) -> bool:
    """
    Validates if a tweet record has all required fields and valid types.
    """
    if not record:
        return False
        
    text = record.get("text")
    if text is None or not isinstance(text, str) or not text.strip():
        return False
        
    tweet_id = record.get("tweet_id")
    if tweet_id is None:
        return False
        
    author_id = record.get("author_id")
    if author_id is None or not str(author_id).strip():
        return False
        
    return True

def clean_tweet_text(text: str, strip_brand_mentions: bool = False, brand_handle: str = "@AppleSupport") -> str:
    """
    Cleans tweet text while strictly preserving user slang, typos, emojis, and punctuation.
    
    Args:
        text: Raw tweet text.
        strip_brand_mentions: If True, removes the initial brand mention tag for intent classification.
        brand_handle: The brand handle to strip if requested.
        
    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
        
    # Replace unicode replacement chars / control chars
    cleaned = CORRUPT_BYTES_REGEX.sub('', text)
    
    # Normalize multiple whitespace / tabs / newlines to single space or clean newline
    cleaned = MULTI_WHITESPACE_REGEX.sub(' ', cleaned).strip()
    
    if strip_brand_mentions and brand_handle:
        # Remove leading brand handle if present (e.g. '@AppleSupport my screen is black' -> 'my screen is black')
        pattern = re.compile(rf'^{re.escape(brand_handle)}\s*', re.IGNORECASE)
        cleaned = pattern.sub('', cleaned).strip()
        
    return cleaned

def clean_brand_reply_text(text: str) -> str:
    """
    Cleans outbound brand response text (stripping leading customer handle tags like @115712).
    """
    if not text:
        return ""
    cleaned = CORRUPT_BYTES_REGEX.sub('', text)
    cleaned = MULTI_WHITESPACE_REGEX.sub(' ', cleaned).strip()
    # Remove leading customer handle mentions (@123456)
    cleaned = re.sub(r'^@\w+\s*', '', cleaned).strip()
    return cleaned
