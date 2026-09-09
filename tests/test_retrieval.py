"""
Unit tests for historical evidence retriever.
"""

import pytest
from src.retrieval.retriever import HistoricalCase, HistoricalRetriever
from sklearn.feature_extraction.text import TfidfVectorizer

def test_retriever_search():
    cases = [
        HistoricalCase(
            case_id="case_001",
            customer_issue="Battery draining fast on iOS 11",
            brand_resolution="Check Settings > Battery for high-usage apps.",
            conversation_context="Context 1",
            intent="battery_power_drain"
        ),
        HistoricalCase(
            case_id="case_002",
            customer_issue="Screen cracked and broken",
            brand_resolution="Please schedule a repair at an Apple Store Genius Bar.",
            conversation_context="Context 2",
            intent="hardware_screen_physical_damage"
        )
    ]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([c.customer_issue for c in cases])
    retriever = HistoricalRetriever(vectorizer, matrix, cases)
    
    results = retriever.search("battery dies very quickly", top_k=1)
    assert len(results) == 1
    assert results[0].case_id == "case_001"
    assert "Settings > Battery" in results[0].brand_resolution
    assert results[0].similarity_score > 0.0

def test_retriever_empty_query():
    cases = [HistoricalCase("c1", "issue", "resolution", "ctx", "intent")]
    vec = TfidfVectorizer().fit(["issue"])
    mat = vec.transform(["issue"])
    retriever = HistoricalRetriever(vec, mat, cases)
    assert retriever.search("") == []
