"""
Retrieval module for indexing and fetching historical resolution evidence.
"""
from .retriever import HistoricalRetriever, RetrievalResult, HistoricalCase
from .index import build_retrieval_index, load_retrieval_index

__all__ = [
    "HistoricalRetriever",
    "RetrievalResult",
    "HistoricalCase",
    "build_retrieval_index",
    "load_retrieval_index"
]
