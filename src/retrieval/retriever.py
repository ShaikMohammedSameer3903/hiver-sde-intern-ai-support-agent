"""
Historical Evidence Retriever for Apple Support Cases.

Provides fast, auditable cosine-similarity retrieval over verified historical
customer support cases (Customer Issue -> Brand Resolution).
Exposes case IDs, similarity scores, historical responses, and grounded evidence.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class HistoricalCase:
    case_id: str
    customer_issue: str
    brand_resolution: str
    conversation_context: str
    intent: Optional[str] = None
    turn_count: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RetrievalResult:
    case_id: str
    similarity_score: float
    customer_issue: str
    brand_resolution: str
    conversation_context: str
    intent: Optional[str] = None
    rank: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class HistoricalRetriever:
    """
    Retrieval engine matching incoming customer messages against historical resolution corpus.
    """
    def __init__(self, vectorizer, matrix, cases: List[HistoricalCase]):
        self.vectorizer = vectorizer
        self.matrix = matrix  # scipy sparse matrix or dense ndarray
        self.cases = cases
        
    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
        intent_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Searches the historical case index for the most relevant historical resolutions.
        """
        if not query or not query.strip():
            return []
            
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        
        # Filter by intent if requested
        if intent_filter:
            valid_indices = [
                i for i in range(len(self.cases))
                if self.cases[i].intent == intent_filter or self.cases[i].intent is None
            ]
            if not valid_indices:
                valid_indices = list(range(len(self.cases)))
        else:
            valid_indices = list(range(len(self.cases)))
            
        ranked = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)
        
        results = []
        rank = 1
        for idx in ranked[:top_k]:
            sim = float(scores[idx])
            if sim < min_similarity:
                continue
            case = self.cases[idx]
            results.append(RetrievalResult(
                case_id=case.case_id,
                similarity_score=round(sim, 4),
                customer_issue=case.customer_issue,
                brand_resolution=case.brand_resolution,
                conversation_context=case.conversation_context,
                intent=case.intent,
                rank=rank
            ))
            rank += 1
            
        return results

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "retriever_data.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "cases": self.cases
            }, f)

    @classmethod
    def load(cls, input_dir: Path) -> "HistoricalRetriever":
        with open(input_dir / "retriever_data.pkl", "rb") as f:
            data = pickle.load(f)
        return cls(
            vectorizer=data["vectorizer"],
            matrix=data["matrix"],
            cases=data["cases"]
        )
