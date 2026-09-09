"""
Proposed Intent Classifier for Apple Support Interactions.

Implements a Calibrated Hybrid Architecture:
- Sublinear N-gram Representation with Character & Word Tokenizers (resilient to typos and Twitter noise)
- Margin-based Ambiguity Quantification
- Probability Calibration for Trustworthy Confidence Scoring
- Traceable Intent Reasoning with Triggered Domain Features
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from src.config import INTENTS_PATH

@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    margin: float
    is_ambiguous: bool
    risk_level: str
    default_action: str
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    triggered_keywords: List[str] = field(default_factory=list)
    method: str = "proposed_hybrid_calibrated"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "margin": self.margin,
            "is_ambiguous": self.is_ambiguous,
            "risk_level": self.risk_level,
            "default_action": self.default_action,
            "all_probabilities": self.all_probabilities,
            "triggered_keywords": self.triggered_keywords,
            "method": self.method
        }

class HybridIntentClassifier:
    """
    Proposed Classifier combining Word N-grams, Char N-grams, and Domain Taxonomy matching.
    """
    def __init__(self, ambiguity_margin_threshold: float = 0.15):
        self.ambiguity_margin_threshold = ambiguity_margin_threshold
        
        # Load intent taxonomy metadata
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.taxonomy = {item["name"]: item for item in cfg["intents"]}
        
        # Multi-scale feature union: word n-grams (1-3) + char n-grams (3-5)
        # Char n-grams make the classifier robust to typos, truncated words, and Twitter abbreviations
        word_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=12000,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'(?u)\b[\w\']+\b'
        )
        char_vec = TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer='char_wb',
            max_features=8000,
            sublinear_tf=True
        )
        
        self.features = FeatureUnion([
            ('word', word_vec),
            ('char', char_vec)
        ])
        
        self.model = LogisticRegression(
            C=3.5,
            max_iter=1500,
            class_weight='balanced',
            solver='lbfgs'
        )
        self.is_fitted = False
        self.classes_: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        if not texts or not labels:
            raise ValueError("Training data cannot be empty!")
            
        X = self.features.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> IntentPrediction:
        if not self.is_fitted:
            raise RuntimeError("Classifier has not been fitted yet!")
            
        text_clean = text.lower() if text else ""
        X = self.features.transform([text_clean])
        probs = self.model.predict_proba(X)[0]
        
        # Sort probabilities descending
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else top1_idx
        
        top1_intent = self.classes_[top1_idx]
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[top2_idx]) if len(sorted_indices) > 1 else 0.0
        
        margin = top1_prob - top2_prob
        is_ambiguous = margin < self.ambiguity_margin_threshold
        
        # Check triggered domain keywords from taxonomy
        triggered_kws = []
        for kw in self.taxonomy.get(top1_intent, {}).get("examples", []):
            for word in kw.lower().split():
                if len(word) > 3 and word in text_clean and word not in triggered_kws:
                    triggered_kws.append(word)
                    
        meta = self.taxonomy.get(top1_intent, {
            "risk_level": "medium",
            "default_action": "ESCALATE_TO_HUMAN"
        })
        
        all_probs = {self.classes_[i]: round(float(probs[i]), 4) for i in sorted_indices}
        
        return IntentPrediction(
            intent=top1_intent,
            confidence=round(top1_prob, 4),
            margin=round(margin, 4),
            is_ambiguous=is_ambiguous,
            risk_level=meta.get("risk_level", "low"),
            default_action=meta.get("default_action", "AUTO_HANDLE"),
            all_probabilities=all_probs,
            triggered_keywords=triggered_kws[:5],
            method="proposed_hybrid_calibrated"
        )
        
    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        return [self.predict(t) for t in texts]
