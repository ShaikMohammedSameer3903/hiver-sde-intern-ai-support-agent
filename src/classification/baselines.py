"""
Baseline Intent Classifiers for Twitter Customer Support.

Baseline 1: Majority Class Predictor (Always predicts the most frequent intent).
Baseline 2: TF-IDF + Logistic Regression / Linear SVM.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

class MajorityBaseline:
    """
    Trivial baseline: Predicts the majority class observed during training.
    """
    def __init__(self):
        self.majority_class: str = "software_update_os"
        self.majority_prob: float = 0.283
        
    def fit(self, texts: List[str], labels: List[str]):
        if not labels:
            return self
        from collections import Counter
        counts = Counter(labels)
        self.majority_class = counts.most_common(1)[0][0]
        self.majority_prob = counts[self.majority_class] / len(labels)
        return self
        
    def predict(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_class,
            "confidence": round(float(self.majority_prob), 4),
            "method": "majority_baseline"
        }
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(t) for t in texts]


class TFIDFBaseline:
    """
    Standard ML Baseline: Sublinear TF-IDF + Calibrated Logistic Regression.
    """
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), max_features: int = 8000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.model = LogisticRegression(
            C=2.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs'
        )
        self.is_fitted = False
        self.classes_: List[str] = []
        
    def fit(self, texts: List[str], labels: List[str]):
        if not texts or not labels:
            raise ValueError("Texts and labels cannot be empty for TF-IDF training!")
            
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self
        
    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("TFIDFBaseline model is not fitted yet!")
            
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        max_idx = int(np.argmax(probs))
        predicted_intent = self.classes_[max_idx]
        confidence = float(probs[max_idx])
        
        # Sort all class probabilities
        all_probs = {self.classes_[i]: round(float(probs[i]), 4) for i in range(len(self.classes_))}
        
        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "method": "tfidf_logistic_regression"
        }
        
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("TFIDFBaseline model is not fitted yet!")
            
        X = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X)
        results = []
        for p in probs:
            max_idx = int(np.argmax(p))
            results.append({
                "intent": self.classes_[max_idx],
                "confidence": round(float(p[max_idx]), 4),
                "method": "tfidf_logistic_regression"
            })
        return results
