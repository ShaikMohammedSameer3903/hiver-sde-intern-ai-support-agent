"""
Intent Classification module containing baselines and proposed classifiers.
"""
from .baselines import MajorityBaseline, TFIDFBaseline
from .classifier import HybridIntentClassifier, IntentPrediction

__all__ = [
    "MajorityBaseline",
    "TFIDFBaseline",
    "HybridIntentClassifier",
    "IntentPrediction"
]
