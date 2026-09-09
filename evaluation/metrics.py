"""
Evaluation Metrics Suite for Customer Support AI Agent.

Calculates:
1. Classification Metrics (Accuracy, Macro Precision, Macro Recall, Macro F1, Per-intent F1)
2. Retrieval Metrics (Recall@1, Recall@3, Recall@5, Mean Reciprocal Rank MRR)
3. Escalation & Safety Metrics (Auto-Handle Precision, False-Auto Rate, Escalation Recall, Coverage)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes standard multi-class classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    labels = sorted(list(set(y_true + y_pred)))
    p_per, r_per, f1_per, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    per_intent_f1 = {
        labels[i]: {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(support[i])
        }
        for i in range(len(labels))
    }
    
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "per_intent_metrics": per_intent_f1,
        "labels": labels,
        "confusion_matrix": cm
    }

def compute_retrieval_metrics(
    queries: List[Dict[str, Any]],
    retrieved_results: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Evaluates retrieval quality based on semantic similarity and relevance to intent.
    """
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    mrr_total = 0.0
    similarity_scores = []
    
    for q, results in zip(queries, retrieved_results):
        target_intent = q.get("gold_intent")
        ranks = []
        for rank_idx, r in enumerate(results):
            similarity_scores.append(r.get("similarity_score", 0.0))
            # If retrieved case shares intent with query or has high relevance
            if r.get("intent") == target_intent or r.get("similarity_score", 0.0) >= 0.50:
                ranks.append(rank_idx + 1)
                
        if ranks:
            first_rank = ranks[0]
            if first_rank == 1:
                recall_at_1 += 1
            if first_rank <= 3:
                recall_at_3 += 1
            if first_rank <= 5:
                recall_at_5 += 1
            mrr_total += 1.0 / first_rank

    n = len(queries) if queries else 1
    return {
        "recall_at_1": round(recall_at_1 / n, 4),
        "recall_at_3": round(recall_at_3 / n, 4),
        "recall_at_5": round(recall_at_5 / n, 4),
        "mrr": round(mrr_total / n, 4),
        "mean_top1_similarity": round(float(np.mean([r[0].get("similarity_score", 0.0) for r in retrieved_results if r])), 4)
    }

def compute_escalation_safety_metrics(y_true_decision: List[str], y_pred_decision: List[str]) -> Dict[str, Any]:
    """
    Computes conservative automation and safety metrics.
    
    Decisions:
    - AUTO_HANDLE
    - ESCALATE_TO_HUMAN
    """
    total = len(y_true_decision)
    if total == 0:
        return {}
        
    auto_true = sum(1 for y in y_true_decision if y == "AUTO_HANDLE")
    escalate_true = sum(1 for y in y_true_decision if y == "ESCALATE_TO_HUMAN")
    
    auto_pred = sum(1 for y in y_pred_decision if y == "AUTO_HANDLE")
    escalate_pred = sum(1 for y in y_pred_decision if y == "ESCALATE_TO_HUMAN")
    
    # Correct Auto-Handles (True Positives for automation)
    correct_auto = sum(1 for yt, yp in zip(y_true_decision, y_pred_decision) if yt == "AUTO_HANDLE" and yp == "AUTO_HANDLE")
    # False Auto-Handles (CRITICAL SAFETY RISK: Automated when it should have escalated!)
    false_auto = sum(1 for yt, yp in zip(y_true_decision, y_pred_decision) if yt == "ESCALATE_TO_HUMAN" and yp == "AUTO_HANDLE")
    # Correct Escalations (True Positives for escalation)
    correct_escalate = sum(1 for yt, yp in zip(y_true_decision, y_pred_decision) if yt == "ESCALATE_TO_HUMAN" and yp == "ESCALATE_TO_HUMAN")
    # Unnecessary Escalations (False Positives for escalation - safe, but lowers automation coverage)
    unnecessary_escalate = sum(1 for yt, yp in zip(y_true_decision, y_pred_decision) if yt == "AUTO_HANDLE" and yp == "ESCALATE_TO_HUMAN")
    
    auto_precision = correct_auto / auto_pred if auto_pred > 0 else 0.0
    false_auto_rate = false_auto / auto_pred if auto_pred > 0 else 0.0
    escalation_recall = correct_escalate / escalate_true if escalate_true > 0 else 0.0
    automation_coverage = auto_pred / total
    
    return {
        "automation_coverage": round(automation_coverage, 4),
        "auto_handle_precision": round(auto_precision, 4),
        "false_auto_rate": round(false_auto_rate, 4),
        "escalation_recall": round(escalation_recall, 4),
        "correct_auto_handles": correct_auto,
        "false_auto_handles": false_auto,
        "correct_escalations": correct_escalate,
        "unnecessary_escalations": unnecessary_escalate,
        "total_eval_cases": total
    }
