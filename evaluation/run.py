"""
Master Evaluation Benchmark Runner.

Executes the complete rigorous benchmark on the 200-example Golden Evaluation Set:
1. Baseline 1 (Majority Class)
2. Baseline 2 (TF-IDF + Logistic Regression)
3. Proposed System (Hybrid Intent Classifier + Retrieval + Grounding + Escalation)
4. LLM Judge Rubric Evaluation
5. Human vs Judge Agreement Analysis
6. Real Failure Mode Analysis
Saves results to artifacts/evaluation_results.json and artifacts/evaluation_report.md.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from typing import List, Dict, Any
import numpy as np

from src.config import EVAL_SET_PATH, ARTIFACTS_DIR
from src.classification.baselines import MajorityBaseline, TFIDFBaseline
from src.pipeline import SupportAgentPipeline
from evaluation.metrics import (
    compute_classification_metrics,
    compute_retrieval_metrics,
    compute_escalation_safety_metrics
)
from evaluation.judge import LLMJudge
from evaluation.human_agreement import evaluate_human_agreement
from evaluation.failure_analysis import extract_real_failure_modes

def run_full_evaluation():
    start_time = time.time()
    print("=" * 80)
    print("STARTING COMPLETE EVALUATION BENCHMARK ON GOLDEN SET")
    print(f"Target Evaluation File: {EVAL_SET_PATH}")
    print("=" * 80)
    
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"Golden Set not found at {EVAL_SET_PATH}")
        
    # 1. Load Golden Set
    golden_set: List[Dict[str, Any]] = []
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            golden_set.append(json.loads(line))
            
    print(f"Loaded {len(golden_set)} held-out Golden Set evaluation examples.")
    
    y_true_intents = [ex["gold_intent"] for ex in golden_set]
    y_true_decisions = [ex["gold_decision"] for ex in golden_set]
    customer_messages = [ex["customer_message"] for ex in golden_set]
    
    # 2. Initialize Support Agent Pipeline
    pipeline = SupportAgentPipeline()
    
    # 3. Train Baselines on same reference data used by pipeline classifier
    # Extract training data from pipeline's classifier
    train_texts = pipeline.classifier.features.transformer_list[0][1].vocabulary_
    # Train baselines using reference corpus sample
    print("\nTraining Baseline Classifiers on reference distribution...")
    maj_baseline = MajorityBaseline().fit([], y_true_intents)
    
    # Train TFIDFBaseline on pipeline's training set
    tfidf_baseline = TFIDFBaseline()
    # Sample 10,000 reference examples from conversations
    from src.config import CONVERSATIONS_PATH, DATA_DIR
    import re
    held_out_path = DATA_DIR / "held_out_evaluation_ids.json"
    held_out_ids = set()
    if held_out_path.exists():
        with open(held_out_path, "r", encoding="utf-8") as f:
            held_out_ids = set(json.load(f))
            
    core_kws = {
        "battery_power_drain": ["battery", "drain", "charge", "dies", "overheating", "percentage", "shut down"],
        "software_update_os": ["update", "updated", "updating", "ios 11", "ios11", "install", "upgrade"],
        "apple_id_account_access": ["apple id", "appleid", "password", "locked", "disabled", "verification code", "2fa", "iforgot"],
        "app_store_billing_subscription": ["app store", "appstore", "billing", "charge", "subscription", "refund", "receipt", "purchase"],
        "hardware_screen_physical_damage": ["screen", "cracked", "broken", "dropped", "glass", "water damage", "repair", "genius bar"],
        "connectivity_wifi_bluetooth_network": ["wifi", "wi-fi", "bluetooth", "cellular", "data", "lte", "signal", "service", "no service", "airdrop"],
        "audio_sound_microphone": ["sound", "audio", "speaker", "earpiece", "volume", "microphone", "mic", "headphones", "airpods"],
        "storage_icloud_backup": ["icloud", "storage", "backup", "photos", "sync", "syncing", "restore", "space", "full storage"],
        "device_freeze_unresponsive_boot": ["freeze", "frozen", "unresponsive", "black screen", "apple logo", "stuck", "boot loop", "crash"],
        "general_inquiry_advice": ["how to", "feature", "recommend", "setting", "wondering", "trade in", "specs"]
    }
    
    ref_texts, ref_labels = [], []
    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["conversation_id"] in held_out_ids:
                continue
            issue = t.get("customer_cleaned_issue", "")
            if len(issue.split()) < 3:
                continue
            matched = [iname for iname, kws in core_kws.items() if any(re.search(rf"\b{re.escape(k)}\b", issue.lower()) for k in kws)]
            if len(matched) == 1:
                ref_texts.append(issue)
                ref_labels.append(matched[0])
                if len(ref_texts) >= 10000:
                    break
                    
    tfidf_baseline.fit(ref_texts, ref_labels)
    
    # 4. Evaluate Baseline 1 (Majority)
    print("Evaluating Baseline 1: Majority Class...")
    maj_preds = [maj_baseline.predict(t)["intent"] for t in customer_messages]
    maj_metrics = compute_classification_metrics(y_true_intents, maj_preds)
    
    # 5. Evaluate Baseline 2 (TF-IDF + Logistic Regression)
    print("Evaluating Baseline 2: TF-IDF Logistic Regression...")
    tfidf_preds = [tfidf_baseline.predict(t)["intent"] for t in customer_messages]
    tfidf_metrics = compute_classification_metrics(y_true_intents, tfidf_preds)
    
    # 6. Evaluate Proposed System (End-to-End Pipeline)
    print("Evaluating Proposed System: Hybrid Intent Classifier + Historical Retrieval + Escalation...")
    pipeline_outputs = []
    proposed_intent_preds = []
    proposed_decision_preds = []
    retrieved_case_lists = []
    
    for ex in golden_set:
        out = pipeline.process(
            customer_message=ex["customer_message"],
            conversation_context=ex.get("conversation_context", "")
        )
        pipeline_outputs.append(out)
        proposed_intent_preds.append(out.predicted_intent)
        proposed_decision_preds.append(out.decision)
        retrieved_case_lists.append(out.historical_evidence)
        
    proposed_cls_metrics = compute_classification_metrics(y_true_intents, proposed_intent_preds)
    
    # 7. Evaluate Retrieval Metrics
    retrieval_metrics = compute_retrieval_metrics(golden_set, retrieved_case_lists)
    
    # 8. Evaluate Escalation & Safety Metrics
    escalation_metrics = compute_escalation_safety_metrics(y_true_decisions, proposed_decision_preds)
    
    # 9. Evaluate with LLM-as-a-Judge
    print("\nRunning LLM-as-a-Judge on generated replies...")
    judge = LLMJudge()
    judge_payloads = [
        {
            "id": ex["id"],
            "customer_message": ex["customer_message"],
            "conversation_context": ex.get("conversation_context", ""),
            "historical_evidence": out.historical_evidence,
            "draft_reply": out.draft_reply
        }
        for ex, out in zip(golden_set, pipeline_outputs)
    ]
    judge_scores = [s.to_dict() for s in judge.evaluate_batch(judge_payloads)]
    
    judge_summary = {
        "mean_correctness": round(float(np.mean([s["correctness"] for s in judge_scores])), 2),
        "mean_groundedness": round(float(np.mean([s["groundedness"] for s in judge_scores])), 2),
        "mean_helpfulness": round(float(np.mean([s["helpfulness"] for s in judge_scores])), 2),
        "mean_unsupported_claims": round(float(np.mean([s["unsupported_claims"] for s in judge_scores])), 2),
        "mean_tone": round(float(np.mean([s["tone"] for s in judge_scores])), 2),
        "mean_overall_score": round(float(np.mean([s["overall_score"] for s in judge_scores])), 2),
        "pass_rate": round(sum(1 for s in judge_scores if s["verdict"] == "PASS") / len(judge_scores) * 100.0, 2),
        "total_evaluated": len(judge_scores)
    }
    
    # 10. Evaluate Human Agreement
    print("Evaluating Human vs. LLM Judge Agreement on 50 sampled replies...")
    agreement_results = evaluate_human_agreement(judge_scores, sample_size=50)
    
    # 11. Real Failure Mode Analysis
    print("Extracting Real Failure Modes...")
    eval_records_for_failure = [
        {
            "id": ex["id"],
            "customer_message": ex["customer_message"],
            "gold_intent": ex["gold_intent"],
            "gold_decision": ex["gold_decision"],
            "pipeline_output": out.to_dict()
        }
        for ex, out in zip(golden_set, pipeline_outputs)
    ]
    real_failures = extract_real_failure_modes(eval_records_for_failure)
    
    # 12. Compile Final Benchmark Results Object
    duration = round(time.time() - start_time, 2)
    
    results = {
        "metadata": {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "golden_set_size": len(golden_set),
            "benchmark_duration_seconds": duration,
            "brand": "AppleSupport"
        },
        "comparative_classification_results": {
            "majority_baseline": {
                "accuracy": maj_metrics["accuracy"],
                "macro_precision": maj_metrics["macro_precision"],
                "macro_recall": maj_metrics["macro_recall"],
                "macro_f1": maj_metrics["macro_f1"]
            },
            "tfidf_baseline": {
                "accuracy": tfidf_metrics["accuracy"],
                "macro_precision": tfidf_metrics["macro_precision"],
                "macro_recall": tfidf_metrics["macro_recall"],
                "macro_f1": tfidf_metrics["macro_f1"]
            },
            "proposed_system": {
                "accuracy": proposed_cls_metrics["accuracy"],
                "macro_precision": proposed_cls_metrics["macro_precision"],
                "macro_recall": proposed_cls_metrics["macro_recall"],
                "macro_f1": proposed_cls_metrics["macro_f1"],
                "per_intent_metrics": proposed_cls_metrics["per_intent_metrics"],
                "labels": proposed_cls_metrics["labels"],
                "confusion_matrix": proposed_cls_metrics["confusion_matrix"]
            }
        },
        "retrieval_evaluation": retrieval_metrics,
        "escalation_safety_evaluation": escalation_metrics,
        "llm_judge_evaluation": judge_summary,
        "human_agreement_evaluation": agreement_results,
        "top_5_real_failures": real_failures
    }
    
    # Save to artifacts/evaluation_results.json
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACTS_DIR / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Generate artifacts/evaluation_report.md
    md_path = ARTIFACTS_DIR / "evaluation_report.md"
    _generate_markdown_report(results, md_path)
    
    # Print Console Summary Table
    _print_console_summary(results)
    print(f"\nEvaluation successfully saved to: {json_path}")
    print(f"Report successfully saved to: {md_path}")
    return results

def _print_console_summary(results: Dict[str, Any]):
    maj = results["comparative_classification_results"]["majority_baseline"]
    tf = results["comparative_classification_results"]["tfidf_baseline"]
    prop = results["comparative_classification_results"]["proposed_system"]
    ret = results["retrieval_evaluation"]
    esc = results["escalation_safety_evaluation"]
    jdg = results["llm_judge_evaluation"]
    agr = results["human_agreement_evaluation"]
    
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS (GOLDEN EVALUATION SET - 200 EXAMPLES)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Majority':<12} | {'TF-IDF ML':<12} | {'Proposed AI':<12}")
    print("-" * 75)
    print(f"{'Intent Accuracy':<30} | {maj['accuracy']:<12.4f} | {tf['accuracy']:<12.4f} | {prop['accuracy']:<12.4f}")
    print(f"{'Macro Precision':<30} | {maj['macro_precision']:<12.4f} | {tf['macro_precision']:<12.4f} | {prop['macro_precision']:<12.4f}")
    print(f"{'Macro Recall':<30} | {maj['macro_recall']:<12.4f} | {tf['macro_recall']:<12.4f} | {prop['macro_recall']:<12.4f}")
    print(f"{'Macro F1':<30} | {maj['macro_f1']:<12.4f} | {tf['macro_f1']:<12.4f} | {prop['macro_f1']:<12.4f}")
    print("=" * 80)
    print("\n--- RETRIEVAL METRICS ---")
    print(f"  Recall@1: {ret['recall_at_1']:.4f} | Recall@3: {ret['recall_at_3']:.4f} | Recall@5: {ret['recall_at_5']:.4f} | MRR: {ret['mrr']:.4f}")
    print("\n--- ESCALATION SAFETY METRICS ---")
    print(f"  Automation Coverage : {esc['automation_coverage']*100:.2f}% ({esc['correct_auto_handles'] + esc['false_auto_handles']}/{esc['total_eval_cases']})")
    print(f"  Auto-Handle Precision: {esc['auto_handle_precision']*100:.2f}% (Safety Accuracy)")
    print(f"  False-Auto Rate     : {esc['false_auto_rate']*100:.2f}% (CRITICAL SAFETY METRIC - Lower is better)")
    print(f"  Escalation Recall   : {esc['escalation_recall']*100:.2f}% (Caught {esc['correct_escalations']}/{esc['correct_escalations'] + esc['false_auto_handles']} high-risk cases)")
    print("\n--- LLM JUDGE & HUMAN AGREEMENT ---")
    print(f"  Judge Pass Rate     : {jdg['pass_rate']}% | Mean Overall Score: {jdg['mean_overall_score']}/5.0")
    print(f"  Groundedness Score  : {jdg['mean_groundedness']}/5.0 | Unsupported Claims Score: {jdg['mean_unsupported_claims']}/5.0")
    print(f"  Human Agreement (r) : {agr['pearson_correlation_r']:.4f} | Cohen's Kappa: {agr['cohen_kappa_verdict']:.4f} | Agreement: {agr['verdict_agreement_percentage']}%")
    print("=" * 80)

def _generate_markdown_report(results: Dict[str, Any], output_path: Path):
    maj = results["comparative_classification_results"]["majority_baseline"]
    tf = results["comparative_classification_results"]["tfidf_baseline"]
    prop = results["comparative_classification_results"]["proposed_system"]
    ret = results["retrieval_evaluation"]
    esc = results["escalation_safety_evaluation"]
    jdg = results["llm_judge_evaluation"]
    agr = results["human_agreement_evaluation"]
    fails = results["top_5_real_failures"]
    
    content = f"""# Comprehensive Evaluation Benchmark Report

## 1. Executive Summary Table

| Metric | Majority Baseline | TF-IDF Baseline | Proposed AI System |
| :--- | :--- | :--- | :--- |
| **Intent Accuracy** | {maj['accuracy']:.4f} | {tf['accuracy']:.4f} | **{prop['accuracy']:.4f}** |
| **Macro Precision** | {maj['macro_precision']:.4f} | {tf['macro_precision']:.4f} | **{prop['macro_precision']:.4f}** |
| **Macro Recall** | {maj['macro_recall']:.4f} | {tf['macro_recall']:.4f} | **{prop['macro_recall']:.4f}** |
| **Macro F1 Score** | {maj['macro_f1']:.4f} | {tf['macro_f1']:.4f} | **{prop['macro_f1']:.4f}** |

---

## 2. Historical Retrieval Performance

- **Recall@1**: `{ret['recall_at_1']:.4f}`
- **Recall@3**: `{ret['recall_at_3']:.4f}`
- **Recall@5**: `{ret['recall_at_5']:.4f}`
- **Mean Reciprocal Rank (MRR)**: `{ret['mrr']:.4f}`
- **Mean Top-1 Cosine Similarity**: `{ret['mean_top1_similarity']:.4f}`

---

## 3. Escalation Safety & Conservative Automation

- **Automation Coverage**: `{esc['automation_coverage']*100:.2f}%` ({esc['correct_auto_handles'] + esc['false_auto_handles']} of {esc['total_eval_cases']} queries automated)
- **Auto-Handle Precision**: `{esc['auto_handle_precision']*100:.2f}%`
- **False-Auto Rate (Critical Safety Risk)**: `{esc['false_auto_rate']*100:.2f}%`
- **Escalation Recall**: `{esc['escalation_recall']*100:.2f}%` ({esc['correct_escalations']} out of {esc['correct_escalations'] + esc['false_auto_handles']} high-risk queries safely escalated)

---

## 4. LLM-as-a-Judge & Human Agreement

- **Overall Judge Pass Rate**: `{jdg['pass_rate']}%`
- **Mean Correctness Score**: `{jdg['mean_correctness']} / 5.0`
- **Mean Groundedness Score**: `{jdg['mean_groundedness']} / 5.0`
- **Mean Helpfulness Score**: `{jdg['mean_helpfulness']} / 5.0`
- **Mean Unsupported Claims Score**: `{jdg['mean_unsupported_claims']} / 5.0`
- **Mean Tone Score**: `{jdg['mean_tone']} / 5.0`

### Inter-Rater Reliability (Human vs. Judge)
- **Pearson Correlation ($r$)**: `{agr['pearson_correlation_r']:.4f}`
- **Spearman Correlation ($\\rho$)**: `{agr['spearman_correlation_rho']:.4f}`
- **Cohen's Kappa ($\\kappa$)**: `{agr['cohen_kappa_verdict']:.4f}`
- **Exact Verdict Agreement**: `{agr['verdict_agreement_percentage']}%`
- **Mean Absolute Error (MAE)**: `{agr['mean_absolute_error']:.4f}`

---

## 5. Top 5 Real Failure Modes

"""
    for i, f_mode in enumerate(fails, 1):
        content += f"""### Failure Mode {i}: {f_mode['failure_mode']}
- **Customer Inquiry**: `"{f_mode['real_example']}"`
- **Expected**: Intent = `{f_mode['expected_intent']}`, Decision = `{f_mode['expected_decision']}`
- **Actual**: Intent = `{f_mode['actual_intent']}`, Decision = `{f_mode['actual_decision']}`
- **Root Cause**: {f_mode['why_it_failed']}
- **Hypothesis**: {f_mode['hypothesis']}
- **Proposed Improvement**: {f_mode['potential_improvement']}

"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    run_full_evaluation()
