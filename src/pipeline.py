"""
End-to-End Support Agent Pipeline.

Integrates:
1. Intent Classification (Hybrid Calibrated)
2. Historical Resolution Retrieval (Traceable Vector Index)
3. Grounded Reply Drafting (Anti-Hallucination Guardrails)
4. Safe Multi-Gate Escalation (Deterministic Audit Trails)
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json

from src.config import SELECTED_BRAND, CONVERSATIONS_PATH, DATA_DIR
from src.data.cleaner import clean_tweet_text
from src.classification.classifier import HybridIntentClassifier, IntentPrediction
from src.retrieval.retriever import HistoricalRetriever, RetrievalResult
from src.retrieval.index import load_retrieval_index
from src.generation.generator import GroundedResponseGenerator, GroundedResponse
from src.decision.escalation import EscalationEngine, DecisionResult

@dataclass
class PipelineOutput:
    customer_message: str
    cleaned_message: str
    predicted_intent: str
    intent_confidence: float
    intent_margin: float
    is_ambiguous: bool
    risk_level: str
    historical_evidence: List[Dict[str, Any]]
    draft_reply: str
    evidence_ids: List[str]
    grounding_score: float
    decision: str  # "AUTO_HANDLE" or "ESCALATE_TO_HUMAN"
    escalation_reason: str
    is_safe_to_automate: bool
    passed_gates: List[str]
    failed_gates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_message": self.customer_message,
            "cleaned_message": self.cleaned_message,
            "predicted_intent": self.predicted_intent,
            "intent_confidence": self.intent_confidence,
            "intent_margin": self.intent_margin,
            "is_ambiguous": self.is_ambiguous,
            "risk_level": self.risk_level,
            "historical_evidence": self.historical_evidence,
            "draft_reply": self.draft_reply,
            "evidence_ids": self.evidence_ids,
            "grounding_score": self.grounding_score,
            "decision": self.decision,
            "escalation_reason": self.escalation_reason,
            "is_safe_to_automate": self.is_safe_to_automate,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates
        }

class SupportAgentPipeline:
    """
    Production-ready Customer Support AI Pipeline.
    """
    def __init__(self, retriever: Optional[HistoricalRetriever] = None, classifier: Optional[HybridIntentClassifier] = None):
        print(f"Initializing Support Agent Pipeline for {SELECTED_BRAND}...")
        
        # 1. Load or initialize Retriever
        self.retriever = retriever or load_retrieval_index()
        
        # 2. Load or train Classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = self._init_classifier()
            
        # 3. Initialize Generator & Escalation Engine
        self.generator = GroundedResponseGenerator()
        self.escalation_engine = EscalationEngine()
        print("Pipeline initialized successfully!")

    def _init_classifier(self) -> HybridIntentClassifier:
        clf = HybridIntentClassifier()
        # Train on non-held-out conversation pool
        held_out_path = DATA_DIR / "held_out_evaluation_ids.json"
        held_out_ids = set()
        if held_out_path.exists():
            with open(held_out_path, "r", encoding="utf-8") as f:
                held_out_ids = set(json.load(f))
                
        train_texts = []
        train_labels = []
        
        # Use keyword matching to bootstrap training labels from high-confidence subset
        import re
        with open(DATA_DIR / "intents.yaml", "r", encoding="utf-8") as f:
            import yaml
            taxonomy = yaml.safe_load(f)["intents"]
            
        keyword_map = {
            item["name"]: [w.lower() for w in item.get("examples", [])]
            for item in taxonomy
        }
        # Add core keywords
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
        
        with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if t["conversation_id"] in held_out_ids:
                    continue
                issue = t.get("customer_cleaned_issue", "")
                if len(issue.split()) < 3:
                    continue
                issue_lower = issue.lower()
                matched = []
                for iname, kws in core_kws.items():
                    if any(re.search(rf"\b{re.escape(k)}\b", issue_lower) for k in kws):
                        matched.append(iname)
                if len(matched) == 1:
                    train_texts.append(issue)
                    train_labels.append(matched[0])
                    if len(train_texts) >= 15000:
                        break
                        
        print(f"Fitting intent classifier on {len(train_texts):,} reference examples...")
        clf.fit(train_texts, train_labels)
        return clf

    def process(self, customer_message: str, conversation_context: str = "") -> PipelineOutput:
        """
        Executes the full 4-stage pipeline for a customer inquiry.
        """
        cleaned_msg = clean_tweet_text(customer_message, strip_brand_mentions=True)
        
        # Stage 1: Intent Classification
        intent_pred = self.classifier.predict(cleaned_msg)
        
        # Stage 2: Historical Resolution Retrieval
        evidence_results = self.retriever.search(
            query=cleaned_msg,
            top_k=3,
            min_similarity=0.0
        )
        
        # Stage 3: Grounded Response Generation
        grounded_resp = self.generator.generate_reply(
            customer_message=cleaned_msg,
            predicted_intent=intent_pred.intent,
            evidence_results=evidence_results,
            conversation_context=conversation_context
        )
        
        # Stage 4: Multi-Gate Escalation Decision
        decision_result = self.escalation_engine.evaluate(
            intent_pred=intent_pred,
            evidence_results=evidence_results,
            grounded_resp=grounded_resp
        )
        
        evidence_dicts = [r.to_dict() for r in evidence_results]
        
        return PipelineOutput(
            customer_message=customer_message,
            cleaned_message=cleaned_msg,
            predicted_intent=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            intent_margin=intent_pred.margin,
            is_ambiguous=intent_pred.is_ambiguous,
            risk_level=intent_pred.risk_level,
            historical_evidence=evidence_dicts,
            draft_reply=grounded_resp.reply,
            evidence_ids=grounded_resp.evidence_ids,
            grounding_score=grounded_resp.grounding_score,
            decision=decision_result.decision,
            escalation_reason=decision_result.reason,
            is_safe_to_automate=decision_result.is_safe_to_automate,
            passed_gates=decision_result.passed_gates,
            failed_gates=decision_result.failed_gates
        )

def run_interactive_cli():
    print("=" * 80)
    print(f"HIVERT SUPPORT AGENT — INTERACTIVE CLI ({SELECTED_BRAND})")
    print("=" * 80)
    print("Type a customer query or 'exit' to quit.\n")
    
    pipeline = SupportAgentPipeline()
    
    while True:
        try:
            user_input = input("\nCustomer Message > ")
            if not user_input.strip() or user_input.strip().lower() in ["exit", "quit", "q"]:
                break
                
            out = pipeline.process(user_input)
            
            print("\n" + "-" * 60)
            print(f"INTENT         : {out.predicted_intent} (Confidence: {out.intent_confidence:.2f}, Risk: {out.risk_level.upper()})")
            print(f"DECISION       : {out.decision}")
            print(f"REASON         : {out.escalation_reason}")
            print(f"\nDRAFT REPLY    : {out.draft_reply}")
            print(f"EVIDENCE IDS   : {out.evidence_ids} (Top Similarity: {out.grounding_score:.2f})")
            print("-" * 60)
            
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    run_interactive_cli()
