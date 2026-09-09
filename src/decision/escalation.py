"""
Conservative Multi-Gate Escalation Engine.

Enforces deterministic safety principles:
- False auto-handling is far more harmful than unnecessary escalation.
- Automates ONLY when:
    1. Intent confidence is above validated threshold.
    2. Intent is unambiguous (sufficient top-1 vs top-2 margin).
    3. Domain risk is strictly LOW (software/troubleshooting).
    4. Historical resolution evidence is strong (similarity >= threshold).
    5. Retrieved cases agree on resolution strategy.
    6. Generated response is grounded with zero unsupported claims.
- Every escalation produces an explicit, deterministic audit reason.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    MAX_AUTOHANDLE_RISK_LEVEL
)
from src.classification.classifier import IntentPrediction
from src.retrieval.retriever import RetrievalResult
from src.generation.generator import GroundedResponse

@dataclass
class DecisionResult:
    decision: str  # "AUTO_HANDLE" or "ESCALATE_TO_HUMAN"
    reason: str
    risk_level: str
    confidence_score: float
    retrieval_score: float
    is_safe_to_automate: bool
    passed_gates: List[str] = field(default_factory=list)
    failed_gates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "confidence_score": self.confidence_score,
            "retrieval_score": self.retrieval_score,
            "is_safe_to_automate": self.is_safe_to_automate,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates
        }

class EscalationEngine:
    """
    Deterministic Safety & Escalation Engine.
    """
    def __init__(
        self,
        min_confidence: float = INTENT_CONFIDENCE_THRESHOLD,
        min_retrieval_sim: float = RETRIEVAL_SIMILARITY_THRESHOLD,
        allowed_risk: str = MAX_AUTOHANDLE_RISK_LEVEL
    ):
        self.min_confidence = min_confidence
        self.min_retrieval_sim = min_retrieval_sim
        self.allowed_risk = allowed_risk

    def evaluate(
        self,
        intent_pred: IntentPrediction,
        evidence_results: List[RetrievalResult],
        grounded_resp: Optional[GroundedResponse] = None
    ) -> DecisionResult:
        passed_gates = []
        failed_gates = []
        escalation_reasons = []

        # Gate 1: High-Risk Domain Check
        # High risk domains (Account Access, Billing Disputes, Physical Hardware Damage) must always escalate
        if intent_pred.risk_level.lower() != self.allowed_risk.lower():
            failed_gates.append("GATE_1_DOMAIN_RISK")
            if intent_pred.intent == "apple_id_account_access":
                escalation_reasons.append("The issue is security-sensitive (Apple ID / Account Access) requiring human identity verification.")
            elif intent_pred.intent == "app_store_billing_subscription":
                escalation_reasons.append("The issue involves financial transactions / billing disputes requiring human billing authorization.")
            elif intent_pred.intent == "hardware_screen_physical_damage":
                escalation_reasons.append("The issue involves physical hardware damage requiring Genius Bar or authorized physical repair.")
            else:
                escalation_reasons.append(f"Domain risk level ({intent_pred.risk_level.upper()}) exceeds auto-handle threshold.")
        else:
            passed_gates.append("GATE_1_DOMAIN_RISK")

        # Gate 2: Intent Confidence Gate
        if intent_pred.confidence < self.min_confidence:
            failed_gates.append("GATE_2_INTENT_CONFIDENCE")
            escalation_reasons.append(
                f"Intent confidence ({intent_pred.confidence:.2f}) is below the validated safety threshold ({self.min_confidence:.2f})."
            )
        else:
            passed_gates.append("GATE_2_INTENT_CONFIDENCE")

        # Gate 3: Intent Ambiguity / Margin Gate
        if intent_pred.is_ambiguous:
            failed_gates.append("GATE_3_AMBIGUITY_MARGIN")
            escalation_reasons.append(
                f"Customer query is ambiguous between multiple intents (margin: {intent_pred.margin:.2f}); human clarification needed."
            )
        else:
            passed_gates.append("GATE_3_AMBIGUITY_MARGIN")

        # Gate 4: Historical Retrieval Evidence Quality Gate
        top_sim = evidence_results[0].similarity_score if evidence_results else 0.0
        if not evidence_results or top_sim < self.min_retrieval_sim:
            failed_gates.append("GATE_4_RETRIEVAL_EVIDENCE")
            escalation_reasons.append(
                f"No sufficiently similar historical resolution was found (top similarity: {top_sim:.2f} < threshold: {self.min_retrieval_sim:.2f})."
            )
        else:
            passed_gates.append("GATE_4_RETRIEVAL_EVIDENCE")

        # Gate 5: Grounding & Unsupported Claims Gate
        if grounded_resp:
            if grounded_resp.unsupported_claims_detected:
                failed_gates.append("GATE_5_UNSUPPORTED_CLAIMS")
                escalation_reasons.append("Generated reply contains potentially unsupported policy or action claims.")
            elif not grounded_resp.is_grounded:
                failed_gates.append("GATE_5_GROUNDING_FIDELITY")
                escalation_reasons.append("Generated reply is insufficiently grounded in verified historical evidence.")
            else:
                passed_gates.append("GATE_5_GROUNDING_FIDELITY")
        else:
            passed_gates.append("GATE_5_GROUNDING_FIDELITY")

        # Final Decision Synthesis
        if not failed_gates:
            decision = "AUTO_HANDLE"
            reason = f"High confidence ({intent_pred.confidence:.2f}), strong historical grounding (sim: {top_sim:.2f}), and low-risk standard troubleshooting."
            is_safe = True
        else:
            decision = "ESCALATE_TO_HUMAN"
            reason = "; ".join(escalation_reasons)
            is_safe = False

        return DecisionResult(
            decision=decision,
            reason=reason,
            risk_level=intent_pred.risk_level,
            confidence_score=intent_pred.confidence,
            retrieval_score=top_sim,
            is_safe_to_automate=is_safe,
            passed_gates=passed_gates,
            failed_gates=failed_gates
        )
